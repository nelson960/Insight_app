from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from backend.core.workspace import get_workspace
from backend.services.planner.agent_tools import rg_search_in_file, read_raw_window
from backend.services.security import KeyManager, decrypt_bytes
from backend.services.storage.sqlite_store import SQLiteMetadataStore

logger = logging.getLogger(__name__)


def _approx_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def _uniq_str(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for v in values:
        s = (v or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "from",
    "this",
    "that",
    "these",
    "those",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "it",
    "as",
    "at",
    "by",
    "about",
    "into",
    "what",
    "why",
    "how",
    "when",
    "where",
    "which",
    "can",
    "could",
    "should",
    "would",
}


def _extract_terms(query: str, *, limit: int = 8) -> list[str]:
    """
    Cheap, deterministic term extraction for raw logs/text.

    Prefer:
    - quoted strings ("ErrorCode 42")
    - id-like tokens (digits / underscores / dashes)
    - a few longer keywords (non-stopwords)
    """
    q = (query or "").strip()
    if not q:
        return []

    out: list[str] = []

    # Quoted phrases first.
    for a, b in re.findall(r"\"([^\"]{2,80})\"|'([^']{2,80})'", q):
        term = (a or b or "").strip()
        if term:
            out.append(term)
        if len(out) >= limit:
            return _uniq_str(out)[:limit]

    # ID-like tokens.
    for tok in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]{3,80}", q):
        if any(ch.isdigit() for ch in tok) or ("_" in tok) or ("-" in tok):
            out.append(tok)
        if len(out) >= limit:
            return _uniq_str(out)[:limit]

    # Fallback keywords.
    for tok in re.findall(r"[A-Za-z][A-Za-z]{4,80}", q):
        t = tok.lower()
        if t in _STOPWORDS:
            continue
        out.append(tok)
        if len(out) >= limit:
            break

    return _uniq_str(out)[:limit]


@dataclass(frozen=True)
class RawLargeAgentConfig:
    """
    Budget knobs for raw-large file analysis (Plan B).
    """

    context_lines: int = 30
    max_patterns: int = 6
    max_results_per_pattern: int = 80
    max_windows: int = 10
    max_bytes_per_window: int = 160_000


class RawLargeAgentLoop:
    """
    Plan B: raw text/log analysis using ripgrep + line-window reads.

    This path is only used for files marked policy.raw_large=True (no ingestion).
    """

    def __init__(self, *, metadata_store: SQLiteMetadataStore) -> None:
        self._store = metadata_store

    def _resolve_plaintext_path(self, *, file_id: str, record: dict[str, Any]) -> Optional[Path]:
        """
        Ensure a plaintext copy exists in the workspace cache (so rg can read it).

        - Preferred: `storage/cache/{file_id}{suffix}` written at upload time.
        - Fallback: decrypt `stored_path` into cache on demand (best-effort).
        """
        filename = str(record.get("filename") or "")
        suffix = Path(filename).suffix or ".txt"
        workspace = get_workspace()
        cache_path = workspace.cache / f"{file_id}{suffix}"
        if cache_path.exists():
            return cache_path

        stored_path = str(record.get("stored_path") or "")
        is_encrypted = bool(int(record.get("is_encrypted") or 0)) if record.get("is_encrypted") is not None else False
        if not stored_path or not is_encrypted:
            p = Path(stored_path)
            return p if p.exists() else None

        try:
            key = KeyManager(workspace).get_key()
            encrypted = Path(stored_path).read_bytes()
            plaintext = decrypt_bytes(key, encrypted)
            cache_path.write_bytes(plaintext)
            return cache_path
        except Exception as exc:
            logger.info("raw_large decrypt failed file_id=%s err=%s", file_id, exc)
            return None

    def build_evidence_hits(
        self,
        query: str,
        *,
        file_id: str,
        request_id: Optional[str] = None,
        config: Optional[RawLargeAgentConfig] = None,
    ) -> list[dict[str, Any]]:
        cfg = config or RawLargeAgentConfig()
        rec = self._store.get_file(file_id) or {}
        filename = str(rec.get("filename") or "Document").strip() or "Document"

        path = self._resolve_plaintext_path(file_id=file_id, record=rec)
        if not path:
            logger.info("raw_large no plaintext path file_id=%s request_id=%s", file_id, request_id)
            return []

        terms = _extract_terms(query, limit=int(cfg.max_patterns))
        if not terms:
            terms = [(query or "").strip()][:1]
        terms = [t for t in terms if t]

        all_hits: list[tuple[int, str]] = []  # (line, text)
        for term in terms[: max(1, int(cfg.max_patterns))]:
            hits = rg_search_in_file(str(path), term, max_results=int(cfg.max_results_per_pattern))
            for h in hits:
                if not h.text:
                    continue
                all_hits.append((int(h.line), h.text))

        # De-dupe by line number (keep first).
        deduped: list[tuple[int, str]] = []
        seen_lines: set[int] = set()
        for line_no, text in all_hits:
            if line_no in seen_lines:
                continue
            seen_lines.add(line_no)
            deduped.append((line_no, text))
            if len(deduped) >= int(cfg.max_windows) * 2:
                break

        # Build merged windows around match lines.
        context = max(2, int(cfg.context_lines))
        ranges: list[tuple[int, int]] = []
        for line_no, _ in deduped:
            start = max(1, int(line_no) - context)
            end = int(line_no) + context
            ranges.append((start, end))
            if len(ranges) >= int(cfg.max_windows) * 2:
                break
        ranges.sort(key=lambda x: (x[0], x[1]))
        merged: list[tuple[int, int]] = []
        for s, e in ranges:
            if not merged:
                merged.append((s, e))
                continue
            ps, pe = merged[-1]
            if s <= pe + 1:
                merged[-1] = (ps, max(pe, e))
            else:
                merged.append((s, e))

        hits_out: list[dict[str, Any]] = []
        for s, e in merged[: max(1, int(cfg.max_windows))]:
            window = read_raw_window(
                str(path),
                line_start=s,
                line_end=e,
                max_bytes=int(cfg.max_bytes_per_window),
            )
            if not window:
                continue
            text = f"Lines {s}-{e}:\n{window}".strip()
            hits_out.append(
                {
                    "doc_id": file_id,
                    "filename": filename,
                    "text": text,
                    "chunk_id": f"raw_window:{s}-{e}",
                    "score": 1.0,
                }
            )

        total_tokens = sum(_approx_tokens(h.get("text") or "") for h in hits_out)
        logger.debug(
            "raw_large windows query_len=%d file=%s terms=%d hits=%d windows=%d tokens=%d request_id=%s",
            len((query or "").strip()),
            filename,
            len(terms),
            len(deduped),
            len(hits_out),
            total_tokens,
            request_id,
        )
        return hits_out


__all__ = ["RawLargeAgentConfig", "RawLargeAgentLoop"]

