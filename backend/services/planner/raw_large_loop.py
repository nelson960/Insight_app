from __future__ import annotations

import logging
import re
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from backend.core.workspace import get_workspace
from backend.services.planner.agent_tools import rg_search_in_file, read_raw_window
from backend.services.extraction import create_extraction_service
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


def _extract_term_sets(query: str, *, limit: int = 8) -> tuple[list[str], list[str], list[str], list[str]]:
    """
    Return (phrases, id_terms, keywords, search_terms).
    """
    q = (query or "").strip()
    if not q:
        return [], [], [], []

    phrases: list[str] = []
    for a, b in re.findall(r"\"([^\"]{2,80})\"|'([^']{2,80})'", q):
        term = (a or b or "").strip()
        if term:
            phrases.append(term)

    id_terms: list[str] = []
    for tok in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]{3,80}", q):
        if any(ch.isdigit() for ch in tok) or ("_" in tok) or ("-" in tok):
            id_terms.append(tok)

    keywords: list[str] = []
    for tok in re.findall(r"[A-Za-z][A-Za-z]{4,80}", q):
        t = tok.lower()
        if t in _STOPWORDS:
            continue
        keywords.append(tok)

    phrases = _uniq_str(phrases)
    id_terms = _uniq_str(id_terms)
    keywords = _uniq_str(keywords)
    search_terms = _uniq_str(phrases + id_terms + keywords)[:limit]
    return phrases, id_terms, keywords, search_terms


def _rg_pattern(term: str) -> str:
    if not term:
        return ""
    # Escape regex metacharacters for phrases / literal tokens.
    if re.search(r"\\s", term) or re.search(r"[\\.^$*+?{}\\[\\]|()]", term):
        return re.escape(term)
    return term


def _score_window(
    text: str,
    *,
    phrases: list[str],
    id_terms: list[str],
    keywords: list[str],
) -> tuple[float, int, int, int]:
    if not text:
        return 0.0, 0, 0, 0
    lower = text.lower()
    phrase_hits = sum(1 for p in phrases if p.lower() in lower)
    id_hits = sum(1 for t in id_terms if t.lower() in lower)
    keyword_hits = sum(1 for k in keywords if k.lower() in lower)
    line_count = max(1, len(text.splitlines()))
    density = (phrase_hits * 3 + id_hits * 2 + keyword_hits) / float(line_count)
    score = phrase_hits * 5.0 + id_hits * 3.0 + keyword_hits * 1.0 + density
    return score, phrase_hits, id_hits, keyword_hits


def _build_overview_hits(
    *,
    path: Path,
    file_id: str,
    filename: str,
    cfg: RawLargeAgentConfig,
) -> list[dict[str, Any]]:
    """
    Build a single "overview" window from the start of the file.
    Used when the query yields no matches so the model can still
    answer "what is this file about" from actual text.
    """
    if not path.exists():
        return []
    overview_lines = max(60, int(cfg.context_lines) * 4)
    window = read_raw_window(
        str(path),
        line_start=1,
        line_end=overview_lines,
        max_bytes=int(cfg.max_bytes_per_window),
    )
    if not window:
        return []
    text = f"Lines 1-{overview_lines}:\n{window}".strip()
    return [
        {
            "doc_id": file_id,
            "filename": filename,
            "text": text,
            "chunk_id": f"raw_overview:1-{overview_lines}",
            "score": 0.1,
            "line_start": 1,
            "line_end": overview_lines,
            "overview": True,
        }
    ]


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
    neighbor_windows: int = 1


def _parse_iso_ts(value: Optional[str]) -> Optional[float]:
    if not value or not isinstance(value, str):
        return None
    try:
        v = value.replace("Z", "+00:00")
        return datetime.fromisoformat(v).timestamp()
    except Exception:
        return None


def _text_cache_is_fresh(cache_path: Path, record: dict[str, Any]) -> bool:
    if not cache_path.exists():
        return False
    try:
        cache_mtime = cache_path.stat().st_mtime
    except Exception:
        return False
    # Prefer DB updated_at if available.
    updated_at = record.get("updated_at") or record.get("created_at")
    ts = _parse_iso_ts(updated_at) if isinstance(updated_at, str) else None
    if ts is not None:
        return cache_mtime >= ts
    # Fallback: compare with stored path mtime if present.
    stored_path = str(record.get("stored_path") or "")
    if stored_path:
        try:
            src_mtime = Path(stored_path).stat().st_mtime
            return cache_mtime >= src_mtime
        except Exception:
            pass
    return True


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

        - Preferred: `storage/cache/{file_id}.txt` if available, else `{file_id}{suffix}`.
        - Fallback: decrypt `stored_path` into cache on demand (best-effort).
        """
        filename = str(record.get("filename") or "")
        suffix = Path(filename).suffix or ".txt"
        workspace = get_workspace()
        text_cache = workspace.cache / f"{file_id}.txt"
        if _text_cache_is_fresh(text_cache, record):
            return text_cache
        cache_path = workspace.cache / f"{file_id}{suffix}"
        if cache_path.exists() and _text_cache_is_fresh(cache_path, record):
            # If we somehow cached plaintext under a suffix, prefer it as a fallback.
            return cache_path if cache_path.suffix.lower() in {".txt", ".log", ".json", ".md", ".csv", ".tsv"} else text_cache

        stored_path = str(record.get("stored_path") or "")
        is_encrypted_raw = record.get("is_encrypted")
        if is_encrypted_raw is None:
            is_encrypted = False
        else:
            try:
                is_encrypted = bool(int(is_encrypted_raw))
            except (ValueError, TypeError):
                is_encrypted = False
        # Try to rebuild plaintext cache if missing/stale.
        # First: use any extracted text stored in SQLite (if available).
        try:
            file_text = self._store.get_file_text(file_id)
            plain = (file_text or {}).get("plain_text") if isinstance(file_text, dict) else None
            if isinstance(plain, str) and plain.strip():
                text_cache.write_text(plain, encoding="utf-8", errors="ignore")
                return text_cache
        except Exception:
            pass

        if not stored_path:
            return None

        source_path: Optional[Path] = None
        if not is_encrypted:
            p = Path(stored_path)
            source_path = p if p.exists() else None
        else:
            try:
                key = KeyManager(workspace).get_key()
                encrypted = Path(stored_path).read_bytes()
                plaintext = decrypt_bytes(key, encrypted)
                cache_path.write_bytes(plaintext)
                source_path = cache_path
            except Exception as exc:
                logger.info("raw_large decrypt failed file_id=%s err=%s", file_id, exc)
                source_path = None

        if source_path is None or not source_path.exists():
            return None

        # If the source is already plain text, cache it and return.
        text_suffixes = {".txt", ".log", ".json", ".md", ".csv", ".tsv", ".yaml", ".yml"}
        mime = str(record.get("mime") or "")
        if mime.startswith("text/") or source_path.suffix.lower() in text_suffixes:
            try:
                text_cache.write_bytes(source_path.read_bytes())
                return text_cache
            except Exception:
                pass

        # Otherwise, extract to plaintext cache using the extractor pipeline.
        try:
            service = create_extraction_service()
            result = service.extract(source_path, mime_type=mime or None)
            text_cache.write_text(result.text or "", encoding="utf-8", errors="ignore")
            return text_cache
        except Exception as exc:
            logger.info("raw_large extract failed file_id=%s err=%s", file_id, exc)
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
            logger.warning("raw_large no plaintext path file_id=%s request_id=%s", file_id, request_id)
            return []

        phrases, id_terms, keywords, terms = _extract_term_sets(query, limit=int(cfg.max_patterns))
        if not terms:
            query_clean = (query or "").strip()
            terms = [query_clean] if query_clean else []
        terms = [t for t in terms if t]
        if not terms:
            logger.info("raw_large no searchable terms; using overview file_id=%s request_id=%s", file_id, request_id)
            return _build_overview_hits(path=path, file_id=file_id, filename=filename, cfg=cfg)

        all_hits: list[tuple[int, str]] = []  # (line, text)
        for term in terms[: max(1, int(cfg.max_patterns))]:
            pat = _rg_pattern(term)
            if not pat:
                continue
            hits = rg_search_in_file(str(path), pat, max_results=int(cfg.max_results_per_pattern))
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

        windows: list[dict[str, Any]] = []
        for s, e in merged:
            window = read_raw_window(
                str(path),
                line_start=s,
                line_end=e,
                max_bytes=int(cfg.max_bytes_per_window),
            )
            if not window:
                continue
            score, phrase_hits, id_hits, keyword_hits = _score_window(
                window,
                phrases=phrases,
                id_terms=id_terms,
                keywords=keywords,
            )
            text = f"Lines {s}-{e}:\n{window}".strip()
            windows.append(
                {
                    "doc_id": file_id,
                    "filename": filename,
                    "text": text,
                    "chunk_id": f"raw_window:{s}-{e}",
                    "score": float(score),
                    "line_start": s,
                    "line_end": e,
                    "phrase_hits": phrase_hits,
                    "id_hits": id_hits,
                    "keyword_hits": keyword_hits,
                }
            )

        windows.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
        base_limit = max(1, int(cfg.max_windows))
        selected = windows[:base_limit]

        # Neighbor expansion (context continuity). Add a limited number of adjacent windows.
        neighbor_budget = max(0, int(cfg.neighbor_windows or 0))
        neighbor_added_before = 0
        neighbor_added_after = 0
        seen_ranges = {(int(w.get("line_start")), int(w.get("line_end"))) for w in selected}
        span = max(2, int(cfg.context_lines)) * 2
        for w in list(selected):
            if neighbor_added_before >= neighbor_budget and neighbor_added_after >= neighbor_budget:
                break
            s = int(w.get("line_start") or 0)
            e = int(w.get("line_end") or 0)
            if neighbor_added_before < neighbor_budget and s > 1:
                ns = max(1, s - span)
                ne = max(1, s - 1)
                if ne >= ns and (ns, ne) not in seen_ranges:
                    window = read_raw_window(str(path), line_start=ns, line_end=ne, max_bytes=int(cfg.max_bytes_per_window))
                    if window:
                        text = f"Lines {ns}-{ne}:\n{window}".strip()
                        selected.append(
                            {
                                "doc_id": file_id,
                                "filename": filename,
                                "text": text,
                                "chunk_id": f"raw_window:{ns}-{ne}",
                                "score": 0.0,
                                "line_start": ns,
                                "line_end": ne,
                                "neighbor": True,
                            }
                        )
                        seen_ranges.add((ns, ne))
                        neighbor_added_before += 1
            if neighbor_added_after < neighbor_budget:
                ns = e + 1
                ne = e + span
                if ne >= ns and (ns, ne) not in seen_ranges:
                    window = read_raw_window(str(path), line_start=ns, line_end=ne, max_bytes=int(cfg.max_bytes_per_window))
                    if window:
                        text = f"Lines {ns}-{ne}:\n{window}".strip()
                        selected.append(
                            {
                                "doc_id": file_id,
                                "filename": filename,
                                "text": text,
                                "chunk_id": f"raw_window:{ns}-{ne}",
                                "score": 0.0,
                                "line_start": ns,
                                "line_end": ne,
                                "neighbor": True,
                            }
                        )
                        seen_ranges.add((ns, ne))
                        neighbor_added_after += 1

        hits_out = selected[: max(1, int(cfg.max_windows))]
        if not hits_out:
            logger.info("raw_large no hits; using overview file_id=%s request_id=%s", file_id, request_id)
            return _build_overview_hits(path=path, file_id=file_id, filename=filename, cfg=cfg)
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


def resolve_raw_large_plaintext_path(
    store: SQLiteMetadataStore,
    file_id: str,
    record: dict[str, Any],
) -> Optional[Path]:
    """
    Resolve (and if needed rebuild) the plaintext cache path for a raw_large file.
    """
    loop = RawLargeAgentLoop(metadata_store=store)
    return loop._resolve_plaintext_path(file_id=file_id, record=record)


__all__ = ["RawLargeAgentConfig", "RawLargeAgentLoop", "resolve_raw_large_plaintext_path"]
