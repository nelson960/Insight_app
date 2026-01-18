from __future__ import annotations

import json
import logging
import math
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from backend.services.retrieval.rag_store import RagStore
from backend.services.storage.sqlite_store import SQLiteMetadataStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RgHit:
    path: str
    line: int
    text: str


def _rg_binary() -> str:
    return os.environ.get("RG_BIN") or "rg"


def rg_search(
    roots: Sequence[str],
    pattern: str,
    *,
    glob: Optional[str] = None,
    max_results: int = 500,
) -> list[RgHit]:
    """
    Fast exact/regex search using ripgrep.

    Returns match line hits only; callers should use `read_raw_window()` if they
    need context lines around a match.
    """
    pat = (pattern or "").strip()
    if not pat:
        return []
    max_results = max(1, min(int(max_results or 500), 5000))

    args: list[str] = [
        _rg_binary(),
        "--json",
        "--no-config",
        "--smart-case",
        "--hidden",
        "--no-heading",
        "--max-count",
        str(max_results),
        pat,
    ]
    if glob:
        args.extend(["--glob", str(glob)])
    args.extend([str(r) for r in roots if str(r).strip()])

    try:
        proc = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=5.0,
        )
    except Exception as exc:
        logger.debug("rg_search failed: %s", exc)
        return []

    hits: list[RgHit] = []
    for line in (proc.stdout or "").splitlines():
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if obj.get("type") != "match":
            continue
        data = obj.get("data") or {}
        path = (data.get("path") or {}).get("text")
        line_no = data.get("line_number")
        lines = (data.get("lines") or {}).get("text")
        if not isinstance(path, str) or not isinstance(line_no, int) or not isinstance(lines, str):
            continue
        text = lines.rstrip("\n")
        hits.append(RgHit(path=path, line=line_no, text=text))
        if len(hits) >= max_results:
            break
    return hits


def rg_search_in_file(path: str, pattern: str, *, max_results: int = 500) -> list[RgHit]:
    return rg_search([path], pattern, max_results=max_results)


def read_raw_window(
    path: str,
    *,
    line_start: int,
    line_end: int,
    max_bytes: int = 256_000,
) -> str:
    """
    Read a line-bounded window from a raw file (logs/text) without loading the full file.
    """
    p = Path(path)
    if not p.exists() or not p.is_file():
        return ""
    line_start = max(1, int(line_start))
    line_end = max(line_start, int(line_end))
    max_bytes = max(4096, int(max_bytes or 256_000))

    out_lines: list[str] = []
    total = 0
    try:
        with p.open("r", encoding="utf-8", errors="replace") as f:
            for i, raw in enumerate(f, start=1):
                if i < line_start:
                    continue
                if i > line_end:
                    break
                total += len(raw)
                if total > max_bytes:
                    out_lines.append("… (truncated)")
                    break
                out_lines.append(raw.rstrip("\n"))
    except Exception:
        logger.warning(
            "read_raw_window failed path=%s lines=%s-%s",
            path,
            line_start,
            line_end,
            exc_info=True,
        )
        return ""
    return "\n".join(out_lines).strip()


def _rank_score(rank: int) -> float:
    r = max(1, int(rank))
    return 1.0 / math.log(2.0 + r)


def _looks_like_exact_term(token: str) -> bool:
    if not token:
        return False
    if len(token) < 4:
        return False
    # IDs/codes often include digits, underscores, dashes.
    return bool(re.search(r"[0-9]", token)) or ("_" in token) or ("-" in token)


def _extract_exact_terms(query: str, *, limit: int = 8) -> list[str]:
    q = (query or "").strip()
    if not q:
        return []
    quoted = re.findall(r"\"([^\"]{2,80})\"|'([^']{2,80})'", q)
    out: list[str] = []
    for a, b in quoted:
        term = (a or b or "").strip()
        if term:
            out.append(term)
    if len(out) >= limit:
        return out[:limit]
    # Fallback: pull "id-like" tokens.
    for tok in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]{3,80}", q):
        if _looks_like_exact_term(tok):
            out.append(tok)
        if len(out) >= limit:
            break
    # Dedup, preserve order.
    seen = set()
    final: list[str] = []
    for t in out:
        if t in seen:
            continue
        seen.add(t)
        final.append(t)
    return final


def _approx_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


@dataclass(frozen=True)
class ChunkAnchor:
    chunk_id: str
    file_id: str
    seq: int
    filename: str
    score: float


@dataclass(frozen=True)
class SeqWindow:
    file_id: str
    filename: str
    seq_start: int
    seq_end: int
    text: str
    page_start: Optional[int]
    page_end: Optional[int]


class AgentToolbox:
    """
    Retrieval + coherence primitives for multi-file analysis.

    This module is *backend-only*. It does NOT expose tool calls to the model
    directly; orchestrator can call these helpers to build safer, more coherent
    ephemeral context packs.
    """

    def __init__(self, *, rag_store: RagStore, metadata_store: SQLiteMetadataStore) -> None:
        self._rag = rag_store
        self._store = metadata_store

    @property
    def rag_store(self) -> RagStore:
        return self._rag

    @property
    def metadata_store(self) -> SQLiteMetadataStore:
        return self._store

    # ------------------------------
    # Chunk retrieval (dense/sparse)
    # ------------------------------
    def vector_search(
        self,
        query: str,
        *,
        file_id: Optional[str],
        top_k: int,
        query_vector: Optional[Sequence[float]] = None,
    ) -> list[dict[str, Any]]:
        if query_vector is not None:
            if file_id:
                return self._rag.retrieve_with_vector(query_vector, chat_id=None, doc_ids=[file_id], top_k=top_k)
            return self._rag.retrieve_with_vector(query_vector, chat_id=None, doc_ids=None, top_k=top_k)
        if file_id:
            return self._rag.retrieve(query, chat_id=None, doc_ids=[file_id], top_k=top_k)
        return self._rag.retrieve(query, chat_id=None, doc_ids=None, top_k=top_k)

    def fts_search(
        self,
        query: str,
        *,
        file_id: Optional[str],
        top_k: int,
    ) -> list[dict[str, Any]]:
        try:
            return self._store.fts_search_chunks(query, file_id=file_id, top_k=top_k)
        except Exception:
            return []

    def hybrid_search(
        self,
        query: str,
        *,
        file_id: Optional[str],
        top_k: int,
        query_vector: Optional[Sequence[float]] = None,
        dense_k: Optional[int] = None,
        sparse_k: Optional[int] = None,
        w_dense: float = 0.6,
        w_sparse: float = 0.4,
    ) -> list[ChunkAnchor]:
        """
        Hybrid anchors: dense (Qdrant) + sparse (FTS) merged with rank-normalized scoring.
        """
        q = (query or "").strip()
        if not q:
            return []

        dense_k = int(dense_k or max(top_k, 12))
        sparse_k = int(sparse_k or max(top_k, 12))
        dense = self.vector_search(q, file_id=file_id, top_k=dense_k, query_vector=query_vector)
        sparse = self.fts_search(q, file_id=file_id, top_k=sparse_k)

        dense_rank: dict[str, int] = {}
        for i, h in enumerate(dense, start=1):
            cid = h.get("chunk_id")
            if isinstance(cid, str) and cid:
                dense_rank[cid] = i

        sparse_rank: dict[str, int] = {}
        for i, h in enumerate(sparse, start=1):
            cid = h.get("chunk_id")
            if isinstance(cid, str) and cid:
                sparse_rank[cid] = i

        # Prefer per-file anchors: merge by chunk_id, and use SQLite to fill seq/filename reliably.
        all_chunk_ids: list[str] = list({*dense_rank.keys(), *sparse_rank.keys()})
        logger.debug(
            "hybrid_search file_id=%s dense=%d sparse=%d merged=%d top_k=%d",
            file_id,
            len(dense_rank),
            len(sparse_rank),
            len(all_chunk_ids),
            top_k,
        )
        details = self._store.fetch_chunks(all_chunk_ids)
        by_id: dict[str, dict[str, Any]] = {str(d.get("id")): d for d in details if isinstance(d.get("id"), str)}

        exact_terms = _extract_exact_terms(q)

        scored: list[Tuple[float, ChunkAnchor]] = []
        for cid in all_chunk_ids:
            detail = by_id.get(cid) or {}
            fid = detail.get("file_id") if isinstance(detail.get("file_id"), str) else ""
            if file_id and fid and fid != file_id:
                continue
            filename = detail.get("filename") if isinstance(detail.get("filename"), str) else "Document"
            seq = int(detail.get("seq") or 0)
            # rank-based normalization
            s = 0.0
            if cid in dense_rank:
                s += float(w_dense) * _rank_score(dense_rank[cid])
            if cid in sparse_rank:
                s += float(w_sparse) * _rank_score(sparse_rank[cid])
            # cheap exact-term boosts using preview text (bounded)
            text = detail.get("text") if isinstance(detail.get("text"), str) else ""
            preview = text[:2000].lower() if text else ""
            for t in exact_terms:
                if t.lower() in preview:
                    s += 0.15
            if not fid:
                continue
            scored.append((s, ChunkAnchor(chunk_id=cid, file_id=fid, seq=seq, filename=filename, score=s)))

        scored.sort(key=lambda x: x[0], reverse=True)
        out: list[ChunkAnchor] = []
        for _, a in scored:
            out.append(a)
            if len(out) >= top_k:
                break
        return out

    # ------------------------------
    # Ordered windows (walk by seq)
    # ------------------------------
    def get_seq_window(self, file_id: str, *, center_seq: int, radius: int) -> SeqWindow:
        radius = max(0, int(radius))
        center_seq = max(0, int(center_seq))
        seq_start = max(0, center_seq - radius)
        seq_end = center_seq + radius
        chunks = self._store.fetch_chunks_by_seq_range(file_id, seq_start=seq_start, seq_end=seq_end)
        text_parts: list[str] = []
        page_start: Optional[int] = None
        page_end: Optional[int] = None
        filename = "Document"
        for row in chunks:
            if isinstance(row.get("filename"), str) and row.get("filename"):
                filename = row["filename"]
            t = row.get("text")
            if isinstance(t, str) and t.strip():
                text_parts.append(t.strip())
            meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            ps = meta.get("page_start")
            pe = meta.get("page_end")
            if isinstance(ps, int):
                page_start = ps if page_start is None else min(page_start, ps)
            if isinstance(pe, int):
                page_end = pe if page_end is None else max(page_end, pe)
        text = "\n\n".join(text_parts).strip()
        return SeqWindow(
            file_id=file_id,
            filename=filename,
            seq_start=seq_start,
            seq_end=seq_end,
            text=text,
            page_start=page_start,
            page_end=page_end,
        )

    @staticmethod
    def merge_windows(windows: Iterable[Tuple[int, int]]) -> list[Tuple[int, int]]:
        ranges: list[Tuple[int, int]] = []
        for a, b in windows:
            start = max(0, int(a))
            end = max(start, int(b))
            ranges.append((start, end))
        if not ranges:
            return []
        ranges.sort(key=lambda x: (x[0], x[1]))
        merged: list[Tuple[int, int]] = []
        cur_s, cur_e = ranges[0]
        for s, e in ranges[1:]:
            if s <= cur_e + 1:
                cur_e = max(cur_e, e)
                continue
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))
        return merged

    def windows_from_anchors(
        self,
        anchors: Sequence[ChunkAnchor],
        *,
        radius: int = 2,
        max_tokens_per_file: int = 1000,
    ) -> dict[str, list[SeqWindow]]:
        """
        Convert per-file anchors into merged, ordered seq-windows.
        """
        radius = max(0, int(radius))
        max_tokens_per_file = max(1, int(max_tokens_per_file))

        by_file: dict[str, list[ChunkAnchor]] = {}
        for a in anchors:
            by_file.setdefault(a.file_id, []).append(a)

        out: dict[str, list[SeqWindow]] = {}
        for fid, items in by_file.items():
            # Prefer a small set of diverse anchors per file so windows don't
            # merge into a single huge range (which then gets truncated and
            # loses the actual "anchor" content).
            items_sorted = sorted(items, key=lambda a: a.score, reverse=True)
            min_gap = max(1, (2 * radius) + 1)
            chosen: list[ChunkAnchor] = []
            for a in items_sorted:
                if any(abs(a.seq - b.seq) < min_gap for b in chosen):
                    continue
                chosen.append(a)
                if len(chosen) >= 8:
                    break
            if not chosen and items_sorted:
                chosen = items_sorted[:1]

            ranges: list[tuple[int, int]] = []
            for a in chosen:
                start = max(0, a.seq - radius)
                end = a.seq + radius
                ranges.append((start, end))
            merged_ranges = self.merge_windows(ranges)

            # Encourage multiple windows per file by capping how much budget a
            # single window may consume (unless there's only one window).
            target_windows = max(1, min(3, len(merged_ranges)))
            per_window_cap = max(200, int(math.ceil(max_tokens_per_file / target_windows)))

            windows: list[SeqWindow] = []
            used = 0
            for s, e in merged_ranges:
                remaining = max(0, max_tokens_per_file - used)
                if remaining <= 0:
                    break
                window_budget = min(remaining, per_window_cap)
                if window_budget <= 0:
                    break

                # Pick center as midpoint for page calculations; actual range fetch is [s,e]
                center = (s + e) // 2
                base_radius = max(e - center, center - s)

                # Prefer shrinking the seq radius (fewer chunks) rather than truncating text,
                # so we preserve the anchor chunk coherently.
                best: Optional[SeqWindow] = None
                for r in range(max(0, int(base_radius)), -1, -1):
                    cand = self.get_seq_window(fid, center_seq=center, radius=r)
                    if not cand.text:
                        continue
                    if _approx_tokens(cand.text) <= window_budget:
                        best = cand
                        break

                if best is None:
                    # Even the single-chunk window is too large; truncate to budget.
                    cand = self.get_seq_window(fid, center_seq=center, radius=0)
                    if not cand.text:
                        continue
                    max_chars = max(0, int(window_budget * 4))
                    text = cand.text
                    if len(text) > max_chars:
                        text = text[:max_chars].rstrip() + "\n… (truncated)"
                    best = SeqWindow(
                        file_id=cand.file_id,
                        filename=cand.filename,
                        seq_start=cand.seq_start,
                        seq_end=cand.seq_end,
                        text=text,
                        page_start=cand.page_start,
                        page_end=cand.page_end,
                    )

                tks = _approx_tokens(best.text)
                if tks <= 0:
                    continue
                windows.append(best)
                used += tks
                if used >= max_tokens_per_file:
                    break

            # Ensure deterministic ordering by seq_start
            windows.sort(key=lambda w: w.seq_start)
            out[fid] = windows
        return out
