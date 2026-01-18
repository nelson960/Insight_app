from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from backend.services.planner.agent_tools import AgentToolbox, ChunkAnchor, SeqWindow
from backend.services.retrieval.index_maintenance import validate_file_index, repair_file_index

logger = logging.getLogger(__name__)


def _approx_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def _uniq(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for v in values or []:
        s = (v or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _is_trivial_compare_query(query: str) -> bool:
    """
    Detect underspecified compare prompts like "compare".

    Hybrid retrieval on a query like "compare" is effectively random and often
    produces clustered anchors. For these prompts we switch to deterministic
    seq sampling (start/middle/end) so each file contributes representative
    context for a comparison.
    """
    q = (query or "").strip().lower()
    if not q:
        return False
    q = re.sub(r"\s+", " ", q).strip().strip(".!?;:")
    if q in {"compare", "comparison", "diff"}:
        return True
    if q.startswith("compare ") and len(q.split()) <= 3:
        # e.g. "compare docs", "compare files", "compare them"
        return True
    return False


def _sample_seq_positions(*, min_seq: int, max_seq: int, count: int) -> list[int]:
    min_i = max(0, int(min_seq))
    max_i = max(min_i, int(max_seq))
    count_i = max(0, int(count))

    if count_i <= 1 or max_i <= min_i:
        return [min_i]

    # Use a stable 3-point sample (start/middle/end) so each file contributes
    # comparable coverage in "compare" turns.
    span = max_i - min_i
    mid = min_i + (span // 2)
    raw = [min_i, mid, max_i]

    out: list[int] = []
    seen: set[int] = set()
    for v in raw:
        s = int(v)
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


@dataclass(frozen=True)
class MultiFileAgentConfig:
    """
    Budget + behavior knobs for multi-file analysis retrieval.

    This is deliberately lightweight: it does not run an LLM planning loop, but
    it does provide a stable "plan/act" retrieval structure with fairness and
    bounded context windows.
    """

    total_k: int = 14
    dense_k: int = 16
    sparse_k: int = 16
    radius: int = 2
    max_tokens_total: int = 2400
    max_tokens_per_file_floor: int = 600

    repair_on_empty: bool = True
    max_repairs: int = 2000


class MultiFileAgentLoop:
    """
    Multi-file retrieval loop that builds coherent, ordered evidence windows.

    Responsibilities:
    - Ensure each file contributes evidence (fairness) when scope=all.
    - Keep evidence ordered within each file (walk-by-seq windows).
    - Bound evidence by a strict token budget.
    - Optionally validate/repair Qdrant↔SQLite mismatches when retrieval is empty.

    Non-goals:
    - No hierarchical summarization (too expensive locally).
    - No general-purpose tool-calling from the model (this is a backend helper).
    """

    def __init__(
        self,
        *,
        tools: AgentToolbox,
        embedder: Optional[Callable[[str], Sequence[float]]] = None,
    ) -> None:
        self._tools = tools
        self._embedder = embedder

    def _maybe_repair_index_for_file(self, file_id: str, *, max_repairs: int) -> bool:
        if not self._embedder:
            return False
        rag_store = self._tools.rag_store
        retrieval_service = getattr(rag_store, "retrieval", None)
        qdrant_client = getattr(retrieval_service, "qdrant_client", None)
        collection_name = getattr(retrieval_service, "collection_name", None)
        sqlite_store = self._tools.metadata_store
        if not qdrant_client or not collection_name or not sqlite_store:
            return False

        v = validate_file_index(
            sqlite_store=sqlite_store,
            qdrant_client=qdrant_client,
            collection_name=str(collection_name),
            file_id=file_id,
        )
        if not v.missing_in_qdrant or v.truncated:
            return False

        res = repair_file_index(
            sqlite_store=sqlite_store,
            qdrant_client=qdrant_client,
            collection_name=str(collection_name),
            file_id=file_id,
            embedder=self._embedder,
            max_repairs=max(1, int(max_repairs)),
            delete_orphans=False,
            dry_run=False,
        )
        return bool(res.get("ok"))

    def build_evidence_windows(
        self,
        query: str,
        *,
        file_ids: Sequence[str],
        config: Optional[MultiFileAgentConfig] = None,
        request_id: Optional[str] = None,
    ) -> dict[str, list[SeqWindow]]:
        cfg = config or MultiFileAgentConfig()
        fids = _uniq(list(file_ids))
        if not fids:
            return {}

        total_k = max(1, int(cfg.total_k))
        per_file_k = max(1, (total_k + len(fids) - 1) // max(1, len(fids)))

        compare_mode = _is_trivial_compare_query(query) and len(fids) >= 2

        # Compute query embedding ONCE per turn (dense retrieval is per-file but the query is the same).
        query_vector: Optional[Sequence[float]] = None
        if self._embedder:
            try:
                query_vector = self._embedder(query)
            except Exception:
                query_vector = None

        # Token budget split by file with a conservative floor.
        max_tokens_total = max(200, int(cfg.max_tokens_total))
        per_file_cap = max(
            int(cfg.max_tokens_per_file_floor),
            int(max_tokens_total // max(1, len(fids))),
        )

        if compare_mode:
            # In compare/overview turns, "compare" is not a meaningful retrieval query.
            # Instead, build ordered, bounded windows by walking each document by seq.
            #
            # This guarantees:
            # - each file contributes evidence (fairness)
            # - evidence is in source order (coherence)
            # - evidence is bounded per file (token budget)
            target_windows_per_file = 3
            per_window_cap = max(200, int(per_file_cap // max(1, target_windows_per_file)))
            windows_map: dict[str, list[SeqWindow]] = {}

            for fid in fids:
                try:
                    stats = self._tools.metadata_store.chunk_seq_stats_for_file(fid)
                except Exception:
                    stats = {"count": 0, "min_seq": 0, "max_seq": 0}
                cnt = int(stats.get("count") or 0)
                if cnt <= 0:
                    continue
                min_seq = int(stats.get("min_seq") or 0)
                max_seq = int(stats.get("max_seq") or 0)
                min_seq = max(0, min_seq)
                max_seq = max(min_seq, max_seq)
                total_seq = max_seq - min_seq + 1

                file_windows: list[SeqWindow] = []
                if total_seq >= target_windows_per_file:
                    # Split the file into N sequential windows (bounded).
                    n = int(target_windows_per_file)
                    seg = max(1, (total_seq + n - 1) // n)  # ceil

                    for i in range(n):
                        s = min_seq + (i * seg)
                        e = min(max_seq, s + seg - 1)
                        if s > max_seq:
                            break

                        rows = self._tools.metadata_store.fetch_chunks_by_seq_range(fid, seq_start=s, seq_end=e)
                        text_parts: list[str] = []
                        page_start: Optional[int] = None
                        page_end: Optional[int] = None
                        filename = "Document"
                        for row in rows:
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

                        text_full = "\n\n".join(text_parts).strip()
                        if not text_full:
                            continue

                        # For compare turns, avoid always truncating from the start of a large
                        # segment. Sample a slice (start/middle/end) depending on segment position.
                        max_chars = max(0, int(per_window_cap * 4))
                        text = text_full
                        if max_chars > 0 and len(text_full) > max_chars:
                            if i <= 0:
                                text = text_full[:max_chars].rstrip()
                            elif i >= (n - 1):
                                text = text_full[-max_chars:].lstrip()
                            else:
                                mid = len(text_full) // 2
                                start = max(0, mid - (max_chars // 2))
                                end = min(len(text_full), start + max_chars)
                                text = text_full[start:end].strip()
                            text = text + "\n… (truncated)"

                        file_windows.append(
                            SeqWindow(
                                file_id=fid,
                                filename=filename,
                                seq_start=s,
                                seq_end=e,
                                text=text,
                                page_start=page_start,
                                page_end=page_end,
                            )
                        )
                else:
                    # If a file has too few seq chunks (1–2), we still want fair, balanced
                    # coverage per file for "compare". Build multiple windows by slicing the
                    # combined text (start/middle/end) instead of relying on seq boundaries.
                    rows = self._tools.metadata_store.fetch_chunks_by_seq_range(fid, seq_start=min_seq, seq_end=max_seq)
                    text_parts: list[str] = []
                    page_start: Optional[int] = None
                    page_end: Optional[int] = None
                    filename = "Document"
                    for row in rows:
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

                    full_text = "\n\n".join(text_parts).strip()
                    if full_text:
                        max_chars = max(0, int(per_window_cap * 4))
                        if max_chars <= 0:
                            max_chars = max(1, len(full_text))

                        # Build up to 3 distinct slices; avoid duplicates for short text.
                        starts = [0]
                        if len(full_text) > max_chars:
                            mid = max(0, (len(full_text) // 2) - (max_chars // 2))
                            end = max(0, len(full_text) - max_chars)
                            starts.extend([mid, end])
                        uniq_starts: list[int] = []
                        seen_starts: set[int] = set()
                        for s in starts:
                            si = int(max(0, min(s, max(0, len(full_text) - 1))))
                            if si in seen_starts:
                                continue
                            seen_starts.add(si)
                            uniq_starts.append(si)
                        max_iterations = max(1, int(target_windows_per_file) * 2)
                        iterations = 0
                        while len(uniq_starts) < int(target_windows_per_file) and len(full_text) > max_chars:
                            iterations += 1
                            if iterations > max_iterations:
                                break
                            # Add an evenly spaced start position if we don't have enough unique slices.
                            frac = (len(uniq_starts) + 1) / float(target_windows_per_file + 1)
                            pos = int(max(0, min(len(full_text) - max_chars, int(len(full_text) * frac))))
                            if pos in seen_starts:
                                break
                            seen_starts.add(pos)
                            uniq_starts.append(pos)

                        for idx, start in enumerate(uniq_starts[: int(target_windows_per_file)]):
                            end = min(len(full_text), start + max_chars)
                            piece = full_text[start:end].strip()
                            if not piece:
                                continue
                            if end < len(full_text):
                                piece = piece + "\n… (truncated)"
                            # Use a stable pseudo seq range per slice to preserve ordering.
                            pseudo = min_seq + idx
                            file_windows.append(
                                SeqWindow(
                                    file_id=fid,
                                    filename=filename,
                                    seq_start=pseudo,
                                    seq_end=pseudo,
                                    text=piece,
                                    page_start=page_start,
                                    page_end=page_end,
                                )
                            )

                file_windows.sort(key=lambda w: w.seq_start)
                if file_windows:
                    windows_map[fid] = file_windows
        else:
            # Fairness: each file gets its own hybrid anchor search.
            anchors: list[ChunkAnchor] = []
            for fid in fids:
                a = self._tools.hybrid_search(
                    query,
                    file_id=fid,
                    top_k=per_file_k,
                    query_vector=query_vector,
                    dense_k=max(per_file_k, int(cfg.dense_k)),
                    sparse_k=max(per_file_k, int(cfg.sparse_k)),
                )
                if not a and cfg.repair_on_empty:
                    try:
                        repaired = self._maybe_repair_index_for_file(fid, max_repairs=int(cfg.max_repairs))
                    except Exception:
                        repaired = False
                    if repaired:
                        a = self._tools.hybrid_search(
                            query,
                            file_id=fid,
                            top_k=per_file_k,
                            query_vector=query_vector,
                            dense_k=max(per_file_k, int(cfg.dense_k)),
                            sparse_k=max(per_file_k, int(cfg.sparse_k)),
                        )
                anchors.extend(a)

            windows_map = self._tools.windows_from_anchors(
                anchors,
                radius=int(cfg.radius),
                max_tokens_per_file=per_file_cap,
            )

        # Log (backend only) for debugging mix and budgets.
        try:
            mix: dict[str, dict[str, int]] = {}
            for fid in fids:
                used = 0
                hits = 0
                if fid not in windows_map:
                    continue
                for w in windows_map.get(fid, []):
                    if not w.text:
                        continue
                    hits += 1
                    used += _approx_tokens(w.text)
                name = windows_map[fid][0].filename if windows_map.get(fid) else "Document"
                mix[name] = {"windows": hits, "tokens": used}
            logger.debug(
                "agent_loop windows query_len=%d files=%d per_file_k=%d per_file_cap=%d compare=%s mix=%s request_id=%s",
                len((query or "").strip()),
                len(fids),
                per_file_k,
                per_file_cap,
                bool(compare_mode),
                mix,
                request_id,
            )
        except Exception:
            pass

        return windows_map

    @staticmethod
    def windows_to_rag_hits(windows_map: dict[str, list[SeqWindow]]) -> list[dict[str, Any]]:
        hits: list[dict[str, Any]] = []
        for fid, windows in (windows_map or {}).items():
            for idx, w in enumerate(windows or []):
                if not w.text:
                    continue
                hits.append(
                    {
                        "doc_id": fid,
                        "text": w.text,
                        "filename": w.filename,
                        "page": w.page_start,
                        "page_start": w.page_start,
                        "page_end": w.page_end,
                        # synthetic chunk_id to prevent accidental cross-doc dedupe
                        "chunk_id": f"seq_window:{w.seq_start}-{w.seq_end}:{idx}",
                        "score": 1.0,
                    }
                )
        return hits


__all__ = ["MultiFileAgentConfig", "MultiFileAgentLoop"]
