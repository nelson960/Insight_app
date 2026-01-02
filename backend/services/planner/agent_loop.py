from __future__ import annotations

import logging
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

        # Compute query embedding ONCE per turn (dense retrieval is per-file but the query is the same).
        query_vector: Optional[Sequence[float]] = None
        if self._embedder:
            try:
                query_vector = self._embedder(query)
            except Exception:
                query_vector = None

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

        # Token budget split by file with a conservative floor.
        max_tokens_total = max(200, int(cfg.max_tokens_total))
        per_file_cap = max(
            int(cfg.max_tokens_per_file_floor),
            int(max_tokens_total // max(1, len(fids))),
        )

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
                for w in windows_map.get(fid, []):
                    if not w.text:
                        continue
                    hits += 1
                    used += _approx_tokens(w.text)
                name = windows_map.get(fid, [SeqWindow(fid, "Document", 0, 0, "", None, None)])[0].filename
                mix[name] = {"windows": hits, "tokens": used}
            logger.debug(
                "agent_loop windows query_len=%d files=%d per_file_k=%d per_file_cap=%d mix=%s request_id=%s",
                len((query or "").strip()),
                len(fids),
                per_file_k,
                per_file_cap,
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
            for w in windows or []:
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
                        "chunk_id": f"seq_window:{w.seq_start}-{w.seq_end}",
                        "score": 1.0,
                    }
                )
        return hits


__all__ = ["MultiFileAgentConfig", "MultiFileAgentLoop"]
