from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

from backend.services.retrieval import RetrievalService, RetrievalQuery, RetrievalContext


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class RagStore:
    """Thin wrapper around RetrievalService to return text chunks + similarity."""

    def __init__(self, retrieval_service: RetrievalService, embedder: Optional[Any]) -> None:
        self.retrieval = retrieval_service
        self.embedder = embedder

    def embed(self, text: str) -> Optional[Any]:
        if not self.embedder:
            return None
        return self.embedder(text)

    def _results_to_hits(self, results: List[Any]) -> List[Dict[str, Any]]:
        hits: List[Dict[str, Any]] = []
        for res in results:
            filename = None
            page = None
            page_start = None
            page_end = None
            section_path: list[str] | None = None
            if isinstance(res.metadata, dict):
                fn = res.metadata.get("filename")
                if isinstance(fn, str) and fn.strip():
                    filename = fn
                meta = res.metadata.get("metadata")
                if isinstance(meta, dict):
                    p = meta.get("page")
                    if isinstance(p, int):
                        page = p
                    ps = meta.get("page_start")
                    if isinstance(ps, int):
                        page_start = ps
                    pe = meta.get("page_end")
                    if isinstance(pe, int):
                        page_end = pe
                    sp = meta.get("section_path")
                    if isinstance(sp, list):
                        cleaned = [str(x).strip() for x in sp if str(x).strip()]
                        section_path = cleaned or None
                    elif isinstance(sp, str) and sp.strip():
                        section_path = [sp.strip()]

            # Keep legacy single-page field populated for older callers.
            if page is None and isinstance(page_start, int):
                page = page_start
            if page_start is None and isinstance(page, int):
                page_start = page
            if page_end is None and isinstance(page_start, int):
                page_end = page_start
            hits.append(
                {
                    "doc_id": res.file_id,
                    "chunk_id": res.chunk_id,
                    "text": res.text or "",
                    "score": res.score or 0.0,
                    "filename": filename,
                    "page": page,
                    "page_start": page_start,
                    "page_end": page_end,
                    "section_path": section_path,
                }
            )
        return hits

    def retrieve(self, query: str, *, chat_id: Optional[str] = None, doc_ids: Optional[List[str]] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.embedder:
            return []
        vector = self.embedder(query)
        ctx = RetrievalContext(user_id=None, chat_id=chat_id, file_ids=doc_ids or [])
        results = self.retrieval.search(
            RetrievalQuery(vector=vector, limit=top_k, with_chunks=True, with_metadata=True),
            context=ctx,
        )
        return self._results_to_hits(results)

    def retrieve_with_vector(
        self,
        vector: Sequence[float],
        *,
        chat_id: Optional[str] = None,
        doc_ids: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        if vector is None:
            return []
        # Defensive: avoid ambiguous truthiness for numpy arrays / sequences.
        vec_list = list(vector)
        if not vec_list:
            return []
        ctx = RetrievalContext(user_id=None, chat_id=chat_id, file_ids=doc_ids or [])
        results = self.retrieval.search(
            RetrievalQuery(vector=vec_list, limit=top_k, with_chunks=True, with_metadata=True),
            context=ctx,
        )
        return self._results_to_hits(results)
