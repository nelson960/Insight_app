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

    def retrieve(self, query: str, *, chat_id: Optional[str] = None, doc_ids: Optional[List[str]] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.embedder:
            return []
        vector = self.embedder(query)
        ctx = RetrievalContext(user_id=None, chat_id=chat_id, file_ids=doc_ids or [])
        results = self.retrieval.search(
            RetrievalQuery(vector=vector, limit=top_k, with_chunks=True, with_metadata=True),
            context=ctx,
        )
        hits: List[Dict[str, Any]] = []
        for res in results:
            hits.append(
                {
                    "doc_id": res.file_id,
                    "chunk_id": res.chunk_id,
                    "text": res.text or "",
                    "score": res.score or 0.0,
                }
            )
        return hits
