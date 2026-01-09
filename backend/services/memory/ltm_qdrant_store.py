from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from backend.services.memory.ltm_store import MemoryHit

logger = logging.getLogger(__name__)


class LtmQdrantStore:
    """
    Long-term memory store backed by a dedicated Qdrant collection.
    Stores memories/summaries keyed by chat_id with payload filters.
    """

    def __init__(
        self,
        *,
        client: QdrantClient,
        collection: str = "insight_memories",
        embedder: Optional[Any],
    ) -> None:
        self._client = client
        self._collection = collection
        self._embedder = embedder
        self._vector_size = None
        self._max_memories_per_chat = 5
        self._ensure_collection()

    # ---- Public API ----
    def retrieve(self, chat_id: str, query: str, *, top_k: int = 5) -> List[MemoryHit]:
        if not self._embedder:
            return []
        vec = self._embedder(query)
        self._ensure_vector_size(len(vec))
        flt = qmodels.Filter(
            must=[
                qmodels.FieldCondition(key="chat_id", match=qmodels.MatchValue(value=chat_id)),
                qmodels.FieldCondition(key="type", match=qmodels.MatchValue(value="memory")),
            ]
        )
        hits = self._client.search(
            collection_name=self._collection,
            query_vector=vec,
            query_filter=flt,
            limit=top_k,
            with_payload=True,
        )
        results: List[MemoryHit] = []
        for h in hits:
            payload = h.payload or {}
            results.append(MemoryHit(text=payload.get("text", ""), score=h.score or 0.0))
        return results

    def get_conv_summary(self, chat_id: str) -> str:
        flt = qmodels.Filter(
            must=[
                qmodels.FieldCondition(key="chat_id", match=qmodels.MatchValue(value=chat_id)),
                qmodels.FieldCondition(key="type", match=qmodels.MatchValue(value="summary")),
            ]
        )
        points, _ = self._client.scroll(
            collection_name=self._collection,
            scroll_filter=flt,
            limit=1,
            with_payload=True,
        )
        if points:
            return (points[0].payload or {}).get("text", "") or ""
        return ""

    def save_memories(self, chat_id: str, texts: List[str]) -> None:
        if not self._embedder:
            return
        now = int(time.time())
        points = []
        for text in texts:
            if not text:
                continue
            vec = self._embedder(text)
            self._ensure_vector_size(len(vec))
            points.append(
                qmodels.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec,
                    payload={
                        "chat_id": chat_id,
                        "type": "memory",
                        "text": text,
                        "created_at": now,
                    },
                )
            )
        if points:
            self._client.upsert(collection_name=self._collection, points=points)
            self._prune_memories(chat_id)

    def update_conv_summary(self, chat_id: str, summary_text: str) -> None:
        if not self._embedder or not summary_text:
            return
        vec = self._embedder(summary_text)
        self._ensure_vector_size(len(vec))
        # delete existing summary for chat_id
        flt = qmodels.Filter(
            must=[
                qmodels.FieldCondition(key="chat_id", match=qmodels.MatchValue(value=chat_id)),
                qmodels.FieldCondition(key="type", match=qmodels.MatchValue(value="summary")),
            ]
        )
        try:
            self._client.delete(collection_name=self._collection, points_selector=qmodels.FilterSelector(filter=flt))
        except Exception:
            logger.warning("Failed to delete LTM summary for chat_id=%s", chat_id, exc_info=True)
        self._client.upsert(
            collection_name=self._collection,
            points=[
            qmodels.PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "chat_id": chat_id,
                    "type": "summary",
                        "text": summary_text,
                    },
                )
            ],
        )

    def delete_chat(self, chat_id: str) -> None:
        """Remove all memories/summaries for a chat_id."""
        flt = qmodels.Filter(
            must=[
                qmodels.FieldCondition(key="chat_id", match=qmodels.MatchValue(value=chat_id)),
            ]
        )
        try:
            self._client.delete(collection_name=self._collection, points_selector=qmodels.FilterSelector(filter=flt))
        except Exception:
            logger.warning("Failed to delete LTM entries for chat_id=%s", chat_id, exc_info=True)

    # ---- Internal ----
    def _prune_memories(self, chat_id: str) -> None:
        if self._max_memories_per_chat <= 0:
            return
        flt = qmodels.Filter(
            must=[
                qmodels.FieldCondition(key="chat_id", match=qmodels.MatchValue(value=chat_id)),
                qmodels.FieldCondition(key="type", match=qmodels.MatchValue(value="memory")),
            ]
        )
        points, _ = self._client.scroll(
            collection_name=self._collection,
            scroll_filter=flt,
            limit=5000,
            with_payload=True,
        )
        if len(points) <= self._max_memories_per_chat:
            return
        scored = []
        for p in points:
            payload = p.payload or {}
            created_at = payload.get("created_at")
            try:
                created_val = int(created_at)
            except (TypeError, ValueError):
                created_val = 0
            scored.append((created_val, p.id))
        scored.sort(key=lambda x: x[0])
        to_delete = [pid for _, pid in scored[:-self._max_memories_per_chat]]
        if not to_delete:
            return
        try:
            self._client.delete(
                collection_name=self._collection,
                points_selector=qmodels.PointIdsList(points=to_delete),
            )
        except Exception:
            logger.warning("Failed to prune LTM memories for chat_id=%s", chat_id, exc_info=True)

    def _ensure_collection(self) -> None:
        try:
            self._client.get_collection(self._collection)
        except Exception:
            # will lazily set vector size on first embed; default to 768
            size = self._vector_size or 768
            self._client.recreate_collection(
                collection_name=self._collection,
                vectors_config=qmodels.VectorParams(size=size, distance=qmodels.Distance.COSINE),
            )

    def _ensure_vector_size(self, size: int) -> None:
        if self._vector_size == size:
            return
        if self._vector_size is None:
            self._vector_size = size
            try:
                self._client.get_collection(self._collection)
            except Exception:
                self._client.recreate_collection(
                    collection_name=self._collection,
                    vectors_config=qmodels.VectorParams(size=size, distance=qmodels.Distance.COSINE),
                )


__all__ = ["LtmQdrantStore"]
