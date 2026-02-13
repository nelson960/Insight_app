from __future__ import annotations

import hashlib
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from backend.services.memory.ltm_store import MemoryHit, sanitize_memory_text
from backend.services.security import KeyManager, decrypt_text, encrypt_text
from backend.core.workspace import get_workspace

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int, *, min_value: int, max_value: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except Exception:
        value = default
    return max(min_value, min(max_value, value))


def _env_float(name: str, default: float, *, min_value: float, max_value: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except Exception:
        value = default
    return max(min_value, min(max_value, value))


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.lower().encode("utf-8")).hexdigest()


class LtmQdrantStore:
    """
    Qdrant-backed LTM store with summary-only, recency-first retrieval.
    """

    def __init__(
        self,
        *,
        client: QdrantClient,
        collection: str = "insight_memories",
        embedder: Optional[Any],
        encryption_key: Optional[bytes] = None,
    ) -> None:
        if embedder is None:
            raise RuntimeError("LTM query embedder unavailable for Qdrant LTM.")
        self._client = client
        self._collection = collection
        self._embedder = embedder
        self._encryption_key = encryption_key
        if self._encryption_key is None:
            try:
                self._encryption_key = KeyManager(get_workspace()).get_key()
            except Exception:
                self._encryption_key = None
        self._vector_size: Optional[int] = None
        self._max_summaries_per_chat = _env_int("INSIGHT_LTM_SUMMARY_KEEP", 8, min_value=1, max_value=500)
        self._summary_max_age_days = _env_float(
            "INSIGHT_LTM_SUMMARY_MAX_AGE_DAYS",
            30.0,
            min_value=1.0,
            max_value=3650.0,
        )
        self._text_max_chars = _env_int("INSIGHT_LTM_TEXT_MAX_CHARS", 1200, min_value=120, max_value=16000)
        self._ensure_collection()

    # ---- Public API ----
    def retrieve(self, chat_id: str, query: str, *, top_k: int = 5) -> List[MemoryHit]:
        del query
        self._prune_expired(chat_id)
        target_k = max(1, int(top_k))
        points = self._scroll_summaries(chat_id, limit=max(target_k * 4, self._max_summaries_per_chat * 2, 50))
        points.sort(key=lambda p: int((p.payload or {}).get("created_at") or 0), reverse=True)
        out: List[MemoryHit] = []
        seen = set()
        for i, p in enumerate(points):
            payload = p.payload or {}
            raw_text = payload.get("text")
            raw_text_enc = payload.get("text_enc")
            if isinstance(raw_text_enc, str) and raw_text_enc:
                decrypted = decrypt_text(raw_text_enc, key=self._encryption_key)
                text = sanitize_memory_text(decrypted, max_chars=self._text_max_chars)
            else:
                text = sanitize_memory_text(raw_text, max_chars=self._text_max_chars)
            if not text:
                continue
            h = _hash_text(text)
            if h in seen:
                continue
            seen.add(h)
            out.append(
                MemoryHit(
                    text=text,
                    score=max(0.0, 1.0 - (0.01 * i)),
                    created_at=int(payload.get("created_at") or 0),
                    memory_type="summary",
                )
            )
            if len(out) >= target_k:
                break
        return out

    def get_conv_summary(self, chat_id: str) -> str:
        rows = self.retrieve(chat_id, "", top_k=1)
        if rows:
            return rows[0].text
        return ""

    def save_memories(
        self,
        chat_id: str,
        texts: List[str],
        *,
        memory_type: str = "summary",
        confidence: Optional[float] = None,
        importance: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        del memory_type, confidence, importance
        now = int(time.time())
        existing = self._scroll_summaries(chat_id, limit=5000)
        existing_by_hash: Dict[str, object] = {}
        for p in existing:
            payload = p.payload or {}
            h = payload.get("hash")
            if isinstance(h, str) and h:
                existing_by_hash[h] = p.id

        points: List[qmodels.PointStruct] = []
        for raw in texts or []:
            text = sanitize_memory_text(raw, max_chars=self._text_max_chars)
            if not text:
                continue
            h = _hash_text(text)
            old_id = existing_by_hash.get(h)
            if old_id is not None:
                self._delete_point_ids([old_id])
            vec = self._embedder(text) or []
            if not vec:
                continue
            self._ensure_vector_size(len(vec))
            payload = dict(metadata or {})
            payload["origin"] = payload.get("origin") or "compaction_summary"
            text_enc = encrypt_text(text, key=self._encryption_key)
            points.append(
                qmodels.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec,
                    payload={
                        "chat_id": chat_id,
                        "type": "summary",
                        "text_enc": text_enc,
                        "hash": h,
                        "created_at": now,
                        "metadata": payload,
                    },
                )
            )
        if points:
            self._client.upsert(collection_name=self._collection, points=points)
            self._prune_keep(chat_id)
            self._prune_expired(chat_id)

    def update_conv_summary(self, chat_id: str, summary_text: str) -> None:
        if not summary_text:
            return
        self.save_memories(chat_id, [summary_text], memory_type="summary")

    def delete_chat(self, chat_id: str) -> None:
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
    def _scroll_summaries(self, chat_id: str, *, limit: int) -> List[qmodels.Record]:
        flt = qmodels.Filter(
            must=[
                qmodels.FieldCondition(key="chat_id", match=qmodels.MatchValue(value=chat_id)),
                qmodels.FieldCondition(key="type", match=qmodels.MatchValue(value="summary")),
            ]
        )
        points, _ = self._client.scroll(
            collection_name=self._collection,
            scroll_filter=flt,
            limit=max(1, int(limit)),
            with_payload=True,
        )
        return points or []

    def _delete_point_ids(self, ids: List[object]) -> None:
        ids = [i for i in ids if i is not None]
        if not ids:
            return
        try:
            self._client.delete(
                collection_name=self._collection,
                points_selector=qmodels.PointIdsList(points=ids),
            )
        except Exception:
            logger.warning("Failed deleting LTM summary points", exc_info=True)

    def _prune_keep(self, chat_id: str) -> None:
        points = self._scroll_summaries(chat_id, limit=5000)
        if len(points) <= self._max_summaries_per_chat:
            return
        points.sort(key=lambda p: int((p.payload or {}).get("created_at") or 0), reverse=True)
        stale = [p.id for p in points[self._max_summaries_per_chat :]]
        self._delete_point_ids(stale)

    def _prune_expired(self, chat_id: str) -> None:
        cutoff = int(time.time() - (self._summary_max_age_days * 86400.0))
        points = self._scroll_summaries(chat_id, limit=5000)
        stale = []
        for p in points:
            created_at = int((p.payload or {}).get("created_at") or 0)
            if created_at and created_at < cutoff:
                stale.append(p.id)
        self._delete_point_ids(stale)

    def _ensure_collection(self) -> None:
        try:
            self._client.get_collection(self._collection)
        except Exception:
            size = self._vector_size or 768
            self._vector_size = size
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
