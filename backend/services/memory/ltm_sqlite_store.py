from __future__ import annotations

import hashlib
import logging
import math
import re
import time
import uuid
from array import array
from typing import Any, List, Optional, Sequence

from backend.services.memory.ltm_store import MemoryHit
from backend.services.storage.sqlite_store import SQLiteMetadataStore

logger = logging.getLogger(__name__)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _hash_text(text: str) -> str:
    return hashlib.sha256(_normalize_text(text).encode("utf-8")).hexdigest()


def _serialize_embedding(vec: Sequence[float]) -> bytes:
    arr = array("f", list(vec))
    return arr.tobytes()


def _deserialize_embedding(blob: Optional[bytes], dim: Optional[int]) -> Sequence[float]:
    if not blob:
        return []
    arr = array("f")
    try:
        arr.frombytes(blob)
    except Exception:
        return []
    if dim and len(arr) != dim:
        return []
    return arr


class LtmSqliteStore:
    """
    Persistent long-term memory store backed by SQLite.
    Stores compacted summaries with embeddings for recency-filtered similarity retrieval.
    """

    def __init__(
        self,
        *,
        metadata_store: SQLiteMetadataStore,
        embedder: Optional[Any],
    ) -> None:
        self._store = metadata_store
        self._embedder = embedder
        self._max_memories_per_chat = 5

    # ---- Public API ----
    def retrieve(self, chat_id: str, query: str, *, top_k: int = 5) -> List[MemoryHit]:
        if not self._embedder:
            return []
        qv = self._embedder(query)
        if not qv:
            return []
        recent_n = max(self._max_memories_per_chat, int(top_k))
        rows = self._store.list_ltm_memories(chat_id, memory_type="memory", limit=recent_n, order_desc=True)
        candidates: List[MemoryHit] = []
        for row in rows:
            emb = _deserialize_embedding(row.get("embedding"), row.get("embedding_dim"))
            if not emb:
                continue
            score = _cosine(qv, emb)
            candidates.append(MemoryHit(text=row.get("text", "") or "", score=score))
        candidates.sort(key=lambda m: m.score, reverse=True)
        return candidates[: int(top_k)]

    def get_conv_summary(self, chat_id: str) -> str:
        rows = self._store.list_ltm_memories(chat_id, memory_type="summary", limit=1, order_desc=True)
        if rows:
            return rows[0].get("text", "") or ""
        return ""

    def save_memories(self, chat_id: str, texts: List[str]) -> None:
        if not self._embedder:
            return
        now = int(time.time())
        for text in texts:
            if not text:
                continue
            emb = self._embedder(text)
            if not emb:
                continue
            mem_id = str(uuid.uuid4())
            hash_value = _hash_text(text)
            try:
                self._store.insert_ltm_memory(
                    memory_id=mem_id,
                    chat_id=chat_id,
                    user_id=None,
                    memory_type="memory",
                    text=text,
                    embedding=_serialize_embedding(emb),
                    embedding_dim=len(emb),
                    embedder_id=None,
                    confidence=None,
                    hash_value=hash_value,
                    created_at=now,
                    metadata=None,
                )
            except Exception:
                logger.warning("Failed to store LTM memory for chat_id=%s", chat_id, exc_info=True)
        self._store.prune_ltm_memories(chat_id, memory_type="memory", keep=self._max_memories_per_chat)

    def update_conv_summary(self, chat_id: str, summary_text: str) -> None:
        if not self._embedder or not summary_text:
            return
        try:
            # delete existing summary rows for chat
            existing = self._store.list_ltm_memories(chat_id, memory_type="summary", limit=50, order_desc=False)
            ids = [row.get("id") for row in existing if row.get("id")]
            if ids:
                self._store.delete_ltm_memories(ids)
        except Exception:
            logger.warning("Failed to delete existing LTM summary for chat_id=%s", chat_id, exc_info=True)

        emb = self._embedder(summary_text)
        if not emb:
            return
        mem_id = str(uuid.uuid4())
        hash_value = _hash_text(summary_text)
        try:
            self._store.insert_ltm_memory(
                memory_id=mem_id,
                chat_id=chat_id,
                user_id=None,
                memory_type="summary",
                text=summary_text,
                embedding=_serialize_embedding(emb),
                embedding_dim=len(emb),
                embedder_id=None,
                confidence=None,
                hash_value=hash_value,
                created_at=int(time.time()),
                metadata=None,
            )
        except Exception:
            logger.warning("Failed to store LTM summary for chat_id=%s", chat_id, exc_info=True)

    def delete_chat(self, chat_id: str) -> None:
        try:
            self._store.delete_ltm_for_chat(chat_id)
        except Exception:
            logger.warning("Failed to delete LTM entries for chat_id=%s", chat_id, exc_info=True)


__all__ = ["LtmSqliteStore"]
