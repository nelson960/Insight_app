from __future__ import annotations

import hashlib
import logging
import os
import time
import uuid
from array import array
from typing import Any, Dict, List, Optional, Sequence

from backend.services.memory.ltm_store import MemoryHit, sanitize_memory_text
from backend.services.storage.sqlite_store import SQLiteMetadataStore

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


def _serialize_embedding(vec: Sequence[float]) -> bytes:
    arr = array("f", list(vec))
    return arr.tobytes()


class LtmSqliteStore:
    """
    SQLite LTM store with summary-only, recency-first retrieval.
    """

    def __init__(
        self,
        *,
        metadata_store: SQLiteMetadataStore,
        embedder: Optional[Any],
    ) -> None:
        self._store = metadata_store
        self._embedder = embedder
        self._max_summaries_per_chat = _env_int("INSIGHT_LTM_SUMMARY_KEEP", 8, min_value=1, max_value=500)
        self._summary_max_age_days = _env_float(
            "INSIGHT_LTM_SUMMARY_MAX_AGE_DAYS",
            30.0,
            min_value=1.0,
            max_value=3650.0,
        )
        self._text_max_chars = _env_int("INSIGHT_LTM_TEXT_MAX_CHARS", 1200, min_value=120, max_value=16000)

    # ---- Public API ----
    def retrieve(self, chat_id: str, query: str, *, top_k: int = 5) -> List[MemoryHit]:
        del query
        self._prune_expired(chat_id)
        target_k = max(1, int(top_k))
        rows = self._store.list_ltm_memories(chat_id, memory_type="summary", limit=target_k, order_desc=True)
        out: List[MemoryHit] = []
        for i, row in enumerate(rows):
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            out.append(
                MemoryHit(
                    text=text.strip(),
                    score=max(0.0, 1.0 - (0.01 * i)),
                    created_at=int(row.get("created_at") or 0),
                    memory_type="summary",
                )
            )
        return out

    def get_conv_summary(self, chat_id: str) -> str:
        rows = self._store.list_ltm_memories(chat_id, memory_type="summary", limit=1, order_desc=True)
        if rows:
            text = rows[0].get("text")
            if isinstance(text, str):
                return text
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
        existing = self._store.list_ltm_memories(chat_id, memory_type="summary", limit=5000, order_desc=False)
        existing_by_hash = {}
        for row in existing:
            h = row.get("hash")
            rid = row.get("id")
            if isinstance(h, str) and h and isinstance(rid, str) and rid:
                existing_by_hash[h] = rid

        for raw in texts or []:
            text = sanitize_memory_text(raw, max_chars=self._text_max_chars)
            if not text:
                continue
            hash_value = _hash_text(text)
            old_id = existing_by_hash.get(hash_value)
            if old_id:
                try:
                    self._store.delete_ltm_memories([old_id])
                except Exception:
                    pass

            emb: Sequence[float] = []
            if self._embedder:
                try:
                    emb = self._embedder(text) or []
                except Exception:
                    emb = []
            embedding_blob = _serialize_embedding(emb) if emb else b""
            embedding_dim = len(emb) if emb else 0
            meta = dict(metadata or {})
            meta["origin"] = meta.get("origin") or "compaction_summary"
            try:
                self._store.insert_ltm_memory(
                    memory_id=str(uuid.uuid4()),
                    chat_id=chat_id,
                    user_id=None,
                    memory_type="summary",
                    text=text,
                    embedding=embedding_blob,
                    embedding_dim=embedding_dim,
                    embedder_id=None,
                    confidence=None,
                    hash_value=hash_value,
                    created_at=now,
                    metadata=meta,
                )
            except Exception:
                logger.warning("Failed to store LTM summary for chat_id=%s", chat_id, exc_info=True)
        self._store.prune_ltm_memories(chat_id, memory_type="summary", keep=self._max_summaries_per_chat)
        self._prune_expired(chat_id)

    def update_conv_summary(self, chat_id: str, summary_text: str) -> None:
        if not summary_text:
            return
        self.save_memories(chat_id, [summary_text], memory_type="summary")

    def delete_chat(self, chat_id: str) -> None:
        try:
            self._store.delete_ltm_for_chat(chat_id)
        except Exception:
            logger.warning("Failed to delete LTM entries for chat_id=%s", chat_id, exc_info=True)

    def _prune_expired(self, chat_id: str) -> None:
        cutoff = int(time.time() - (self._summary_max_age_days * 86400.0))
        rows = self._store.list_ltm_memories(chat_id, memory_type="summary", limit=5000, order_desc=False)
        stale_ids = []
        for row in rows:
            created_at = int(row.get("created_at") or 0)
            if created_at and created_at < cutoff:
                rid = row.get("id")
                if isinstance(rid, str):
                    stale_ids.append(rid)
        if stale_ids:
            self._store.delete_ltm_memories(stale_ids)


__all__ = ["LtmSqliteStore"]
