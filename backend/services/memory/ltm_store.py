from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from backend.services.storage.sqlite_store import SQLiteMetadataStore


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


@dataclass
class MemoryHit:
    text: str
    score: float


class LongTermMemoryStore:
    """
    Minimal long-term memory store.
    For now uses SQLite with an in-memory vector cache per process; can be swapped to Qdrant later.
    """

    def __init__(self, metadata_store: SQLiteMetadataStore, embedder: Optional[Any]) -> None:
        self._store = metadata_store
        self._embedder = embedder
        self._memories: List[Dict[str, Any]] = []  # in-memory list of {chat_id, text, embedding, type}
        self._summaries: Dict[str, str] = {}

    def retrieve(self, chat_id: str, query: str, *, top_k: int = 5) -> List[MemoryHit]:
        if not self._embedder:
            return []
        qv = self._embedder(query)
        candidates: List[Tuple[float, Dict[str, Any]]] = []
        for mem in self._memories:
            if mem.get("chat_id") != chat_id:
                continue
            score = _cosine(qv, mem.get("embedding") or [])
            candidates.append((score, mem))
        candidates.sort(key=lambda x: x[0], reverse=True)
        hits: List[MemoryHit] = []
        for score, mem in candidates[:top_k]:
            hits.append(MemoryHit(text=mem.get("text", ""), score=score))
        return hits

    def get_conv_summary(self, chat_id: str) -> str:
        return self._summaries.get(chat_id, "")

    def save_memories(self, chat_id: str, texts: List[str]) -> None:
        if not self._embedder:
            return
        for text in texts:
            if not text:
                continue
            emb = self._embedder(text)
            self._memories.append(
                {
                    "chat_id": chat_id,
                    "text": text,
                    "embedding": emb,
                    "type": "memory",
                }
            )

    def update_conv_summary(self, chat_id: str, summary_text: str) -> None:
        if summary_text:
            self._summaries[chat_id] = summary_text

    def delete_chat(self, chat_id: str) -> None:
        """Remove all memories and summaries for a chat_id."""
        self._memories = [m for m in self._memories if m.get("chat_id") != chat_id]
        if chat_id in self._summaries:
            self._summaries.pop(chat_id, None)


__all__ = ["LongTermMemoryStore", "MemoryHit"]
