from __future__ import annotations

import hashlib
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from backend.services.storage.sqlite_store import SQLiteMetadataStore


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


def sanitize_memory_text(text: object, *, max_chars: int) -> str:
    if not isinstance(text, str):
        return ""
    out = re.sub(r"\s+", " ", text).strip()
    if not out:
        return ""
    if len(out) > max_chars:
        out = out[:max_chars].rstrip()
    return out


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.lower().encode("utf-8")).hexdigest()


@dataclass
class MemoryHit:
    text: str
    score: float = 1.0
    created_at: int = 0
    memory_type: str = "summary"


class LongTermMemoryStore:
    """
    In-memory fallback LTM store.

    Policy:
    - Save compaction summaries only.
    - Retrieve newest summaries first (LIFO).
    """

    def __init__(self, metadata_store: SQLiteMetadataStore, embedder: Optional[Any]) -> None:
        self._store = metadata_store
        self._embedder = embedder
        self._rows: List[Dict[str, Any]] = []
        self._max_summaries_per_chat = _env_int("INSIGHT_LTM_SUMMARY_KEEP", 8, min_value=1, max_value=500)
        self._summary_max_age_days = _env_float(
            "INSIGHT_LTM_SUMMARY_MAX_AGE_DAYS",
            30.0,
            min_value=1.0,
            max_value=3650.0,
        )
        self._text_max_chars = _env_int("INSIGHT_LTM_TEXT_MAX_CHARS", 1200, min_value=120, max_value=16000)

    def retrieve(self, chat_id: str, query: str, *, top_k: int = 5) -> List[MemoryHit]:
        del query  # Retrieval is recency-first (LIFO), not semantic.
        self._prune_expired(chat_id)
        target_k = max(1, int(top_k))
        rows = [r for r in self._rows if r.get("chat_id") == chat_id and r.get("type") == "summary"]
        rows.sort(key=lambda r: int(r.get("created_at") or 0), reverse=True)
        out: List[MemoryHit] = []
        for i, row in enumerate(rows[:target_k]):
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            score = max(0.0, 1.0 - (0.01 * i))
            out.append(
                MemoryHit(
                    text=text,
                    score=score,
                    created_at=int(row.get("created_at") or 0),
                    memory_type="summary",
                )
            )
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
        del memory_type, confidence, importance, metadata
        now = int(time.time())
        for raw in texts or []:
            text = sanitize_memory_text(raw, max_chars=self._text_max_chars)
            if not text:
                continue
            hash_value = _hash_text(text)
            # Refresh existing duplicate summary to most-recent.
            self._rows = [
                r
                for r in self._rows
                if not (
                    r.get("chat_id") == chat_id and r.get("type") == "summary" and r.get("hash") == hash_value
                )
            ]
            self._rows.append(
                {
                    "chat_id": chat_id,
                    "type": "summary",
                    "text": text,
                    "hash": hash_value,
                    "created_at": now,
                }
            )
        self._prune_expired(chat_id)
        self._prune_keep(chat_id)

    def update_conv_summary(self, chat_id: str, summary_text: str) -> None:
        if not summary_text:
            return
        self.save_memories(chat_id, [summary_text], memory_type="summary")

    def delete_chat(self, chat_id: str) -> None:
        self._rows = [r for r in self._rows if r.get("chat_id") != chat_id]

    def _prune_keep(self, chat_id: str) -> None:
        rows = [r for r in self._rows if r.get("chat_id") == chat_id and r.get("type") == "summary"]
        if len(rows) <= self._max_summaries_per_chat:
            return
        rows.sort(key=lambda r: int(r.get("created_at") or 0), reverse=True)
        keep_hashes = {r.get("hash") for r in rows[: self._max_summaries_per_chat]}
        self._rows = [
            r
            for r in self._rows
            if r.get("chat_id") != chat_id
            or r.get("type") != "summary"
            or r.get("hash") in keep_hashes
        ]

    def _prune_expired(self, chat_id: str) -> None:
        cutoff = int(time.time() - (self._summary_max_age_days * 86400.0))
        self._rows = [
            r
            for r in self._rows
            if r.get("chat_id") != chat_id
            or r.get("type") != "summary"
            or int(r.get("created_at") or 0) >= cutoff
        ]


__all__ = ["LongTermMemoryStore", "MemoryHit", "sanitize_memory_text"]
