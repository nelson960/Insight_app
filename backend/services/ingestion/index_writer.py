from __future__ import annotations

import logging
import threading
from typing import Optional, Protocol, Sequence

from .models import ChunkIndexStatus, IndexWriteBatch

logger = logging.getLogger(__name__)


class VectorIndex(Protocol):
    """Protocol defining the minimal vector index operations we rely on."""

    def add(
        self,
        ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
        payloads: Optional[Sequence[dict]] = None,
    ) -> None:
        ...

    def flush(self) -> None:
        ...


class ChunkStore(Protocol):
    """Protocol for persisting chunk indexing state to SQLite."""

    def mark_indexed(self, chunk_ids: Sequence[str]) -> None:
        ...

    def mark_stale(self, chunk_ids: Sequence[str]) -> None:
        ...


class VectorIndexWriter:
    """
    Handles batched, lock-guarded writes to the vector index with SQLite updates.
    """

    def __init__(self, index: VectorIndex, chunk_store: ChunkStore) -> None:
        self._index = index
        self._chunk_store = chunk_store
        self._lock = threading.Lock()

    def write_batch(self, batch: IndexWriteBatch) -> None:
        if not batch.entries:
            logger.debug("Skipping empty index batch for model %s", batch.embedding_model)
            return

        with self._lock:
            chunk_ids = [chunk_id for chunk_id, _, _ in batch.entries]
            vectors = [vector for _, vector, _ in batch.entries]
            payloads = [payload for _, _, payload in batch.entries]
            logger.debug(
                "Writing %d vectors to index (model=%s, version=%s)",
                len(batch.entries),
                batch.embedding_model,
                batch.embedding_version,
            )
            try:
                self._index.add(chunk_ids, vectors, payloads)
                try:
                    self._index.flush()
                except AttributeError:
                    logger.debug("Vector index flush no-op for current backend")
            except Exception:
                logger.exception("Vector index write failed; marking chunks as stale")
                self._chunk_store.mark_stale(chunk_ids)
                raise
            else:
                self._chunk_store.mark_indexed(chunk_ids)


__all__ = ["VectorIndex", "ChunkStore", "VectorIndexWriter"]
