from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Protocol, Sequence

from .models import EmbeddedChunk, EmbeddingTask

logger = logging.getLogger(__name__)


class EmbeddingConnector(Protocol):
    """Abstract connector for embedding providers (local or cloud)."""

    def supports(self, model: str) -> bool:
        ...

    def embed(self, model: str, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        ...


@dataclass
class EmbeddingConfig:
    """Configuration knobs for embedding execution."""

    # Keep batches small for local CPU embedding to avoid huge ORT calls that can
    # appear "hung" on long-sequence models (e.g., 2048 token encoders).
    batch_size: int = 8
    prefer_local: bool = True


class EmbeddingClient:
    """
    Orchestrates embedding requests across local/cloud connectors with batching.
    """

    def __init__(
        self,
        *,
        connectors: Iterable[EmbeddingConnector],
        config: EmbeddingConfig | None = None,
    ) -> None:
        connectors = list(connectors)
        if not connectors:
            raise ValueError("At least one embedding connector must be provided")

        self._connectors = connectors
        self._config = config or EmbeddingConfig()

    def embed_tasks(self, tasks: Sequence[EmbeddingTask]) -> List[EmbeddedChunk]:
        if not tasks:
            return []

        grouped: Dict[str, List[EmbeddingTask]] = {}
        for task in tasks:
            grouped.setdefault(task.embedding_model, []).append(task)

        results: List[EmbeddedChunk] = []
        for model_name, model_tasks in grouped.items():
            connector = self._select_connector(model_name)
            logger.debug(
                "Embedding %d chunks with model %s via %s",
                len(model_tasks),
                model_name,
                type(connector).__name__,
            )
            batches = self._batched(model_tasks, self._config.batch_size)
            for batch in batches:
                vectors = connector.embed(model_name, [item.text for item in batch])
                if len(vectors) != len(batch):
                    raise RuntimeError(
                        f"Connector returned {len(vectors)} vectors for {len(batch)} inputs"
                    )
                for task, vector in zip(batch, vectors):
                    results.append(
                        EmbeddedChunk(
                            chunk_id=task.chunk_id,
                            vector=tuple(vector),
                            embedding_model=model_name,
                            embedding_version=task.embedding_version,
                            metadata=task.metadata,
                        )
                    )

        return results

    def _select_connector(self, model: str) -> EmbeddingConnector:
        candidates = [connector for connector in self._connectors if connector.supports(model)]
        if not candidates:
            raise RuntimeError(f"No embedding connector available for model {model!r}")
        if self._config.prefer_local and len(candidates) > 1:
            for connector in candidates:
                if getattr(connector, "is_local", False):
                    return connector
        return candidates[0]

    @staticmethod
    def _batched(sequence: Sequence[EmbeddingTask], batch_size: int) -> List[List[EmbeddingTask]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        batches: List[List[EmbeddingTask]] = []
        batch: List[EmbeddingTask] = []
        for item in sequence:
            batch.append(item)
            if len(batch) >= batch_size:
                batches.append(batch)
                batch = []
        if batch:
            batches.append(batch)
        return batches


__all__ = ["EmbeddingConnector", "EmbeddingConfig", "EmbeddingClient"]
