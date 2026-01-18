from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

logger = logging.getLogger(__name__)


@dataclass
class QdrantConfig:
    collection_name: str = "insight_chunks"
    vector_size: int = 768
    distance: str = "Cosine"
    path: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    prefer_grpc: bool = False


class QdrantVectorIndex:
    """Lightweight Qdrant adapter with automatic collection creation."""

    def __init__(self, config: QdrantConfig) -> None:
        self._config = config
        self._vector_size = config.vector_size
        self._client = self._create_client(config)
        self._ensure_collection()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @property
    def client(self) -> QdrantClient:
        return self._client

    @property
    def collection_name(self) -> str:
        return self._config.collection_name

    def add(
        self,
        ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
        payloads: Optional[Sequence[dict]] = None,
    ) -> None:
        if not ids:
            return
        if len(ids) != len(vectors):
            raise ValueError("ids and vectors must have the same length")

        vector_size = len(vectors[0]) if vectors else self._vector_size
        if vector_size != self._vector_size:
            logger.info("Recreating Qdrant collection %s with vector size %s", self.collection_name, vector_size)
            self._vector_size = vector_size
            self._recreate_collection()

        points = []
        for idx, vec in enumerate(vectors):
            payload = payloads[idx].copy() if payloads and idx < len(payloads) and payloads[idx] else {}
            raw_id = ids[idx]
            try:
                point_id = str(uuid.UUID(str(raw_id)))
            except Exception:
                point_id = uuid.uuid5(uuid.NAMESPACE_DNS, str(raw_id)).hex
            if isinstance(payload, dict) and "chunk_id" not in payload:
                payload["chunk_id"] = str(raw_id)
            points.append(
                qmodels.PointStruct(
                    id=point_id,
                    vector=vec,
                    payload=payload,
                )
            )
        self._client.upsert(collection_name=self.collection_name, points=points)

    def flush(self) -> None:
        try:
            self._client.get_collection(self.collection_name)
        except Exception:
            return

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _create_client(self, config: QdrantConfig) -> QdrantClient:
        if config.path:
            Path(config.path).mkdir(parents=True, exist_ok=True)
            return QdrantClient(path=str(config.path), prefer_grpc=config.prefer_grpc)
        host = config.host or "localhost"
        port = config.port or 6333
        return QdrantClient(host=host, port=port, prefer_grpc=config.prefer_grpc)

    def _resolve_distance(self) -> qmodels.Distance:
        try:
            return getattr(qmodels.Distance, self._config.distance.upper())
        except Exception:
            return qmodels.Distance.COSINE

    def _ensure_collection(self) -> None:
        try:
            self._client.get_collection(self.collection_name)
        except Exception:
            self._recreate_collection()

    def _recreate_collection(self) -> None:
        vectors_config = qmodels.VectorParams(size=self._vector_size, distance=self._resolve_distance())
        self._client.recreate_collection(collection_name=self.collection_name, vectors_config=vectors_config)


def create_qdrant_index(config: Optional[QdrantConfig] = None) -> QdrantVectorIndex:
    return QdrantVectorIndex(config or QdrantConfig())


__all__ = ["QdrantVectorIndex", "QdrantConfig", "create_qdrant_index"]
