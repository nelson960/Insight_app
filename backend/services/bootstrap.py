from __future__ import annotations

from pathlib import Path
from typing import Optional

from .ingestion import IngestionPipeline, IngestionPipelineConfig, create_ingestion_pipeline
from .logging_config import configure_logging
from .retrieval import RetrievalService
from .storage import (
    QdrantConfig,
    SQLiteConfig,
    create_qdrant_index,
    create_sqlite_store,
    QdrantVectorIndex,
    SQLiteMetadataStore,
)


def create_storage_backends(
    *,
    sqlite_path: Path | str,
    sqlite_config: Optional[SQLiteConfig] = None,
    qdrant_config: Optional[QdrantConfig] = None,
) -> tuple[SQLiteMetadataStore, QdrantVectorIndex]:
    """Instantiate storage backends for metadata (SQLite) and vectors (Qdrant)."""

    metadata_store = create_sqlite_store(sqlite_path, config=sqlite_config)
    vector_index = create_qdrant_index(config=qdrant_config)
    return metadata_store, vector_index


def create_core_services(
    *,
    sqlite_path: Path | str,
    pipeline_config: IngestionPipelineConfig,
    sqlite_config: Optional[SQLiteConfig] = None,
    qdrant_config: Optional[QdrantConfig] = None,
    nomic_model_dir: Optional[Path] = None,
    nomic_auto_download: bool = False,
) -> tuple[IngestionPipeline, RetrievalService]:
    """
    Assemble ingestion and retrieval services sharing the same storage backends.
    """

    configure_logging()
    metadata_store, vector_index = create_storage_backends(
        sqlite_path=sqlite_path,
        sqlite_config=sqlite_config,
        qdrant_config=qdrant_config,
    )
    ingestion = create_ingestion_pipeline(
        metadata_store=metadata_store,
        vector_index=vector_index,
        pipeline_config=pipeline_config,
        nomic_model_dir=nomic_model_dir,
        nomic_auto_download=nomic_auto_download,
    )
    retrieval = RetrievalService(
        qdrant_client=vector_index.client,
        collection_name=vector_index.collection_name,
        metadata_store=metadata_store,
    )
    return ingestion, retrieval


__all__ = ["create_storage_backends", "create_core_services"]
