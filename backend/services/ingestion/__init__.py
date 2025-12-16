"""
Factory helpers and public symbols for the ingestion service package.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

from ..extraction import FileExtractionService, create_extraction_service
from .chunker import ChunkerConfig, SimpleChunker
from ..connectors import NomicEmbedTextConnector, NomicOnnxEmbedTextConnector, NomicOnnxConfig
from .embedder import EmbeddingClient, EmbeddingConfig, EmbeddingConnector
from .index_writer import ChunkStore, VectorIndex, VectorIndexWriter
from .models import (
    ChunkIndexStatus,
    ChunkMetadata,
    ChunkPayload,
    EmbeddedChunk,
    EmbeddingTask,
    FileIngestionStatus,
    FilePolicy,
    FileRecord,
    IngestionJobContext,
    IngestionRequest,
    IndexWriteBatch,
    NormalizedDocument,
    RetryState,
)
from .pipeline import (
    FileNormalizer,
    IngestionPipeline,
    IngestionPipelineConfig,
    IngestionPipelineError,
    MetadataStore,
)
from .retry import RetryPolicy
from .worker import IngestionJob, IngestionWorker, JobQueue


def create_ingestion_pipeline(
    *,
    metadata_store: MetadataStore,
    vector_index: VectorIndex,
    embedding_connectors: Optional[Iterable[EmbeddingConnector]] = None,
    pipeline_config: IngestionPipelineConfig,
    retry_policy: Optional[RetryPolicy] = None,
    chunker_config: Optional[ChunkerConfig] = None,
    embedding_config: Optional[EmbeddingConfig] = None,
    normalizer: Optional[FileNormalizer] = None,
    nomic_model_dir: Optional[Path] = None,
    nomic_auto_download: bool = False,
    extraction_service: Optional[FileExtractionService] = None,
    use_onnx_embeddings: bool = True,
) -> IngestionPipeline:
    """
    Assemble an `IngestionPipeline` with sensible defaults for dependencies.
    """

    extraction = extraction_service or create_extraction_service()
    chunker = SimpleChunker(config=chunker_config)
    connectors = list(embedding_connectors) if embedding_connectors else _default_embedding_connectors(
        nomic_model_dir=nomic_model_dir,
        auto_download=nomic_auto_download,
        use_onnx=use_onnx_embeddings,
    )
    embedder = EmbeddingClient(connectors=connectors, config=embedding_config)
    index_writer = VectorIndexWriter(index=vector_index, chunk_store=metadata_store)
    retry = retry_policy or RetryPolicy()

    return IngestionPipeline(
        extraction_service=extraction,
        chunker=chunker,
        embedder=embedder,
        index_writer=index_writer,
        metadata_store=metadata_store,
        retry_policy=retry,
        normalizer=normalizer,
        config=pipeline_config,
    )


__all__ = [
    "ChunkerConfig",
    "SimpleChunker",
    "EmbeddingClient",
    "EmbeddingConfig",
    "EmbeddingConnector",
    "NomicEmbedTextConnector",
    "NomicOnnxEmbedTextConnector",
    "NomicOnnxConfig",
    "VectorIndexWriter",
    "VectorIndex",
    "ChunkStore",
    "FileIngestionStatus",
    "ChunkIndexStatus",
    "FilePolicy",
    "FileRecord",
    "ChunkMetadata",
    "ChunkPayload",
    "EmbeddedChunk",
    "EmbeddingTask",
    "IngestionRequest",
    "IndexWriteBatch",
    "NormalizedDocument",
    "RetryState",
    "IngestionPipeline",
    "IngestionPipelineConfig",
    "IngestionPipelineError",
    "FileNormalizer",
    "MetadataStore",
    "RetryPolicy",
    "IngestionWorker",
    "IngestionJob",
    "JobQueue",
    "create_ingestion_pipeline",
    "FileExtractionService",
]


def _default_embedding_connectors(
    *,
    nomic_model_dir: Optional[Path],
    auto_download: bool,
    use_onnx: bool,
) -> Sequence[EmbeddingConnector]:
    """
    Provide default embedding connectors (currently Nomic embed text only).
    """

    model_dir = nomic_model_dir or Path("backend/models/nomic-embed-text")
    if use_onnx:
        return [NomicOnnxEmbedTextConnector(model_dir=model_dir, config=NomicOnnxConfig())]
    return [
        NomicEmbedTextConnector(
            model_dir=model_dir,
            auto_download=auto_download,
        )
    ]
