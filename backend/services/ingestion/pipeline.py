from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, Sequence

from ..extraction import FileExtractionResult, FileExtractionService, ExtractionError
from .embedder import EmbeddingClient
from .index_writer import ChunkStore, VectorIndexWriter
from .models import (
    ChunkPayload,
    EmbeddedChunk,
    EmbeddingTask,
    FileIngestionStatus,
    IngestionErrorCode,
    IngestionJobContext,
    IngestionRequest,
    IndexWriteBatch,
    NormalizedDocument,
    ExtractionResult,
)
from .retry import RetryPolicy, classify_exception

logger = logging.getLogger(__name__)

CancelCheck = Callable[[], bool]


class Chunker(Protocol):
    """Chunker interface (block-aware or plain-text)."""

    def chunk(self, extraction: ExtractionResult) -> Sequence[ChunkPayload]:
        ...


class IngestionCancelled(RuntimeError):
    """Raised when an ingestion job is cancelled mid-flight."""


class FileNormalizer(Protocol):
    """Produces a normalized document ready for extraction."""

    def normalize(self, request: IngestionRequest) -> NormalizedDocument:
        ...


class MetadataStore(ChunkStore, Protocol):
    """Abstract persistence APIs for ingestion metadata."""

    def mark_file_status(self, file_id: str, status: FileIngestionStatus) -> None:
        ...

    def record_extraction(self, file_id: str, *, pages: int | None, metadata: dict[str, object]) -> None:
        ...

    def upsert_file_text(self, file_id: str, *, text: str, blocks: Sequence[dict]) -> None:
        ...

    def stage_chunks(
        self,
        file_id: str,
        chunks: Sequence[ChunkPayload],
        *,
        embedding_model: str,
        embedding_version: int,
    ) -> None:
        ...

    def mark_file_completed(self, file_id: str) -> None:
        ...

    def record_failure(
        self,
        file_id: str,
        *,
        error_code: IngestionErrorCode,
        error_message: str,
        retry_state: object,
    ) -> None:
        ...


@dataclass
class IngestionPipelineConfig:
    """Static configuration required for the ingestion pipeline."""

    embedding_model: str
    embedding_version: int
    require_encryption: bool = False


@dataclass
class PassthroughNormalizer:
    """Default normalizer that trusts the supplied source path."""

    def normalize(self, request: IngestionRequest) -> NormalizedDocument:
        return NormalizedDocument(
            file_id=request.file_id,
            normalized_path=request.source_path,
            mime=request.mime,
            is_encrypted=False,
        )


class IngestionPipeline:
    """Full ingestion pipeline coordinating normalization through indexing."""

    def __init__(
        self,
        *,
        extraction_service: FileExtractionService,
        chunker: Chunker,
        embedder: EmbeddingClient,
        index_writer: VectorIndexWriter,
        metadata_store: MetadataStore,
        retry_policy: RetryPolicy,
        normalizer: FileNormalizer | None = None,
        config: IngestionPipelineConfig,
    ) -> None:
        self._extraction_service = extraction_service
        self._chunker = chunker
        self._embedder = embedder
        self._index_writer = index_writer
        self._metadata_store = metadata_store
        self._retry_policy = retry_policy
        self._normalizer = normalizer or PassthroughNormalizer()
        self._config = config

    def run(self, request: IngestionRequest, *, cancel_check: CancelCheck | None = None) -> IngestionJobContext:
        ctx = IngestionJobContext(request=request)
        logger.info("Starting ingestion for file %s", request.file_id)
        self._metadata_store.mark_file_status(request.file_id, FileIngestionStatus.PROCESSING)

        try:
            if cancel_check and cancel_check():
                raise IngestionCancelled("cancelled before start")

            ctx.normalized = self._normalizer.normalize(request)
            ctx.extraction = self._run_extraction(ctx.normalized)
            if cancel_check and cancel_check():
                raise IngestionCancelled("cancelled after extraction")
            self._metadata_store.upsert_file_text(
                request.file_id,
                text=ctx.extraction.plain_text,
                blocks=ctx.extraction.blocks,
            )
            self._metadata_store.record_extraction(
                request.file_id,
                pages=ctx.extraction.pages,
                metadata=ctx.extraction.metadata,
            )

            ctx.chunks = self._chunker.chunk(ctx.extraction)
            self._apply_chunk_context(ctx.chunks, request, ctx.extraction)
            if cancel_check and cancel_check():
                raise IngestionCancelled("cancelled before staging chunks")
            self._metadata_store.stage_chunks(
                request.file_id,
                ctx.chunks,
                embedding_model=self._config.embedding_model,
                embedding_version=self._config.embedding_version,
            )
            if cancel_check and cancel_check():
                raise IngestionCancelled("cancelled before embedding")

            embedding_tasks = [
                EmbeddingTask(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    embedding_model=self._config.embedding_model,
                    embedding_version=self._config.embedding_version,
                    metadata=self._build_payload(chunk),
                )
                for chunk in ctx.chunks
            ]
            ctx.embedded_chunks = self._embedder.embed_tasks(embedding_tasks)
            if cancel_check and cancel_check():
                raise IngestionCancelled("cancelled before index write")

            entries = []
            for chunk_payload, embedded in zip(ctx.chunks, ctx.embedded_chunks):
                payload = self._build_payload(chunk_payload)
                entries.append((embedded.chunk_id, embedded.vector, payload))
            index_batch = IndexWriteBatch(
                embedding_model=self._config.embedding_model,
                embedding_version=self._config.embedding_version,
                entries=entries,
            )
            self._index_writer.write_batch(index_batch)

        except IngestionCancelled:
            logger.info("Ingestion cancelled for file %s", request.file_id)
            # Mark file failed; the caller may clean up and/or delete chat data.
            self._metadata_store.mark_file_status(request.file_id, FileIngestionStatus.FAILED)
            raise
        except Exception as exc:
            error_code = classify_exception(exc)
            error_message = str(exc)
            logger.exception("Ingestion failed for file %s", request.file_id)
            if self._retry_policy.should_retry(error_code=error_code, state=ctx.retry):
                ctx.retry = self._retry_policy.compute_next_state(ctx.retry)
            self._metadata_store.record_failure(
                request.file_id,
                error_code=error_code,
                error_message=error_message,
                retry_state=ctx.retry,
            )
            self._metadata_store.mark_file_status(request.file_id, FileIngestionStatus.FAILED)
            raise IngestionPipelineError(
                file_id=request.file_id,
                error_code=error_code,
                context=ctx,
            ) from exc

        self._metadata_store.mark_file_completed(request.file_id)
        logger.info("Ingestion completed for file %s", request.file_id)
        return ctx

    def _run_extraction(self, document: NormalizedDocument) -> ExtractionResult:
        try:
            extracted = self._extraction_service.extract(
                Path(document.normalized_path),
                mime_type=document.mime,
            )
            logger.info(
                "Extraction result for file %s: %d chars (%s)",
                document.file_id,
                len(extracted.text),
                extracted.mime_type,
            )
        except ExtractionError:
            raise
        segments = self._segment_text(extracted)
        blocks = [
            {"kind": b.kind, "text": b.text, "metadata": b.metadata}
            for b in (getattr(extracted, "blocks", None) or [])
        ]
        return ExtractionResult(
            file_id=document.file_id,
            text_segments=segments,
            pages=extracted.metadata.get("page_count"),
            metadata=extracted.metadata,
            plain_text=extracted.text,
            blocks=blocks,
        )

    @staticmethod
    def _segment_text(extracted: FileExtractionResult) -> list[str]:
        text = extracted.text or ""
        chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
        if not chunks and text:
            chunks = [text]
        return chunks

    def _apply_chunk_context(
        self,
        chunks: Sequence[ChunkPayload],
        request: IngestionRequest,
        extraction: ExtractionResult,
    ) -> None:
        for chunk in chunks:
            chunk.user_id = request.user_id
            chunk.chat_id = request.chat_id
            chunk.source = request.source
            chunk.tags = sorted(set(request.tags + chunk.metadata.get("tags", [])))
            chunk.created_at = request.created_at
            chunk.metadata = {
                **chunk.metadata,
                "file_id": request.file_id,
                "page": chunk.page,
                "extraction_meta": extraction.metadata,
                "extra": request.extra_metadata,
            }

    @staticmethod
    def _build_payload(chunk: ChunkPayload) -> dict:
        return {
            "chunk_id": chunk.chunk_id,
            "file_id": chunk.file_id,
            "user_id": chunk.user_id,
            "chat_id": chunk.chat_id,
            "source": chunk.source,
            "tags": chunk.tags,
            "created_at": chunk.created_at.isoformat(),
            "metadata": chunk.metadata,
        }


class IngestionPipelineError(RuntimeError):
    """Wrapper that surfaces pipeline failures with context for the worker."""

    def __init__(
        self,
        *,
        file_id: str,
        error_code: IngestionErrorCode,
        context: IngestionJobContext,
    ) -> None:
        super().__init__(f"Ingestion pipeline failed for file {file_id} with {error_code.value}")
        self.file_id = file_id
        self.error_code = error_code
        self.context = context


__all__ = [
    "IngestionPipeline",
    "IngestionPipelineConfig",
    "FileNormalizer",
    "MetadataStore",
    "IngestionPipelineError",
]
