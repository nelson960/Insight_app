from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


class FileIngestionStatus(str, enum.Enum):
    """Lifecycle state for a file during ingestion."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ChunkIndexStatus(str, enum.Enum):
    """Indexing status for individual chunks."""

    PENDING = "pending"
    INDEXED = "indexed"
    STALE = "stale"


class JobType(str, enum.Enum):
    """Supported ingestion-related job types."""

    INGESTION = "ingestion"
    REINDEX = "reindex"
    MIGRATION = "migration"
    CLEANUP = "cleanup"


class JobStatus(str, enum.Enum):
    """State progression for ingestion background jobs."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class IngestionErrorCode(str, enum.Enum):
    """Categorised error codes aligned with the ingestion retry policy."""

    EMBEDDING_TIMEOUT = "EMBEDDING_TIMEOUT"
    RATE_LIMIT = "RATE_LIMIT"
    INDEX_WRITE_FAIL = "INDEX_WRITE_FAIL"
    CORRUPT_FILE = "CORRUPT_FILE"
    EXTRACTION_FAILED = "EXTRACTION_FAILED"
    UNKNOWN = "UNKNOWN"


@dataclass
class FilePolicy:
    """Policy controls applied to a file (PII flags, confidentiality, etc.)."""

    pii: bool = False
    confidential: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FileRecord:
    """Metadata representation of a file stored in SQLite."""

    id: str
    filename: str
    mime: str
    size_bytes: int
    hash: str
    stored_path: str
    is_encrypted: bool
    status: FileIngestionStatus
    policy: FilePolicy
    pages: Optional[int]
    img_meta: Optional[Dict[str, Any]]
    error_code: Optional[IngestionErrorCode]
    error_message: Optional[str]
    retry_count: int
    next_retry_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime


@dataclass
class ChunkMetadata:
    """Lightweight representation of a chunk row in SQLite."""

    id: str
    file_id: str
    seq: int
    start_offset: int
    end_offset: int
    page: Optional[int]
    embedding_model: str
    embedding_version: int
    embedding_id: Optional[str]
    index_status: ChunkIndexStatus
    meta: Dict[str, Any]
    created_at: datetime


@dataclass
class IngestionRequest:
    """Input contract for the ingestion pipeline."""

    file_id: str
    source_path: str
    filename: str
    mime: str
    size_bytes: int
    hash: str
    policy: FilePolicy
    created_at: datetime
    user_id: str = "default"
    chat_id: Optional[str] = None
    source: str = "upload"
    tags: List[str] = field(default_factory=list)
    extra_metadata: Dict[str, Any] = field(default_factory=dict)
    cleanup_path: Optional[str] = None


@dataclass
class NormalizedDocument:
    """Document representation after normalization but before extraction."""

    file_id: str
    normalized_path: str
    mime: str
    is_encrypted: bool
    temp_files: List[str] = field(default_factory=list)


@dataclass
class ExtractionResult:
    """Structured output of the extraction stage."""

    file_id: str
    text_segments: List[str]
    pages: Optional[int]
    metadata: Dict[str, Any]
    plain_text: str = ""
    blocks: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ChunkPayload:
    """Chunk content and metadata ready for embedding."""

    chunk_id: str
    file_id: str
    sequence: int
    text: str
    metadata: Dict[str, Any]
    page: Optional[int] = None
    user_id: str = "default"
    chat_id: Optional[str] = None
    source: str = "upload"
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class EmbeddingTask:
    """Payload that must be embedded."""

    chunk_id: str
    text: str
    embedding_model: str
    embedding_version: int
    metadata: Dict[str, Any]


@dataclass
class EmbeddedChunk:
    """Result of an embedding call."""

    chunk_id: str
    vector: Sequence[float]
    embedding_model: str
    embedding_version: int
    metadata: Dict[str, Any]


@dataclass
class IndexWriteBatch:
    """Batch wrapper for atomic writes to the vector index."""

    embedding_model: str
    embedding_version: int
    entries: List[Tuple[str, Sequence[float], Dict[str, Any]]]  # (chunk_id, vector, payload)


@dataclass
class RetryState:
    """Tracks exponential backoff metadata for retries."""

    retry_count: int = 0
    next_retry_at: Optional[datetime] = None


@dataclass
class IngestionJobContext:
    """Aggregated state passed between pipeline stages."""

    request: IngestionRequest
    file_record: Optional[FileRecord] = None
    normalized: Optional[NormalizedDocument] = None
    extraction: Optional[ExtractionResult] = None
    chunks: List[ChunkPayload] = field(default_factory=list)
    embedded_chunks: List[EmbeddedChunk] = field(default_factory=list)
    retry: RetryState = field(default_factory=RetryState)


__all__ = [
    "FileIngestionStatus",
    "ChunkIndexStatus",
    "JobType",
    "JobStatus",
    "IngestionErrorCode",
    "FilePolicy",
    "FileRecord",
    "ChunkMetadata",
    "IngestionRequest",
    "NormalizedDocument",
    "ExtractionResult",
    "ChunkPayload",
    "EmbeddingTask",
    "EmbeddedChunk",
    "IndexWriteBatch",
    "RetryState",
    "IngestionJobContext",
]
