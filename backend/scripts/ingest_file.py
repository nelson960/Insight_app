#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.workspace import get_workspace
from backend.services.bootstrap import create_storage_backends
from backend.services.extraction.detector import detect_mime_type
from backend.services.ingestion import (
    FilePolicy,
    IngestionPipelineConfig,
    IngestionRequest,
    create_ingestion_pipeline,
)
from backend.services.logging_config import configure_logging
from backend.services.storage.qdrant_index import QdrantConfig
from backend.services.storage.sqlite_store import SQLiteConfig

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path("backend/models/nomic-embed-text")
DEFAULT_COLLECTION = "insight_chunks"
DEFAULT_EMBED_MODEL = "nomic-embed-text-v1.5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest a file into the Insight pipeline.")
    parser.add_argument("file", type=Path, help="Path to the file to ingest")
    parser.add_argument("--workspace", type=Path, default=None, help="Override workspace root directory")
    parser.add_argument("--sqlite", type=Path, default=None, help="Path to SQLite metadata database")
    parser.add_argument("--qdrant-path", type=Path, default=None, help="Directory for embedded Qdrant storage")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Local directory containing the nomic embedding model")
    parser.add_argument("--collection", type=str, default=DEFAULT_COLLECTION, help="Qdrant collection name")
    parser.add_argument("--embedding-model", type=str, default=DEFAULT_EMBED_MODEL, help="Embedding model identifier")
    parser.add_argument("--embedding-version", type=int, default=1, help="Embedding model version")
    parser.add_argument("--file-id", type=str, default=None, help="Optional explicit file ID")
    parser.add_argument("--user-id", type=str, default="default", help="Owning user ID for multi-tenant isolation")
    parser.add_argument("--chat-id", type=str, default=None, help="Chat/session identifier")
    parser.add_argument("--source", type=str, default="upload", help="Source label for the file (upload, sync, etc.)")
    parser.add_argument("--tag", dest="tags", action="append", default=[], help="Tag to attach to all chunks (repeatable)")
    parser.add_argument("--metadata", type=str, default=None, help="JSON blob merged into chunk metadata")
    parser.add_argument("--created-at", type=str, default=None, help="ISO8601 timestamp override for file ingest time")
    return parser.parse_args()


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> None:
    args = parse_args()
    configure_logging()
    workspace = get_workspace(args.workspace)
    sqlite_path = Path(args.sqlite) if args.sqlite else workspace.db
    qdrant_path = Path(args.qdrant_path) if args.qdrant_path else workspace.qdrant

    file_path = args.file.expanduser().resolve()
    if not file_path.exists():
        logger.error("File %s does not exist", file_path)
        raise SystemExit(1)

    workspace.base.mkdir(parents=True, exist_ok=True)
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    qdrant_path.mkdir(parents=True, exist_ok=True)

    metadata_store, vector_index = create_storage_backends(
        sqlite_path=sqlite_path,
        sqlite_config=SQLiteConfig(),
        qdrant_config=QdrantConfig(
            path=str(qdrant_path),
            collection_name=args.collection,
            vector_size=768,
        ),
    )

    pipeline = create_ingestion_pipeline(
        metadata_store=metadata_store,
        vector_index=vector_index,
        pipeline_config=IngestionPipelineConfig(
            embedding_model=args.embedding_model,
            embedding_version=args.embedding_version,
        ),
        nomic_model_dir=args.model_dir,
    )

    file_id = args.file_id or f"file_{uuid4().hex}"
    mime = detect_mime_type(file_path)
    size_bytes = file_path.stat().st_size
    hash_value = compute_sha256(file_path)
    metadata_store.register_file(
        file_id=file_id,
        filename=file_path.name,
        mime=mime,
        size_bytes=size_bytes,
        hash_value=hash_value,
        stored_path=str(file_path),
        is_encrypted=False,
        policy={"pii": False, "confidential": False},
    )

    try:
        extra_meta = json.loads(args.metadata) if args.metadata else {}
    except json.JSONDecodeError as exc:
        logger.error("Invalid metadata JSON: %s", exc)
        raise SystemExit(3)

    created_at = datetime.fromisoformat(args.created_at) if args.created_at else datetime.now(timezone.utc)

    request = IngestionRequest(
        file_id=file_id,
        source_path=str(file_path),
        filename=file_path.name,
        mime=mime,
        size_bytes=size_bytes,
        hash=hash_value,
        policy=FilePolicy(),
        created_at=created_at,
        user_id=args.user_id,
        chat_id=args.chat_id,
        source=args.source,
        tags=args.tags,
        extra_metadata=extra_meta,
    )

    logger.info("Starting ingestion for %s (id=%s)", file_path, file_id)
    try:
        pipeline.run(request)
    except Exception:
        logger.exception("Ingestion failed for %s", file_id)
        raise SystemExit(2)

    logger.info("Ingestion succeeded for %s", file_id)
    print(f"Ingestion succeeded for file_id={file_id}")


if __name__ == "__main__":
    main()
