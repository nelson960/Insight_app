#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.workspace import get_workspace
from backend.services.bootstrap import create_storage_backends
from backend.services.ingestion import NomicOnnxEmbedTextConnector, NomicOnnxConfig
from backend.services.logging_config import configure_logging
from backend.services.retrieval import RetrievalQuery, RetrievalService, RetrievalContext
from backend.services.storage import QdrantConfig, SQLiteConfig

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path("backend/models/nomic-embed-text")
DEFAULT_COLLECTION = "insight_chunks"
DEFAULT_EMBED_MODEL = "nomic-embed-text-v1.5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search ingested chunks using Insight retrieval.")
    parser.add_argument("query", type=str, help="Text query to search for")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--workspace", type=Path, default=None, help="Override workspace root directory")
    parser.add_argument("--sqlite", type=Path, default=None, help="Path to SQLite metadata database")
    parser.add_argument("--qdrant-path", type=Path, default=None, help="Directory for embedded Qdrant storage")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Local directory containing the nomic embedding model")
    parser.add_argument("--collection", type=str, default=DEFAULT_COLLECTION, help="Qdrant collection name")
    parser.add_argument("--embedding-model", type=str, default=DEFAULT_EMBED_MODEL, help="Embedding model identifier")
    parser.add_argument("--user-id", type=str, default=None, help="Filter by user ID")
    parser.add_argument("--chat-id", type=str, default=None, help="Filter by chat/session ID")
    parser.add_argument("--file-id", dest="file_ids", action="append", default=[], help="Filter by file ID (repeatable)")
    parser.add_argument("--tag", dest="tags", action="append", default=[], help="Filter by tag (repeatable)")
    parser.add_argument("--after", type=str, default=None, help="Filter chunks created after timestamp (ISO8601)")
    parser.add_argument("--before", type=str, default=None, help="Filter chunks created before timestamp (ISO8601)")
    return parser.parse_args()


def embed_query(query: str, model_dir: Path, model_name: str) -> list[float]:
    connector = NomicOnnxEmbedTextConnector(model_dir=model_dir, config=NomicOnnxConfig())
    vectors = connector.embed(model_name, [query])
    return list(vectors[0])


def _build_context(args: argparse.Namespace) -> RetrievalContext | None:
    has_filters = any([args.user_id, args.chat_id, args.file_ids, args.tags, args.after, args.before])
    if not has_filters:
        return None
    date_range = None
    if args.after or args.before:
        start = args.after or "1970-01-01T00:00:00Z"
        end = args.before or datetime.now(timezone.utc).isoformat()
        date_range = (start, end)
    return RetrievalContext(
        user_id=args.user_id,
        chat_id=args.chat_id,
        file_ids=args.file_ids,
        tags=args.tags,
        date_range=date_range,
    )


def main() -> None:
    args = parse_args()
    configure_logging()
    workspace = get_workspace(args.workspace)
    sqlite_path = Path(args.sqlite) if args.sqlite else workspace.db
    qdrant_path = Path(args.qdrant_path) if args.qdrant_path else workspace.qdrant
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
    retrieval = RetrievalService(
        qdrant_client=vector_index.client,
        collection_name=vector_index.collection_name,
        metadata_store=metadata_store,
    )

    logger.info("Embedding query: %s", args.query)
    query_vector = embed_query(args.query, args.model_dir, args.embedding_model)
    results = retrieval.search(
        RetrievalQuery(
            vector=query_vector,
            limit=args.top_k,
            with_chunks=True,
            with_metadata=True,
        ),
        context=_build_context(args),
    )
    if not results:
        logger.info("No retrieval results for query %s", args.query)
        print("No matches found.")
        return

    logger.info("Retrieved %d results", len(results))
    for idx, result in enumerate(results, start=1):
        print(f"Result #{idx}")
        print(f"  chunk_id: {result.chunk_id}")
        print(f"  file_id : {result.file_id}")
        print(f"  score   : {result.score:.4f}")
        if result.text:
            snippet = result.text[:400].replace("\n", " ")
            print(f"  text    : {snippet}...")
        if result.metadata:
            print(f"  metadata: {result.metadata}")
        print("-" * 40)


if __name__ == "__main__":
    main()
