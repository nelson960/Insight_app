from .sqlite_store import SQLiteMetadataStore, SQLiteConfig, create_sqlite_store

__all__ = [
    "SQLiteMetadataStore",
    "SQLiteConfig",
    "create_sqlite_store",
    "QdrantConfig",
    "QdrantVectorIndex",
    "create_qdrant_index",
]


def __getattr__(name: str):
    # Keep Qdrant imports lazy so callers that only need SQLite don't require
    # optional qdrant-client at import time (important for lightweight CI jobs).
    if name in {"QdrantConfig", "QdrantVectorIndex", "create_qdrant_index"}:
        from .qdrant_index import QdrantConfig, QdrantVectorIndex, create_qdrant_index

        globals().update(
            {
                "QdrantConfig": QdrantConfig,
                "QdrantVectorIndex": QdrantVectorIndex,
                "create_qdrant_index": create_qdrant_index,
            }
        )
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
