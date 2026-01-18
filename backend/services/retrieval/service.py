from __future__ import annotations

import logging
from time import perf_counter
from typing import Dict, List, Sequence, Optional

from qdrant_client import QdrantClient

from ..storage.sqlite_store import SQLiteMetadataStore
from .models import RetrievalCandidate, RetrievalQuery, RetrievalResult
from .context import RetrievalContext, build_filter

logger = logging.getLogger(__name__)


class RetrievalService:
    """Performs dense retrieval using Qdrant plus SQLite metadata enrichment."""

    def __init__(
        self,
        *,
        qdrant_client: QdrantClient,
        collection_name: str,
        metadata_store: SQLiteMetadataStore,
    ) -> None:
        self._qdrant = qdrant_client
        self._collection = collection_name
        self._metadata_store = metadata_store

    @property
    def qdrant_client(self) -> QdrantClient:
        return self._qdrant

    @property
    def collection_name(self) -> str:
        return self._collection

    def search(self, query: RetrievalQuery, *, context: Optional[RetrievalContext] = None) -> List[RetrievalResult]:
        logger.debug(
            "Searching Qdrant collection %s (limit=%s)",
            self._collection,
            query.limit,
        )
        q_filter = build_filter(context)
        qdrant_start = perf_counter()
        points = self._qdrant.search(
            collection_name=self._collection,
            query_vector=list(query.vector),
            limit=query.limit,
            with_vectors=False,
            with_payload=query.with_metadata or q_filter is not None,
            query_filter=q_filter,
        )
        qdrant_ms = (perf_counter() - qdrant_start) * 1000
        candidates = [
            RetrievalCandidate(
                chunk_id=(point.payload or {}).get("chunk_id") or str(point.id),
                score=point.score,
                payload=point.payload or {},
            )
            for point in points
        ]
        if not candidates:
            logger.info(
                "Retrieval timings collection=%s qdrant=%.1fms candidates=0",
                self._collection,
                qdrant_ms,
            )
            return []
        sqlite_start = perf_counter()
        chunk_details = self._metadata_store.fetch_chunks([c.chunk_id for c in candidates])
        sqlite_ms = (perf_counter() - sqlite_start) * 1000
        logger.info(
            "Retrieval timings collection=%s qdrant=%.1fms sqlite=%.1fms candidates=%d",
            self._collection,
            qdrant_ms,
            sqlite_ms,
            len(candidates),
        )
        details_by_id: Dict[str, Dict[str, object]] = {row["id"]: row for row in chunk_details}

        results: List[RetrievalResult] = []
        for candidate in candidates:
            detail = details_by_id.get(candidate.chunk_id, {})
            detail_meta = detail.get("metadata", {}) or {}
            payload_meta = (candidate.payload or {}).get("metadata", {}) if candidate.payload else {}
            combined_meta = {**payload_meta, **detail_meta}
            merged_metadata: Dict[str, object] = {**(candidate.payload or {})}
            if "metadata" in merged_metadata:
                merged_metadata.pop("metadata")
            merged_metadata.update(
                {
                    "user_id": detail.get("user_id") or (candidate.payload or {}).get("user_id"),
                    "chat_id": detail.get("chat_id") or (candidate.payload or {}).get("chat_id"),
                    "source": detail.get("source") or (candidate.payload or {}).get("source"),
                    "tags": detail.get("tags") or (candidate.payload or {}).get("tags"),
                    "created_at": detail.get("chunk_created_at") or (candidate.payload or {}).get("created_at"),
                    "filename": detail.get("filename") or (candidate.payload or {}).get("filename"),
                    "metadata": combined_meta,
                }
            )
            results.append(
                RetrievalResult(
                    chunk_id=candidate.chunk_id,
                    score=candidate.score,
                    text=detail.get("text") if query.with_chunks else None,
                    metadata=merged_metadata,
                    file_id=detail.get("file_id"),
                    seq=detail.get("seq"),
                )
            )
        return results

    def fetch_chunks_for_files(self, file_ids: Sequence[str], *, limit_per_file: int = 8) -> dict[str, list[dict[str, object]]]:
        """
        Fetch chunk texts for given file_ids ordered by seq.

        Note: this uses SQLite as the source of truth (stable ordering + avoids dummy vectors).
        """
        if not file_ids:
            return {}
        out: Dict[str, list[dict[str, object]]] = {}
        try:
            out = self._metadata_store.fetch_chunks_for_files(file_ids, limit_per_file=limit_per_file)
        except Exception:
            logger.warning("fetch_chunks_for_files failed", exc_info=True)
            out = {fid: [] for fid in file_ids}

        # Enrich with filename (best-effort) for UI/debug consumers.
        for fid in file_ids:
            chunks = out.get(fid) or []
            try:
                file_row = self._metadata_store.get_file(fid) or {}
                filename = file_row.get("filename")
            except Exception:
                filename = None
            if filename:
                for c in chunks:
                    if isinstance(c, dict) and "filename" not in c:
                        c["filename"] = filename
        return out

    def delete_chunks_for_chat(self, chat_id: str) -> None:
        """
        Delete all chunks in this collection that belong to a chat_id.
        """
        try:
            self._qdrant.delete(
                collection_name=self._collection,
                points_selector={
                    "filter": {
                        "must": [
                            {"key": "chat_id", "match": {"value": chat_id}}
                        ]
                    }
                },
            )
        except Exception as exc:
            logger.warning("Failed to delete chunks for chat_id=%s: %s", chat_id, exc)

    def delete_chunks_for_file(self, file_id: str) -> None:
        """Delete all chunks in this collection that belong to a file_id."""
        if not file_id:
            return
        try:
            self._qdrant.delete(
                collection_name=self._collection,
                points_selector={
                    "filter": {
                        "must": [
                            {"key": "file_id", "match": {"value": file_id}}
                        ]
                    }
                },
            )
        except Exception as exc:
            logger.warning("Failed to delete chunks for file_id=%s: %s", file_id, exc)

    def delete_chunks_for_chat_file(self, chat_id: str, file_id: str) -> None:
        """Delete all chunks in this collection that belong to a (chat_id, file_id)."""
        if not chat_id or not file_id:
            return
        try:
            self._qdrant.delete(
                collection_name=self._collection,
                points_selector={
                    "filter": {
                        "must": [
                            {"key": "chat_id", "match": {"value": chat_id}},
                            {"key": "file_id", "match": {"value": file_id}},
                        ]
                    }
                },
            )
        except Exception as exc:
            logger.warning(
                "Failed to delete chunks for chat_id=%s file_id=%s: %s",
                chat_id,
                file_id,
                exc,
            )


__all__ = ["RetrievalService"]
