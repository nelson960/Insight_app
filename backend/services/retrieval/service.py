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
        Fetch chunk texts for given file_ids directly from Qdrant payloads, ordered by seq if present.
        """
        if not file_ids:
            return {}
        out: Dict[str, list[dict[str, object]]] = {fid: [] for fid in file_ids}
        for fid in file_ids:
            try:
                points = self._qdrant.search(
                    collection_name=self._collection,
                    query_vector=[0.0] * 4,  # dummy vector; relies on filter
                    limit=limit_per_file,
                    with_payload=True,
                    with_vectors=False,
                    query_filter={
                        "must": [
                            {"key": "file_id", "match": {"value": fid}},
                        ]
                    },
                )
                sorted_points = sorted(points, key=lambda p: (p.payload or {}).get("seq", 0))
                for p in sorted_points:
                    payload = p.payload or {}
                    text = payload.get("text")
                    if not text:
                        continue
                    out[fid].append(
                        {
                            "text": text,
                            "seq": payload.get("seq", 0),
                            "filename": payload.get("filename"),
                            "file_id": fid,
                        }
                    )
            except Exception:
                out[fid] = []
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


__all__ = ["RetrievalService"]
