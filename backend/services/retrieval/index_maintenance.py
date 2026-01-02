from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Iterable, Optional, Sequence

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from backend.services.storage.sqlite_store import SQLiteMetadataStore

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _point_id_for_chunk_id(chunk_id: str) -> str:
    try:
        return str(uuid.UUID(str(chunk_id)))
    except Exception:
        return uuid.uuid5(uuid.NAMESPACE_DNS, str(chunk_id)).hex


def _build_filter(*, file_id: str, chat_id: Optional[str] = None) -> qmodels.Filter:
    must: list[qmodels.FieldCondition] = [
        qmodels.FieldCondition(key="file_id", match=qmodels.MatchValue(value=file_id))
    ]
    if chat_id:
        must.append(qmodels.FieldCondition(key="chat_id", match=qmodels.MatchValue(value=chat_id)))
    return qmodels.Filter(must=must)


@dataclass(frozen=True)
class IndexValidation:
    file_id: str
    sqlite_chunks: int
    qdrant_points: int
    missing_in_qdrant: list[str]
    orphan_in_qdrant: list[str]
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return not self.missing_in_qdrant and not self.orphan_in_qdrant and not self.truncated

    def to_dict(self) -> dict[str, object]:
        return {
            "file_id": self.file_id,
            "sqlite_chunks": self.sqlite_chunks,
            "qdrant_points": self.qdrant_points,
            "missing_in_qdrant": self.missing_in_qdrant,
            "orphan_in_qdrant": self.orphan_in_qdrant,
            "truncated": self.truncated,
            "ok": self.ok,
        }


def validate_file_index(
    *,
    sqlite_store: SQLiteMetadataStore,
    qdrant_client: QdrantClient,
    collection_name: str,
    file_id: str,
    chat_id: Optional[str] = None,
    max_scan: int = 50_000,
) -> IndexValidation:
    fid = (file_id or "").strip()
    if not fid:
        return IndexValidation(
            file_id="",
            sqlite_chunks=0,
            qdrant_points=0,
            missing_in_qdrant=[],
            orphan_in_qdrant=[],
            truncated=False,
        )

    sqlite_chunk_ids = sqlite_store.list_chunk_ids_for_file(fid)
    sqlite_ids = set(sqlite_chunk_ids)

    q_filter = _build_filter(file_id=fid, chat_id=(chat_id or "").strip() or None)
    try:
        count_res = qdrant_client.count(collection_name=collection_name, count_filter=q_filter, exact=True)
        qdrant_points = int(getattr(count_res, "count", 0) or 0)
    except Exception:
        qdrant_points = 0

    qdrant_ids: set[str] = set()
    truncated = False
    offset: Optional[object] = None
    try:
        while True:
            records, next_offset = qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter=q_filter,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for rec in records:
                payload = getattr(rec, "payload", None) or {}
                cid = payload.get("chunk_id") if isinstance(payload, dict) else None
                if not isinstance(cid, str) or not cid:
                    cid = str(getattr(rec, "id", "") or "")
                if cid:
                    qdrant_ids.add(cid)
                if len(qdrant_ids) >= max_scan:
                    truncated = True
                    break
            if truncated:
                break
            if next_offset is None:
                break
            offset = next_offset
    except Exception as exc:
        logger.debug("validate_file_index scroll failed file_id=%s: %s", fid, exc)

    missing = sorted(sqlite_ids - qdrant_ids)
    orphan = sorted(qdrant_ids - sqlite_ids)
    return IndexValidation(
        file_id=fid,
        sqlite_chunks=len(sqlite_ids),
        qdrant_points=qdrant_points,
        missing_in_qdrant=missing,
        orphan_in_qdrant=orphan,
        truncated=truncated,
    )


def repair_file_index(
    *,
    sqlite_store: SQLiteMetadataStore,
    qdrant_client: QdrantClient,
    collection_name: str,
    file_id: str,
    embedder: Callable[[str], Sequence[float]],
    chat_id: Optional[str] = None,
    max_repairs: int = 5_000,
    delete_orphans: bool = False,
    dry_run: bool = False,
) -> dict[str, object]:
    """
    Best-effort repair for Qdrant↔SQLite inconsistency for a single file_id.

    - Re-embeds and upserts chunks that exist in SQLite but are missing in Qdrant.
    - Optionally deletes orphaned Qdrant points (disabled by default).

    This is intended as a safety valve for rare corruption/partial writes, not
    the normal ingestion path.
    """
    fid = (file_id or "").strip()
    if not fid:
        return {"ok": False, "error": "file_id required"}

    validation = validate_file_index(
        sqlite_store=sqlite_store,
        qdrant_client=qdrant_client,
        collection_name=collection_name,
        file_id=fid,
        chat_id=chat_id,
    )
    missing = list(validation.missing_in_qdrant)
    orphan = list(validation.orphan_in_qdrant)

    repaired: list[str] = []
    skipped: list[str] = []
    errors: list[str] = []

    if missing:
        missing = missing[: max(0, int(max_repairs or 0))]
        rows = sqlite_store.fetch_chunks(missing)
        by_id: dict[str, dict[str, object]] = {
            str(r.get("id")): r for r in rows if isinstance(r.get("id"), str)
        }
        points: list[qmodels.PointStruct] = []
        for cid in missing:
            row = by_id.get(cid) or {}
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                skipped.append(cid)
                continue
            try:
                vector = list(embedder(text))
            except Exception as exc:
                errors.append(f"{cid}: embed failed: {exc}")
                continue
            payload = {
                "chunk_id": cid,
                "file_id": row.get("file_id") or fid,
                "user_id": row.get("user_id"),
                "chat_id": row.get("chat_id"),
                "source": row.get("source"),
                "tags": row.get("tags") if isinstance(row.get("tags"), list) else [],
                "created_at": row.get("chunk_created_at") or _now_iso(),
                "metadata": row.get("metadata") if isinstance(row.get("metadata"), dict) else {},
            }
            points.append(qmodels.PointStruct(id=_point_id_for_chunk_id(cid), vector=vector, payload=payload))

        if points and not dry_run:
            try:
                qdrant_client.upsert(collection_name=collection_name, points=points)
                repaired = [p.payload.get("chunk_id") for p in points if isinstance(p.payload, dict) and p.payload.get("chunk_id")]
                sqlite_store.mark_indexed(repaired)
            except Exception as exc:
                errors.append(f"upsert failed: {exc}")

    deleted_orphans = 0
    if delete_orphans and orphan and not dry_run:
        try:
            # Delete by point IDs (derived from chunk_id) to avoid another scroll pass.
            point_ids = [_point_id_for_chunk_id(cid) for cid in orphan if isinstance(cid, str) and cid]
            if point_ids:
                qdrant_client.delete(
                    collection_name=collection_name,
                    points_selector=qmodels.PointIdsList(points=point_ids),
                )
                deleted_orphans = len(point_ids)
        except Exception as exc:
            errors.append(f"delete orphans failed: {exc}")

    after = validate_file_index(
        sqlite_store=sqlite_store,
        qdrant_client=qdrant_client,
        collection_name=collection_name,
        file_id=fid,
        chat_id=chat_id,
    )
    return {
        "ok": after.ok,
        "file_id": fid,
        "before": validation.to_dict(),
        "after": after.to_dict(),
        "repaired": repaired,
        "skipped": skipped,
        "deleted_orphans": deleted_orphans,
        "dry_run": bool(dry_run),
        "errors": errors,
    }


__all__ = ["IndexValidation", "validate_file_index", "repair_file_index"]

