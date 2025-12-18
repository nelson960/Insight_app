from __future__ import annotations

import json
import logging
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from backend.services.ingestion.models import (
    ChunkIndexStatus,
    FileIngestionStatus,
    IngestionErrorCode,
)

logger = logging.getLogger(__name__)


@dataclass
class SQLiteConfig:
    journal_mode: str = "WAL"
    busy_timeout_ms: int = 5000


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteMetadataStore:
    """
    Persistent SQLite store for ingestion metadata, chunk payloads, and basic job tracking.
    """

    def __init__(self, db_path: Path | str, *, config: Optional[SQLiteConfig] = None) -> None:
        self._db_path = Path(db_path)
        self._config = config or SQLiteConfig()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self._db_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._initialize()
        logger.info("SQLiteMetadataStore ready at %s", self._db_path)

    # ------------------------------------------------------------------ #
    # Schema setup
    # ------------------------------------------------------------------ #
    def _initialize(self) -> None:
        conn = self._connection
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                chat_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content_json TEXT,
                model TEXT,
                mode TEXT,
                planner_payload_json TEXT,
                citations_json TEXT,
                created_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS file_text (
                file_id TEXT PRIMARY KEY,
                blocks_json TEXT,
                plain_text TEXT,
                created_at TEXT,
                updated_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                job_type TEXT,
                file_id TEXT,
                status TEXT,
                error TEXT,
                created_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS files (
                id TEXT PRIMARY KEY,
                filename TEXT,
                mime TEXT,
                size_bytes INTEGER,
                hash TEXT,
                stored_path TEXT,
                is_encrypted INTEGER,
                status TEXT,
                policy_json TEXT,
                pages INTEGER,
                metadata_json TEXT,
                error_code TEXT,
                error_message TEXT,
                retry_count INTEGER,
                next_retry_at TEXT,
                created_at TEXT,
                updated_at TEXT,
                user_id TEXT,
                chat_id TEXT,
                tags TEXT,
                source TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                file_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                text TEXT,
                user_id TEXT,
                chat_id TEXT,
                source TEXT,
                tags TEXT,
                created_at TEXT,
                metadata TEXT,
                embedding_model TEXT,
                embedding_version INTEGER,
                embedding_id TEXT,
                index_status TEXT
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_file_seq ON chunks(file_id, seq)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_chat ON chunks(chat_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_created ON chunks(created_at)")
        conn.execute(f"PRAGMA journal_mode={self._config.journal_mode}")
        conn.execute(f"PRAGMA busy_timeout={self._config.busy_timeout_ms}")

    # ------------------------------------------------------------------ #
    # Message APIs (minimal passthrough for compatibility)
    # ------------------------------------------------------------------ #
    def insert_message(
        self,
        message_id: str,
        chat_id: str,
        role: str,
        content_json: str,
        model: Optional[str],
        mode: Optional[str],
        planner_payload_json: Optional[str],
        citations_json: Optional[str],
        created_at: str,
    ) -> None:
        with self._lock:
            self._connection.execute(
                """
                INSERT OR REPLACE INTO messages
                    (id, chat_id, role, content_json, model, mode, planner_payload_json, citations_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    chat_id,
                    role,
                    content_json,
                    model,
                    mode,
                    planner_payload_json,
                    citations_json,
                    created_at,
                ),
            )
            self._connection.commit()

    def insert_message_documents(self, _message_id: str, _file_ids: Sequence[str]) -> None:  # pragma: no cover - not used
        return

    def fetch_recent_messages(self, chat_id: str, limit: int = 5) -> List[Dict[str, object]]:
        cursor = self._connection.execute(
            """
            SELECT id, role, content_json, created_at, model, mode
            FROM messages
            WHERE chat_id=?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (chat_id, limit),
        )
        rows = cursor.fetchall()
        results: List[Dict[str, object]] = []
        for row in reversed(rows):
            results.append(
                {
                    "id": row["id"],
                    "role": row["role"],
                    "content": row["content_json"],
                    "created_at": row["created_at"],
                    "model": row["model"],
                    "mode": row["mode"],
                }
            )
        return results

    def fetch_messages_for_chat(self, chat_id: str, *, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Fetch UI transcript messages for a chat, ordered by created_at ASC.

        Returns a simplified list of {"role": ..., "content": ...} where content is extracted
        from content_json["text"] when possible.
        """
        sql = """
            SELECT role, content_json
            FROM messages
            WHERE chat_id=?
            ORDER BY created_at ASC
        """
        params: list[object] = [chat_id]
        if isinstance(limit, int) and limit > 0:
            sql += " LIMIT ?"
            params.append(limit)

        cursor = self._connection.execute(sql, tuple(params))
        rows = cursor.fetchall()
        out: List[Dict[str, str]] = []
        for row in rows:
            role = (row["role"] or "user").strip()
            content_json = row["content_json"] or ""
            text = ""
            try:
                v = json.loads(content_json) if content_json else {}
                if isinstance(v, dict) and isinstance(v.get("text"), str):
                    text = v["text"]
                elif isinstance(v, str):
                    text = v
                else:
                    text = content_json
            except Exception:
                text = content_json
            text = (text or "").strip()
            if not text:
                continue
            out.append({"role": role, "content": text})
        return out

    def delete_messages_for_chat(self, chat_id: str) -> int:
        with self._lock:
            cur = self._connection.execute("DELETE FROM messages WHERE chat_id=?", (chat_id,))
            self._connection.commit()
            return int(cur.rowcount or 0)

    # ------------------------------------------------------------------ #
    # Ingestion / chunk persistence
    # ------------------------------------------------------------------ #
    def register_file(
        self,
        *,
        file_id: str,
        filename: str,
        mime: str,
        size_bytes: int,
        hash_value: str,
        stored_path: str,
        is_encrypted: bool,
        policy: dict | None = None,
        user_id: Optional[str] = None,
        chat_id: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        source: Optional[str] = None,
    ) -> None:
        now = _now_iso()
        policy_json = json.dumps(policy or {})
        tags_json = json.dumps(sorted(set(tags))) if tags else json.dumps([])

        with self._lock:
            cursor = self._connection.execute("SELECT created_at, status FROM files WHERE id=?", (file_id,))
            row = cursor.fetchone()
            created_at = row["created_at"] if row else now
            status = row["status"] if row else FileIngestionStatus.PENDING.value

            self._connection.execute(
                """
                INSERT OR REPLACE INTO files (
                    id, filename, mime, size_bytes, hash, stored_path, is_encrypted, status,
                    policy_json, pages, metadata_json, error_code, error_message, retry_count,
                    next_retry_at, created_at, updated_at, user_id, chat_id, tags, source
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    file_id,
                    filename,
                    mime,
                    size_bytes,
                    hash_value,
                    stored_path,
                    1 if is_encrypted else 0,
                    status,
                    policy_json,
                    None,
                    None,
                    None,
                    None,
                    0,
                    None,
                    created_at,
                    now,
                    user_id,
                    chat_id,
                    tags_json,
                    source,
                ),
            )
            self._connection.commit()

    def mark_file_status(self, file_id: str, status: FileIngestionStatus | str) -> None:
        with self._lock:
            self._connection.execute(
                "UPDATE files SET status=?, updated_at=? WHERE id=?",
                (str(status), _now_iso(), file_id),
            )
            self._connection.commit()

    def record_extraction(self, file_id: str, *, pages: int | None, metadata: dict[str, object]) -> None:
        with self._lock:
            self._connection.execute(
                "UPDATE files SET pages=?, metadata_json=?, updated_at=? WHERE id=?",
                (pages, json.dumps(metadata or {}), _now_iso(), file_id),
            )
            self._connection.commit()

    def upsert_file_text(self, file_id: str, *, text: str, blocks: Sequence[dict]) -> None:
        now = _now_iso()
        blocks_json = json.dumps(list(blocks) if blocks else [])
        with self._lock:
            cursor = self._connection.execute("SELECT created_at FROM file_text WHERE file_id=?", (file_id,))
            row = cursor.fetchone()
            created_at = row["created_at"] if row else now
            self._connection.execute(
                """
                INSERT OR REPLACE INTO file_text (file_id, blocks_json, plain_text, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (file_id, blocks_json, text or "", created_at, now),
            )
            self._connection.commit()

    def get_file_text(self, file_id: str) -> Optional[dict[str, object]]:
        cursor = self._connection.execute(
            "SELECT file_id, blocks_json, plain_text, created_at, updated_at FROM file_text WHERE file_id=?",
            (file_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        blocks = []
        try:
            blocks = json.loads(row["blocks_json"]) if row["blocks_json"] else []
        except Exception:
            blocks = []
        return {
            "file_id": row["file_id"],
            "blocks": blocks,
            "plain_text": row["plain_text"] or "",
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def stage_chunks(
        self,
        file_id: str,
        chunks: Sequence[dict],
        *,
        embedding_model: str,
        embedding_version: int,
    ) -> None:
        if not chunks:
            return
        now = _now_iso()
        def _get(obj: Any, name: str, default: Any = None) -> Any:
            if isinstance(obj, dict):
                return obj.get(name, default)
            return getattr(obj, name, default)

        with self._lock:
            for chunk in chunks:
                chunk_id = _get(chunk, "chunk_id") or _get(chunk, "id")
                seq = _get(chunk, "sequence")
                seq = seq if seq is not None else _get(chunk, "seq", 0)
                tags_val = _get(chunk, "tags", []) or []
                metadata_val = _get(chunk, "metadata", {}) or {}
                tags_json = json.dumps(sorted(set(tags_val)))
                metadata_json = json.dumps(metadata_val)
                created_at = None
                created = _get(chunk, "created_at")
                if created:
                    try:
                        created_at = created.isoformat() if hasattr(created, "isoformat") else str(created)
                    except Exception:
                        created_at = None
                self._connection.execute(
                    """
                    INSERT OR REPLACE INTO chunks (
                        id, file_id, seq, text, user_id, chat_id, source, tags,
                        created_at, metadata, embedding_model, embedding_version, embedding_id, index_status
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk_id,
                        file_id,
                        seq or 0,
                        _get(chunk, "text"),
                        _get(chunk, "user_id"),
                        _get(chunk, "chat_id"),
                        _get(chunk, "source"),
                        tags_json,
                        created_at or now,
                        metadata_json,
                        embedding_model,
                        embedding_version,
                        _get(chunk, "embedding_id"),
                        ChunkIndexStatus.PENDING.value,
                    ),
                )
            self._connection.commit()

    def mark_indexed(self, chunk_ids: Sequence[str]) -> None:
        if not chunk_ids:
            return
        placeholders = ",".join("?" for _ in chunk_ids)
        with self._lock:
            self._connection.execute(
                f"UPDATE chunks SET index_status=? WHERE id IN ({placeholders})",
                (ChunkIndexStatus.INDEXED.value, *chunk_ids),
            )
            self._connection.commit()

    def mark_stale(self, chunk_ids: Sequence[str]) -> None:
        if not chunk_ids:
            return
        placeholders = ",".join("?" for _ in chunk_ids)
        with self._lock:
            self._connection.execute(
                f"UPDATE chunks SET index_status=? WHERE id IN ({placeholders})",
                (ChunkIndexStatus.STALE.value, *chunk_ids),
            )
            self._connection.commit()

    def mark_file_completed(self, file_id: str) -> None:
        with self._lock:
            self._connection.execute(
                "UPDATE files SET status=?, updated_at=? WHERE id=?",
                (FileIngestionStatus.COMPLETED.value, _now_iso(), file_id),
            )
            self._connection.commit()

    def record_failure(
        self,
        file_id: str,
        *,
        error_code: IngestionErrorCode,
        error_message: str,
        retry_state: object,
    ) -> None:
        retry_count = getattr(retry_state, "retry_count", 0) if retry_state else 0
        next_retry_at = getattr(retry_state, "next_retry_at", None)
        next_retry_str = next_retry_at.isoformat() if hasattr(next_retry_at, "isoformat") else None
        with self._lock:
            self._connection.execute(
                """
                UPDATE files
                SET status=?, error_code=?, error_message=?, retry_count=?, next_retry_at=?, updated_at=?
                WHERE id=?
                """,
                (
                    FileIngestionStatus.FAILED.value,
                    str(error_code),
                    error_message,
                    retry_count,
                    next_retry_str,
                    _now_iso(),
                    file_id,
                ),
            )
            self._connection.commit()

    def count_chunks(self) -> int:
        cursor = self._connection.execute("SELECT COUNT(*) AS cnt FROM chunks")
        (count,) = cursor.fetchone()
        return int(count or 0)

    # ------------------------------------------------------------------ #
    # Retrieval helpers
    # ------------------------------------------------------------------ #
    def fetch_chunks(self, chunk_ids: Sequence[str]) -> list[dict[str, object]]:
        if not chunk_ids:
            return []
        placeholders = ",".join("?" for _ in chunk_ids)
        query = f"""
            SELECT
                c.id,
                c.file_id,
                c.seq,
                c.text,
                c.user_id,
                c.chat_id,
                c.source,
                c.tags,
                c.created_at AS chunk_created_at,
                c.metadata,
                f.filename
            FROM chunks c
            LEFT JOIN files f ON f.id = c.file_id
            WHERE c.id IN ({placeholders})
        """
        cursor = self._connection.execute(query, tuple(chunk_ids))
        rows = cursor.fetchall()
        results: list[dict[str, object]] = []
        for row in rows:
            tags = json.loads(row["tags"]) if row["tags"] else []
            metadata = json.loads(row["metadata"]) if row["metadata"] else {}
            results.append(
                {
                    "id": row["id"],
                    "file_id": row["file_id"],
                    "seq": row["seq"],
                    "text": row["text"],
                    "user_id": row["user_id"],
                    "chat_id": row["chat_id"],
                    "source": row["source"],
                    "tags": tags,
                    "chunk_created_at": row["chunk_created_at"],
                    "metadata": metadata,
                    "filename": row["filename"],
                }
            )
        return results

    def fetch_chunks_for_files(self, file_ids: Sequence[str], *, limit_per_file: int = 8) -> dict[str, list[dict[str, object]]]:
        if not file_ids:
            return {}
        results: dict[str, list[dict[str, object]]] = {fid: [] for fid in file_ids}
        for fid in file_ids:
            cursor = self._connection.execute(
                """
                SELECT id, file_id, seq, text, metadata, created_at
                FROM chunks
                WHERE file_id=?
                ORDER BY seq ASC
                LIMIT ?
                """,
                (fid, limit_per_file),
            )
            rows = cursor.fetchall()
            for row in rows:
                meta = json.loads(row["metadata"]) if row["metadata"] else {}
                results[fid].append(
                    {
                        "chunk_id": row["id"],
                        "file_id": row["file_id"],
                        "seq": row["seq"],
                        "text": row["text"],
                        "metadata": meta,
                        "created_at": row["created_at"],
                    }
                )
        return results

    def fetch_chunk_texts_for_file(self, file_id: str, *, limit: Optional[int] = None) -> list[str]:
        sql = """
            SELECT text
            FROM chunks
            WHERE file_id=?
            ORDER BY seq ASC
        """
        params: list[object] = [file_id]
        if isinstance(limit, int) and limit > 0:
            sql += " LIMIT ?"
            params.append(limit)
        cursor = self._connection.execute(sql, tuple(params))
        rows = cursor.fetchall()
        out: list[str] = []
        for row in rows:
            t = row["text"] if isinstance(row, sqlite3.Row) else row[0]
            if isinstance(t, str) and t.strip():
                out.append(t)
        return out

    def bump_chunk_usage(self, _chunk_ids: Sequence[str], _weight: int = 1) -> None:  # pragma: no cover - usage accounting not required
        return

    def delete_chunks_for_chat(self, chat_id: str) -> None:
        with self._lock:
            self._connection.execute("DELETE FROM chunks WHERE chat_id=?", (chat_id,))
            self._connection.commit()

    def get_file(self, file_id: str) -> Optional[dict]:
        cursor = self._connection.execute(
            "SELECT * FROM files WHERE id=?",
            (file_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return dict(row)

    def list_files_for_chat(self, chat_id: str) -> list[dict[str, object]]:
        cursor = self._connection.execute(
            "SELECT id, filename, stored_path FROM files WHERE chat_id=?",
            (chat_id,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def delete_files_for_chat(self, chat_id: str) -> int:
        # Also delete extracted text payloads for files tied to this chat.
        file_ids = [f.get("id") for f in self.list_files_for_chat(chat_id)]
        ids = [x for x in file_ids if isinstance(x, str) and x]
        if ids:
            placeholders = ",".join(["?"] * len(ids))
            with self._lock:
                self._connection.execute(f"DELETE FROM file_text WHERE file_id IN ({placeholders})", ids)
                self._connection.commit()
        with self._lock:
            cur = self._connection.execute("DELETE FROM files WHERE chat_id=?", (chat_id,))
            self._connection.commit()
            return int(cur.rowcount or 0)

    def delete_jobs_for_chat(self, chat_id: str) -> int:
        # Jobs only store file_id; delete by file_ids tied to this chat.
        file_ids = [f.get("id") for f in self.list_files_for_chat(chat_id)]
        ids = [x for x in file_ids if isinstance(x, str) and x]
        if not ids:
            return 0
        placeholders = ",".join(["?"] * len(ids))
        with self._lock:
            cur = self._connection.execute(f"DELETE FROM jobs WHERE file_id IN ({placeholders})", ids)
            self._connection.commit()
            return int(cur.rowcount or 0)

    # ------------------------------------------------------------------ #
    # Job helpers
    # ------------------------------------------------------------------ #
    def create_job(self, job_id: str, job_type: str, file_id: str, created_at: Optional[str] = None) -> None:
        with self._lock:
            self._connection.execute(
                """
                INSERT OR REPLACE INTO jobs (id, job_type, file_id, status, error, created_at)
                VALUES (?, ?, ?, 'queued', NULL, COALESCE(?, datetime('now')))
                """,
                (job_id, job_type, file_id, created_at),
            )
            self._connection.commit()

    def update_job_status(self, job_id: str, status: str, error: Optional[str] = None) -> None:
        with self._lock:
            self._connection.execute(
                """
                UPDATE jobs
                SET status = ?, error = ?
                WHERE id = ?
                """,
                (status, error, job_id),
            )
            self._connection.commit()

    def list_jobs(self, limit: int = 50) -> list[dict[str, object]]:
        cursor = self._connection.execute(
            """
            SELECT id, job_type, file_id, status, error, created_at
            FROM jobs
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(row) for row in cursor.fetchall()]


def create_sqlite_store(db_path: Any, *, config: Optional[SQLiteConfig] = None) -> SQLiteMetadataStore:
    return SQLiteMetadataStore(db_path, config=config)


__all__ = ["SQLiteMetadataStore", "SQLiteConfig", "create_sqlite_store"]
