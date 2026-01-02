from __future__ import annotations

import json
import logging
import re
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
from backend.services.ipc_events import emit_event

logger = logging.getLogger(__name__)


@dataclass
class SQLiteConfig:
    journal_mode: str = "WAL"
    busy_timeout_ms: int = 5000


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_file_ingestion_status(status: FileIngestionStatus | str) -> str:
    """
    Normalize status values stored in SQLite and emitted over IPC.

    Historically, some code paths used `str(FileIngestionStatus.PROCESSING)` which
    yields "FileIngestionStatus.PROCESSING" instead of the intended "processing".
    Keep this tolerant so older rows don't break UI logic.
    """
    if isinstance(status, FileIngestionStatus):
        return status.value
    if isinstance(status, str):
        prefix = "FileIngestionStatus."
        if status.startswith(prefix):
            name = status[len(prefix) :]
            try:
                return FileIngestionStatus[name].value
            except Exception:
                return name.lower()
        return status
    return str(status)


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
            CREATE TABLE IF NOT EXISTS app_settings (
                key TEXT PRIMARY KEY,
                value_json TEXT,
                updated_at TEXT
            )
            """
        )
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
        # Map file_ids to one or more chats. This replaces the legacy `files.chat_id`
        # field for association checks and listing, enabling document sharing across
        # branched cards without re-ingestion.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_files (
                chat_id TEXT NOT NULL,
                file_id TEXT NOT NULL,
                created_at TEXT,
                PRIMARY KEY (chat_id, file_id)
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
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS doc_pages (
                chat_id TEXT NOT NULL,
                file_id TEXT NOT NULL,
                title TEXT,
                doc_json TEXT,
                is_user_edited INTEGER,
                source_file_updated_at TEXT,
                created_at TEXT,
                updated_at TEXT,
                PRIMARY KEY (chat_id, file_id)
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_pages_chat ON doc_pages(chat_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_file_seq ON chunks(file_id, seq)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_chat ON chunks(chat_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_created ON chunks(created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_files_chat ON chat_files(chat_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_files_file ON chat_files(file_id)")
        conn.execute(f"PRAGMA journal_mode={self._config.journal_mode}")
        conn.execute(f"PRAGMA busy_timeout={self._config.busy_timeout_ms}")

        # Backfill chat_files mapping from legacy files.chat_id for older databases.
        # This is idempotent and safe to run at startup.
        try:
            conn.execute(
                """
                INSERT OR IGNORE INTO chat_files (chat_id, file_id, created_at)
                SELECT chat_id, id, created_at
                FROM files
                WHERE chat_id IS NOT NULL AND chat_id != ''
                """
            )
            conn.commit()
        except Exception:
            # Best effort; older/broken DBs should still boot.
            pass

        # Best-effort FTS index for chunks so the backend can do fast keyword search
        # without loading full documents (used for hybrid retrieval and agent loops).
        self._ensure_chunks_fts()

    def _ensure_chunks_fts(self) -> None:
        """
        Ensure an FTS5 index exists for chunk text.

        This is best-effort: if the local SQLite build lacks FTS5, the app should
        still run (dense retrieval continues to work).
        """
        conn = self._connection
        try:
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
                USING fts5(
                    chunk_id UNINDEXED,
                    file_id UNINDEXED,
                    text
                )
                """
            )
            conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS chunks_fts_ai
                AFTER INSERT ON chunks
                BEGIN
                    INSERT INTO chunks_fts(chunk_id, file_id, text)
                    VALUES (new.id, new.file_id, COALESCE(new.text, ''));
                END
                """
            )
            conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS chunks_fts_ad
                AFTER DELETE ON chunks
                BEGIN
                    DELETE FROM chunks_fts WHERE chunk_id = old.id;
                END
                """
            )
            conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS chunks_fts_au
                AFTER UPDATE OF text, file_id ON chunks
                BEGIN
                    DELETE FROM chunks_fts WHERE chunk_id = old.id;
                    INSERT INTO chunks_fts(chunk_id, file_id, text)
                    VALUES (new.id, new.file_id, COALESCE(new.text, ''));
                END
                """
            )

            # One-time backfill (or repair) if the FTS index is behind.
            try:
                (fts_cnt,) = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()
            except Exception:
                fts_cnt = 0
            try:
                (chunks_cnt,) = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
            except Exception:
                chunks_cnt = 0

            if int(fts_cnt or 0) < int(chunks_cnt or 0):
                conn.execute("DELETE FROM chunks_fts")
                conn.execute(
                    """
                    INSERT INTO chunks_fts(chunk_id, file_id, text)
                    SELECT id, file_id, COALESCE(text, '')
                    FROM chunks
                    """
                )
            conn.commit()
        except sqlite3.OperationalError as exc:
            # SQLite build likely lacks FTS5.
            logger.info("FTS disabled (chunks_fts unavailable): %s", exc)
        except Exception as exc:
            logger.warning("Failed to initialize chunks_fts: %s", exc)

    # ------------------------------------------------------------------ #
    # App settings (simple key/value JSON)
    # ------------------------------------------------------------------ #
    def get_setting(self, key: str, default: Any = None) -> Any:
        if not key:
            return default
        cursor = self._connection.execute("SELECT value_json FROM app_settings WHERE key=?", (key,))
        row = cursor.fetchone()
        if not row:
            return default
        raw = row["value_json"]
        if raw is None:
            return default
        try:
            return json.loads(raw)
        except Exception:
            return default

    def set_setting(self, key: str, value: Any) -> None:
        if not key:
            raise ValueError("settings key is required")
        payload = json.dumps(value)
        with self._lock:
            self._connection.execute(
                """
                INSERT OR REPLACE INTO app_settings (key, value_json, updated_at)
                VALUES (?, ?, ?)
                """,
                (key, payload, _now_iso()),
            )
            self._connection.commit()

    def delete_setting(self, key: str) -> None:
        if not key:
            return
        with self._lock:
            self._connection.execute("DELETE FROM app_settings WHERE key=?", (key,))
            self._connection.commit()

    def list_settings(self) -> dict[str, Any]:
        cursor = self._connection.execute("SELECT key, value_json FROM app_settings")
        out: dict[str, Any] = {}
        for row in cursor.fetchall():
            k = row["key"]
            raw = row["value_json"]
            try:
                out[k] = json.loads(raw) if raw is not None else None
            except Exception:
                out[k] = None
        return out

    def close(self) -> None:
        try:
            with self._lock:
                self._connection.close()
        except Exception:
            pass

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
            status = _normalize_file_ingestion_status(status)

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
            if chat_id:
                self._connection.execute(
                    """
                    INSERT OR IGNORE INTO chat_files (chat_id, file_id, created_at)
                    VALUES (?, ?, ?)
                    """,
                    (chat_id, file_id, created_at),
                )
            self._connection.commit()

    def mark_file_status(self, file_id: str, status: FileIngestionStatus | str) -> None:
        status_value = _normalize_file_ingestion_status(status)
        with self._lock:
            self._connection.execute(
                "UPDATE files SET status=?, updated_at=? WHERE id=?",
                (status_value, _now_iso(), file_id),
            )
            self._connection.commit()
            try:
                cursor = self._connection.execute("SELECT filename FROM files WHERE id=?", (file_id,))
                row = cursor.fetchone()
                filename = (row["filename"] if row else "") or ""
                for chat_id in self.list_chat_ids_for_file(file_id):
                    emit_event(
                        "file_status",
                        chat_id=chat_id,
                        file_id=file_id,
                        filename=filename,
                        status=status_value,
                    )
            except Exception:
                pass

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
            filename = ""
            try:
                cursor = self._connection.execute("SELECT filename FROM files WHERE id=?", (file_id,))
                row = cursor.fetchone()
                if row:
                    filename = row["filename"] or ""
            except Exception:
                filename = ""
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
            for chat_id in self.list_chat_ids_for_file(file_id):
                emit_event(
                    "file_text_ready",
                    chat_id=chat_id,
                    file_id=file_id,
                    filename=filename,
                )

    def get_file_text_updated_at(self, file_id: str) -> str | None:
        with self._lock:
            cursor = self._connection.execute(
                "SELECT updated_at FROM file_text WHERE file_id=?",
                (file_id,),
            )
            row = cursor.fetchone()
        if not row:
            return None
        return row["updated_at"] or None

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
        status_value = FileIngestionStatus.COMPLETED.value
        with self._lock:
            self._connection.execute(
                "UPDATE files SET status=?, updated_at=? WHERE id=?",
                (status_value, _now_iso(), file_id),
            )
            self._connection.commit()
            try:
                cursor = self._connection.execute(
                    "SELECT chat_id, filename FROM files WHERE id=?",
                    (file_id,),
                )
                row = cursor.fetchone()
                if row and row["chat_id"]:
                    emit_event(
                        "file_status",
                        chat_id=row["chat_id"],
                        file_id=file_id,
                        filename=row["filename"] or "",
                        status=status_value,
                    )
            except Exception:
                pass

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

    def list_chunk_ids_for_file(self, file_id: str) -> list[str]:
        """
        Return all chunk ids for a file, ordered by seq ASC.

        Used for index validation/repair (SQLite ↔ Qdrant consistency).
        """
        fid = (file_id or "").strip()
        if not fid:
            return []
        cursor = self._connection.execute(
            "SELECT id FROM chunks WHERE file_id=? ORDER BY seq ASC",
            (fid,),
        )
        out: list[str] = []
        for row in cursor.fetchall():
            cid = row["id"] if isinstance(row, sqlite3.Row) else row[0]
            if isinstance(cid, str) and cid:
                out.append(cid)
        return out

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

    def fetch_chunks_by_seq_range(
        self,
        file_id: str,
        *,
        seq_start: int,
        seq_end: int,
    ) -> list[dict[str, object]]:
        """
        Fetch chunk rows for a file, ordered by seq ASC, within an inclusive range.
        """
        fid = (file_id or "").strip()
        if not fid:
            return []
        seq_start_i = max(0, int(seq_start))
        seq_end_i = max(seq_start_i, int(seq_end))
        cursor = self._connection.execute(
            """
            SELECT
                c.id,
                c.file_id,
                c.seq,
                c.text,
                c.metadata,
                f.filename
            FROM chunks c
            LEFT JOIN files f ON f.id = c.file_id
            WHERE c.file_id = ? AND c.seq BETWEEN ? AND ?
            ORDER BY c.seq ASC
            """,
            (fid, seq_start_i, seq_end_i),
        )
        rows = cursor.fetchall()
        out: list[dict[str, object]] = []
        for row in rows:
            meta = {}
            try:
                meta = json.loads(row["metadata"]) if row["metadata"] else {}
            except Exception:
                meta = {}
            out.append(
                {
                    "id": row["id"],
                    "file_id": row["file_id"],
                    "seq": row["seq"],
                    "text": row["text"],
                    "metadata": meta,
                    "filename": row["filename"],
                }
            )
        return out

    @staticmethod
    def _to_fts_query(query: str, *, max_terms: int = 8) -> str:
        """
        Convert a user query into a conservative FTS5 MATCH expression.

        This avoids common syntax errors from raw user input (quotes/operators).
        """
        q = (query or "").strip()
        if not q:
            return ""
        terms = []
        for tok in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]{1,80}", q):
            t = tok.strip()
            if not t:
                continue
            terms.append(t)
            if len(terms) >= max_terms:
                break
        if not terms:
            return ""
        # Quote each token to treat punctuation literally (best effort).
        return " AND ".join(f"\"{t}\"" for t in terms)

    def fts_search_chunks(
        self,
        query: str,
        *,
        file_id: Optional[str] = None,
        top_k: int = 50,
    ) -> list[dict[str, object]]:
        """
        Keyword search over chunk text using SQLite FTS5.

        Returns lightweight hit records containing chunk_id + file_id + score.
        """
        q = (query or "").strip()
        if not q:
            return []
        match = self._to_fts_query(q)
        if not match:
            return []
        limit = max(1, min(int(top_k or 50), 500))

        sql = """
            SELECT chunk_id, file_id, bm25(chunks_fts) AS score
            FROM chunks_fts
            WHERE chunks_fts MATCH ?
        """
        params: list[object] = [match]
        if isinstance(file_id, str) and file_id.strip():
            sql += " AND file_id = ?"
            params.append(file_id.strip())
        sql += " ORDER BY score ASC LIMIT ?"
        params.append(limit)

        try:
            cursor = self._connection.execute(sql, tuple(params))
            rows = cursor.fetchall()
        except Exception:
            return []

        out: list[dict[str, object]] = []
        for row in rows:
            cid = row["chunk_id"]
            fid = row["file_id"]
            if not isinstance(cid, str) or not cid:
                continue
            if not isinstance(fid, str) or not fid:
                continue
            try:
                score = float(row["score"])
            except Exception:
                score = 0.0
            out.append({"chunk_id": cid, "file_id": fid, "score": score})
        return out

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

    # ------------------------------------------------------------------ #
    # Chat ↔ file association (multi-chat document sharing)
    # ------------------------------------------------------------------ #
    def add_file_to_chat(self, chat_id: str, file_id: str) -> None:
        if not chat_id or not file_id:
            return
        with self._lock:
            self._connection.execute(
                """
                INSERT OR IGNORE INTO chat_files (chat_id, file_id, created_at)
                VALUES (?, ?, ?)
                """,
                (chat_id, file_id, _now_iso()),
            )
            self._connection.commit()

    def chat_has_file(self, chat_id: str, file_id: str) -> bool:
        if not chat_id or not file_id:
            return False
        cursor = self._connection.execute(
            "SELECT 1 FROM chat_files WHERE chat_id=? AND file_id=? LIMIT 1",
            (chat_id, file_id),
        )
        return cursor.fetchone() is not None

    def list_file_ids_for_chat(self, chat_id: str) -> list[str]:
        if not chat_id:
            return []
        cursor = self._connection.execute(
            "SELECT file_id FROM chat_files WHERE chat_id=? ORDER BY created_at ASC",
            (chat_id,),
        )
        out: list[str] = []
        for row in cursor.fetchall():
            fid = row["file_id"] if isinstance(row, sqlite3.Row) else row[0]
            if isinstance(fid, str) and fid:
                out.append(fid)
        return out

    def list_chat_ids_for_file(self, file_id: str) -> list[str]:
        if not file_id:
            return []
        cursor = self._connection.execute(
            "SELECT chat_id FROM chat_files WHERE file_id=? ORDER BY created_at ASC",
            (file_id,),
        )
        out: list[str] = []
        for row in cursor.fetchall():
            cid = row["chat_id"] if isinstance(row, sqlite3.Row) else row[0]
            if isinstance(cid, str) and cid:
                out.append(cid)
        return out

    def delete_chat_files(self, chat_id: str) -> int:
        """
        Remove all chat↔file links for a chat.

        Note: this does NOT delete the underlying files. Use `delete_file_everywhere`
        for fully unreferenced file cleanup.
        """
        if not chat_id:
            return 0
        with self._lock:
            cur = self._connection.execute("DELETE FROM chat_files WHERE chat_id=?", (chat_id,))
            self._connection.commit()
            return int(cur.rowcount or 0)

    def delete_chat_file(self, chat_id: str, file_id: str) -> int:
        """
        Remove a single chat↔file link for a chat.

        This does NOT delete the underlying file or any chunks; callers should do
        additional cleanup as needed.
        """
        if not chat_id or not file_id:
            return 0
        with self._lock:
            cur = self._connection.execute(
                "DELETE FROM chat_files WHERE chat_id=? AND file_id=?",
                (chat_id, file_id),
            )
            self._connection.commit()
            return int(cur.rowcount or 0)

    def delete_doc_pages_for_chat(self, chat_id: str) -> int:
        """Delete all per-chat editor pages for a chat_id."""
        if not chat_id:
            return 0
        with self._lock:
            cur = self._connection.execute("DELETE FROM doc_pages WHERE chat_id=?", (chat_id,))
            self._connection.commit()
            return int(cur.rowcount or 0)

    def delete_doc_page(self, chat_id: str, file_id: str) -> int:
        """Delete a single per-chat editor page (chat_id + file_id)."""
        if not chat_id or not file_id:
            return 0
        with self._lock:
            cur = self._connection.execute(
                "DELETE FROM doc_pages WHERE chat_id=? AND file_id=?",
                (chat_id, file_id),
            )
            self._connection.commit()
            return int(cur.rowcount or 0)

    def delete_chunks_for_chat_file(self, chat_id: str, file_id: str) -> int:
        """
        Delete all chunk rows for a given (chat_id, file_id).

        This keeps other chats' chunk rows intact even if they reference the same file_id.
        """
        if not chat_id or not file_id:
            return 0
        with self._lock:
            cur = self._connection.execute(
                "DELETE FROM chunks WHERE chat_id=? AND file_id=?",
                (chat_id, file_id),
            )
            self._connection.commit()
            return int(cur.rowcount or 0)

    def delete_file_everywhere(self, file_id: str) -> dict[str, int]:
        """
        Delete a file and all associated database rows.

        This is only safe to call once a file is no longer referenced by ANY chat.
        """
        if not file_id:
            return {"file_id": "", "deleted": 0}
        with self._lock:
            deleted_chat_files = int(
                (self._connection.execute("DELETE FROM chat_files WHERE file_id=?", (file_id,)).rowcount or 0)
            )
            deleted_doc_pages = int(
                (self._connection.execute("DELETE FROM doc_pages WHERE file_id=?", (file_id,)).rowcount or 0)
            )
            deleted_file_text = int(
                (self._connection.execute("DELETE FROM file_text WHERE file_id=?", (file_id,)).rowcount or 0)
            )
            deleted_chunks = int(
                (self._connection.execute("DELETE FROM chunks WHERE file_id=?", (file_id,)).rowcount or 0)
            )
            deleted_jobs = int(
                (self._connection.execute("DELETE FROM jobs WHERE file_id=?", (file_id,)).rowcount or 0)
            )
            deleted_files = int(
                (self._connection.execute("DELETE FROM files WHERE id=?", (file_id,)).rowcount or 0)
            )
            self._connection.commit()
        return {
            "file_id": file_id,
            "chat_files": deleted_chat_files,
            "doc_pages": deleted_doc_pages,
            "file_text": deleted_file_text,
            "chunks": deleted_chunks,
            "jobs": deleted_jobs,
            "files": deleted_files,
        }

    def list_files_for_chat(self, chat_id: str) -> list[dict[str, object]]:
        cursor = self._connection.execute(
            """
            SELECT f.id, f.filename, f.stored_path, cf.created_at AS linked_at
            FROM chat_files cf
            JOIN files f ON f.id = cf.file_id
            WHERE cf.chat_id=?
            ORDER BY cf.created_at ASC
            """,
            (chat_id,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def list_files_status_for_chat(self, chat_id: str) -> list[dict[str, object]]:
        """
        List files for a chat including ingestion status and basic metadata.

        This is used by desktop UI and ingestion readiness checks.
        """
        cursor = self._connection.execute(
            """
            SELECT id, filename, mime, size_bytes, status, pages, created_at, updated_at, source
            FROM files
            WHERE id IN (SELECT file_id FROM chat_files WHERE chat_id=?)
            ORDER BY created_at ASC
            """,
            (chat_id,),
        )
        out: list[dict[str, object]] = []
        for row in cursor.fetchall():
            d = dict(row)
            d["status"] = _normalize_file_ingestion_status(d.get("status") or "")
            out.append(d)
        return out

    def chat_ingestion_progress(self, chat_id: str) -> dict[str, object]:
        """
        Return a cheap ingestion progress snapshot for a single chat_id.

        This avoids polling global busy state and gives the UI an explicit readiness signal.
        """
        if not chat_id:
            return {
                "chat_id": chat_id,
                "total": 0,
                "done": 0,
                "failed": 0,
                "active_jobs": 0,
                "counts": {},
                "files": [],
                "jobs": [],
                "percent": 100.0,
            }

        files = self.list_files_status_for_chat(chat_id)
        total = len(files)
        counts: dict[str, int] = {}
        done = 0
        failed = 0
        for f in files:
            st = f.get("status") or ""
            st = str(st)
            counts[st] = counts.get(st, 0) + 1
            if st in ("completed", "failed"):
                done += 1
            if st == "failed":
                failed += 1

        # Jobs table does not include chat_id; join via files table.
        active_jobs = 0
        jobs: list[dict[str, object]] = []
        try:
            cursor = self._connection.execute(
                """
                SELECT j.id, j.job_type, j.file_id, j.status, j.error, j.created_at, f.filename
                FROM jobs j
                JOIN chat_files cf ON cf.file_id = j.file_id
                JOIN files f ON f.id = j.file_id
                WHERE cf.chat_id=? AND j.status IN ('queued', 'running')
                ORDER BY j.created_at DESC
                LIMIT 200
                """,
                (chat_id,),
            )
            jobs = [dict(r) for r in cursor.fetchall()]
            active_jobs = len(jobs)
        except Exception:
            jobs = []
            active_jobs = 0

        percent = 100.0
        if total > 0:
            percent = round((done / total) * 100.0, 2)

        return {
            "chat_id": chat_id,
            "total": total,
            "done": done,
            "failed": failed,
            "active_jobs": active_jobs,
            "counts": counts,
            "files": [
                {
                    "file_id": f.get("id"),
                    "filename": f.get("filename") or "",
                    "status": f.get("status") or "",
                    "mime": f.get("mime") or "",
                    "size_bytes": f.get("size_bytes") or 0,
                    "pages": f.get("pages"),
                    "created_at": f.get("created_at"),
                    "updated_at": f.get("updated_at"),
                    "source": f.get("source") or "",
                }
                for f in files
            ],
            "jobs": jobs,
            "percent": percent,
        }

    def latest_file_id_for_chat(self, chat_id: str) -> Optional[str]:
        cursor = self._connection.execute(
            """
            SELECT file_id
            FROM chat_files
            WHERE chat_id=?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (chat_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        fid = row["file_id"] if isinstance(row, sqlite3.Row) else row[0]
        return fid if isinstance(fid, str) and fid else None

    def delete_files_for_chat(self, chat_id: str) -> int:
        # Also delete extracted text payloads for files tied to this chat.
        file_ids = [f.get("id") for f in self.list_files_for_chat(chat_id)]
        ids = [x for x in file_ids if isinstance(x, str) and x]
        if ids:
            placeholders = ",".join(["?"] * len(ids))
            with self._lock:
                self._connection.execute(f"DELETE FROM file_text WHERE file_id IN ({placeholders})", ids)
                self._connection.execute(f"DELETE FROM doc_pages WHERE file_id IN ({placeholders})", ids)
                self._connection.commit()
        with self._lock:
            cur = self._connection.execute("DELETE FROM files WHERE chat_id=?", (chat_id,))
            self._connection.commit()
            return int(cur.rowcount or 0)

    # ------------------------------------------------------------------ #
    # Document pages (ProseMirror JSON)
    # ------------------------------------------------------------------ #
    def get_doc_page(self, chat_id: str, file_id: str) -> Optional[dict[str, object]]:
        cursor = self._connection.execute(
            """
            SELECT chat_id, file_id, title, doc_json, is_user_edited, source_file_updated_at, created_at, updated_at
            FROM doc_pages
            WHERE chat_id=? AND file_id=?
            """,
            (chat_id, file_id),
        )
        row = cursor.fetchone()
        if not row:
            return None
        doc = None
        try:
            doc = json.loads(row["doc_json"]) if row["doc_json"] else None
        except Exception:
            doc = None
        return {
            "chat_id": row["chat_id"],
            "file_id": row["file_id"],
            "title": row["title"] or "",
            "doc": doc,
            "doc_json": row["doc_json"] or "",
            "is_user_edited": bool(row["is_user_edited"]) if row["is_user_edited"] is not None else False,
            "source_file_updated_at": row["source_file_updated_at"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def upsert_doc_page(
        self,
        chat_id: str,
        file_id: str,
        *,
        title: str,
        doc: object,
        is_user_edited: bool,
        source_file_updated_at: str | None,
    ) -> None:
        now = _now_iso()
        doc_json = json.dumps(doc or {})
        with self._lock:
            cursor = self._connection.execute(
                "SELECT created_at FROM doc_pages WHERE chat_id=? AND file_id=?",
                (chat_id, file_id),
            )
            row = cursor.fetchone()
            created_at = row["created_at"] if row else now
            self._connection.execute(
                """
                INSERT OR REPLACE INTO doc_pages (
                    chat_id, file_id, title, doc_json, is_user_edited, source_file_updated_at, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chat_id,
                    file_id,
                    title or "",
                    doc_json,
                    1 if is_user_edited else 0,
                    source_file_updated_at,
                    created_at,
                    now,
                ),
            )
            self._connection.commit()

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

    def list_jobs_with_status(self, statuses: Sequence[str], *, limit: int = 200) -> list[dict[str, object]]:
        statuses = [s for s in (statuses or []) if isinstance(s, str) and s]
        if not statuses:
            return []
        placeholders = ",".join(["?"] * len(statuses))
        cursor = self._connection.execute(
            f"""
            SELECT id, job_type, file_id, status, error, created_at
            FROM jobs
            WHERE status IN ({placeholders})
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (*tuple(statuses), int(limit)),
        )
        return [dict(row) for row in cursor.fetchall()]

    def count_jobs_with_status(self, statuses: Sequence[str]) -> int:
        statuses = [s for s in (statuses or []) if isinstance(s, str) and s]
        if not statuses:
            return 0
        placeholders = ",".join(["?"] * len(statuses))
        cursor = self._connection.execute(
            f"SELECT COUNT(*) AS c FROM jobs WHERE status IN ({placeholders})",
            tuple(statuses),
        )
        row = cursor.fetchone()
        try:
            return int(row["c"] if row and "c" in row.keys() else 0)
        except Exception:
            try:
                return int(row[0] if row else 0)
            except Exception:
                return 0


def create_sqlite_store(db_path: Any, *, config: Optional[SQLiteConfig] = None) -> SQLiteMetadataStore:
    return SQLiteMetadataStore(db_path, config=config)


__all__ = ["SQLiteMetadataStore", "SQLiteConfig", "create_sqlite_store"]
