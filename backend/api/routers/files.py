from __future__ import annotations

import asyncio
import hashlib
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Body

from backend.api.deps import AppDependencies

import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/files", tags=["Files"])

def _max_multi_file_bytes() -> int:
    # Default matches the desktop picker guardrail; backend must still enforce it.
    raw = os.environ.get("INSIGHT_MAX_MULTI_FILE_BYTES", str(5 * 1024 * 1024))
    try:
        val = int(raw)
    except ValueError:
        val = 5 * 1024 * 1024
    return max(0, val)


def _max_single_large_file_bytes() -> int:
    # Absolute safety cap for the current upload pipeline which reads files into memory.
    raw = os.environ.get("INSIGHT_MAX_SINGLE_LARGE_FILE_BYTES", str(50 * 1024 * 1024))
    try:
        val = int(raw)
    except ValueError:
        val = 50 * 1024 * 1024
    return max(0, val)

def _is_large_for_chat(size_bytes: int) -> bool:
    return int(size_bytes) > _max_multi_file_bytes()


def _upload_size_bytes(upload: UploadFile) -> int:
    try:
        f = upload.file
        f.seek(0, os.SEEK_END)
        size = int(f.tell())
        f.seek(0)
        return max(0, size)
    except Exception:
        return 0


async def _stream_upload_to_cache(upload: UploadFile, dest_path: Path) -> tuple[str, int]:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    hasher = hashlib.sha256()
    size_bytes = 0
    with dest_path.open("wb") as f:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            hasher.update(chunk)
            size_bytes += len(chunk)
    return hasher.hexdigest(), size_bytes


def _guess_mime_from_suffix(filename: str) -> str:
    suf = Path(filename or "").suffix.lower()
    if suf in {".txt", ".log", ".md", ".json", ".csv", ".tsv", ".yaml", ".yml"}:
        return "text/plain"
    return "application/octet-stream"


def _enforce_chat_upload_size_policy(
    *,
    store: Any,
    chat_id: str,
    incoming: list[tuple[str, int]],
) -> None:
    """
    Enforce a strict upload policy:
      - "Large" file uploads (> max_multi_file_bytes) are only allowed in a brand-new chat with no files.
      - A chat that contains a large file is locked (no further uploads of any size).
      - A chat that contains any file(s) cannot accept a large file later.
      - "Large" file mode only supports raw text inputs (txt/log/json) because it uses rg_search.
      - Always reject files above the absolute safety cap.
    """
    max_multi = _max_multi_file_bytes()
    max_large = _max_single_large_file_bytes()
    allowed_large_suffixes = {".txt", ".log", ".json"}

    for filename, size_bytes in incoming:
        if int(size_bytes) > max_large:
            raise HTTPException(
                status_code=413,
                detail={
                    "error": "file_too_large",
                    "filename": filename,
                    "size_bytes": int(size_bytes),
                    "max_bytes": max_large,
                },
            )
        if int(size_bytes) > max_multi:
            suffix = Path(str(filename or "")).suffix.lower()
            if suffix not in allowed_large_suffixes:
                raise HTTPException(
                    status_code=415,
                    detail={
                        "error": "large_file_type_not_supported",
                        "message": "Large files are supported only for raw text formats (.txt, .log, .json).",
                        "filename": filename,
                        "suffix": suffix,
                        "allowed_suffixes": sorted(allowed_large_suffixes),
                        "size_bytes": int(size_bytes),
                        "max_multi_file_bytes": max_multi,
                        "max_single_large_file_bytes": max_large,
                    },
                )

    existing = store.list_files_status_for_chat(chat_id)
    existing_sizes = [int(r.get("size_bytes") or 0) for r in (existing or [])]

    if any(size > max_multi for size in existing_sizes):
        raise HTTPException(
            status_code=409,
            detail={
                "error": "large_file_chat_locked",
                "message": "This chat already contains a large file and does not allow additional uploads.",
                "chat_id": chat_id,
                "max_multi_file_bytes": max_multi,
            },
        )

    if existing_sizes and any(int(size) > max_multi for _, size in incoming):
        raise HTTPException(
            status_code=409,
            detail={
                "error": "large_file_requires_new_chat",
                "message": "Large files can only be uploaded into a new chat with no existing files.",
                "chat_id": chat_id,
                "max_multi_file_bytes": max_multi,
            },
        )

    large_incoming = [(f, int(s)) for f, s in incoming if int(s) > max_multi]
    if not existing_sizes and large_incoming and len(incoming) != 1:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "large_file_must_be_alone",
                "message": "When uploading a large file, upload exactly one file into a new chat.",
                "chat_id": chat_id,
                "max_multi_file_bytes": max_multi,
            },
        )


@router.post("/upload")
async def upload_files(
    chat_id: str = Form(...),
    user_id: str = Form("default"),
    files: List[UploadFile] = File(...),
):
    from backend.services.extraction.detector import detect_mime_type
    from backend.services.ingestion import FilePolicy, IngestionRequest
    from backend.services.ipc_events import emit_event
    from backend.services.security import encrypt_bytes

    workspace = AppDependencies.workspace()
    scheduler = AppDependencies.ingestion_scheduler()
    key = AppDependencies.key_manager().get_key()
    metadata_store, _ = AppDependencies.storage()

    incoming_sizes = [(str(f.filename or "upload"), _upload_size_bytes(f)) for f in files]
    _enforce_chat_upload_size_policy(store=metadata_store, chat_id=chat_id, incoming=incoming_sizes)

    stored_files = []
    for upload in files:
        suffix = Path(upload.filename).suffix
        temp_path = workspace.cache / f"upload_{uuid.uuid4().hex}{suffix or '.tmp'}"
        hash_value, size_bytes = await _stream_upload_to_cache(upload, temp_path)
        file_id = f"file_{hash_value}"
        enc_path = workspace.uploads / f"{file_id}.enc"
        cache_path = workspace.cache / f"{file_id}{suffix or '.tmp'}"
        if temp_path != cache_path:
            try:
                temp_path.replace(cache_path)
            except Exception:
                cache_path.write_bytes(temp_path.read_bytes())
                try:
                    temp_path.unlink()
                except Exception:
                    pass
        logger.info("Encrypting upload %s as %s", upload.filename, enc_path.name)
        enc_path.write_bytes(encrypt_bytes(key, cache_path.read_bytes()))
        is_large = _is_large_for_chat(size_bytes)
        if is_large:
            mime = (upload.content_type or "").strip() or _guess_mime_from_suffix(upload.filename)
            # Path B ("raw_large") needs a plaintext file on disk for rg_search + window reads.
            # Store a cache copy keyed by file_id so it can be cleaned up with the rest of cache.
        else:
            mime = detect_mime_type(cache_path)
        logger.info("Registering file %s (%s)", file_id, upload.filename)
        metadata_store.register_file(
            file_id=file_id,
            filename=upload.filename,
            mime=mime,
            size_bytes=size_bytes,
            hash_value=hash_value,
            stored_path=str(enc_path),
            is_encrypted=True,
            policy={"pii": False, "raw_large": bool(is_large), "ingest": not bool(is_large)},
            user_id=user_id or "default",
            chat_id=chat_id,
            source="upload_raw_large" if is_large else "upload",
        )
        emit_event(
            "files_changed",
            chat_id=chat_id,
            file_id=file_id,
            filename=upload.filename,
            status="registered",
        )
        if is_large:
            # Path B ("raw_large") — do not run extraction/chunking/embedding. Mark completed
            # so UI doesn't wait on ingestion progress.
            try:
                metadata_store.mark_file_status(file_id, "completed")
            except Exception:
                pass
            stored_files.append({"file_id": file_id, "filename": upload.filename, "job_id": "raw_large"})
            continue

        request = IngestionRequest(
            file_id=file_id,
            source_path=str(cache_path),
            filename=upload.filename,
            mime=mime,
            size_bytes=size_bytes,
            hash=hash_value,
            policy=FilePolicy(),
            created_at=datetime.now(timezone.utc),
            user_id=user_id or "default",
            chat_id=chat_id,
            source="upload",
            cleanup_path=str(cache_path),
        )
        job_id = scheduler.schedule(request)
        logger.info("Queued ingestion job %s for %s", job_id, file_id)
        stored_files.append({"file_id": file_id, "filename": upload.filename, "job_id": job_id})
    return {"files": stored_files}


@router.post("/ingest_path")
async def ingest_paths(payload: Dict[str, Any] = Body(...)):
    """
    Desktop-friendly ingestion entrypoint (JSON-only):
      { "chat_id": "...", "user_id": "...", "paths": ["/abs/path/to/file.pdf", ...] }

    Reads each path locally, encrypts + stores it under workspace uploads, and schedules ingestion.
    """
    from backend.services.extraction.detector import detect_mime_type
    from backend.services.ingestion import FilePolicy, IngestionRequest
    from backend.services.ipc_events import emit_event, is_ipc_mode
    from backend.services.security import encrypt_bytes

    if not is_ipc_mode():
        # Reading arbitrary local filesystem paths is a desktop-only feature. In HTTP
        # server mode, callers must upload bytes instead.
        raise HTTPException(status_code=403, detail="ingest_path is only available in desktop IPC mode")

    chat_id = payload.get("chat_id")
    user_id = payload.get("user_id") or "default"
    paths = payload.get("paths")
    if not chat_id or not isinstance(chat_id, str):
        raise HTTPException(status_code=400, detail="chat_id is required")
    if not isinstance(paths, list) or not paths:
        raise HTTPException(status_code=400, detail="paths must be a non-empty list")

    workspace = AppDependencies.workspace()
    scheduler = AppDependencies.ingestion_scheduler()
    key = AppDependencies.key_manager().get_key()
    metadata_store, _ = AppDependencies.storage()

    stored_files: list[dict[str, str]] = []

    validated: list[Path] = []
    incoming_sizes: list[tuple[str, int]] = []
    for raw in paths:
        if not isinstance(raw, str) or not raw:
            continue
        src_path = Path(raw).expanduser()
        if not src_path.exists() or not src_path.is_file():
            raise HTTPException(status_code=400, detail=f"Invalid file path: {raw}")
        try:
            size_bytes = int(src_path.stat().st_size)
        except OSError:
            size_bytes = 0
        validated.append(src_path)
        incoming_sizes.append((src_path.name, size_bytes))

    if not validated:
        raise HTTPException(status_code=400, detail="No valid paths provided")

    _enforce_chat_upload_size_policy(store=metadata_store, chat_id=chat_id, incoming=incoming_sizes)

    for src_path in validated:

        filename = src_path.name
        suffix = src_path.suffix

        content = await asyncio.to_thread(src_path.read_bytes)
        hash_value = hashlib.sha256(content).hexdigest()
        file_id = f"file_{hash_value}"

        enc_path = workspace.uploads / f"{file_id}.enc"
        await asyncio.to_thread(enc_path.write_bytes, encrypt_bytes(key, content))
        size_bytes = len(content)
        is_large = _is_large_for_chat(size_bytes)
        if is_large:
            mime = await asyncio.to_thread(detect_mime_type, src_path)
            try:
                cache_path = workspace.cache / f"{file_id}{suffix or '.txt'}"
                await asyncio.to_thread(cache_path.write_bytes, content)
            except Exception:
                pass
        else:
            # Copy plaintext to cache for extraction/indexing; scheduler cleans it up.
            temp_path = workspace.cache / f"{file_id}{suffix or '.tmp'}"
            await asyncio.to_thread(temp_path.write_bytes, content)
            mime = await asyncio.to_thread(detect_mime_type, temp_path)

        metadata_store.register_file(
            file_id=file_id,
            filename=filename,
            mime=mime,
            size_bytes=size_bytes,
            hash_value=hash_value,
            stored_path=str(enc_path),
            is_encrypted=True,
            policy={"pii": False, "raw_large": bool(is_large), "ingest": not bool(is_large)},
            user_id=user_id,
            chat_id=chat_id,
            source="desktop_path_raw_large" if is_large else "desktop_path",
        )
        emit_event(
            "files_changed",
            chat_id=chat_id,
            file_id=file_id,
            filename=filename,
            status="registered",
        )

        if is_large:
            try:
                metadata_store.mark_file_status(file_id, "completed")
            except Exception:
                pass
            stored_files.append({"file_id": file_id, "filename": filename, "job_id": "raw_large"})
            continue

        temp_path = workspace.cache / f"{file_id}{suffix or '.tmp'}"
        request = IngestionRequest(
            file_id=file_id,
            source_path=str(temp_path),
            filename=filename,
            mime=mime,
            size_bytes=size_bytes,
            hash=hash_value,
            policy=FilePolicy(),
            created_at=datetime.now(timezone.utc),
            user_id=user_id,
            chat_id=chat_id,
            source="desktop_path",
            cleanup_path=str(temp_path),
        )
        job_id = scheduler.schedule(request)
        stored_files.append({"file_id": file_id, "filename": filename, "job_id": job_id})

    if not stored_files:
        raise HTTPException(status_code=400, detail="No valid paths provided")

    return {"files": stored_files}


@router.get("/chat/{chat_id}")
async def list_files_for_chat(chat_id: str):
    """
    List files associated with a chat (used by the desktop Documents pane).
    """
    store = AppDependencies.sqlite_store()
    rows = store.list_files_for_chat(chat_id)
    files: list[dict[str, object]] = []
    for row in rows:
        file_id = row.get("id")
        if not isinstance(file_id, str) or not file_id:
            continue
        record = store.get_file(file_id) or {}
        files.append(
            {
                "file_id": file_id,
                "filename": record.get("filename") or row.get("filename") or "",
                "mime": record.get("mime") or "",
                "size_bytes": record.get("size_bytes") or 0,
                "status": record.get("status") or "",
                "pages": record.get("pages"),
                "created_at": record.get("created_at"),
                "updated_at": record.get("updated_at"),
                "source": record.get("source") or "",
            }
        )
    return {"files": files}

@router.delete("/chat/{chat_id}/{file_id}")
async def delete_file_from_chat(chat_id: str, file_id: str):
    """
    Remove a file from a chat (and delete its chat-scoped index data).

    If this was the last chat referencing the file_id, this also deletes the underlying
    file record, extracted text, chunks, jobs, and encrypted bytes on disk.
    """
    from backend.services.ipc_events import emit_event

    if not chat_id:
        raise HTTPException(status_code=400, detail="chat_id is required")
    if not file_id:
        raise HTTPException(status_code=400, detail="file_id is required")

    store = AppDependencies.sqlite_store()
    if not store.chat_has_file(chat_id, file_id):
        raise HTTPException(status_code=404, detail="file not linked to chat")

    record = store.get_file(file_id) or {}
    filename = str(record.get("filename") or "")
    stored_path = str(record.get("stored_path") or "")

    # Cancel any in-flight ingestion for this file in this chat (best-effort).
    cancelled_jobs = 0
    try:
        scheduler = AppDependencies.ingestion_scheduler()
        cancelled_jobs = int(scheduler.cancel_file(file_id, chat_id=chat_id))
    except Exception:
        cancelled_jobs = 0

    # Delete chat-scoped chunks in both SQLite and Qdrant.
    deleted_sqlite_chunks = 0
    try:
        deleted_sqlite_chunks = int(store.delete_chunks_for_chat_file(chat_id, file_id))
    except Exception:
        deleted_sqlite_chunks = 0

    try:
        rag_store = AppDependencies.rag_store()
        retrieval = getattr(rag_store, "retrieval", None)
        if retrieval is not None and hasattr(retrieval, "delete_chunks_for_chat_file"):
            retrieval.delete_chunks_for_chat_file(chat_id, file_id)
    except Exception:
        pass

    # Remove the per-chat editor page (edited representation).
    deleted_doc_page = 0
    try:
        deleted_doc_page = int(store.delete_doc_page(chat_id, file_id))
    except Exception:
        deleted_doc_page = 0

    # Unlink from this chat.
    deleted_chat_file_link = 0
    try:
        deleted_chat_file_link = int(store.delete_chat_file(chat_id, file_id))
    except Exception:
        deleted_chat_file_link = 0

    remaining_chats = store.list_chat_ids_for_file(file_id)
    fully_deleted = False
    deleted_file_rows: dict[str, int] | None = None
    deleted_encrypted_bytes = False

    if not remaining_chats:
        fully_deleted = True
        try:
            deleted_file_rows = store.delete_file_everywhere(file_id)
        except Exception:
            deleted_file_rows = None
        try:
            rag_store = AppDependencies.rag_store()
            retrieval = getattr(rag_store, "retrieval", None)
            if retrieval is not None and hasattr(retrieval, "delete_chunks_for_file"):
                retrieval.delete_chunks_for_file(file_id)
        except Exception:
            pass

        if stored_path:
            try:
                p = Path(stored_path).expanduser()
                if p.exists() and p.is_file():
                    p.unlink()
                    deleted_encrypted_bytes = True
            except Exception as exc:
                logger.warning("Failed to delete encrypted file bytes for %s: %s", file_id, exc)

    emit_event(
        "files_changed",
        chat_id=chat_id,
        file_id=file_id,
        filename=filename,
        status="deleted",
    )

    return {
        "ok": True,
        "chat_id": chat_id,
        "file_id": file_id,
        "filename": filename,
        "cancelled_jobs": cancelled_jobs,
        "deleted": {
            "chat_file_link": deleted_chat_file_link,
            "doc_page": deleted_doc_page,
            "sqlite_chunks": deleted_sqlite_chunks,
            "fully_deleted": fully_deleted,
            "encrypted_bytes": deleted_encrypted_bytes,
            "sqlite_file_rows": deleted_file_rows,
        },
        "remaining_chats": remaining_chats,
    }


@router.get("/progress/{chat_id}")
async def ingestion_progress(chat_id: str):
    """
    Return per-chat ingestion progress (desktop UI helper).

    This is intentionally cheap and deterministic: it reflects SQLite file/job rows,
    not transient in-memory state.
    """
    store = AppDependencies.sqlite_store()
    return store.chat_ingestion_progress(chat_id)

@router.get("/extracted/{file_id}")
async def get_extracted_view(file_id: str):
    """
    Return a display-ready extracted representation for a file.

    The desktop UI should render these blocks instead of attempting to embed PDFs/DOCX directly.
    """
    from backend.services.docs import blocks_to_plain_text, prosemirror_doc_to_blocks
    from backend.services.extraction.service import blocks_from_text

    store = AppDependencies.sqlite_store()
    record = store.get_file(file_id)
    if not record:
        raise HTTPException(status_code=404, detail="file not found")

    file_text = store.get_file_text(file_id) or {}
    blocks = file_text.get("blocks") or []
    plain_text = file_text.get("plain_text") or ""

    # If the user has edited this file's representation in the current chat, prefer the
    # saved ProseMirror page (per chat_id + file_id) over the raw extractor output.
    chat_id = record.get("chat_id")
    if isinstance(chat_id, str) and chat_id:
        page = store.get_doc_page(chat_id, file_id)
        if page and page.get("is_user_edited") and isinstance(page.get("doc"), dict):
            derived_blocks = prosemirror_doc_to_blocks(page["doc"])
            derived_text = blocks_to_plain_text(derived_blocks)
            if derived_blocks:
                blocks = derived_blocks
                plain_text = derived_text

    # Backfill for older ingestions: reconstruct from stored chunks if we don't have
    # persisted extraction output yet.
    if not blocks and not plain_text:
        chunk_texts = store.fetch_chunk_texts_for_file(file_id)
        if chunk_texts:
            reconstructed = "\n\n".join(chunk_texts)
            extracted_blocks = [
                {"kind": b.kind, "text": b.text, "metadata": b.metadata} for b in blocks_from_text(reconstructed)
            ]
            store.upsert_file_text(file_id, text=reconstructed, blocks=extracted_blocks)
            blocks = extracted_blocks
            plain_text = reconstructed

    return {
        "file_id": file_id,
        "filename": record.get("filename") or "",
        "mime": record.get("mime") or "application/octet-stream",
        "status": record.get("status") or "",
        "pages": record.get("pages"),
        "blocks": blocks,
        "plain_text": plain_text,
    }
