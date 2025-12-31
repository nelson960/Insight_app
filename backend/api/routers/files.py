from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Body

from backend.api.deps import AppDependencies
from backend.services.ingestion import IngestionRequest, FilePolicy
from backend.services.extraction.detector import detect_mime_type
from backend.services.security import encrypt_bytes
from backend.services.extraction.service import blocks_from_text
from backend.services.ipc_events import emit_event, is_ipc_mode
from backend.services.docs import blocks_to_plain_text, prosemirror_doc_to_blocks

import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/files", tags=["Files"])


@router.post("/upload")
async def upload_files(
    chat_id: str = Form(...),
    user_id: str = Form("default"),
    files: List[UploadFile] = File(...),
):
    workspace = AppDependencies.workspace()
    scheduler = AppDependencies.ingestion_scheduler()
    key = AppDependencies.key_manager().get_key()
    stored_files = []
    for upload in files:
        suffix = Path(upload.filename).suffix
        content = await upload.read()
        hash_value = hashlib.sha256(content).hexdigest()
        file_id = f"file_{hash_value}"
        enc_path = workspace.uploads / f"{file_id}.enc"
        logger.info("Encrypting upload %s as %s", upload.filename, enc_path.name)
        enc_path.write_bytes(encrypt_bytes(key, content))
        temp_path = workspace.cache / f"{file_id}{suffix or '.tmp'}"
        temp_path.write_bytes(content)
        size_bytes = len(content)
        mime = detect_mime_type(temp_path)
        metadata_store, _ = AppDependencies.storage()
        logger.info("Registering file %s (%s)", file_id, upload.filename)
        metadata_store.register_file(
            file_id=file_id,
            filename=upload.filename,
            mime=mime,
            size_bytes=size_bytes,
            hash_value=hash_value,
            stored_path=str(enc_path),
            is_encrypted=True,
            policy={"pii": False},
            user_id=user_id or "default",
            chat_id=chat_id,
            source="upload",
        )
        emit_event(
            "files_changed",
            chat_id=chat_id,
            file_id=file_id,
            filename=upload.filename,
            status="registered",
        )
        request = IngestionRequest(
            file_id=file_id,
            source_path=str(temp_path),
            filename=upload.filename,
            mime=mime,
            size_bytes=size_bytes,
            hash=hash_value,
            policy=FilePolicy(),
            created_at=datetime.now(timezone.utc),
            user_id=user_id or "default",
            chat_id=chat_id,
            source="upload",
            cleanup_path=str(temp_path),
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

    for raw in paths:
        if not isinstance(raw, str) or not raw:
            continue
        src_path = Path(raw).expanduser()
        if not src_path.exists() or not src_path.is_file():
            raise HTTPException(status_code=400, detail=f"Invalid file path: {raw}")

        filename = src_path.name
        suffix = src_path.suffix

        content = await asyncio.to_thread(src_path.read_bytes)
        hash_value = hashlib.sha256(content).hexdigest()
        file_id = f"file_{hash_value}"

        enc_path = workspace.uploads / f"{file_id}.enc"
        await asyncio.to_thread(enc_path.write_bytes, encrypt_bytes(key, content))

        # Copy plaintext to cache for extraction/indexing; scheduler cleans it up.
        temp_path = workspace.cache / f"{file_id}{suffix or '.tmp'}"
        await asyncio.to_thread(temp_path.write_bytes, content)
        size_bytes = len(content)
        mime = await asyncio.to_thread(detect_mime_type, temp_path)

        metadata_store.register_file(
            file_id=file_id,
            filename=filename,
            mime=mime,
            size_bytes=size_bytes,
            hash_value=hash_value,
            stored_path=str(enc_path),
            is_encrypted=True,
            policy={"pii": False},
            user_id=user_id,
            chat_id=chat_id,
            source="desktop_path",
        )
        emit_event(
            "files_changed",
            chat_id=chat_id,
            file_id=file_id,
            filename=filename,
            status="registered",
        )

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
    store, _ = AppDependencies.storage()
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
    if not chat_id:
        raise HTTPException(status_code=400, detail="chat_id is required")
    if not file_id:
        raise HTTPException(status_code=400, detail="file_id is required")

    store, _ = AppDependencies.storage()
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
    store, _ = AppDependencies.storage()
    return store.chat_ingestion_progress(chat_id)

@router.get("/extracted/{file_id}")
async def get_extracted_view(file_id: str):
    """
    Return a display-ready extracted representation for a file.

    The desktop UI should render these blocks instead of attempting to embed PDFs/DOCX directly.
    """
    store, _ = AppDependencies.storage()
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
