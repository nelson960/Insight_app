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
