from __future__ import annotations

import asyncio
import logging
import threading
import os
from typing import Any, Dict, Optional, List, Annotated

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Form, UploadFile, File, Request, Body
from fastapi.responses import StreamingResponse

from backend.services.planner import PlannerRequest
from backend.services.planner.summarizer import summarize_text as do_summarize_text
from backend.api.deps import AppDependencies
from backend.services.ingestion import create_extraction_service, IngestionRequest, FilePolicy
from backend.services.extraction.detector import detect_mime_type
from backend.services.security import encrypt_bytes
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import asyncio

router = APIRouter(prefix="/chat", tags=["Chat"])
logger = logging.getLogger(__name__)
_upload_locks: Dict[str, asyncio.Lock] = {}

# Minimal backend-enforced defaults (avoid “frontend-only” limits).
# Kept here (not a full config system) to stay simple and consistent across IPC/HTTP.
MAX_ATTACHMENT_BYTES = 5 * 1024 * 1024  # 5 MiB


def _enforce_max_attachment_size(*, filename: str, size_bytes: int) -> None:
    if size_bytes <= MAX_ATTACHMENT_BYTES:
        return
    raise HTTPException(
        status_code=413,
        detail={
            "error": "file_too_large",
            "filename": filename,
            "size_bytes": size_bytes,
            "max_bytes": MAX_ATTACHMENT_BYTES,
        },
    )


@router.post("")
async def chat_entry(
    request: Request,
    chat_id: Optional[str] = Form(default=None),
    query: Optional[str] = Form(default=None),
    include_context_status: Optional[bool] = Form(default=False),
    files: Optional[List[UploadFile]] = File(default=None),
):
    """
    Unified chat endpoint:
    - If multipart with files is provided, extract text inline, queue ingestion, and answer with those docs.
    - Otherwise, treat body as JSON payload for regular chat.
    """
    # If files were provided, do upload+chat
    if files:
        if not chat_id or not query:
            raise HTTPException(status_code=400, detail="chat_id and query are required with file upload")
        return await _upload_and_chat_internal(chat_id, query, files, include_context_status=bool(include_context_status))

    # If form fields present without files, treat as a normal chat call
    if chat_id and query:
        payload = {
            "chat_id": chat_id,
            "query": query,
            "include_context_status": include_context_status,
        }
        return _run_planner(payload)

    # Otherwise expect JSON body
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON or missing payload")

    # Desktop/IPC-friendly attachments: user provides local file paths.
    # We only ingest when a query is sent (same call), and we attach extracted text inline.
    if payload.get("paths"):
        payload = await _ingest_paths_for_chat(payload)
    return _run_planner(payload)


async def _upload_and_chat_internal(chat_id: str, query: str, files: List[UploadFile], include_context_status: bool = False):
    # Use a dedicated async lock for upload flow to avoid deadlocks with session locks.
    lock = _upload_locks.setdefault(chat_id, asyncio.Lock())
    async with lock:
        workspace = AppDependencies.workspace()
        scheduler = AppDependencies.ingestion_scheduler()
        key = AppDependencies.key_manager().get_key()
        extraction_service = create_extraction_service()
        doc_texts: list[str] = []
        doc_ids: list[str] = []
        for upload in files:
            filename = upload.filename or ""
            suffix = Path(filename).suffix if filename else ""
            content = await upload.read()
            _enforce_max_attachment_size(filename=filename or "upload", size_bytes=len(content))
            hash_value = hashlib.sha256(content).hexdigest()
            file_id = f"file_{hash_value}"
            enc_path = workspace.uploads / f"{file_id}.enc"
            logger.info("Encrypting upload %s as %s", upload.filename, enc_path.name)
            await asyncio.to_thread(enc_path.write_bytes, encrypt_bytes(key, content))
            temp_path = workspace.cache / f"{file_id}{suffix or '.tmp'}"
            await asyncio.to_thread(temp_path.write_bytes, content)
            size_bytes = len(content)
            mime = await asyncio.to_thread(detect_mime_type, temp_path)

            # Synchronous extraction for immediate use
            extracted = await asyncio.to_thread(extraction_service.extract, temp_path, mime_type=mime)
            truncated = _truncate_doc_text(extracted.text or "")
            if truncated.strip():
                doc_texts.append(truncated)
            else:
                logger.warning("Extraction returned empty text for upload filename=%s mime=%s", filename, mime)
            doc_ids.append(file_id)

            # Register and queue ingestion for indexing
            metadata_store, _ = AppDependencies.storage()
            metadata_store.register_file(
                file_id=file_id,
                filename=filename,
                mime=mime,
                size_bytes=size_bytes,
                hash_value=hash_value,
                stored_path=str(enc_path),
                is_encrypted=True,
                policy={"pii": False},
            )
            ingestion_request = IngestionRequest(
                file_id=file_id,
                source_path=str(temp_path),
                filename=filename,
                mime=mime,
                size_bytes=size_bytes,
                hash=hash_value,
                policy=FilePolicy(),
                created_at=datetime.now(timezone.utc),
                user_id="default",
                chat_id=chat_id,
                source="upload",
                cleanup_path=str(temp_path),
            )
            job_id = scheduler.schedule(ingestion_request)
            logger.info("Queued ingestion job %s for %s", job_id, file_id)

        # Invoke planner with inline documents_text
        service = AppDependencies.planner_service()
        req = PlannerRequest(
            chat_id=chat_id,
            query=query,
            documents=doc_ids,
            documents_text=doc_texts,
        )
        result = service.handle_request(req)
        resp: Dict[str, Any] = {
            "answer": result.answer,
        }
        if include_context_status:
            try:
                mgr = AppDependencies.session_manager()
                resp["context"] = mgr.get_context_status(chat_id)
            except Exception as exc:
                logger.warning("Failed to build context status: %s", exc)
        return resp


async def _ingest_paths_for_chat(payload: Dict[str, Any]) -> Dict[str, Any]:
    chat_id = payload.get("chat_id")
    query = payload.get("query")
    paths = payload.get("paths") or []

    if not isinstance(chat_id, str) or not chat_id:
        raise HTTPException(status_code=400, detail="chat_id is required")
    if not isinstance(query, str) or not query:
        raise HTTPException(status_code=400, detail="query is required when using paths")
    if not isinstance(paths, list) or not paths:
        raise HTTPException(status_code=400, detail="paths must be a non-empty list")

    # Use the same upload lock strategy as multipart upload+chat.
    lock = _upload_locks.setdefault(chat_id, asyncio.Lock())
    async with lock:
        workspace = AppDependencies.workspace()
        scheduler = AppDependencies.ingestion_scheduler()
        key = AppDependencies.key_manager().get_key()
        extraction_service = create_extraction_service()
        metadata_store, _ = AppDependencies.storage()

        doc_texts: list[str] = []
        doc_ids: list[str] = []
        attachment_names: list[str] = []

        for raw in paths:
            if not isinstance(raw, str) or not raw:
                continue
            src_path = Path(raw).expanduser()
            if not src_path.exists() or not src_path.is_file():
                raise HTTPException(status_code=400, detail=f"Invalid file path: {raw}")

            filename = src_path.name
            suffix = src_path.suffix
            attachment_names.append(filename)

            try:
                size_bytes = src_path.stat().st_size
            except OSError as exc:
                raise HTTPException(status_code=400, detail=f"Cannot stat file {raw}: {exc}") from exc
            _enforce_max_attachment_size(filename=filename, size_bytes=size_bytes)

            content = await asyncio.to_thread(src_path.read_bytes)
            hash_value = hashlib.sha256(content).hexdigest()
            file_id = f"file_{hash_value}"

            enc_path = workspace.uploads / f"{file_id}.enc"
            logger.info("Encrypting desktop path %s as %s", filename, enc_path.name)
            await asyncio.to_thread(enc_path.write_bytes, encrypt_bytes(key, content))

            temp_path = workspace.cache / f"{file_id}{suffix or '.tmp'}"
            await asyncio.to_thread(temp_path.write_bytes, content)
            mime = await asyncio.to_thread(detect_mime_type, temp_path)

            # Synchronous extraction for immediate use (inline docs in prompt)
            extracted = await asyncio.to_thread(extraction_service.extract, temp_path, mime_type=mime)
            truncated = _truncate_doc_text(extracted.text or "")
            if truncated.strip():
                doc_texts.append(truncated)
            else:
                logger.warning("Extraction returned empty text for desktop path filename=%s mime=%s", filename, mime)
            doc_ids.append(file_id)

            metadata_store.register_file(
                file_id=file_id,
                filename=filename,
                mime=mime,
                size_bytes=size_bytes,
                hash_value=hash_value,
                stored_path=str(enc_path),
                is_encrypted=True,
                policy={"pii": False},
                user_id="default",
                chat_id=chat_id,
                source="desktop_path",
            )

            ingestion_request = IngestionRequest(
                file_id=file_id,
                source_path=str(temp_path),
                filename=filename,
                mime=mime,
                size_bytes=size_bytes,
                hash=hash_value,
                policy=FilePolicy(),
                created_at=datetime.now(timezone.utc),
                user_id="default",
                chat_id=chat_id,
                source="desktop_path",
                cleanup_path=str(temp_path),
            )
            job_id = scheduler.schedule(ingestion_request)
            logger.info("Queued ingestion job %s for %s", job_id, file_id)

        if not doc_ids:
            raise HTTPException(status_code=400, detail="No valid paths provided")

        # Mutate into planner-compatible fields and remove paths so it won't be reprocessed.
        payload = dict(payload)
        payload.pop("paths", None)
        payload["documents"] = doc_ids
        payload["documents_text"] = doc_texts
        payload["attachments"] = attachment_names
        return payload


@router.websocket("/stream")
async def chat_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        payload = await websocket.receive_json()
        req = _build_request(payload)
    except Exception as exc:  # pragma: no cover
        await websocket.send_json({"event": "error", "detail": str(exc)})
        await websocket.close(code=1003)
        return

    service = AppDependencies.planner_service()
    stream = service.stream_request(req)  # type: ignore[attr-defined]

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue()

    def _producer() -> None:
        try:
            for chunk in stream:
                asyncio.run_coroutine_threadsafe(queue.put(chunk), loop).result()
        except Exception as exc:
            logger.exception("Streaming planner error")
            asyncio.run_coroutine_threadsafe(
                queue.put({"event": "error", "detail": str(exc)}),
                loop,
            ).result()
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

    threading.Thread(target=_producer, daemon=True).start()

    try:
        while True:
            message = await queue.get()
            if message is None:
                break
            await websocket.send_json(message)
    except WebSocketDisconnect:
        logger.info("Client disconnected from /chat/stream")
    finally:
        if hasattr(stream, "close"):
            try:
                stream.close()
            except Exception:
                pass
        await websocket.close()


def _run_planner(payload: dict):
    service = AppDependencies.planner_service()
    req = _build_request(payload)
    logger.info(
        "chat _run_planner chat=%s stream=%s docs=%d",
        req.chat_id,
        payload.get("stream", True),
        len(req.documents or []) if hasattr(req, "documents") else 0,
    )
    # Default to streaming unless explicitly disabled.
    if payload.get("stream", True):
        logger.info("chat streaming start chat=%s", req.chat_id)
        blocking_stream = service.stream_request(req)

        async def token_streamer():
            queue: asyncio.Queue[str | None] = asyncio.Queue()

            loop = asyncio.get_running_loop()

            def producer():
                count = 0
                try:
                    for token in blocking_stream:
                        count += 1
                        if count == 1:
                            logger.info("chat streaming first token chat=%s", req.chat_id)
                        # asyncio.Queue is not thread-safe; schedule puts on the event loop.
                        asyncio.run_coroutine_threadsafe(queue.put(token or ""), loop).result()
                except Exception as exc:
                    logger.exception("Streaming planner error")
                    asyncio.run_coroutine_threadsafe(queue.put(f"[ERROR] {exc}"), loop).result()
                finally:
                    logger.info("chat streaming producer done chat=%s tokens=%d", req.chat_id, count)
                    asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

            loop.run_in_executor(None, producer)
            while True:
                token = await queue.get()
                if token is None:
                    break
                if token:
                    # Yield each token immediately for true real-time streaming.
                    yield token

            yield "\n[DONE]\n"

        return StreamingResponse(token_streamer(), media_type="text/plain")

    result = service.handle_request(req)
    resp: Dict[str, Any] = {"answer": result.answer}
    include_ctx = payload.get("include_context_status")
    if include_ctx:
        try:
            mgr = AppDependencies.session_manager()
            resp["context"] = mgr.get_context_status(req.chat_id)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to build context status: %s", exc)
    return resp


@router.get("/sessions")
def list_sessions():
    mgr = AppDependencies.session_manager()
    return {"sessions": mgr.list_sessions()}


@router.get("/context/{chat_id}")
def context_status(chat_id: str):
    mgr = AppDependencies.session_manager()
    try:
        status = mgr.get_context_status(chat_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Chat not found")
    return status

@router.post("/summarize_text")
def summarize_text(payload: Dict[str, Any] = Body(...)):
    """
    Summarize arbitrary text without persisting any session state.
    """
    text = payload.get("text")
    if not text:
        raise HTTPException(status_code=400, detail="Field 'text' is required")
    max_tokens = payload.get("max_tokens", 512)

    mgr = AppDependencies.session_manager()
    summary = do_summarize_text(text, mgr, max_tokens=max_tokens)
    return {"summary": summary}

@router.delete("/sessions/{chat_id}")
def delete_session(chat_id: str):
    deleted: Dict[str, Any] = {"chat_id": chat_id}

    # 0) Cancel any in-flight ingestion jobs for this chat so they don't recreate chunks
    # after we've deleted Qdrant/SQLite rows.
    try:
        scheduler = AppDependencies.ingestion_scheduler()
        deleted["ingestion_jobs_cancelled"] = scheduler.cancel_chat(chat_id)
    except Exception as exc:
        logger.warning("Failed to cancel ingestion jobs for chat_id=%s: %s", chat_id, exc)
        deleted["ingestion_jobs_cancelled"] = 0

    # 1) Gather file records before deleting metadata (for disk cleanup).
    file_rows: list[dict[str, object]] = []
    try:
        store, _ = AppDependencies.storage()
        file_rows = store.list_files_for_chat(chat_id)
        deleted["files_found"] = len(file_rows)
    except Exception as exc:
        logger.warning("Failed to list files for chat_id=%s: %s", chat_id, exc)

    # 2) Delete UI transcript + ingestion metadata in SQLite FIRST so any reload that
    # happens during KV deletion cannot "resurrect" the chat from transcript rows.
    try:
        store, _ = AppDependencies.storage()
        deleted["sqlite_messages_deleted"] = store.delete_messages_for_chat(chat_id)
        deleted["sqlite_chunks_deleted"] = None
        try:
            store.delete_chunks_for_chat(chat_id)
            deleted["sqlite_chunks_deleted"] = True
        except Exception:
            deleted["sqlite_chunks_deleted"] = False
        deleted["sqlite_jobs_deleted"] = store.delete_jobs_for_chat(chat_id)
        deleted["sqlite_files_deleted"] = store.delete_files_for_chat(chat_id)
    except Exception as exc:
        logger.warning("Failed to delete SQLite data for chat_id=%s: %s", chat_id, exc)

    # 3) Delete KV session (model state + kv snapshots) AFTER SQLite transcript deletion.
    # This is what triggers the kv_sessions watcher event used by the UI.
    try:
        mgr = AppDependencies.session_manager()
        mgr.delete_session(chat_id)
        deleted["kv_session_deleted"] = True
    except Exception as exc:
        logger.warning("Failed to delete KV session for chat_id=%s: %s", chat_id, exc)
        deleted["kv_session_deleted"] = False

    # 4) Delete RAG vectors from Qdrant (insight_chunks) for this chat
    try:
        rag_store = AppDependencies.rag_store()
        retrieval = getattr(rag_store, "retrieval", None)
        if retrieval is not None and hasattr(retrieval, "delete_chunks_for_chat"):
            retrieval.delete_chunks_for_chat(chat_id)
            deleted["qdrant_chunks_deleted"] = True
        else:
            deleted["qdrant_chunks_deleted"] = False
    except Exception as exc:
        logger.warning("Failed to delete Qdrant chunks for chat_id=%s: %s", chat_id, exc)
        deleted["qdrant_chunks_deleted"] = False

    # 5) Delete LTM entries for this chat (Qdrant insight_memories)
    try:
        ltm = AppDependencies.ltm_store()
        if hasattr(ltm, "delete_chat"):
            ltm.delete_chat(chat_id)
            deleted["ltm_deleted"] = True
    except Exception as exc:
        logger.warning("Failed to delete LTM for chat_id=%s: %s", chat_id, exc)
        deleted["ltm_deleted"] = False

    # 6) Remove encrypted raw uploads from disk (best effort)
    try:
        removed = 0
        workspace = AppDependencies.workspace()
        for row in file_rows:
            stored_path = row.get("stored_path")
            file_id = row.get("id")
            if isinstance(stored_path, str) and stored_path:
                try:
                    p = Path(stored_path)
                    if p.exists():
                        p.unlink()
                        removed += 1
                except Exception:
                    pass
            # Also try to remove any leftover cache temp for this file_id.
            if isinstance(file_id, str) and file_id:
                try:
                    for candidate in workspace.cache.glob(f"{file_id}*"):
                        if candidate.is_file():
                            candidate.unlink()
                except Exception:
                    pass
        deleted["uploads_deleted"] = removed
    except Exception as exc:
        logger.warning("Failed to delete uploads for chat_id=%s: %s", chat_id, exc)

    return {"deleted": deleted}


def _summarize_citations(citations: list[Dict[str, Any]]) -> list[str]:
    if not citations:
        return []
    names: list[str] = []
    seen = set()
    for item in citations:
        name = item.get("filename") or item.get("file_id")
        if not name:
            continue
        if name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def _build_request(payload: Dict[str, Any]) -> PlannerRequest:
    try:
        docs = payload.get("documents")
        if docs is None:
            docs = payload.get("document_ids") or []
        return PlannerRequest(
            chat_id=payload["chat_id"],
            query=payload["query"],
            documents=docs,
            documents_text=payload.get("documents_text", []),
            attachments=payload.get("attachments", []) or [],
            screenshot=payload.get("screenshot"),
            request_id=payload.get("request_id"),
        )
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"Missing field {exc.args[0]}") from exc


def _truncate_doc_text(text: str) -> str:
    # Only truncate very long inline docs to avoid overloading prompts.
    if not text:
        return ""
    if len(text) > 20000:
        return text[:20000]
    return text
