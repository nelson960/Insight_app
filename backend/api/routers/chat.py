from __future__ import annotations

import asyncio
import json
import logging
import threading
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List, Annotated

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Form, UploadFile, File, Request, Body
from fastapi.responses import StreamingResponse

from backend.services.planner import PlannerRequest
from backend.services.planner.summarizer import summarize_text as do_summarize_text
from backend.api.deps import AppDependencies
from backend.services.ipc_events import emit_event
from pathlib import Path

router = APIRouter(prefix="/chat", tags=["Chat"])
logger = logging.getLogger(__name__)

def _required_file_ids_for_turn(payload: dict) -> list[str]:
    """
    Determine which file_ids must be "RAG ready" for this /chat turn.

    This is a defensive backend guard. The desktop UI already queues sends until
    ingestion completes, but API callers (or UI edge cases) can still hit /chat
    early and get low-quality answers with zero retrieval candidates.
    """
    docs_payload = payload.get("documents") or payload.get("document_ids") or []
    _attachments = payload.get("attachments") or []
    doc_scope_mode = payload.get("doc_scope_mode")
    doc_scope_mode = doc_scope_mode.strip().lower() if isinstance(doc_scope_mode, str) else ""

    # Selection hard-focuses a single file.
    selection = payload.get("selection")
    if isinstance(selection, dict):
        fid = selection.get("file_id")
        if isinstance(fid, str) and fid.strip():
            return [fid.strip()]

    focus_document_id = payload.get("focus_document_id")
    focus_document_id = focus_document_id.strip() if isinstance(focus_document_id, str) else ""
    doc_pane_open = payload.get("doc_pane_open")
    if not isinstance(doc_pane_open, bool):
        doc_pane_open = None
    doc_pane_open_bool = bool(doc_pane_open) if doc_pane_open is not None else bool(focus_document_id)

    # Multi-upload in the chat pane: require all uploaded docs for this turn.
    turn_doc_ids: list[str] = []
    if isinstance(docs_payload, list):
        for d in docs_payload:
            if isinstance(d, str) and d.strip():
                turn_doc_ids.append(d.strip())

    # Explicit "all docs" mode (UI hint): require all files in the chat.
    # This prevents low-quality answers when a compare-style turn is requested
    # before ingestion completes for all relevant files.
    if doc_scope_mode == "all":
        chat_id = payload.get("chat_id")
        if isinstance(chat_id, str) and chat_id:
            try:
                store, _ = AppDependencies.storage()
                all_ids = store.list_file_ids_for_chat(chat_id)
                out_all = []
                for fid in (all_ids or []):
                    if isinstance(fid, str) and fid.strip():
                        out_all.append(fid.strip())
                # Ensure newly attached docs for this turn are also included.
                out_all.extend(turn_doc_ids)
                seen = set()
                uniq: list[str] = []
                for fid in out_all:
                    if fid in seen:
                        continue
                    seen.add(fid)
                    uniq.append(fid)
                return uniq
            except Exception:
                # Fall through to default logic.
                pass
    if (not doc_pane_open_bool) and len(turn_doc_ids) > 1:
        # De-dup while preserving order.
        seen = set()
        uniq: list[str] = []
        for fid in turn_doc_ids:
            if fid in seen:
                continue
            seen.add(fid)
            uniq.append(fid)
        return uniq

    # Focused mode: require the focused file if provided, plus any newly attached docs.
    out: list[str] = []
    if doc_pane_open_bool and focus_document_id and not turn_doc_ids:
        out.append(focus_document_id)

    # Always include new attachments for this turn.
    out.extend(turn_doc_ids)

    # If the docs pane is hidden and no explicit focus is set, default to the latest file in the chat.
    if (not doc_pane_open_bool) and not out:
        chat_id = payload.get("chat_id")
        if isinstance(chat_id, str) and chat_id:
            try:
                store, _ = AppDependencies.storage()
                rows = store.list_files_for_chat(chat_id)
                last_fid: Optional[str] = None
                best_fid: Optional[str] = None
                best_ts: Optional[float] = None
                for r in rows or []:
                    if not isinstance(r, dict):
                        continue
                    fid = r.get("id")
                    if not isinstance(fid, str) or not fid.strip():
                        continue
                    last_fid = fid.strip()
                    ts = r.get("created_at") or r.get("created") or r.get("ts")
                    try:
                        ts_val = float(ts) if ts is not None else None
                    except Exception:
                        ts_val = None
                    if ts_val is not None and (best_ts is None or ts_val >= best_ts):
                        best_fid = fid.strip()
                        best_ts = ts_val
                chosen = best_fid or last_fid
                if chosen:
                    out.append(chosen)
            except Exception:
                pass

    # Defensive: some clients might send attachment names without documents; ignore those.
    _ = _attachments

    # De-dup while preserving order.
    seen = set()
    uniq: list[str] = []
    for fid in out:
        if fid in seen:
            continue
        seen.add(fid)
        uniq.append(fid)
    return uniq


def _enforce_rag_ready(payload: dict) -> None:
    """
    Enforce "RAG always" on the backend: if the user is asking in a doc-scoped
    context but ingestion hasn't finished, fail fast with a structured error.
    """
    chat_id = payload.get("chat_id")
    if not isinstance(chat_id, str) or not chat_id:
        return

    required = _required_file_ids_for_turn(payload)
    if not required:
        return

    store, _ = AppDependencies.storage()
    files = store.list_files_status_for_chat(chat_id)
    status_by_id: dict[str, str] = {}
    for f in files or []:
        fid = f.get("id")
        if not isinstance(fid, str) or not fid:
            continue
        st = f.get("status") or ""
        status_by_id[fid] = str(st)

    missing = [fid for fid in required if fid not in status_by_id]
    not_ready = [fid for fid in required if status_by_id.get(fid) not in {"completed", "failed"}]
    failed = [fid for fid in required if status_by_id.get(fid) == "failed"]

    if missing or not_ready or failed:
        progress = store.chat_ingestion_progress(chat_id)
        raise HTTPException(
            status_code=409,
            detail={
                "error": "ingestion_not_ready",
                "chat_id": chat_id,
                "required_file_ids": required,
                "missing_file_ids": missing,
                "not_ready_file_ids": not_ready,
                "failed_file_ids": failed,
                "progress": progress,
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
    - Uploads and local path ingestion are handled by the /files router (RAG-only flow).
    - Otherwise, treat body as JSON payload for regular chat.
    """
    if files:
        raise HTTPException(
            status_code=400,
            detail="Upload via /files/upload then call /chat once ingestion is complete.",
        )

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

    if payload.get("paths"):
        raise HTTPException(
            status_code=400,
            detail="Use /files/ingest_path then call /chat once ingestion is complete.",
        )
    return _run_planner(payload)


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
    # Backend guard: if this turn is doc-scoped, require ingestion to be complete.
    # The desktop UI also waits, but this prevents silent low-quality answers when
    # /chat is called early (e.g. API callers, edge cases).
    _enforce_rag_ready(payload)

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
    try:
        mgr = AppDependencies.session_manager()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
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

@router.post("/branch")
def branch_chat(payload: Dict[str, Any] = Body(...)):
    """
    Create a new child chat/card branched from a single message.

    - Seeds the child chat transcript with ONLY the selected message.
    - Optionally shares all documents from the parent chat (no re-ingestion).
    """
    parent_chat_id = payload.get("parent_chat_id")
    if not isinstance(parent_chat_id, str) or not parent_chat_id.strip():
        raise HTTPException(status_code=400, detail="parent_chat_id is required")
    parent_chat_id = parent_chat_id.strip()

    msg = payload.get("message") or {}
    if not isinstance(msg, dict):
        raise HTTPException(status_code=400, detail="message must be an object")
    role = msg.get("role")
    if role not in {"user", "assistant"}:
        raise HTTPException(status_code=400, detail="message.role must be 'user' or 'assistant'")
    content = msg.get("content")
    if not isinstance(content, str):
        content = msg.get("text")
    if not isinstance(content, str) or not content.strip():
        raise HTTPException(status_code=400, detail="message.content is required")
    content = content.strip()

    share_docs = payload.get("share_docs")
    share_docs = True if share_docs is None else bool(share_docs)

    # Create a stable-ish ID that matches the rest of the app's "chat-*" convention.
    child_chat_id = f"chat-{int(datetime.now(timezone.utc).timestamp() * 1000)}-{uuid.uuid4().hex[:6]}"

    store, _ = AppDependencies.storage()

    shared_file_ids: list[str] = []
    if share_docs:
        try:
            shared_file_ids = store.list_file_ids_for_chat(parent_chat_id)
        except Exception:
            shared_file_ids = []

        for fid in shared_file_ids:
            try:
                store.add_file_to_chat(child_chat_id, fid)
            except Exception:
                continue

        # Copy any per-chat edited doc pages (best effort) so the child card opens with
        # the same document view as the parent at branch time.
        for fid in shared_file_ids:
            try:
                page = store.get_doc_page(parent_chat_id, fid)
                if not page:
                    continue
                doc = page.get("doc")
                if doc is None:
                    continue
                store.upsert_doc_page(
                    child_chat_id,
                    fid,
                    title=str(page.get("title") or ""),
                    doc=doc,
                    is_user_edited=bool(page.get("is_user_edited") or False),
                    source_file_updated_at=page.get("source_file_updated_at") if isinstance(page.get("source_file_updated_at"), str) else None,
                )
            except Exception:
                continue

    # Persist the seeded UI transcript message so the card shows up immediately.
    message_id = f"msg_{uuid.uuid4().hex}"
    created_at = datetime.now(timezone.utc).isoformat()
    store.insert_message(
        message_id=message_id,
        chat_id=child_chat_id,
        role=role,
        content_json=json.dumps({"text": content}, ensure_ascii=False),
        model="branch",
        mode="chat",
        planner_payload_json=None,
        citations_json=None,
        created_at=created_at,
    )

    # Seed the model session (clean KV) with the same message so the next turn
    # continues naturally from the branched content.
    try:
        mgr = AppDependencies.session_manager()
        mgr.seed_session_messages(
            child_chat_id,
            system_prompt="You are Insight, a local privacy-first AI assistant.",
            messages=[{"role": role, "content": content}],
        )
    except Exception as exc:
        logger.warning("Failed to seed branched KV session chat=%s: %s", child_chat_id, exc)

    return {
        "ok": True,
        "child_chat_id": child_chat_id,
        "parent_chat_id": parent_chat_id,
        "shared_docs": share_docs,
        "shared_file_ids": shared_file_ids if share_docs else [],
    }

@router.delete("/sessions/{chat_id}")
def delete_session(chat_id: str):
    deleted: Dict[str, Any] = {"chat_id": chat_id}

    store, _ = AppDependencies.storage()

    # 0) Cancel any in-flight ingestion jobs for this chat so they don't recreate chunks
    # after we've deleted Qdrant/SQLite rows. The scheduler will skip cancellation for
    # files that are still shared by other chats.
    try:
        scheduler = AppDependencies.ingestion_scheduler()
        deleted["ingestion_jobs_cancelled"] = scheduler.cancel_chat(chat_id)
    except Exception as exc:
        logger.warning("Failed to cancel ingestion jobs for chat_id=%s: %s", chat_id, exc)
        deleted["ingestion_jobs_cancelled"] = 0

    # 1) Gather file records and sharing info before deleting anything (for safe cleanup).
    file_rows: list[dict[str, object]] = []
    file_ids: list[str] = []
    file_ids_to_delete: list[str] = []
    shared_file_ids: list[str] = []
    try:
        file_rows = store.list_files_for_chat(chat_id)
        deleted["files_found"] = len(file_rows)
        file_ids = [r.get("id") for r in file_rows if isinstance(r.get("id"), str) and r.get("id")]
        file_ids = [str(x) for x in file_ids if isinstance(x, str) and x]
        for fid in file_ids:
            chats = store.list_chat_ids_for_file(fid)
            others = [c for c in chats if c != chat_id]
            if others:
                shared_file_ids.append(fid)
            else:
                file_ids_to_delete.append(fid)
    except Exception as exc:
        logger.warning("Failed to list files for chat_id=%s: %s", chat_id, exc)
        file_rows = []
        file_ids = []
        file_ids_to_delete = []
        shared_file_ids = []

    deleted["shared_file_ids"] = shared_file_ids
    deleted["file_ids_deleted"] = file_ids_to_delete

    # 2) Delete UI transcript + per-chat editor state in SQLite FIRST so any reload that
    # happens during KV deletion cannot "resurrect" the chat from transcript rows.
    try:
        deleted["sqlite_messages_deleted"] = store.delete_messages_for_chat(chat_id)
        deleted["sqlite_doc_pages_deleted"] = store.delete_doc_pages_for_chat(chat_id)
        deleted["sqlite_chat_files_deleted"] = store.delete_chat_files(chat_id)
    except Exception as exc:
        logger.warning("Failed to delete SQLite data for chat_id=%s: %s", chat_id, exc)

    # 2b) Delete underlying file data ONLY if unreferenced by any other chat.
    # This preserves shared documents across branched cards.
    deleted["sqlite_file_cleanup"] = []
    if file_ids_to_delete:
        for fid in file_ids_to_delete:
            try:
                deleted["sqlite_file_cleanup"].append(store.delete_file_everywhere(fid))
            except Exception as exc:
                logger.warning("Failed to delete file rows for file_id=%s: %s", fid, exc)

    # 3) Delete KV session (model state + kv snapshots) AFTER SQLite transcript deletion.
    # This is what triggers the kv_sessions watcher event used by the UI.
    try:
        mgr = AppDependencies.session_manager()
        mgr.delete_session(chat_id)
        deleted["kv_session_deleted"] = True
    except Exception as exc:
        logger.warning("Failed to delete KV session for chat_id=%s: %s", chat_id, exc)
        deleted["kv_session_deleted"] = False

    # 4) Delete RAG vectors from Qdrant (insight_chunks) for files that are no longer
    # referenced by any chat. (Never delete by chat_id; documents can be shared.)
    try:
        rag_store = AppDependencies.rag_store()
        retrieval = getattr(rag_store, "retrieval", None)
        deleted["qdrant_files_deleted"] = []
        if retrieval is not None and hasattr(retrieval, "delete_chunks_for_file"):
            for fid in file_ids_to_delete:
                try:
                    retrieval.delete_chunks_for_file(fid)
                    deleted["qdrant_files_deleted"].append(fid)
                except Exception as exc:
                    logger.warning("Failed to delete Qdrant chunks for file_id=%s: %s", fid, exc)
    except Exception as exc:
        logger.warning("Failed to delete Qdrant chunks for deleted files in chat_id=%s: %s", chat_id, exc)

    # 5) Delete LTM entries for this chat (Qdrant insight_memories)
    try:
        ltm = AppDependencies.ltm_store()
        if hasattr(ltm, "delete_chat"):
            ltm.delete_chat(chat_id)
            deleted["ltm_deleted"] = True
    except Exception as exc:
        logger.warning("Failed to delete LTM for chat_id=%s: %s", chat_id, exc)
        deleted["ltm_deleted"] = False

    # 6) Remove encrypted raw uploads from disk (best effort, unreferenced files only)
    try:
        removed = 0
        workspace = AppDependencies.workspace()
        for row in file_rows:
            stored_path = row.get("stored_path")
            file_id = row.get("id")
            if file_id not in file_ids_to_delete:
                continue
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
        focus_document_id = (
            payload.get("focus_document_id")
            or payload.get("focus_file_id")
            or payload.get("active_document_id")
            or payload.get("focus_document")
        )
        selection = payload.get("selection")
        if selection is not None and not isinstance(selection, dict):
            selection = None
        doc_pane_open = payload.get("doc_pane_open")
        if not isinstance(doc_pane_open, bool):
            doc_pane_open = None
        return PlannerRequest(
            chat_id=payload["chat_id"],
            query=payload["query"],
            documents=docs,
            attachments=payload.get("attachments", []) or [],
            focus_document_id=focus_document_id if isinstance(focus_document_id, str) else None,
            doc_pane_open=doc_pane_open,
            selection=selection,
            screenshot=payload.get("screenshot"),
            request_id=payload.get("request_id"),
        )
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"Missing field {exc.args[0]}") from exc
