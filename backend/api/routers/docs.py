from __future__ import annotations

import json
from typing import Any, Dict

from fastapi import APIRouter, Body, HTTPException

from backend.api.deps import AppDependencies
from backend.services.docs import blocks_to_prosemirror_doc

router = APIRouter(prefix="/docs", tags=["Docs"])


@router.get("/page/{chat_id}/{file_id}")
def get_doc_page(chat_id: str, file_id: str):
    store = AppDependencies.sqlite_store()

    record = store.get_file(file_id)
    if not record:
        raise HTTPException(status_code=404, detail="file not found")
    if not store.chat_has_file(chat_id, file_id):
        raise HTTPException(status_code=404, detail="file not found for chat")

    existing = store.get_doc_page(chat_id, file_id)
    file_text = store.get_file_text(file_id) or {}
    source_updated_at = file_text.get("updated_at") if isinstance(file_text.get("updated_at"), str) else None
    title = record.get("filename") or ""
    blocks = file_text.get("blocks") if isinstance(file_text.get("blocks"), list) else []

    source_is_stale = False
    if source_updated_at and existing:
        source_is_stale = source_updated_at != existing.get("source_file_updated_at")

    if existing and existing.get("doc") and not source_is_stale:
        return {
            "chat_id": chat_id,
            "file_id": file_id,
            "title": existing.get("title") or title,
            "doc": existing.get("doc"),
            "updated_at": existing.get("updated_at"),
            "source_file_updated_at": existing.get("source_file_updated_at"),
            "source_is_stale": False,
            "is_user_edited": bool(existing.get("is_user_edited")),
            "bootstrapped": False,
        }

    if existing and existing.get("doc") and bool(existing.get("is_user_edited")):
        return {
            "chat_id": chat_id,
            "file_id": file_id,
            "title": existing.get("title") or title,
            "doc": existing.get("doc"),
            "updated_at": existing.get("updated_at"),
            "source_file_updated_at": existing.get("source_file_updated_at"),
            "source_is_stale": source_is_stale,
            "is_user_edited": True,
            "bootstrapped": False,
        }

    policy_raw = record.get("policy_json")
    policy: Dict[str, Any] = {}
    if isinstance(policy_raw, str) and policy_raw.strip():
        try:
            policy = json.loads(policy_raw) or {}
        except Exception:
            policy = {}

    # Raw-large files do not render a cached preview. Return a minimal placeholder.
    if not blocks and bool(policy.get("raw_large")):
        placeholder = blocks_to_prosemirror_doc(
            [
                {
                    "text": "Preview not available for large files. Ask a question to search the content.",
                }
            ]
        )
        return {
            "chat_id": chat_id,
            "file_id": file_id,
            "title": title,
            "doc": placeholder,
            "updated_at": source_updated_at,
            "source_file_updated_at": source_updated_at,
            "source_is_stale": False,
            "is_user_edited": False,
            "bootstrapped": False,
        }

    doc = blocks_to_prosemirror_doc(blocks)
    store.upsert_doc_page(
        chat_id,
        file_id,
        title=title,
        doc=doc,
        is_user_edited=False,
        source_file_updated_at=source_updated_at,
    )
    return {
        "chat_id": chat_id,
        "file_id": file_id,
        "title": title,
        "doc": doc,
        "updated_at": source_updated_at,
        "source_file_updated_at": source_updated_at,
        "source_is_stale": False,
        "is_user_edited": False,
        "bootstrapped": True,
    }


@router.put("/page/{chat_id}/{file_id}")
def save_doc_page(chat_id: str, file_id: str, payload: Dict[str, Any] = Body(...)):
    store = AppDependencies.sqlite_store()

    record = store.get_file(file_id)
    if not record:
        raise HTTPException(status_code=404, detail="file not found")
    if not store.chat_has_file(chat_id, file_id):
        raise HTTPException(status_code=404, detail="file not found for chat")

    title = payload.get("title")
    title = title if isinstance(title, str) else (record.get("filename") or "")

    doc = payload.get("doc")
    if isinstance(doc, str):
        try:
            doc = json.loads(doc)
        except Exception:
            raise HTTPException(status_code=400, detail="doc must be valid JSON")
    if not isinstance(doc, dict):
        raise HTTPException(status_code=400, detail="doc must be a JSON object")

    file_text = store.get_file_text(file_id) or {}
    source_updated_at = file_text.get("updated_at")
    store.upsert_doc_page(
        chat_id,
        file_id,
        title=title,
        doc=doc,
        is_user_edited=True,
        source_file_updated_at=source_updated_at if isinstance(source_updated_at, str) else None,
    )
    return {"ok": True}


__all__ = ["router"]
