from __future__ import annotations

from fastapi import APIRouter, Query

from backend.api.deps import AppDependencies

router = APIRouter(prefix="/search", tags=["Search"])


@router.get("/file/{file_id}")
def search_file(
    file_id: str,
    q: str = Query(..., min_length=1, description="Literal text to search for"),
    limit: int = Query(200, ge=1, le=1000),
    case_sensitive: bool = Query(False),
    whole_word: bool = Query(False),
):
    svc = AppDependencies.search_service()
    matches = svc.search_file(
        file_id,
        query=q,
        limit=limit,
        case_sensitive=case_sensitive,
        whole_word=whole_word,
    )
    return {
        "file_id": file_id,
        "q": q,
        "limit": limit,
        "case_sensitive": case_sensitive,
        "whole_word": whole_word,
        "matches": [
            {
                "block_index": m.block_index,
                "start": m.start,
                "end": m.end,
                "snippet": m.snippet,
                "page": m.page,
            }
            for m in matches
        ],
    }


@router.get("/doc/{chat_id}/{file_id}")
def search_doc(
    chat_id: str,
    file_id: str,
    q: str = Query(..., min_length=1, description="Literal text to search for"),
    limit: int = Query(200, ge=1, le=1000),
    case_sensitive: bool = Query(False),
    whole_word: bool = Query(False),
):
    store, _ = AppDependencies.storage()
    record = store.get_file(file_id) or {}
    if record.get("chat_id") != chat_id:
        return {
            "chat_id": chat_id,
            "file_id": file_id,
            "q": q,
            "limit": limit,
            "case_sensitive": case_sensitive,
            "whole_word": whole_word,
            "matches": [],
        }

    svc = AppDependencies.doc_search_service()
    matches = svc.search_doc_page(
        chat_id,
        file_id,
        query=q,
        limit=limit,
        case_sensitive=case_sensitive,
        whole_word=whole_word,
    )
    return {
        "chat_id": chat_id,
        "file_id": file_id,
        "q": q,
        "limit": limit,
        "case_sensitive": case_sensitive,
        "whole_word": whole_word,
        "matches": [
            {"from": m.from_pos, "to": m.to_pos, "snippet": m.snippet} for m in matches
        ],
    }


__all__ = ["router"]
