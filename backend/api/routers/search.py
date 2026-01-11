from __future__ import annotations

from fastapi import APIRouter, Query

from backend.api.deps import AppDependencies

router = APIRouter(prefix="/search", tags=["Search"])


@router.get("/doc/{chat_id}/{file_id}")
def search_doc(
    chat_id: str,
    file_id: str,
    q: str = Query(..., min_length=1, description="Literal text to search for"),
    limit: int = Query(200, ge=1, le=1000),
    case_sensitive: bool = Query(False),
    whole_word: bool = Query(False),
):
    store = AppDependencies.sqlite_store()
    if not store.chat_has_file(chat_id, file_id):
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
