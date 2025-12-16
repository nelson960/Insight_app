from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict

from fastapi import APIRouter

from backend.api.deps import AppDependencies

router = APIRouter(prefix="/sessions", tags=["Sessions"])


@router.api_route("", methods=["GET", "POST"])
def list_sessions() -> Dict[str, List[Dict[str, str]]]:
    """
    List known chat sessions by scanning persisted KV snapshots.
    """
    workspace = AppDependencies.workspace()
    sessions_dir = Path(workspace.base) / "kv_sessions"
    sessions: list[dict[str, str]] = []
    if sessions_dir.exists():
        seen = set()
        for entry in os.listdir(sessions_dir):
            if entry.endswith(".kv") or entry.endswith(".json") or entry.endswith(".bin"):
                chat_id = Path(entry).stem
                if chat_id in seen:
                    continue
                seen.add(chat_id)
                sessions.append({"chat_id": chat_id})
    return {"sessions": sessions}
