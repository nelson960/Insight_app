from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class PlannerRequest:
    chat_id: str
    query: str
    documents: List[str] = field(default_factory=list)
    attachments: List[str] = field(default_factory=list)  # UI-only: filenames attached to this turn
    focus_document_id: Optional[str] = None  # UI hint: currently viewed/active doc
    doc_pane_open: Optional[bool] = None  # UI hint: whether the Documents pane is visible/open
    selection: Optional[Dict[str, Any]] = None  # UI hint: selected excerpt, highest priority
    screenshot: Optional[str] = None
    request_id: Optional[str] = None
    skip_user_message: bool = False
    target_assistant_id: Optional[str] = None
    user_message_id: Optional[str] = None


@dataclass(slots=True)
class PlannerResult:
    answer: str
    citations: List[str]
    raw_text: str
    model: str
    provider: str
    prompt_payload: dict


__all__ = ["PlannerRequest", "PlannerResult"]
