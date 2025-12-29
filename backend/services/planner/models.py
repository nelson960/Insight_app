from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class PlannerRequest:
    chat_id: str
    query: str
    documents: List[str] = field(default_factory=list)
    documents_text: List[str] = field(default_factory=list)  # optional raw doc texts
    attachments: List[str] = field(default_factory=list)  # UI-only: filenames attached to this turn
    focus_document_id: Optional[str] = None  # UI hint: currently viewed/active doc
    doc_scope_mode: Optional[str] = None  # UI hint: "focused" (bias to focus) or "all" (equal priority)
    selection: Optional[Dict[str, Any]] = None  # UI hint: selected excerpt, highest priority
    screenshot: Optional[str] = None
    request_id: Optional[str] = None


@dataclass(slots=True)
class PlannerResult:
    answer: str
    citations: List[str]
    raw_text: str
    model: str
    provider: str
    prompt_payload: dict


__all__ = ["PlannerRequest", "PlannerResult"]
