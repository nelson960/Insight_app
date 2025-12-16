from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(slots=True)
class PlannerRequest:
    chat_id: str
    query: str
    documents: List[str] = field(default_factory=list)
    documents_text: List[str] = field(default_factory=list)  # optional raw doc texts
    attachments: List[str] = field(default_factory=list)  # UI-only: filenames attached to this turn
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
