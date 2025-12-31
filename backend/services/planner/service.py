from __future__ import annotations

import logging

from backend.services.planner.orchestrator import InsightOrchestrator
from backend.services.planner.models import PlannerRequest, PlannerResult

logger = logging.getLogger(__name__)


class PlannerService:
    """Planner that orchestrates KV sessions, RAG, and long-term memory (regular mode)."""

    def __init__(self, orchestrator: InsightOrchestrator) -> None:
        self._orchestrator = orchestrator

    def handle_request(self, request: PlannerRequest) -> PlannerResult:
        reply = self._orchestrator.handle_message(
            chat_id=request.chat_id,
            user_message=request.query,
            documents=request.documents or [],
            attachments=getattr(request, "attachments", []) or [],
            focus_document_id=getattr(request, "focus_document_id", None),
            doc_pane_open=getattr(request, "doc_pane_open", None),
            selection=getattr(request, "selection", None),
        )
        return PlannerResult(
            answer=reply,
            citations=[],
            raw_text=reply,
            model="llama_cpp",
            provider="llama_cpp",
            prompt_payload={},
        )

    def stream_request(self, request: PlannerRequest):
        return self._orchestrator.handle_message_stream(
            chat_id=request.chat_id,
            user_message=request.query,
            documents=request.documents or [],
            attachments=getattr(request, "attachments", []) or [],
            focus_document_id=getattr(request, "focus_document_id", None),
            doc_pane_open=getattr(request, "doc_pane_open", None),
            selection=getattr(request, "selection", None),
            request_id=getattr(request, "request_id", None),
        )


__all__ = ["PlannerService"]
