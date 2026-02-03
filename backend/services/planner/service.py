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
            attachments=request.attachments or [],
            focus_document_id=request.focus_document_id,
            doc_pane_open=request.doc_pane_open,
            selection=request.selection,
            skip_user_message=request.skip_user_message,
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
            attachments=request.attachments or [],
            focus_document_id=request.focus_document_id,
            doc_pane_open=request.doc_pane_open,
            selection=request.selection,
            request_id=request.request_id,
            skip_user_message=request.skip_user_message,
            target_assistant_id=request.target_assistant_id,
            user_message_id=request.user_message_id,
        )


__all__ = ["PlannerService"]
