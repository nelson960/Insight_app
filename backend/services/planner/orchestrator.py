from __future__ import annotations

import json
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from backend.services.connectors.llama_session_manager import LlamaSessionManager
from backend.services.retrieval.rag_store import RagStore
from backend.services.memory.ltm_store import LongTermMemoryStore, MemoryHit
from backend.services.storage.sqlite_store import SQLiteMetadataStore

logger = logging.getLogger(__name__)

# Minimal inline-doc budgeting defaults (backend-owned; no config system yet).
INLINE_DOC_FRACTION = 0.30
INLINE_DOC_MAX_TOKENS = 1500


def build_turn_prompt(
    system_hint: str,
    user_message: str,
    ltm_hits: List[MemoryHit],
    doc_texts: List[str],
    *,
    turn_index: int = 0,
) -> str:
    first_turn = turn_index == 0
    mem_block = "\n".join(m.text for m in ltm_hits) if ltm_hits else ""
    docs_block = "\n\n".join(doc_texts) if doc_texts else ""

    # Note: `system_hint` is injected as a system message into the session manager.
    # Do not duplicate it inside the per-turn user prompt.
    parts: List[str] = []
    parts.append(f"User Query:\n{user_message}" if first_turn else f"Follow-up Query:\n{user_message}")
    if mem_block:
        parts.append(f"Long-Term Memory:\n{mem_block}")
    if doc_texts:
        parts.append(f"Documents:\n{docs_block}")
    return "\n\n".join(parts).strip()


class InsightOrchestrator:
    """
    Combines KV sessions, RAG, and long-term memory into a single turn handler.
    """

    def __init__(
        self,
        *,
        session_mgr: LlamaSessionManager,
        rag_store: RagStore,
        ltm_store: LongTermMemoryStore,
        metadata_store: SQLiteMetadataStore,
        system_hint: str = "You are Insight, a local privacy-first AI assistant.",
    ) -> None:
        self.session_mgr = session_mgr
        self.rag_store = rag_store
        self.ltm_store = ltm_store
        self.metadata_store = metadata_store
        self.system_hint = system_hint
        self._last_compaction_tick: Dict[str, int] = {}

    def _approx_token_count(self, text: str) -> int:
        # Cheap estimate: ~4 chars per token on average.
        # Used only for budgeting/truncation, not for reporting.
        if not text:
            return 0
        return max(1, len(text) // 4)

    def _apply_inline_doc_budget(self, chat_id: str, doc_texts: List[str]) -> List[str]:
        if not doc_texts:
            return []

        # Session must exist so context status includes system prompt.
        self.session_mgr.get_or_create_session(chat_id, system_prompt=self.system_hint)
        try:
            status = self.session_mgr.get_context_status(chat_id)
            used = int(status.get("used_tokens") or 0)
            capacity = int(status.get("capacity_tokens") or 0)
        except Exception:
            used = 0
            capacity = int(getattr(self.session_mgr, "ctx_size", 0) or 0)

        remaining = max(0, capacity - used)
        budget = min(int(remaining * INLINE_DOC_FRACTION), INLINE_DOC_MAX_TOKENS)
        if budget <= 0:
            logger.info("Inline docs omitted (no budget) chat=%s used=%d cap=%d", chat_id, used, capacity)
            return []

        approx_doc_tokens = sum(self._approx_token_count(t) for t in doc_texts if t)
        if approx_doc_tokens <= budget:
            return doc_texts
        # IMPORTANT: if the attachment is too large to inline, we do NOT include
        # a partial prefix in the prompt (it’s often irrelevant). We fall back
        # to RAG retrieval instead.
        logger.info(
            "Inline docs skipped (too large) chat=%s approx_doc_tokens=%d budget_tokens=%d used=%d cap=%d",
            chat_id,
            approx_doc_tokens,
            budget,
            used,
            capacity,
        )
        return []

    def _persist_ui_message(
        self,
        *,
        chat_id: str,
        role: str,
        text: str,
        attachments: Optional[List[str]] = None,
        mode: str = "chat",
        model: str = "llama_cpp",
    ) -> None:
        payload: Dict[str, Any] = {"text": text}
        if attachments:
            payload["attachments"] = attachments
        message_id = f"msg_{uuid.uuid4().hex}"
        created_at = datetime.now(timezone.utc).isoformat()
        self.metadata_store.insert_message(
            message_id=message_id,
            chat_id=chat_id,
            role=role,
            content_json=json.dumps(payload, ensure_ascii=False),
            model=model,
            mode=mode,
            planner_payload_json=None,
            citations_json=None,
            created_at=created_at,
        )

    def handle_message(
        self,
        chat_id: str,
        user_message: str,
        documents: Optional[List[str]] = None,
        documents_text: Optional[List[str]] = None,
        attachments: Optional[List[str]] = None,
    ) -> str:
        start = time.perf_counter()
        documents = documents or []
        documents_text = [t for t in (documents_text or []) if isinstance(t, str) and t.strip()]
        attachments = attachments or []
        analysis = self._analyze_query(user_message, has_docs=bool(documents or documents_text))
        # Ensure deterministic session per chat_id (required even when no docs are attached).
        self.session_mgr.get_or_create_session(chat_id, system_prompt=self.system_hint)

        ltm_hits = self.ltm_store.retrieve(chat_id, user_message, top_k=5)
        # If documents are provided, fetch their content; otherwise rely on RAG.
        doc_texts = documents_text or self._fetch_doc_texts(documents)
        doc_texts = self._apply_inline_doc_budget(chat_id, doc_texts)
        if doc_texts:
            rag_hits_raw = []
        elif documents:
            rag_hits_raw = self.rag_store.retrieve(user_message, chat_id=chat_id, doc_ids=documents, top_k=6)
        else:
            rag_hits_raw = self.rag_store.retrieve(user_message, chat_id=chat_id, top_k=6)
        selected_rag = self._dedup_rag(rag_hits_raw)

        # Turn index (exclude system messages)
        session_meta = self.session_mgr.sessions.get(chat_id, {})
        turn_index = len([m for m in session_meta.get("messages", []) if m.get("role") != "system"])

        turn_prompt = build_turn_prompt(
            system_hint=self.system_hint,
            ltm_hits=ltm_hits,
            user_message=user_message,
            doc_texts=doc_texts,
            turn_index=turn_index,
        )
        if not doc_texts and selected_rag:
            rag_block = "\n".join((c.get("text", "") or "") for c in selected_rag)
            turn_prompt += "\n\nContext:\n" + rag_block

        # Persist clean UI user message (not the giant turn_prompt).
        try:
            self._persist_ui_message(chat_id=chat_id, role="user", text=user_message, attachments=attachments)
        except Exception:
            logger.exception("Failed to persist UI user message chat=%s", chat_id)

        # Allow a bit more room for longer answers (cap still applies)
        reply = self.session_mgr.ask(chat_id, turn_prompt, max_tokens=768, temperature=0.2)

        try:
            if reply:
                self._persist_ui_message(chat_id=chat_id, role="assistant", text=reply)
        except Exception:
            logger.exception("Failed to persist UI assistant message chat=%s", chat_id)

        total_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "Planner prompt chat=%s docs=%d rag_hits=%d ltm=%d target_words=%d time_ms=%.1f",
            chat_id,
            len(documents) if documents else len(doc_texts),
            len(selected_rag),
            len(ltm_hits),
            analysis["target_words"],
            total_ms,
        )
        logger.info("Planner prompt payload chat=%s turn=%d:\n%s", chat_id, turn_index, turn_prompt)

        # If a new compaction summary was produced, persist it to LTM and clear marker
        session_meta = self.session_mgr.sessions.get(chat_id, {})
        tick = session_meta.get("compaction_tick", 0)
        if tick and self._last_compaction_tick.get(chat_id) != tick:
            summary_text = session_meta.get("ltm_summary")
            if summary_text:
                self.ltm_store.update_conv_summary(chat_id, summary_text)
            self._last_compaction_tick[chat_id] = tick
        else:
            # If no compaction, update summary lightly with last turn
            self.ltm_store.update_conv_summary(chat_id, f"Last turn: user='{user_message[:200]}', assistant='{reply[:200]}'")
        return reply

    def handle_message_stream(
        self,
        chat_id: str,
        user_message: str,
        documents: Optional[List[str]] = None,
        documents_text: Optional[List[str]] = None,
        attachments: Optional[List[str]] = None,
        *,
        request_id: Optional[str] = None,
    ):
        """
        Streaming variant of handle_message yielding tokens.
        """
        documents = documents or []
        documents_text = [t for t in (documents_text or []) if isinstance(t, str) and t.strip()]
        attachments = attachments or []
        analysis = self._analyze_query(user_message, has_docs=bool(documents or documents_text))
        # Ensure deterministic session per chat_id (required even when no docs are attached).
        self.session_mgr.get_or_create_session(chat_id, system_prompt=self.system_hint)

        ltm_hits = self.ltm_store.retrieve(chat_id, user_message, top_k=5)
        doc_texts = documents_text or self._fetch_doc_texts(documents)
        doc_texts = self._apply_inline_doc_budget(chat_id, doc_texts)
        if doc_texts:
            rag_hits_raw = []
        elif documents:
            rag_hits_raw = self.rag_store.retrieve(user_message, chat_id=chat_id, doc_ids=documents, top_k=6)
        else:
            rag_hits_raw = self.rag_store.retrieve(user_message, chat_id=chat_id, top_k=6)
        selected_rag = self._dedup_rag(rag_hits_raw)

        session_meta = self.session_mgr.sessions.get(chat_id, {})
        turn_index = len([m for m in session_meta.get("messages", []) if m.get("role") != "system"])

        turn_prompt = build_turn_prompt(
            system_hint=self.system_hint,
            ltm_hits=ltm_hits,
            user_message=user_message,
            doc_texts=doc_texts,
            turn_index=turn_index,
        )
        if not doc_texts and selected_rag:
            rag_block = "\n".join((c.get("text", "") or "") for c in selected_rag)
            turn_prompt += "\n\nContext:\n" + rag_block

        def generator():
            tokens: List[str] = []
            # Persist clean UI user message immediately so history is instant.
            try:
                self._persist_ui_message(chat_id=chat_id, role="user", text=user_message, attachments=attachments)
            except Exception:
                logger.exception("Failed to persist UI user message chat=%s", chat_id)

            for token in self.session_mgr.ask_stream(
                chat_id,
                turn_prompt,
                max_tokens=768,
                temperature=0.2,
                request_id=request_id,
            ):
                tokens.append(token)
                yield token

            reply = "".join(tokens)
            try:
                if reply.strip():
                    self._persist_ui_message(chat_id=chat_id, role="assistant", text=reply)
            except Exception:
                logger.exception("Failed to persist UI assistant message chat=%s", chat_id)

            session_meta = self.session_mgr.sessions.get(chat_id, {})
            tick = session_meta.get("compaction_tick", 0)
            if tick and self._last_compaction_tick.get(chat_id) != tick:
                summary_text = session_meta.get("ltm_summary")
                if summary_text:
                    self.ltm_store.update_conv_summary(chat_id, summary_text)
                self._last_compaction_tick[chat_id] = tick
            else:
                self.ltm_store.update_conv_summary(chat_id, f"Last turn: user='{user_message[:200]}', assistant='{reply[:200]}'")

        return generator()

    def _should_use_docs(self, rag_hits: List[Dict[str, Any]], user_hint_doc: bool = False) -> bool:
        if user_hint_doc:
            return True
        if not rag_hits:
            return False
        max_score = max((h.get("score", 0.0) for h in rag_hits), default=0.0)
        return len(rag_hits) >= 2 or max_score >= 0.55

    def _dedup_rag(self, rag_hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        deduped: List[Dict[str, Any]] = []
        for h in rag_hits:
            key = h.get("chunk_id") or h.get("text") or ""
            if key in seen:
                continue
            seen.add(key)
            deduped.append(h)
        return deduped

    def _fetch_doc_texts(self, document_ids: List[str], *, max_chars_per_doc: int = 12000, max_docs: int = 3) -> List[str]:
        texts: List[str] = []
        if not document_ids:
            return texts
        # Stitch using our normalized chunk store (SQLite), ordered by seq.
        try:
            if hasattr(self.metadata_store, "fetch_chunks_for_files"):
                chunks_by_file = self.metadata_store.fetch_chunks_for_files(document_ids[:max_docs], limit_per_file=999)
                for _fid, chunks in chunks_by_file.items():
                    doc_text = "\n".join((c.get("text") or "") for c in chunks)
                    doc_text = doc_text[:max_chars_per_doc]
                    if doc_text:
                        texts.append(doc_text)
        except Exception:
            pass
        return texts

    def _analyze_query(self, query: str, has_docs: bool) -> Dict[str, Any]:
        q = query.lower()
        target_words = 120
        m = re.search(r"(\\d+)\\s*words", q)
        if m:
            try:
                target_words = int(m.group(1))
            except Exception:
                target_words = 120
        elif any(k in q for k in ["short", "brief", "summary"]):
            target_words = 60
        elif any(k in q for k in ["detailed", "long", "full"]):
            target_words = 180
        hint_doc = has_docs or any(k in q for k in ["document", "file", "pdf", "upload"])
        return {
            "target_words": target_words,
            "hint_doc": hint_doc,
        }


__all__ = ["InsightOrchestrator"]
