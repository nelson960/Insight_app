from __future__ import annotations

import json
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from backend.services.connectors.llama_session_manager import LlamaSessionManager
from backend.services.docs import blocks_to_plain_text, prosemirror_doc_to_blocks
from backend.services.retrieval.rag_store import RagStore
from backend.services.memory.ltm_store import LongTermMemoryStore, MemoryHit
from backend.services.storage.sqlite_store import SQLiteMetadataStore

logger = logging.getLogger(__name__)

# Minimal inline-doc budgeting defaults (backend-owned; no config system yet).
INLINE_DOC_FRACTION = 0.30
INLINE_DOC_MAX_TOKENS = 1500
RAG_CONTEXT_FRACTION = 0.30
RAG_CONTEXT_MAX_TOKENS = 1800
# Output budgeting (dynamic; avoids hard-coded long generations).
#
# NOTE: The previous defaults (120 words → ~244 tokens) were frequently too small and
# caused the model to hit max_tokens on normal queries (truncated answers). These
# settings bias toward more complete "chat-sized" answers while still keeping a hard
# ceiling for local compute.
OUTPUT_MIN_TOKENS = 128
OUTPUT_MAX_TOKENS = 1024
OUTPUT_TOKENS_PER_WORD = 2.0
OUTPUT_TOKENS_BUFFER = 96


def build_turn_prompt(
    system_hint: str,
    user_message: str,
    ltm_hits: List[MemoryHit],
    doc_texts: List[str],
    *,
    selection: Optional[Dict[str, Any]] = None,	
    turn_index: int = 0,
) -> str:
    first_turn = turn_index == 0
    mem_block = "\n".join(m.text for m in ltm_hits) if ltm_hits else ""
    docs_block = "\n\n".join(doc_texts) if doc_texts else ""

    # Note: `system_hint` is injected as a system message into the session manager.
    # Do not duplicate it inside the per-turn user prompt.
    parts: List[str] = []
    if selection and isinstance(selection, dict):
        sel_text = selection.get("text")
        if isinstance(sel_text, str) and sel_text.strip():
            sel_file = selection.get("file_id") if isinstance(selection.get("file_id"), str) else None
            sel_page = selection.get("page") if isinstance(selection.get("page"), int) else None
            header = "Selected Excerpt (highest priority):"
            if sel_file and sel_page:
                header = f"Selected Excerpt (highest priority) from {sel_file} page {sel_page}:"
            elif sel_file:
                header = f"Selected Excerpt (highest priority) from {sel_file}:"
            elif sel_page:
                header = f"Selected Excerpt (highest priority) page {sel_page}:"
            # Prevent extremely large selections from polluting context.
            sel_trimmed = sel_text.strip()
            if len(sel_trimmed) > 5000:
                sel_trimmed = sel_trimmed[:5000] + "…"
            parts.append(f"{header}\n{sel_trimmed}")
    parts.append(f"User Query:\n{user_message}" if first_turn else f"Follow-up Query:\n{user_message}")
    if mem_block:
        parts.append(f"Long-Term Memory:\n{mem_block}")
    if doc_texts:
        parts.append(f"Documents:\n{docs_block}")
    return "\n\n".join(parts).strip()


def build_context_pack(
    *,
    ltm_hits: List[MemoryHit],
    doc_texts: List[str],
    rag_hits: List[Dict[str, Any]],
    selection: Optional[Dict[str, Any]] = None,
    effective_focus: Optional[str] = None,
    include_selection_excerpt: bool = True,
) -> str:
    """
    Build an ephemeral context pack for a single generation.

    This text is injected for the current model call only and MUST NOT be
    persisted into session messages/KV. Keeping it separate prevents retrieval
    and document blobs from "sticking" into the KV cache across turns.
    """
    parts: List[str] = []

    if selection and isinstance(selection, dict) and isinstance(selection.get("file_id"), str):
        parts.append(f"SCOPE:\n- mode: selection\n- file_id: {selection.get('file_id')}")
    elif effective_focus:
        parts.append(f"SCOPE:\n- mode: focused_document\n- file_id: {effective_focus}")

    # NOTE: Selected excerpts can be injected as part of the *dirty user turn* so they
    # can't be diluted by large context packs. In that mode, keep the excerpt out of
    # this pack to avoid duplication and wasted budget.
    if include_selection_excerpt and selection and isinstance(selection, dict):
        sel_text = selection.get("text")
        if isinstance(sel_text, str) and sel_text.strip():
            sel_trimmed = sel_text.strip()
            if len(sel_trimmed) > 5000:
                sel_trimmed = sel_trimmed[:5000] + "…"
            header = "SELECTED EXCERPT (highest priority):"
            sel_file = selection.get("file_id") if isinstance(selection.get("file_id"), str) else None
            sel_page = selection.get("page") if isinstance(selection.get("page"), int) else None
            if sel_file and sel_page is not None:
                header = f"SELECTED EXCERPT (highest priority) from {sel_file} page {sel_page}:"
            elif sel_file:
                header = f"SELECTED EXCERPT (highest priority) from {sel_file}:"
            parts.append(f"{header}\n{sel_trimmed}")

    if ltm_hits:
        mem_block = "\n".join(m.text for m in ltm_hits if getattr(m, "text", None)) or ""
        if mem_block.strip():
            parts.append(f"LONG-TERM MEMORY (relevant):\n{mem_block.strip()}")

    if doc_texts:
        docs_block = "\n\n".join(t for t in doc_texts if isinstance(t, str) and t.strip())
        if docs_block.strip():
            parts.append(f"INLINE DOCUMENT TEXT (truncated):\n{docs_block.strip()}")

    if rag_hits:
        lines: List[str] = []
        for i, hit in enumerate(rag_hits, start=1):
            text = (hit.get("text") or "").strip()
            if not text:
                continue
            doc_id = hit.get("doc_id") or ""
            chunk_id = hit.get("chunk_id") or ""
            score = hit.get("score") or 0.0
            lines.append(f"[E{i}] doc_id={doc_id} chunk_id={chunk_id} score={score}")
            lines.append(text)
            lines.append("")
        evidence = "\n".join(lines).strip()
        if evidence:
            parts.append(f"EVIDENCE (RAG chunks):\n{evidence}")

    return "\n\n".join(p.strip() for p in parts if p and p.strip()).strip()


def build_dirty_user_turn(user_message: str, *, selection: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Build a "dirty" user turn that strongly weights a selected excerpt.

    This string is used for the current model run ONLY and is NOT persisted into
    the clean session transcript/KV.
    """
    if not isinstance(selection, dict):
        return None
    sel_text = selection.get("text")
    if not isinstance(sel_text, str) or not sel_text.strip():
        return None

    sel_trimmed = sel_text.strip()
    if len(sel_trimmed) > 5000:
        sel_trimmed = sel_trimmed[:5000] + "…"

    sel_file = selection.get("file_id") if isinstance(selection.get("file_id"), str) else None
    sel_page = selection.get("page") if isinstance(selection.get("page"), int) else None
    if sel_file and sel_page is not None:
        header = f"SELECTED EXCERPT (highest priority) from {sel_file} page {sel_page}:"
    elif sel_file:
        header = f"SELECTED EXCERPT (highest priority) from {sel_file}:"
    else:
        header = "SELECTED EXCERPT (highest priority):"

    # Keep this compact and directive. We want the model to anchor on the excerpt,
    # but we still want the rest of the ephemeral context pack (RAG/LTM) available.
    return "\n\n".join(
        [
            f"{header}\n{sel_trimmed}",
            "QUESTION:\n" + (user_message or "").strip(),
        ]
    ).strip()


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

    def _compute_output_max_tokens(self, analysis: Dict[str, Any]) -> int:
        """
        Dynamic output budget based on the user's requested/implicit target length.

        This controls the model's *maximum* generated tokens. The model may stop earlier.
        """
        try:
            target_words = int(analysis.get("target_words") or 120)
        except Exception:
            target_words = 120
        target_words = max(20, min(1200, target_words))
        est = int(target_words * OUTPUT_TOKENS_PER_WORD) + int(OUTPUT_TOKENS_BUFFER)
        return max(int(OUTPUT_MIN_TOKENS), min(int(OUTPUT_MAX_TOKENS), int(est)))

    @staticmethod
    def _is_summary_request(user_message: str) -> bool:
        q = (user_message or "").strip().lower()
        if not q:
            return False
        # Handle short "mode" prompts used in testing ("brief", "summary", etc.).
        if q in {"brief", "summary", "summarize", "tldr", "tl;dr"}:
            return True
        return any(k in q for k in ("summarize", "summary", "tl;dr", "tldr"))

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

    def _apply_inline_doc_budget_partial(self, chat_id: str, text: str) -> str:
        """
        Like `_apply_inline_doc_budget`, but returns a truncated prefix instead of dropping the doc.

        This is used for summary-style queries where vector search is not meaningful
        (e.g. user asks just "brief"), so we must provide some sequential doc text.
        """
        if not text:
            return ""

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
            return ""
        budget_chars = max(0, budget * 4)
        if len(text) <= budget_chars:
            return text
        return text[:budget_chars]

    def _resolve_focus_for_turn(
        self,
        chat_id: str,
        *,
        documents: List[str],
        focus_document_id: Optional[str],
        selection: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        """
        Resolve the "effective" focused file for this turn.

        Priority:
          1) explicit selection.file_id
          2) explicit focus_document_id from UI
          3) last document id in the request payload (upload/attach turn)
          4) last file registered for this chat (SQLite)
        """
        sel_file_id = (
            selection.get("file_id")
            if isinstance(selection, dict) and isinstance(selection.get("file_id"), str)
            else None
        )
        if isinstance(sel_file_id, str) and sel_file_id:
            return sel_file_id

        if isinstance(focus_document_id, str) and focus_document_id:
            return focus_document_id

        # If the request included documents (e.g. upload+chat), default to the most recent
        # doc in that list so "upload → ask" works without extra UI wiring.
        for fid in reversed(documents or []):
            if isinstance(fid, str) and fid:
                return fid

        # Fallback: when the frontend doesn't send focus_document_id, use the most recently
        # registered file for this chat so we don't accidentally keep answering from the
        # first uploaded doc forever.
        try:
            if hasattr(self.metadata_store, "latest_file_id_for_chat"):
                fid = self.metadata_store.latest_file_id_for_chat(chat_id)  # type: ignore[attr-defined]
                if isinstance(fid, str) and fid:
                    return fid
        except Exception:
            return None

        return None

    def _is_user_edited_doc(self, chat_id: str, file_id: str) -> bool:
        if not isinstance(chat_id, str) or not chat_id:
            return False
        if not isinstance(file_id, str) or not file_id:
            return False
        try:
            row = self.metadata_store.get_doc_page(chat_id, file_id)
            return bool(row.get("is_user_edited")) if row else False
        except Exception:
            return False

    def _fetch_focus_doc_text(self, chat_id: str, file_id: str) -> str:
        """
        Best-effort: get canonical extracted text for a single file.
        """
        if not isinstance(file_id, str) or not file_id:
            return ""
        # If the user edited the doc page, treat the editor content as canonical for this file.
        try:
            doc_page = self.metadata_store.get_doc_page(chat_id, file_id)
            if doc_page and bool(doc_page.get("is_user_edited")):
                doc = doc_page.get("doc")
                if isinstance(doc, dict):
                    blocks = prosemirror_doc_to_blocks(doc)
                    text = blocks_to_plain_text(blocks)
                    if isinstance(text, str) and text.strip():
                        return text
        except Exception:
            pass
        try:
            row = self.metadata_store.get_file_text(file_id)
            if row and isinstance(row.get("plain_text"), str):
                return row.get("plain_text") or ""
        except Exception:
            pass
        # Fallback: stitch chunk texts if file_text isn't available.
        try:
            if hasattr(self.metadata_store, "fetch_chunk_texts_for_file"):
                texts = self.metadata_store.fetch_chunk_texts_for_file(file_id, limit=None)
                return "\n\n".join(t for t in texts if isinstance(t, str) and t.strip())
        except Exception:
            pass
        return ""

    def _edited_doc_evidence(self, chat_id: str, file_id: str, query: str, *, max_windows: int = 6) -> List[Dict[str, Any]]:
        """
        Retrieve "chunk-like" evidence from an edited doc page without embeddings.

        We score blocks by literal overlap with the user query, then return small block windows
        (block + neighbors) in document order. This avoids mixing stale Qdrant chunks with the
        user-edited doc content.
        """
        if not isinstance(chat_id, str) or not chat_id:
            return []
        if not isinstance(file_id, str) or not file_id:
            return []
        q = (query or "").strip()
        if not q:
            return []
        try:
            doc_page = self.metadata_store.get_doc_page(chat_id, file_id)
        except Exception:
            doc_page = None
        if not doc_page or not bool(doc_page.get("is_user_edited")):
            return []
        doc = doc_page.get("doc")
        if not isinstance(doc, dict):
            return []

        blocks_raw = prosemirror_doc_to_blocks(doc)
        blocks: List[Dict[str, Any]] = []
        for b in blocks_raw:
            if not isinstance(b, dict):
                continue
            text = b.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            blocks.append(b)
        if not blocks:
            return []

        phrase = q.casefold()
        # Keep this cheap: use only word-like tokens of length >= 3.
        terms = [t for t in re.findall(r"[a-z0-9_]{3,}", phrase) if t]
        terms = terms[:12]

        scored: List[tuple[int, float]] = []
        for i, b in enumerate(blocks):
            text = (b.get("text") or "")
            text_ci = text.casefold()
            score = 0.0
            if phrase and phrase in text_ci:
                score += 10.0
            for t in terms:
                if t in text_ci:
                    # Cheap frequency-ish score.
                    score += float(text_ci.count(t))
            if score > 0:
                scored.append((i, score))
        if not scored:
            return []

        scored.sort(key=lambda x: (-x[1], x[0]))
        picked: List[int] = []
        for idx, _score in scored:
            # Avoid picking blocks that are immediately adjacent to already picked ones;
            # we merge them via windows anyway.
            if any(abs(idx - p) <= 1 for p in picked):
                continue
            picked.append(idx)
            if len(picked) >= max_windows:
                break

        if not picked:
            return []

        # Build windows (block + neighbors) and merge overlaps.
        windows: List[tuple[int, int]] = []
        for idx in sorted(picked):
            start = max(0, idx - 1)
            end = min(len(blocks) - 1, idx + 1)
            windows.append((start, end))

        merged: List[tuple[int, int]] = []
        for start, end in sorted(windows):
            if not merged:
                merged.append((start, end))
                continue
            last_s, last_e = merged[-1]
            if start <= last_e + 1:
                merged[-1] = (last_s, max(last_e, end))
            else:
                merged.append((start, end))

        evidence: List[Dict[str, Any]] = []
        for start, end in merged:
            lines: List[str] = []
            for b in blocks[start : end + 1]:
                kind = str(b.get("kind") or "paragraph")
                text = str(b.get("text") or "").strip()
                meta = b.get("metadata") if isinstance(b.get("metadata"), dict) else {}
                if kind == "heading":
                    level = meta.get("level", 2)
                    try:
                        level = int(level)
                    except Exception:
                        level = 2
                    level = max(1, min(level, 6))
                    lines.append(f"{'#' * level} {text}")
                elif kind == "list" and isinstance(meta.get("items"), list):
                    items = [str(x).strip() for x in meta.get("items") if str(x).strip()]
                    if items:
                        lines.extend([f"- {it}" for it in items])
                    else:
                        lines.append(text)
                elif kind == "code":
                    lines.append("```")
                    lines.append(text)
                    lines.append("```")
                else:
                    lines.append(text)
                lines.append("")
            excerpt = "\n".join(lines).strip()
            if not excerpt:
                continue
            evidence.append(
                {
                    "doc_id": file_id,
                    "chunk_id": f"doc_page:{start}-{end}",
                    "score": 1.0,
                    "text": excerpt,
                }
            )

        return evidence

    def _doc_fallback_preview(self, chat_id: str, text: str, *, max_chars: int = 4000) -> str:
        """
        Small, safe excerpt used when focused RAG returns no hits.

        Keeps the UI/chat responsive for queries like "brief" where embeddings may not
        retrieve anything meaningful, while still avoiding prompt pollution.
        """
        if not text:
            return ""
        trimmed = self._apply_inline_doc_budget_partial(chat_id, text)
        if max_chars > 0 and len(trimmed) > max_chars:
            return trimmed[:max_chars]
        return trimmed

    def _list_chat_file_ids(self, chat_id: str) -> List[str]:
        try:
            rows = self.metadata_store.list_files_for_chat(chat_id)
        except Exception:
            return []
        out: List[str] = []
        for r in rows:
            fid = r.get("id")
            if isinstance(fid, str) and fid:
                out.append(fid)
        return out

    def _apply_rag_budget(self, chat_id: str, rag_hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not rag_hits:
            return []

        # Use the same context status mechanism as inline docs budgeting.
        self.session_mgr.get_or_create_session(chat_id, system_prompt=self.system_hint)
        try:
            status = self.session_mgr.get_context_status(chat_id)
            used = int(status.get("used_tokens") or 0)
            capacity = int(status.get("capacity_tokens") or 0)
        except Exception:
            used = 0
            capacity = int(getattr(self.session_mgr, "ctx_size", 0) or 0)

        remaining = max(0, capacity - used)
        budget_tokens = min(int(remaining * RAG_CONTEXT_FRACTION), RAG_CONTEXT_MAX_TOKENS)
        if budget_tokens <= 0:
            return []

        # Approximate chars budget (~4 chars/token).
        budget_chars = max(0, budget_tokens * 4)
        remaining_chars = budget_chars

        trimmed: List[Dict[str, Any]] = []
        for hit in rag_hits:
            if remaining_chars <= 0:
                break
            text = (hit.get("text") or "")
            if not isinstance(text, str) or not text.strip():
                continue
            if len(text) > remaining_chars:
                text = text[:remaining_chars]
            trimmed.append({**hit, "text": text})
            remaining_chars -= len(text)

        if len(trimmed) != len(rag_hits):
            logger.info(
                "RAG context trimmed chat=%s hits_in=%d hits_out=%d budget_tokens=%d",
                chat_id,
                len(rag_hits),
                len(trimmed),
                budget_tokens,
            )
        return trimmed

    def _persist_ui_message(
        self,
        *,
        chat_id: str,
        role: str,
        text: str,
        attachments: Optional[List[str]] = None,
        focus_document_id: Optional[str] = None,
        selection: Optional[Dict[str, Any]] = None,
        mode: str = "chat",
        model: str = "llama_cpp",
    ) -> None:
        payload: Dict[str, Any] = {"text": text}
        if attachments:
            payload["attachments"] = attachments
        if isinstance(focus_document_id, str) and focus_document_id:
            payload["focus_document_id"] = focus_document_id
        if isinstance(selection, dict) and selection.get("text"):
            # Store a lightweight selection object for UI rehydration/debugging.
            sel: Dict[str, Any] = {"text": str(selection.get("text"))}
            if isinstance(selection.get("file_id"), str):
                sel["file_id"] = selection.get("file_id")
            if isinstance(selection.get("page"), int):
                sel["page"] = selection.get("page")
            payload["selection"] = sel
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
        focus_document_id: Optional[str] = None,
        selection: Optional[Dict[str, Any]] = None,
    ) -> str:
        start = time.perf_counter()
        documents = documents or []
        documents_text = [t for t in (documents_text or []) if isinstance(t, str) and t.strip()]
        attachments = attachments or []
        analysis = self._analyze_query(user_message, has_docs=bool(documents or documents_text))
        max_tokens = self._compute_output_max_tokens(analysis)
        summary_request = self._is_summary_request(user_message)
        # Ensure deterministic session per chat_id (required even when no docs are attached).
        self.session_mgr.get_or_create_session(chat_id, system_prompt=self.system_hint)

        ltm_k = 2 if selection else 5
        ltm_hits = self.ltm_store.retrieve(chat_id, user_message, top_k=ltm_k)
        sel_file_id = selection.get("file_id") if isinstance(selection, dict) and isinstance(selection.get("file_id"), str) else None
        effective_focus = self._resolve_focus_for_turn(
            chat_id,
            documents=documents,
            focus_document_id=focus_document_id,
            selection=selection,
        )
        logger.debug(
            "Planner turn chat=%s summary=%s focus=%s docs_payload=%d docs_text=%d attachments=%d",
            chat_id,
            summary_request,
            effective_focus,
            len(documents),
            len(documents_text),
            len(attachments),
        )

        focus_user_edited = bool(effective_focus and self._is_user_edited_doc(chat_id, effective_focus))

        # If documents are provided, fetch their content; otherwise rely on RAG.
        # IMPORTANT: if the focused file is user-edited, do not inline cached chunk text
        # (it may not match the edited doc page). We’ll pull evidence from doc_pages instead.
        if focus_user_edited:
            doc_texts = []
        elif documents_text:
            doc_texts = documents_text
        elif effective_focus:
            doc_texts = self._fetch_doc_texts([effective_focus])
        else:
            doc_texts = self._fetch_doc_texts(documents)
        doc_texts = self._apply_inline_doc_budget(chat_id, doc_texts)

        # Summary-style queries (e.g. "brief") should not rely on vector search;
        # they need sequential doc text for the focused document.
        if summary_request and effective_focus:
            focus_text = self._fetch_focus_doc_text(chat_id, effective_focus) if focus_user_edited else ""
            if documents and documents_text and len(documents_text) == len(documents):
                try:
                    idx = documents.index(effective_focus)
                    focus_text = documents_text[idx] if idx < len(documents_text) else ""
                except ValueError:
                    focus_text = ""
            if not focus_text:
                focus_text = self._fetch_focus_doc_text(chat_id, effective_focus)
            focus_text = self._apply_inline_doc_budget_partial(chat_id, focus_text)
            doc_texts = [focus_text] if focus_text.strip() else []
            rag_hits_raw = []
        elif doc_texts:
            rag_hits_raw = []
        else:
            primary_k = 4 if selection else 6
            rag_hits_raw = []
            if effective_focus:
                if self._is_user_edited_doc(chat_id, effective_focus):
                    rag_hits_raw = self._edited_doc_evidence(chat_id, effective_focus, user_message, max_windows=primary_k)
                    logger.info("Focused doc is user-edited; skipping Qdrant for file_id=%s chat=%s", effective_focus, chat_id)
                else:
                    rag_hits_raw.extend(
                        self.rag_store.retrieve(
                            user_message,
                            chat_id=chat_id,
                            doc_ids=[effective_focus],
                            top_k=primary_k,
                        )
                    )
            elif documents:
                rag_hits_raw = self.rag_store.retrieve(user_message, chat_id=chat_id, doc_ids=documents, top_k=primary_k)
            else:
                rag_hits_raw = self.rag_store.retrieve(user_message, chat_id=chat_id, top_k=primary_k)
        selected_rag = self._dedup_rag(rag_hits_raw)
        selected_rag = self._apply_rag_budget(chat_id, selected_rag)
        if effective_focus and not doc_texts and not selected_rag:
            # Focused retrieval yielded nothing; fall back to a small sequential excerpt.
            preview = self._doc_fallback_preview(chat_id, self._fetch_focus_doc_text(chat_id, effective_focus))
            if preview.strip():
                doc_texts = [preview]
        has_selection_text = bool(
            isinstance(selection, dict)
            and isinstance(selection.get("text"), str)
            and str(selection.get("text")).strip()
        )
        context_pack = build_context_pack(
            ltm_hits=ltm_hits,
            doc_texts=doc_texts,
            rag_hits=selected_rag,
            selection=selection,
            effective_focus=effective_focus,
            include_selection_excerpt=not has_selection_text,
        )
        if summary_request and effective_focus:
            context_pack = (
                "TASK:\n- Summarize the focused document briefly.\n"
                "- Use only the provided document text/evidence.\n\n"
                + (context_pack or "")
            ).strip()

        # Persist clean UI user message (not the giant turn_prompt).
        try:
            self._persist_ui_message(
                chat_id=chat_id,
                role="user",
                text=user_message,
                attachments=attachments,
                focus_document_id=effective_focus,
                selection=selection,
            )
        except Exception:
            logger.exception("Failed to persist UI user message chat=%s", chat_id)

        # Allow a bit more room for longer answers (cap still applies)
        dirty_user = build_dirty_user_turn(user_message, selection=selection) if has_selection_text else None
        reply = self.session_mgr.ask_with_context(
            chat_id,
            user_text=user_message,
            run_user_text=dirty_user,
            context_pack=context_pack,
            max_tokens=max_tokens,
            temperature=0.2,
        )

        try:
            if reply:
                self._persist_ui_message(chat_id=chat_id, role="assistant", text=reply)
        except Exception:
            logger.exception("Failed to persist UI assistant message chat=%s", chat_id)

        total_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "Planner prompt chat=%s focus=%s summary=%s docs=%d rag_hits=%d ltm=%d target_words=%d time_ms=%.1f",
            chat_id,
            effective_focus,
            summary_request,
            len(documents) if documents else len(doc_texts),
            len(selected_rag),
            len(ltm_hits),
            analysis["target_words"],
            total_ms,
        )

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
        focus_document_id: Optional[str] = None,
        selection: Optional[Dict[str, Any]] = None,
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
        max_tokens = self._compute_output_max_tokens(analysis)
        summary_request = self._is_summary_request(user_message)
        # Ensure deterministic session per chat_id (required even when no docs are attached).
        self.session_mgr.get_or_create_session(chat_id, system_prompt=self.system_hint)

        ltm_k = 2 if selection else 5
        ltm_hits = self.ltm_store.retrieve(chat_id, user_message, top_k=ltm_k)
        effective_focus = self._resolve_focus_for_turn(
            chat_id,
            documents=documents,
            focus_document_id=focus_document_id,
            selection=selection,
        )
        logger.debug(
            "Planner stream turn chat=%s summary=%s focus=%s docs_payload=%d docs_text=%d attachments=%d request_id=%s",
            chat_id,
            summary_request,
            effective_focus,
            len(documents),
            len(documents_text),
            len(attachments),
            request_id,
        )

        focus_user_edited = bool(effective_focus and self._is_user_edited_doc(chat_id, effective_focus))

        if focus_user_edited:
            doc_texts = []
        elif documents_text:
            doc_texts = documents_text
        elif effective_focus:
            doc_texts = self._fetch_doc_texts([effective_focus])
        else:
            doc_texts = self._fetch_doc_texts(documents)
        doc_texts = self._apply_inline_doc_budget(chat_id, doc_texts)

        if summary_request and effective_focus:
            focus_text = self._fetch_focus_doc_text(chat_id, effective_focus) if focus_user_edited else ""
            if documents and documents_text and len(documents_text) == len(documents):
                try:
                    idx = documents.index(effective_focus)
                    focus_text = documents_text[idx] if idx < len(documents_text) else ""
                except ValueError:
                    focus_text = ""
            if not focus_text:
                focus_text = self._fetch_focus_doc_text(chat_id, effective_focus)
            focus_text = self._apply_inline_doc_budget_partial(chat_id, focus_text)
            doc_texts = [focus_text] if focus_text.strip() else []
            rag_hits_raw = []
        elif doc_texts:
            rag_hits_raw = []
        else:
            primary_k = 4 if selection else 6
            rag_hits_raw = []
            if effective_focus:
                if self._is_user_edited_doc(chat_id, effective_focus):
                    rag_hits_raw = self._edited_doc_evidence(chat_id, effective_focus, user_message, max_windows=primary_k)
                    logger.info("Focused doc is user-edited; skipping Qdrant for file_id=%s chat=%s", effective_focus, chat_id)
                else:
                    rag_hits_raw.extend(
                        self.rag_store.retrieve(
                            user_message,
                            chat_id=chat_id,
                            doc_ids=[effective_focus],
                            top_k=primary_k,
                        )
                    )
            elif documents:
                rag_hits_raw = self.rag_store.retrieve(user_message, chat_id=chat_id, doc_ids=documents, top_k=primary_k)
            else:
                rag_hits_raw = self.rag_store.retrieve(user_message, chat_id=chat_id, top_k=primary_k)
        selected_rag = self._dedup_rag(rag_hits_raw)
        selected_rag = self._apply_rag_budget(chat_id, selected_rag)
        if effective_focus and not doc_texts and not selected_rag:
            preview = self._doc_fallback_preview(chat_id, self._fetch_focus_doc_text(chat_id, effective_focus))
            if preview.strip():
                doc_texts = [preview]
        has_selection_text = bool(
            isinstance(selection, dict)
            and isinstance(selection.get("text"), str)
            and str(selection.get("text")).strip()
        )
        context_pack = build_context_pack(
            ltm_hits=ltm_hits,
            doc_texts=doc_texts,
            rag_hits=selected_rag,
            selection=selection,
            effective_focus=effective_focus,
            include_selection_excerpt=not has_selection_text,
        )
        if summary_request and effective_focus:
            context_pack = (
                "TASK:\n- Summarize the focused document briefly.\n"
                "- Use only the provided document text/evidence.\n\n"
                + (context_pack or "")
            ).strip()

        def generator():
            tokens: List[str] = []
            # Persist clean UI user message immediately so history is instant.
            try:
                self._persist_ui_message(
                    chat_id=chat_id,
                    role="user",
                    text=user_message,
                    attachments=attachments,
                    focus_document_id=effective_focus,
                    selection=selection,
                )
            except Exception:
                logger.exception("Failed to persist UI user message chat=%s", chat_id)

            dirty_user = build_dirty_user_turn(user_message, selection=selection) if has_selection_text else None
            for token in self.session_mgr.ask_stream_with_context(
                chat_id,
                user_text=user_message,
                run_user_text=dirty_user,
                context_pack=context_pack,
                max_tokens=max_tokens,
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
        # Default target length for a "normal" chat answer.
        target_words = 180
        m = re.search(r"(\\d+)\\s*words", q)
        if m:
            try:
                target_words = int(m.group(1))
            except Exception:
                target_words = 180
        # Prefer "detailed/long" over "brief/summary" if both appear (e.g. "detailed summary").
        elif any(k in q for k in ["detailed", "long", "full"]):
            target_words = 260
        elif any(k in q for k in ["short", "brief", "summary"]):
            target_words = 90
        hint_doc = has_docs or any(k in q for k in ["document", "file", "pdf", "upload"])
        return {
            "target_words": target_words,
            "hint_doc": hint_doc,
        }


__all__ = ["InsightOrchestrator"]
