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
from backend.services.ipc_events import emit_event
from backend.services.storage.sqlite_store import SQLiteMetadataStore

logger = logging.getLogger(__name__)

# Minimal inline-doc budgeting defaults (backend-owned; no config system yet).
INLINE_DOC_FRACTION = 0.30
INLINE_DOC_MAX_TOKENS = 1500
RAG_CONTEXT_FRACTION = 0.30
RAG_CONTEXT_MAX_TOKENS = 2400
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
            sel_file = selection.get("filename") if isinstance(selection.get("filename"), str) else None
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
    effective_focus_name: Optional[str] = None,
    scope_files: Optional[List[str]] = None,
    include_selection_excerpt: bool = True,
) -> str:
    """
    Build an ephemeral context pack for a single generation.

    This text is injected for the current model call only and MUST NOT be
    persisted into session messages/KV. Keeping it separate prevents retrieval
    and document blobs from "sticking" into the KV cache across turns.
    """
    parts: List[str] = []

    if selection and isinstance(selection, dict) and isinstance(selection.get("text"), str) and selection.get("text"):
        # Never include internal IDs in model-visible text.
        sel_name = selection.get("filename") if isinstance(selection.get("filename"), str) else None
        if sel_name:
            parts.append(f"SCOPE:\n- mode: selection\n- file: {sel_name}")
        else:
            parts.append("SCOPE:\n- mode: selection")
    else:
        # Optional override: allow the caller to define an explicit "document scope"
        # using safe, user-facing filenames (never internal IDs). This is useful for:
        # - doc_scope_mode="all" turns (compare/multi-doc)
        # - multi-upload compare turns where UI focus would otherwise "hide" scope
        names_in_scope: List[str] = []
        if isinstance(scope_files, list) and scope_files:
            for raw in scope_files:
                if not isinstance(raw, str):
                    continue
                s = raw.strip()
                if not s:
                    continue
                names_in_scope.append(s)
                if len(names_in_scope) >= 12:
                    break

        if names_in_scope:
            # Disambiguate duplicates without exposing internal ids.
            counts: Dict[str, int] = {}
            for n in names_in_scope:
                counts[n] = counts.get(n, 0) + 1
            if any(v > 1 for v in counts.values()):
                seen: Dict[str, int] = {}
                disambiguated: List[str] = []
                for n in names_in_scope:
                    if counts.get(n, 0) <= 1:
                        disambiguated.append(n)
                        continue
                    seen[n] = seen.get(n, 0) + 1
                    disambiguated.append(f"{n} ({seen[n]})")
                names_in_scope = disambiguated

            if len(names_in_scope) >= 2:
                parts.append("SCOPE:\n- mode: all_documents\n- files: " + ", ".join(names_in_scope[:6]))
            else:
                parts.append("SCOPE:\n- mode: document\n- file: " + names_in_scope[0])
        elif effective_focus_name:
            parts.append(f"SCOPE:\n- mode: focused_document\n- file: {effective_focus_name}")
        elif effective_focus:
            # Focus exists but we don't have a safe filename; keep the scope without IDs.
            parts.append("SCOPE:\n- mode: focused_document")
        else:
            # No explicit focus: if evidence spans multiple documents, tell the model.
            # Use filenames only (never internal IDs).
            names: List[str] = []
            seen = set()
            for h in rag_hits or []:
                fn = h.get("filename")
                if isinstance(fn, str):
                    fn = fn.strip()
                if not fn:
                    continue
                if fn in seen:
                    continue
                seen.add(fn)
                names.append(fn)
                if len(names) >= 6:
                    break
            if len(names) >= 2:
                parts.append("SCOPE:\n- mode: all_documents\n- files: " + ", ".join(names))
            elif len(names) == 1:
                parts.append("SCOPE:\n- mode: document\n- file: " + names[0])

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
            sel_file = selection.get("filename") if isinstance(selection.get("filename"), str) else None
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
        # Group evidence by *doc_id* to avoid mixing when filenames collide or are missing.
        # Never include internal identifiers (file_id/chunk_id/score) in model-visible text.
        buckets: Dict[str, List[Dict[str, Any]]] = {}
        order: List[str] = []
        name_by_doc: Dict[str, str] = {}
        unknown_counter = 0

        for hit in rag_hits:
            text = (hit.get("text") or "").strip()
            if not text:
                continue

            doc_id = hit.get("doc_id")
            if not isinstance(doc_id, str) or not doc_id:
                unknown_counter += 1
                doc_id = f"unknown_{unknown_counter}"

            filename = hit.get("filename")
            if isinstance(filename, str):
                filename = filename.strip()
            if not filename:
                filename = "Document"

            if doc_id not in buckets:
                buckets[doc_id] = []
                order.append(doc_id)
                name_by_doc[doc_id] = filename
            else:
                # Prefer a real filename over a placeholder.
                if name_by_doc.get(doc_id) in {"", "Document"} and filename not in {"", "Document"}:
                    name_by_doc[doc_id] = filename

            buckets[doc_id].append(hit)

        # Disambiguate duplicate filenames without exposing internal ids.
        display_name: Dict[str, str] = {}
        counts: Dict[str, int] = {}
        for doc_id in order:
            base = name_by_doc.get(doc_id) or "Document"
            counts[base] = counts.get(base, 0) + 1
        seen: Dict[str, int] = {}
        for doc_id in order:
            base = name_by_doc.get(doc_id) or "Document"
            if counts.get(base, 0) <= 1:
                display_name[doc_id] = base
                continue
            seen[base] = seen.get(base, 0) + 1
            display_name[doc_id] = f"{base} ({seen[base]})"

        lines: List[str] = []
        for doc_id in order:
            hits = buckets.get(doc_id) or []
            if not hits:
                continue
            lines.append(f"DOCUMENT: {display_name.get(doc_id) or 'Document'}")
            ex_i = 0
            for hit in hits:
                text = (hit.get("text") or "").strip()
                if not text:
                    continue
                ex_i += 1
                page = hit.get("page")
                if not isinstance(page, int):
                    page = None
                if page is None:
                    m = re.search(r"---\s*Page\s+(\d+)\s*---", text, flags=re.IGNORECASE)
                    if m:
                        try:
                            page = int(m.group(1))
                        except Exception:
                            page = None

                if page is not None:
                    lines.append(f"Excerpt {ex_i} (page {page}):")
                else:
                    lines.append(f"Excerpt {ex_i}:")
                lines.append(text)
                lines.append("")
            lines.append("")

        evidence = "\n".join(lines).strip()
        if evidence:
            parts.append(f"EVIDENCE (by document):\n{evidence}")

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

    sel_file = selection.get("filename") if isinstance(selection.get("filename"), str) else None
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

    @staticmethod
    def _is_compare_request(user_message: str) -> bool:
        q = (user_message or "").strip().lower()
        if not q:
            return False
        if q in {"compare", "compare them", "compare both", "compare files", "compare documents"}:
            return True
        return any(k in q for k in ("compare", "contrast", "difference", "differences", "vs", "versus", "similarity"))

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
        doc_scope_mode: Optional[str] = None,
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

        # If the UI explicitly requests "all documents" scope, do not invent a focus.
        # This enables compare-style retrieval across every file in the chat.
        if isinstance(doc_scope_mode, str) and doc_scope_mode.strip().lower() == "all":
            return None

        attached_docs = [fid for fid in (documents or []) if isinstance(fid, str) and fid]
        if isinstance(focus_document_id, str) and focus_document_id:
            # Upload/attach turns include `documents`. If the UI sent a stale focus id
            # (not in this request's document list), prefer the most recently attached
            # doc so "upload → ask" matches the documents pane selection.
            if not attached_docs:
                return focus_document_id
            if focus_document_id in attached_docs:
                return focus_document_id
            return attached_docs[-1]

        # If the request included documents (e.g. upload+chat), default to the most recent
        # doc in that list so "upload → ask" works without extra UI wiring.
        if attached_docs:
            return attached_docs[-1]

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

    def _chat_file_ids(self, chat_id: str) -> List[str]:
        try:
            rows = self.metadata_store.list_files_for_chat(chat_id)
        except Exception:
            rows = []
        out: List[str] = []
        for r in rows or []:
            fid = r.get("id") if isinstance(r, dict) else None
            if isinstance(fid, str) and fid:
                out.append(fid)
        return out

    @staticmethod
    def _rebalance_hits_by_file(rag_hits: List[Dict[str, Any]], *, max_hits: int) -> List[Dict[str, Any]]:
        """
        Rebalance a hit list so multiple documents contribute evidence.

        This is a best-effort round-robin selector (keeps per-file ordering from the input list).
        """
        if not rag_hits or max_hits <= 0:
            return []

        buckets: Dict[str, List[Dict[str, Any]]] = {}
        order: List[str] = []
        for hit in rag_hits:
            fid = hit.get("doc_id") if isinstance(hit.get("doc_id"), str) else ""
            if fid not in buckets:
                buckets[fid] = []
                order.append(fid)
            buckets[fid].append(hit)

        out: List[Dict[str, Any]] = []
        while len(out) < max_hits:
            progressed = False
            for fid in order:
                b = buckets.get(fid) or []
                if not b:
                    continue
                out.append(b.pop(0))
                progressed = True
                if len(out) >= max_hits:
                    break
            if not progressed:
                break
        return out

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

        filename: Optional[str] = None
        try:
            rec = self.metadata_store.get_file(file_id)
            name = rec.get("filename") if isinstance(rec, dict) else None
            if isinstance(name, str) and name.strip():
                filename = name
        except Exception:
            filename = None

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
            pages: List[int] = []
            for b in blocks[start : end + 1]:
                kind = str(b.get("kind") or "paragraph")
                text = str(b.get("text") or "").strip()
                meta = b.get("metadata") if isinstance(b.get("metadata"), dict) else {}
                page = meta.get("page")
                if isinstance(page, int):
                    pages.append(page)
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
            page = min(pages) if pages else None
            evidence.append(
                {
                    "doc_id": file_id,
                    "chunk_id": f"doc_page:{start}-{end}",
                    "score": 1.0,
                    "text": excerpt,
                    "filename": filename,
                    "page": page,
                }
            )

        return evidence

    def _build_upload_doc_evidence(
        self,
        *,
        chat_id: str,
        query: str,
        doc_ids: List[str],
        documents_text: List[str],
        attachments: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Build cheap, deterministic per-file evidence windows from `documents_text`.

        This is used for upload/attach turns so the model can answer immediately,
        even before embeddings are available in Qdrant.

        IMPORTANT: Never embed internal IDs into the *text* we send to the model.
        We keep `doc_id/chunk_id` only as internal metadata for dedupe/debug.
        """
        if not doc_ids or not isinstance(doc_ids, list):
            return []
        if not isinstance(query, str) or not query.strip():
            return []

        # Preserve index alignment: documents_text and attachments are "best effort".
        # (Callers sometimes omit empty extracted text entries.)
        docs_count = len(doc_ids)
        if docs_count <= 0:
            return []

        # Heuristic: cap total upload evidence so we don't crowd out RAG evidence.
        # (~1200 tokens ~= 4800 chars; per-file budget scales with file count.)
        total_chars_budget = 4800
        per_file_chars_budget = max(900, min(2400, total_chars_budget // max(1, docs_count)))

        phrase = query.casefold()
        # Keep this cheap: use only word-like tokens of length >= 3.
        terms = [t for t in re.findall(r"[a-z0-9_]{3,}", phrase) if t]
        # Avoid pathological queries (long pasted text) from exploding work.
        terms = terms[:16]

        def _split_blocks(text: str) -> List[str]:
            # Prefer paragraph-ish blocks; keep page markers inside the excerpt text.
            return [b.strip() for b in re.split(r"\n\s*\n+", text or "") if b and b.strip()]

        def _score_block(block: str) -> float:
            blk = (block or "").casefold()
            if not blk:
                return 0.0
            score = 0.0
            if phrase and phrase in blk:
                score += 10.0
            for t in terms:
                if t in blk:
                    score += float(blk.count(t))
            return score

        def _pick_windows(blocks: List[str], *, desired: int) -> List[tuple[int, int]]:
            if not blocks or desired <= 0:
                return []

            scored: List[tuple[int, float]] = []
            for i, b in enumerate(blocks):
                s = _score_block(b)
                if s > 0:
                    scored.append((i, s))
            scored.sort(key=lambda x: (-x[1], x[0]))

            picked: List[int] = []
            if scored:
                for idx, _s in scored:
                    if any(abs(idx - p) <= 1 for p in picked):
                        continue
                    picked.append(idx)
                    if len(picked) >= desired:
                        break
            else:
                # Fallback (e.g. "compare these") when no lexical hits exist:
                # pick representative positions across the document.
                candidates = [0]
                if desired >= 2:
                    candidates.append(len(blocks) // 2)
                if desired >= 3:
                    candidates.append(max(0, len(blocks) - 1))
                for idx in candidates:
                    if idx < 0 or idx >= len(blocks):
                        continue
                    if any(abs(idx - p) <= 1 for p in picked):
                        continue
                    picked.append(idx)

            if not picked:
                return []

            windows: List[tuple[int, int]] = []
            for idx in sorted(picked):
                start = max(0, idx - 1)
                end = min(len(blocks) - 1, idx + 1)
                windows.append((start, end))

            merged: List[tuple[int, int]] = []
            for start, end in windows:
                if not merged:
                    merged.append((start, end))
                    continue
                last_s, last_e = merged[-1]
                if start <= last_e + 1:
                    merged[-1] = (last_s, max(last_e, end))
                else:
                    merged.append((start, end))
            return merged

        evidence: List[Dict[str, Any]] = []
        for i, doc_id in enumerate(doc_ids):
            if not isinstance(doc_id, str) or not doc_id:
                continue
            text = documents_text[i] if i < len(documents_text) and isinstance(documents_text[i], str) else ""
            if not text.strip():
                continue

            filename = attachments[i] if i < len(attachments) and isinstance(attachments[i], str) else None
            if not filename:
                try:
                    rec = self.metadata_store.get_file(doc_id)
                    name = rec.get("filename") if isinstance(rec, dict) else None
                    if isinstance(name, str) and name.strip():
                        filename = name.strip()
                except Exception:
                    filename = None

            approx_tokens = self._approx_token_count(text)
            if approx_tokens < 600:
                desired_windows = 1
            elif approx_tokens < 2400:
                desired_windows = 2
            else:
                desired_windows = 3

            blocks = _split_blocks(text)
            if not blocks:
                continue

            # Cap windows based on per-file budget.
            max_windows_by_budget = max(1, min(3, per_file_chars_budget // 800))
            desired_windows = max(1, min(desired_windows, max_windows_by_budget))
            windows = _pick_windows(blocks, desired=desired_windows)
            if not windows:
                continue

            per_window_budget = max(320, per_file_chars_budget // max(1, len(windows)))
            for w_i, (start, end) in enumerate(windows):
                excerpt = "\n\n".join(blocks[start : end + 1]).strip()
                if not excerpt:
                    continue
                if len(excerpt) > per_window_budget:
                    excerpt = excerpt[:per_window_budget].rstrip() + "…"
                evidence.append(
                    {
                        "doc_id": doc_id,
                        "chunk_id": f"upload_window:{doc_id}:{i}:{w_i}:{start}-{end}",
                        "score": 1.0,
                        "text": excerpt,
                        "filename": filename,
                        "page": None,
                        "source": "upload",
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

    def _apply_rag_budget_balanced(
        self,
        chat_id: str,
        rag_hits: List[Dict[str, Any]],
        *,
        file_order: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Apply the RAG budget but ensure multiple files contribute evidence.

        This is used for compare-style ("all") scope and multi-upload turns so a single
        document cannot monopolize the evidence window.
        """
        if not rag_hits:
            return []

        # Use the same budget computation as the sequential variant.
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

        budget_chars = max(0, budget_tokens * 4)
        if budget_chars <= 0:
            return []

        # Determine file buckets.
        file_ids: List[str] = []
        if isinstance(file_order, list) and file_order:
            seen = set()
            for fid in file_order:
                if isinstance(fid, str) and fid and fid not in seen:
                    seen.add(fid)
                    file_ids.append(fid)
        else:
            seen = set()
            for h in rag_hits:
                fid = h.get("doc_id")
                if isinstance(fid, str) and fid and fid not in seen:
                    seen.add(fid)
                    file_ids.append(fid)

        # If we don't have at least 2 distinct files, fall back to sequential trim.
        if len(file_ids) < 2:
            return self._apply_rag_budget(chat_id, rag_hits)

        per_file_budget = max(256, budget_chars // max(1, len(file_ids)))
        # Hard cap per-hit excerpt so one chunk cannot consume an entire file's allocation.
        per_hit_cap = min(2400, per_file_budget)

        buckets: Dict[str, List[Dict[str, Any]]] = {fid: [] for fid in file_ids}
        unknown: List[Dict[str, Any]] = []
        for h in rag_hits:
            fid = h.get("doc_id")
            if isinstance(fid, str) and fid in buckets:
                buckets[fid].append(h)
            else:
                unknown.append(h)

        trimmed: List[Dict[str, Any]] = []
        used_chars_total = 0
        for fid in file_ids:
            remaining_chars = per_file_budget
            for hit in buckets.get(fid, []):
                if remaining_chars <= 0:
                    break
                text = hit.get("text") or ""
                if not isinstance(text, str) or not text.strip():
                    continue
                take = min(len(text), remaining_chars, per_hit_cap)
                trimmed.append({**hit, "text": text[:take]})
                used_chars_total += take
                remaining_chars -= take

        # If there is spare budget, include unknown hits (rare) using the remaining pool.
        remaining_pool = max(0, budget_chars - used_chars_total)
        for hit in unknown:
            if remaining_pool <= 0:
                break
            text = hit.get("text") or ""
            if not isinstance(text, str) or not text.strip():
                continue
            take = min(len(text), remaining_pool, 2400)
            trimmed.append({**hit, "text": text[:take]})
            remaining_pool -= take

        if len(trimmed) != len(rag_hits):
            logger.info(
                "RAG context trimmed (balanced) chat=%s hits_in=%d hits_out=%d files=%d budget_tokens=%d",
                chat_id,
                len(rag_hits),
                len(trimmed),
                len(file_ids),
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
        doc_scope_mode: Optional[str] = None,
        selection: Optional[Dict[str, Any]] = None,
    ) -> str:
        start = time.perf_counter()
        documents = documents or []
        # Keep index alignment with `documents` and `attachments` (upload turns rely on it).
        # Do NOT drop empty entries here; callers sometimes omit empty extracted text entries.
        documents_text = [t if isinstance(t, str) else "" for t in (documents_text or [])]
        attachments = attachments or []
        if len(documents_text) < len(documents):
            documents_text = documents_text + [""] * (len(documents) - len(documents_text))
        elif len(documents_text) > len(documents):
            documents_text = documents_text[: len(documents)]
        if len(attachments) < len(documents):
            attachments = attachments + [""] * (len(documents) - len(attachments))
        elif len(attachments) > len(documents):
            attachments = attachments[: len(documents)]
        # Best-effort: fill missing attachment names from metadata store to keep labels stable.
        # (The UI sometimes omits/shortens attachment arrays across turn types.)
        if documents and attachments:
            for i, fid in enumerate(documents):
                if i >= len(attachments):
                    break
                if isinstance(attachments[i], str) and attachments[i].strip():
                    continue
                if not isinstance(fid, str) or not fid:
                    continue
                try:
                    rec = self.metadata_store.get_file(fid)
                    name = rec.get("filename") if isinstance(rec, dict) else None
                    if isinstance(name, str) and name.strip():
                        attachments[i] = name.strip()
                except Exception:
                    continue
        analysis = self._analyze_query(user_message, has_docs=bool(documents or documents_text))
        max_tokens = self._compute_output_max_tokens(analysis)
        summary_request = self._is_summary_request(user_message)
        compare_request = self._is_compare_request(user_message)
        # Ensure deterministic session per chat_id (required even when no docs are attached).
        self.session_mgr.get_or_create_session(chat_id, system_prompt=self.system_hint)

        ltm_k = 2 if selection else 5
        ltm_hits = self.ltm_store.retrieve(chat_id, user_message, top_k=ltm_k)
        sel_file_id = selection.get("file_id") if isinstance(selection, dict) and isinstance(selection.get("file_id"), str) else None
        effective_focus = self._resolve_focus_for_turn(
            chat_id,
            documents=documents,
            focus_document_id=focus_document_id,
            doc_scope_mode=doc_scope_mode,
            selection=selection,
        )
        scope_mode = (doc_scope_mode or "").strip().lower()
        # Selection always hard-focuses, regardless of UI scope mode.
        if sel_file_id:
            scope_mode = "focused"
        if scope_mode not in {"focused", "all"}:
            scope_mode = "focused"
        # Compare queries should consider multiple documents even if the UI is currently focused.
        if compare_request and not sel_file_id and scope_mode == "focused":
            try:
                existing = self._chat_file_ids(chat_id)
            except Exception:
                existing = []
            if len(documents) > 1 or len(set(existing or [])) >= 2:
                scope_mode = "all"
        logger.debug(
            "Planner turn chat=%s summary=%s scope=%s focus=%s docs_payload=%d docs_text=%d attachments=%d",
            chat_id,
            summary_request,
            scope_mode,
            effective_focus,
            len(documents),
            len(documents_text),
            len(attachments),
        )

        focus_user_edited = bool(effective_focus and self._is_user_edited_doc(chat_id, effective_focus))
        effective_focus_name: Optional[str] = None
        if effective_focus:
            try:
                rec = self.metadata_store.get_file(effective_focus)
                name = rec.get("filename") if isinstance(rec, dict) else None
                if isinstance(name, str) and name.strip():
                    effective_focus_name = name.strip()
            except Exception:
                effective_focus_name = None

        upload_has_text = bool(any(isinstance(t, str) and t.strip() for t in documents_text))
        # Multi-doc uploads should never inline full doc blobs; use per-file windows + RAG.
        upload_multi = bool(len(documents) > 1 and upload_has_text)
        upload_evidence: List[Dict[str, Any]] = []

        chat_doc_ids: List[str] = []

        # Inline docs are reserved for summary-style queries and small fallback previews.
        # IMPORTANT: if the focused file is user-edited, do not inline cached chunk text
        # (it may not match the edited doc page). We’ll pull evidence from doc_pages instead.
        if focus_user_edited or upload_multi:
            doc_texts = []
        elif documents_text:
            doc_texts = [t for t in documents_text if isinstance(t, str) and t.strip()]
        elif effective_focus:
            doc_texts = self._fetch_doc_texts([effective_focus])
        else:
            doc_texts = self._fetch_doc_texts(documents)
        doc_texts = self._apply_inline_doc_budget(chat_id, doc_texts)

        # Summary-style queries (e.g. "brief") should not rely on vector search;
        # they need sequential doc text for the focused document.
        if summary_request and effective_focus:
            focus_text = self._fetch_focus_doc_text(chat_id, effective_focus) if focus_user_edited else ""
            if documents and documents_text:
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
            # If documents were attached for this turn (upload/attach), build cheap evidence
            # windows from the extracted text so "upload → ask" works even before Qdrant is ready.
            if documents and upload_has_text:
                upload_evidence = self._build_upload_doc_evidence(
                    chat_id=chat_id,
                    query=user_message,
                    doc_ids=documents,
                    documents_text=documents_text,
                    attachments=attachments,
                )
            rag_hits_raw = list(upload_evidence)
            chat_doc_ids = self._chat_file_ids(chat_id) or [d for d in documents if isinstance(d, str) and d]

            if effective_focus and scope_mode == "focused":
                # Focus-biased retrieval: primary from the focused doc, with a small allowance
                # for other docs in the same chat (lower priority).
                if self._is_user_edited_doc(chat_id, effective_focus):
                    rag_hits_raw.extend(
                        self._edited_doc_evidence(chat_id, effective_focus, user_message, max_windows=primary_k)
                    )
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

                other_doc_ids = [fid for fid in chat_doc_ids if fid != effective_focus]
                secondary_k = 2 if selection else 3
                if other_doc_ids and secondary_k > 0:
                    rag_hits_raw.extend(
                        self.rag_store.retrieve(
                            user_message,
                            chat_id=chat_id,
                            doc_ids=other_doc_ids,
                            top_k=secondary_k,
                        )
                    )
            elif scope_mode == "all" and chat_doc_ids:
                # Compare-style retrieval: ensure every file contributes evidence.
                #
                # A global "top-k across all docs" search can easily bias to a single file,
                # especially for vague queries like "compare". For small doc sets, do a
                # per-file retrieval pass so the model always sees at least some content
                # from each file.
                doc_ids_in_scope: List[str] = []
                seen_docs = set()
                for fid in chat_doc_ids:
                    if isinstance(fid, str) and fid and fid not in seen_docs:
                        seen_docs.add(fid)
                        doc_ids_in_scope.append(fid)
                if not doc_ids_in_scope:
                    doc_ids_in_scope = chat_doc_ids

                max_docs_for_per_file = 6
                if len(doc_ids_in_scope) <= max_docs_for_per_file:
                    q_vec = self.rag_store.embed(user_message)
                    per_doc_k = max(1, (primary_k + len(doc_ids_in_scope) - 1) // max(1, len(doc_ids_in_scope)))
                    for fid in doc_ids_in_scope:
                        if self._is_user_edited_doc(chat_id, fid):
                            rag_hits_raw.extend(self._edited_doc_evidence(chat_id, fid, user_message, max_windows=per_doc_k))
                        elif q_vec is not None:
                            rag_hits_raw.extend(
                                self.rag_store.retrieve_with_vector(
                                    q_vec,
                                    chat_id=chat_id,
                                    doc_ids=[fid],
                                    top_k=per_doc_k,
                                )
                            )
                        else:
                            rag_hits_raw.extend(
                                self.rag_store.retrieve(user_message, chat_id=chat_id, doc_ids=[fid], top_k=per_doc_k)
                            )
                else:
                    # Fallback: pull a larger pool across the whole chat, then rebalance so
                    # multiple documents contribute evidence.
                    k_total = primary_k
                    pool_k = min(24, max(k_total, k_total * max(1, len(doc_ids_in_scope))))
                    pool = self.rag_store.retrieve(
                        user_message,
                        chat_id=chat_id,
                        doc_ids=doc_ids_in_scope,
                        top_k=pool_k,
                    )
                    rag_hits_raw.extend(self._rebalance_hits_by_file(pool, max_hits=k_total))
            elif documents:
                rag_hits_raw.extend(
                    self.rag_store.retrieve(user_message, chat_id=chat_id, doc_ids=documents, top_k=primary_k)
                )
            else:
                rag_hits_raw.extend(self.rag_store.retrieve(user_message, chat_id=chat_id, top_k=primary_k))
        selected_rag = self._dedup_rag(rag_hits_raw)
        # IMPORTANT: when multiple files contribute evidence (multi-upload turns or
        # doc_scope_mode="all"), interleave hits across files so budgeting/truncation
        # can't drop an entire document from the model-visible context pack.
        if selected_rag and (upload_multi or scope_mode == "all"):
            selected_rag = self._rebalance_hits_by_file(selected_rag, max_hits=len(selected_rag))
        if selected_rag and (upload_multi or scope_mode == "all"):
            file_order = documents if upload_multi else chat_doc_ids
            selected_rag = self._apply_rag_budget_balanced(chat_id, selected_rag, file_order=file_order)
        else:
            selected_rag = self._apply_rag_budget(chat_id, selected_rag)
        if selected_rag and (upload_multi or scope_mode == "all"):
            try:
                mix: Dict[str, Dict[str, int]] = {}
                for h in selected_rag:
                    name = h.get("filename") if isinstance(h.get("filename"), str) and h.get("filename") else "unknown"
                    mix.setdefault(name, {"hits": 0, "chars": 0})
                    mix[name]["hits"] += 1
                    txt = h.get("text")
                    if isinstance(txt, str):
                        mix[name]["chars"] += len(txt)
                logger.debug("RAG mix chat=%s scope=%s files=%s", chat_id, scope_mode, mix)
            except Exception:
                pass
        if scope_mode == "focused" and effective_focus and not doc_texts and not selected_rag:
            # Focused retrieval yielded nothing; fall back to a small sequential excerpt.
            preview = self._doc_fallback_preview(chat_id, self._fetch_focus_doc_text(chat_id, effective_focus))
            if preview.strip():
                doc_texts = [preview]
        has_selection_text = bool(
            isinstance(selection, dict)
            and isinstance(selection.get("text"), str)
            and str(selection.get("text")).strip()
        )

        scope_files: Optional[List[str]] = None
        if not sel_file_id:
            # For multi-doc compare turns (including multi-upload), explicitly list filenames
            # in SCOPE so the model doesn't treat the turn as "focused only".
            if compare_request and upload_multi:
                scope_files = [a for a in attachments if isinstance(a, str) and a.strip()]
            elif scope_mode == "all" or (compare_request and scope_mode == "focused"):
                try:
                    rows = self.metadata_store.list_files_for_chat(chat_id)
                except Exception:
                    rows = []
                scope_files = []
                for r in rows or []:
                    name = r.get("filename") if isinstance(r, dict) else None
                    if not isinstance(name, str):
                        continue
                    name = name.strip()
                    if not name:
                        continue
                    scope_files.append(name)
                    if len(scope_files) >= 12:
                        break
                if not scope_files:
                    scope_files = None

        context_pack = build_context_pack(
            ltm_hits=ltm_hits,
            doc_texts=doc_texts,
            rag_hits=selected_rag,
            selection=selection,
            effective_focus=effective_focus,
            effective_focus_name=effective_focus_name,
            scope_files=scope_files,
            include_selection_excerpt=not has_selection_text,
        )
        if (upload_multi or scope_mode == "all") and compare_request:
            context_pack = (
                "TASK:\n"
                "- Compare the documents listed in SCOPE.\n"
                "- Keep document evidence separate; do not blend details across files.\n"
                "- Only claim similarities if supported by BOTH documents.\n"
                "- For anything not supported by the evidence, say you can't tell.\n"
                "- Avoid speculation/marketing; keep it grounded to the excerpts.\n\n"
                + (context_pack or "")
            ).strip()
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
        dirty_user: Optional[str] = None
        if has_selection_text:
            dirty_user = build_dirty_user_turn(user_message, selection=selection)
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
        doc_scope_mode: Optional[str] = None,
        selection: Optional[Dict[str, Any]] = None,
        *,
        request_id: Optional[str] = None,
    ):
        """
        Streaming variant of handle_message yielding tokens.
        """
        documents = documents or []
        # Keep index alignment with `documents` and `attachments` (upload turns rely on it).
        # Do NOT drop empty entries here; callers sometimes omit empty extracted text entries.
        documents_text = [t if isinstance(t, str) else "" for t in (documents_text or [])]
        attachments = attachments or []
        if len(documents_text) < len(documents):
            documents_text = documents_text + [""] * (len(documents) - len(documents_text))
        elif len(documents_text) > len(documents):
            documents_text = documents_text[: len(documents)]
        if len(attachments) < len(documents):
            attachments = attachments + [""] * (len(documents) - len(attachments))
        elif len(attachments) > len(documents):
            attachments = attachments[: len(documents)]
        # Best-effort: fill missing attachment names from metadata store to keep labels stable.
        if documents and attachments:
            for i, fid in enumerate(documents):
                if i >= len(attachments):
                    break
                if isinstance(attachments[i], str) and attachments[i].strip():
                    continue
                if not isinstance(fid, str) or not fid:
                    continue
                try:
                    rec = self.metadata_store.get_file(fid)
                    name = rec.get("filename") if isinstance(rec, dict) else None
                    if isinstance(name, str) and name.strip():
                        attachments[i] = name.strip()
                except Exception:
                    continue
        analysis = self._analyze_query(user_message, has_docs=bool(documents or documents_text))
        max_tokens = self._compute_output_max_tokens(analysis)
        summary_request = self._is_summary_request(user_message)
        compare_request = self._is_compare_request(user_message)
        # Ensure deterministic session per chat_id (required even when no docs are attached).
        self.session_mgr.get_or_create_session(chat_id, system_prompt=self.system_hint)

        ltm_k = 2 if selection else 5
        ltm_hits = self.ltm_store.retrieve(chat_id, user_message, top_k=ltm_k)
        sel_file_id = selection.get("file_id") if isinstance(selection, dict) and isinstance(selection.get("file_id"), str) else None
        effective_focus = self._resolve_focus_for_turn(
            chat_id,
            documents=documents,
            focus_document_id=focus_document_id,
            doc_scope_mode=doc_scope_mode,
            selection=selection,
        )
        scope_mode = (doc_scope_mode or "").strip().lower()
        if sel_file_id:
            scope_mode = "focused"
        if scope_mode not in {"focused", "all"}:
            scope_mode = "focused"
        if compare_request and not sel_file_id and scope_mode == "focused":
            try:
                existing = self._chat_file_ids(chat_id)
            except Exception:
                existing = []
            if len(documents) > 1 or len(set(existing or [])) >= 2:
                scope_mode = "all"
        logger.debug(
            "Planner stream turn chat=%s summary=%s scope=%s focus=%s docs_payload=%d docs_text=%d attachments=%d request_id=%s",
            chat_id,
            summary_request,
            scope_mode,
            effective_focus,
            len(documents),
            len(documents_text),
            len(attachments),
            request_id,
        )

        focus_user_edited = bool(effective_focus and self._is_user_edited_doc(chat_id, effective_focus))
        effective_focus_name: Optional[str] = None
        if effective_focus:
            try:
                rec = self.metadata_store.get_file(effective_focus)
                name = rec.get("filename") if isinstance(rec, dict) else None
                if isinstance(name, str) and name.strip():
                    effective_focus_name = name.strip()
            except Exception:
                effective_focus_name = None

        upload_has_text = bool(any(isinstance(t, str) and t.strip() for t in documents_text))
        upload_multi = bool(len(documents) > 1 and upload_has_text)
        upload_evidence: List[Dict[str, Any]] = []

        chat_doc_ids: List[str] = []

        # Inline docs are reserved for summary-style queries and small fallback previews.
        # IMPORTANT: if the focused file is user-edited, do not inline cached chunk text
        # (it may not match the edited doc page). We’ll pull evidence from doc_pages instead.
        if focus_user_edited or upload_multi:
            doc_texts = []
        elif documents_text:
            doc_texts = [t for t in documents_text if isinstance(t, str) and t.strip()]
        elif effective_focus:
            doc_texts = self._fetch_doc_texts([effective_focus])
        else:
            doc_texts = self._fetch_doc_texts(documents)
        doc_texts = self._apply_inline_doc_budget(chat_id, doc_texts)

        if summary_request and effective_focus:
            focus_text = self._fetch_focus_doc_text(chat_id, effective_focus) if focus_user_edited else ""
            if documents and documents_text:
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
            if documents and upload_has_text:
                upload_evidence = self._build_upload_doc_evidence(
                    chat_id=chat_id,
                    query=user_message,
                    doc_ids=documents,
                    documents_text=documents_text,
                    attachments=attachments,
                )
            rag_hits_raw = list(upload_evidence)
            chat_doc_ids = self._chat_file_ids(chat_id) or [d for d in documents if isinstance(d, str) and d]

            if effective_focus and scope_mode == "focused":
                if self._is_user_edited_doc(chat_id, effective_focus):
                    rag_hits_raw.extend(
                        self._edited_doc_evidence(chat_id, effective_focus, user_message, max_windows=primary_k)
                    )
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

                other_doc_ids = [fid for fid in chat_doc_ids if fid != effective_focus]
                secondary_k = 2 if selection else 3
                if other_doc_ids and secondary_k > 0:
                    rag_hits_raw.extend(
                        self.rag_store.retrieve(
                            user_message,
                            chat_id=chat_id,
                            doc_ids=other_doc_ids,
                            top_k=secondary_k,
                        )
                    )
            elif scope_mode == "all" and chat_doc_ids:
                doc_ids_in_scope: List[str] = []
                seen_docs = set()
                for fid in chat_doc_ids:
                    if isinstance(fid, str) and fid and fid not in seen_docs:
                        seen_docs.add(fid)
                        doc_ids_in_scope.append(fid)
                if not doc_ids_in_scope:
                    doc_ids_in_scope = chat_doc_ids

                max_docs_for_per_file = 6
                if len(doc_ids_in_scope) <= max_docs_for_per_file:
                    q_vec = self.rag_store.embed(user_message)
                    per_doc_k = max(1, (primary_k + len(doc_ids_in_scope) - 1) // max(1, len(doc_ids_in_scope)))
                    for fid in doc_ids_in_scope:
                        if self._is_user_edited_doc(chat_id, fid):
                            rag_hits_raw.extend(self._edited_doc_evidence(chat_id, fid, user_message, max_windows=per_doc_k))
                        elif q_vec is not None:
                            rag_hits_raw.extend(
                                self.rag_store.retrieve_with_vector(
                                    q_vec,
                                    chat_id=chat_id,
                                    doc_ids=[fid],
                                    top_k=per_doc_k,
                                )
                            )
                        else:
                            rag_hits_raw.extend(
                                self.rag_store.retrieve(user_message, chat_id=chat_id, doc_ids=[fid], top_k=per_doc_k)
                            )
                else:
                    k_total = primary_k
                    pool_k = min(24, max(k_total, k_total * max(1, len(doc_ids_in_scope))))
                    pool = self.rag_store.retrieve(
                        user_message,
                        chat_id=chat_id,
                        doc_ids=doc_ids_in_scope,
                        top_k=pool_k,
                    )
                    rag_hits_raw.extend(self._rebalance_hits_by_file(pool, max_hits=k_total))
            elif documents:
                rag_hits_raw.extend(
                    self.rag_store.retrieve(user_message, chat_id=chat_id, doc_ids=documents, top_k=primary_k)
                )
            else:
                rag_hits_raw.extend(self.rag_store.retrieve(user_message, chat_id=chat_id, top_k=primary_k))
        selected_rag = self._dedup_rag(rag_hits_raw)
        if selected_rag and (upload_multi or scope_mode == "all"):
            selected_rag = self._rebalance_hits_by_file(selected_rag, max_hits=len(selected_rag))
        if selected_rag and (upload_multi or scope_mode == "all"):
            file_order = documents if upload_multi else chat_doc_ids
            selected_rag = self._apply_rag_budget_balanced(chat_id, selected_rag, file_order=file_order)
        else:
            selected_rag = self._apply_rag_budget(chat_id, selected_rag)
        if selected_rag and (upload_multi or scope_mode == "all"):
            try:
                mix: Dict[str, Dict[str, int]] = {}
                for h in selected_rag:
                    name = h.get("filename") if isinstance(h.get("filename"), str) and h.get("filename") else "unknown"
                    mix.setdefault(name, {"hits": 0, "chars": 0})
                    mix[name]["hits"] += 1
                    txt = h.get("text")
                    if isinstance(txt, str):
                        mix[name]["chars"] += len(txt)
                logger.debug("RAG mix chat=%s scope=%s files=%s request_id=%s", chat_id, scope_mode, mix, request_id)
            except Exception:
                pass
        if scope_mode == "focused" and effective_focus and not doc_texts and not selected_rag:
            preview = self._doc_fallback_preview(chat_id, self._fetch_focus_doc_text(chat_id, effective_focus))
            if preview.strip():
                doc_texts = [preview]
        has_selection_text = bool(
            isinstance(selection, dict)
            and isinstance(selection.get("text"), str)
            and str(selection.get("text")).strip()
        )

        scope_files: Optional[List[str]] = None
        if not sel_file_id:
            if compare_request and upload_multi:
                scope_files = [a for a in attachments if isinstance(a, str) and a.strip()]
            elif scope_mode == "all" or (compare_request and scope_mode == "focused"):
                try:
                    rows = self.metadata_store.list_files_for_chat(chat_id)
                except Exception:
                    rows = []
                scope_files = []
                seen_names = set()
                for r in rows or []:
                    name = r.get("filename") if isinstance(r, dict) else None
                    if not isinstance(name, str):
                        continue
                    name = name.strip()
                    if not name or name in seen_names:
                        continue
                    seen_names.add(name)
                    scope_files.append(name)
                    if len(scope_files) >= 12:
                        break
                if not scope_files:
                    scope_files = None

        context_pack = build_context_pack(
            ltm_hits=ltm_hits,
            doc_texts=doc_texts,
            rag_hits=selected_rag,
            selection=selection,
            effective_focus=effective_focus,
            effective_focus_name=effective_focus_name,
            scope_files=scope_files,
            include_selection_excerpt=not has_selection_text,
        )
        if (upload_multi or scope_mode == "all") and compare_request:
            context_pack = (
                "TASK:\n"
                "- Compare the documents listed in SCOPE.\n"
                "- Keep document evidence separate; do not blend details across files.\n"
                "- Only claim similarities if supported by BOTH documents.\n"
                "- For anything not supported by the evidence, say you can't tell.\n"
                "- Avoid speculation/marketing; keep it grounded to the excerpts.\n\n"
                + (context_pack or "")
            ).strip()
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
            doc_id = h.get("doc_id") if isinstance(h.get("doc_id"), str) else ""
            chunk_id = h.get("chunk_id") if isinstance(h.get("chunk_id"), str) else ""
            text = h.get("text") if isinstance(h.get("text"), str) else ""

            # Avoid cross-document dedupe collisions. Chunk ids for edited/upload windows
            # aren't guaranteed globally unique, and boilerplate text repeats across files.
            if doc_id and chunk_id:
                key = f"{doc_id}:{chunk_id}"
            elif doc_id and text:
                key = f"{doc_id}:{hash(text)}"
            else:
                key = chunk_id or text or ""
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
                target_words = 280
        # Prefer "detailed/long" over "brief/summary" if both appear (e.g. "detailed summary").
        elif any(k in q for k in ["detailed", "long", "full"]):
            target_words = 350
        elif any(k in q for k in ["short", "brief", "summary"]):
            target_words = 100
        hint_doc = has_docs or any(k in q for k in ["document", "file", "pdf", "upload"])
        return {
            "target_words": target_words,
            "hint_doc": hint_doc,
        }


__all__ = ["InsightOrchestrator"]
