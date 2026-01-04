from __future__ import annotations

import json
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from backend.services.connectors.llama_session_manager import LlamaSessionManager
from backend.services.planner.agent_loop import MultiFileAgentConfig, MultiFileAgentLoop
from backend.services.planner.agent_tools import AgentToolbox, ChunkAnchor
from backend.services.planner.raw_large_loop import RawLargeAgentLoop
from backend.services.retrieval.rag_store import RagStore
from backend.services.memory.ltm_store import LongTermMemoryStore, MemoryHit
from backend.services.ipc_events import emit_event
from backend.services.storage.sqlite_store import SQLiteMetadataStore

logger = logging.getLogger(__name__)

# RAG budgeting defaults (backend-owned; no config system yet).
#
# Keep this conservative: the session manager has a hard safety net that will
# clamp the final "context pack" to the remaining context window. These values
# primarily control how much *document evidence* we try to retrieve/pack.
RAG_CONTEXT_FRACTION = 0.35
RAG_CONTEXT_MAX_TOKENS_SINGLE = 3000
RAG_CONTEXT_MAX_TOKENS_MULTI = 10_000

# Retrieval policy (UI-driven focus/scope).
RAG_PRIMARY_K = 12
RAG_PRIMARY_K_SELECTION = 6
RAG_PRIMARY_K_SUMMARY = 16
RAG_ALL_K_TOTAL = 14
# Focused mode should hard-scope retrieval to the focused file only.
# Cross-file mixing is allowed only in scope=all.
RAG_SECONDARY_K_TOTAL = 0
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


def build_context_pack(
    *,
    ltm_hits: List[MemoryHit],
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
        # - compare/multi-doc turns where UI focus would otherwise "hide" scope
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

    # IMPORTANT: Strict RAG mode — never inject raw/extracted document blobs into the
    # model-visible context pack. Documents should only enter the prompt via retrieved
    # evidence chunks (Qdrant) or edited-doc lexical windows (doc_pages).

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
        self.agent_tools = AgentToolbox(rag_store=rag_store, metadata_store=metadata_store)
        self.agent_loop = MultiFileAgentLoop(tools=self.agent_tools, embedder=getattr(rag_store, "embedder", None))
        self.raw_large_loop = RawLargeAgentLoop(metadata_store=metadata_store)
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

    def _compute_rag_budget_tokens(
        self,
        chat_id: str,
        *,
        files_in_scope: int,
        reserved_output_tokens: int,
    ) -> int:
        """
        Compute a doc-evidence token budget for this turn.

        This is used to size multi-file windows (scope=all) and to trim the final
        RAG hits, while still leaving room for:
        - clean session history (already in KV)
        - the user's turn
        - LTM / scope headers
        - model output (reserved_output_tokens)

        The session manager enforces the real hard limit; this is a best-effort
        allocator that tries to keep multi-file answers coherent (enough text per file).
        """
        try:
            files_in_scope = int(files_in_scope)
        except Exception:
            files_in_scope = 0
        if files_in_scope <= 0:
            return 0

        # Ensure session exists so context status is meaningful.
        self.session_mgr.get_or_create_session(chat_id, system_prompt=self.system_hint)
        try:
            status = self.session_mgr.get_context_status(chat_id)
            used = int(status.get("used_tokens") or 0)
            capacity = int(status.get("capacity_tokens") or 0)
        except Exception:
            used = 0
            capacity = int(getattr(self.session_mgr, "ctx_size", 0) or 0)

        margin = 256
        remaining = max(0, capacity - used - int(reserved_output_tokens) - margin)
        if remaining <= 0:
            return 0

        cap = RAG_CONTEXT_MAX_TOKENS_SINGLE if files_in_scope <= 1 else RAG_CONTEXT_MAX_TOKENS_MULTI
        # Keep a stable fraction of remaining capacity for doc evidence; leave the rest for
        # the rest of the prompt (LTM, scope lines, tool-safe instructions, etc).
        budget = min(int(remaining * RAG_CONTEXT_FRACTION), int(cap))
        return max(0, int(budget))

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
        """
        Detect short compare prompts ("compare", "compare docs", etc.).

        These turns typically need:
        - broader, balanced evidence windows across files
        - slightly larger output budget
        """
        q = (user_message or "").strip().lower()
        if not q:
            return False
        q = re.sub(r"\s+", " ", q).strip().strip(".!?;:")
        if q in {"compare", "comparison", "diff"}:
            return True
        if q.startswith("compare ") and len(q.split()) <= 3:
            return True
        return False

    @staticmethod
    def _uniq_file_ids(values: List[Any]) -> List[str]:
        out: List[str] = []
        seen = set()
        for v in values or []:
            if not isinstance(v, str):
                continue
            s = v.strip()
            if not s or s in seen:
                continue
            seen.add(s)
            out.append(s)
        return out

    def _resolve_scope_for_turn(
        self,
        chat_id: str,
        *,
        documents: List[str],
        focus_document_id: Optional[str],
        doc_pane_open: Optional[bool],
        selection: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Resolve the retrieval plan for this turn based on UI state.

        Rules (high level):
        - Selection from docs (selection.file_id): hard-focus that file only.
        - Chat pane only + multi-upload in this turn: scope=all across uploaded docs (equal priority).
        - Otherwise: scope=focused with a primary focus file.
        """
        turn_doc_ids = self._uniq_file_ids(documents or [])

        # Infer doc pane visibility if the UI didn't send it yet.
        doc_pane_open_bool = bool(doc_pane_open) if doc_pane_open is not None else bool(focus_document_id)

        chat_doc_ids = self._uniq_file_ids(self._chat_file_ids(chat_id))
        for fid in turn_doc_ids:
            if fid not in chat_doc_ids:
                chat_doc_ids.append(fid)

        latest_doc_id: Optional[str] = None
        if turn_doc_ids:
            latest_doc_id = turn_doc_ids[-1]
        elif chat_doc_ids:
            latest_doc_id = chat_doc_ids[-1]

        sel_file_id = (
            selection.get("file_id")
            if isinstance(selection, dict) and isinstance(selection.get("file_id"), str)
            else None
        )
        if isinstance(sel_file_id, str) and sel_file_id:
            return {
                "doc_pane_open": doc_pane_open_bool,
                "scope": "focused",
                "focus": sel_file_id,
                "scope_doc_ids": [],
                "secondary_doc_ids": [],
                "chat_doc_ids": chat_doc_ids,
                "latest_doc_id": latest_doc_id,
            }

        # If the user uploaded document(s) in *this* turn, always prioritize the new upload(s)
        # even if the docs pane is open and still focused on an older file at send-time.
        #
        # - Single upload: focus that new file.
        # - Multi-upload: scope=all across the uploaded files for this turn.
        if turn_doc_ids:
            if len(turn_doc_ids) > 1:
                return {
                    "doc_pane_open": doc_pane_open_bool,
                    "scope": "all",
                    "focus": None,
                    "scope_doc_ids": turn_doc_ids,
                    "secondary_doc_ids": [],
                    "chat_doc_ids": chat_doc_ids,
                    "latest_doc_id": latest_doc_id,
                }

            focus = turn_doc_ids[-1]
            secondary_doc_ids = [fid for fid in chat_doc_ids if fid != focus]
            return {
                "doc_pane_open": doc_pane_open_bool,
                "scope": "focused",
                "focus": focus,
                "scope_doc_ids": [],
                "secondary_doc_ids": secondary_doc_ids,
                "chat_doc_ids": chat_doc_ids,
                "latest_doc_id": latest_doc_id,
            }

        # Multi-upload in the chat pane: treat all uploaded files equally.
        if (not doc_pane_open_bool) and len(turn_doc_ids) > 1:
            return {
                "doc_pane_open": doc_pane_open_bool,
                "scope": "all",
                "focus": None,
                "scope_doc_ids": turn_doc_ids,
                "secondary_doc_ids": [],
                "chat_doc_ids": chat_doc_ids,
                "latest_doc_id": latest_doc_id,
            }

        # Default: focused mode with a primary focus file.
        focus: Optional[str] = None
        if doc_pane_open_bool and isinstance(focus_document_id, str) and focus_document_id.strip():
            cand = focus_document_id.strip()
            if cand in chat_doc_ids or cand in turn_doc_ids:
                focus = cand
        if not focus:
            focus = latest_doc_id

        secondary_doc_ids: List[str] = []
        if focus:
            secondary_doc_ids = [fid for fid in chat_doc_ids if fid != focus]

        return {
            "doc_pane_open": doc_pane_open_bool,
            "scope": "focused",
            "focus": focus,
            "scope_doc_ids": [],
            "secondary_doc_ids": secondary_doc_ids,
            "chat_doc_ids": chat_doc_ids,
            "latest_doc_id": latest_doc_id,
        }

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
        return self._uniq_file_ids(out)

    def _raw_large_file_id(self, file_ids: List[str]) -> Optional[str]:
        """
        Return the first file_id marked as policy.raw_large=True (Plan B).

        By policy, a chat may contain at most one raw_large file.
        """
        for fid in file_ids or []:
            if not isinstance(fid, str) or not fid:
                continue
            try:
                rec = self.metadata_store.get_file(fid) or {}
                policy_raw = rec.get("policy_json")
                policy: Dict[str, Any] = {}
                if isinstance(policy_raw, str) and policy_raw.strip():
                    try:
                        policy = json.loads(policy_raw) or {}
                    except Exception:
                        policy = {}
                if bool(policy.get("raw_large")):
                    return fid
            except Exception:
                continue
        return None

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

    def _apply_rag_budget(self, chat_id: str, rag_hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return self._apply_rag_budget_with_tokens(chat_id, rag_hits, budget_tokens=None)

    def _apply_rag_budget_with_tokens(
        self,
        chat_id: str,
        rag_hits: List[Dict[str, Any]],
        *,
        budget_tokens: Optional[int],
    ) -> List[Dict[str, Any]]:
        if not rag_hits:
            return []

        if budget_tokens is None:
            # Infer doc count from hits and compute a budget without assuming scope mode.
            doc_ids = {h.get("doc_id") for h in rag_hits if isinstance(h.get("doc_id"), str) and h.get("doc_id")}
            inferred_files = max(1, len(doc_ids))
            # Use a moderate default output reserve when called outside the main turn path.
            budget_tokens = self._compute_rag_budget_tokens(
                chat_id,
                files_in_scope=inferred_files,
                reserved_output_tokens=OUTPUT_MAX_TOKENS,
            )

        try:
            budget_tokens = int(budget_tokens)
        except Exception:
            budget_tokens = 0
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
        budget_tokens: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Apply the RAG budget but ensure multiple files contribute evidence.

        This is used for compare-style ("all") scope and multi-upload turns so a single
        document cannot monopolize the evidence window.
        """
        if not rag_hits:
            return []

        if budget_tokens is None:
            # Compute from inferred doc count.
            doc_ids = {h.get("doc_id") for h in rag_hits if isinstance(h.get("doc_id"), str) and h.get("doc_id")}
            inferred_files = max(1, len(doc_ids))
            budget_tokens = self._compute_rag_budget_tokens(
                chat_id,
                files_in_scope=inferred_files,
                reserved_output_tokens=OUTPUT_MAX_TOKENS,
            )
        try:
            budget_tokens = int(budget_tokens)
        except Exception:
            budget_tokens = 0
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
                take = min(len(text), remaining_chars)
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
            take = min(len(text), remaining_pool)
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
        attachments: Optional[List[str]] = None,
        focus_document_id: Optional[str] = None,
        doc_pane_open: Optional[bool] = None,
        selection: Optional[Dict[str, Any]] = None,
    ) -> str:
        tokens: List[str] = []
        for tok in self.handle_message_stream(
            chat_id=chat_id,
            user_message=user_message,
            documents=documents,
            attachments=attachments,
            focus_document_id=focus_document_id,
            doc_pane_open=doc_pane_open,
            selection=selection,
            request_id=None,
        ):
            tokens.append(tok)
        return "".join(tokens)

    def handle_message_stream(
        self,
        chat_id: str,
        user_message: str,
        documents: Optional[List[str]] = None,
        attachments: Optional[List[str]] = None,
        focus_document_id: Optional[str] = None,
        doc_pane_open: Optional[bool] = None,
        selection: Optional[Dict[str, Any]] = None,
        *,
        request_id: Optional[str] = None,
    ):
        """
        Streaming variant of handle_message yielding tokens.
        """
        start = time.perf_counter()
        documents = documents or []
        attachments = attachments or []

        # Keep index alignment with `documents` for UI display (upload turns rely on it).
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

        analysis = self._analyze_query(user_message, has_docs=bool(documents))
        max_tokens = self._compute_output_max_tokens(analysis)
        summary_request = self._is_summary_request(user_message)
        compare_request = self._is_compare_request(user_message)

        # Ensure deterministic session per chat_id (required even when no docs are attached).
        self.session_mgr.get_or_create_session(chat_id, system_prompt=self.system_hint)

        selection_obj: Optional[Dict[str, Any]] = selection if isinstance(selection, dict) else None
        has_selection_text = bool(
            selection_obj
            and isinstance(selection_obj.get("text"), str)
            and str(selection_obj.get("text")).strip()
        )
        sel_file_id = (
            selection_obj.get("file_id")
            if selection_obj and isinstance(selection_obj.get("file_id"), str)
            else None
        )

        # Fill selection filename for better scope headers (never internal ids).
        selection_for_prompt = selection_obj
        if sel_file_id and selection_obj and not isinstance(selection_obj.get("filename"), str):
            try:
                rec = self.metadata_store.get_file(sel_file_id)
                name = rec.get("filename") if isinstance(rec, dict) else None
                if isinstance(name, str) and name.strip():
                    selection_for_prompt = {**selection_obj, "filename": name.strip()}
            except Exception:
                pass

        ltm_k = 2 if has_selection_text else 5
        ltm_hits = self.ltm_store.retrieve(chat_id, user_message, top_k=ltm_k)

        scope = self._resolve_scope_for_turn(
            chat_id,
            documents=documents,
            focus_document_id=focus_document_id,
            doc_pane_open=doc_pane_open,
            selection=selection_obj,
        )
        scope_mode = scope.get("scope") or "focused"
        effective_focus = scope.get("focus") if isinstance(scope.get("focus"), str) else None
        scope_doc_ids = scope.get("scope_doc_ids") if isinstance(scope.get("scope_doc_ids"), list) else []
        secondary_doc_ids = scope.get("secondary_doc_ids") if isinstance(scope.get("secondary_doc_ids"), list) else []
        chat_doc_ids = scope.get("chat_doc_ids") if isinstance(scope.get("chat_doc_ids"), list) else []

        primary_k = RAG_PRIMARY_K
        if summary_request:
            primary_k = RAG_PRIMARY_K_SUMMARY
        elif has_selection_text:
            primary_k = RAG_PRIMARY_K_SELECTION

        logger.debug(
            "Planner stream turn chat=%s summary=%s scope=%s doc_pane_open=%s focus=%s docs_payload=%d attachments=%d request_id=%s",
            chat_id,
            summary_request,
            scope_mode,
            bool(scope.get("doc_pane_open")),
            effective_focus,
            len(documents),
            len(attachments),
            request_id,
        )

        effective_focus_name: Optional[str] = None
        if effective_focus:
            try:
                rec = self.metadata_store.get_file(effective_focus)
                name = rec.get("filename") if isinstance(rec, dict) else None
                if isinstance(name, str) and name.strip():
                    effective_focus_name = name.strip()
            except Exception:
                effective_focus_name = None

        rag_hits_raw: List[Dict[str, Any]] = []
        scope_files: Optional[List[str]] = None
        query_vector: Any = None

        # Plan B ("raw_large"): skip ingestion/RAG, use rg_search + read_raw_window evidence.
        raw_large_fid = self._raw_large_file_id(chat_doc_ids)
        if raw_large_fid:
            scope_mode = "raw_large"
            effective_focus = raw_large_fid
            scope_doc_ids = []
            secondary_doc_ids = []
            try:
                rec = self.metadata_store.get_file(raw_large_fid) or {}
                name = rec.get("filename") if isinstance(rec, dict) else None
                if isinstance(name, str) and name.strip():
                    effective_focus_name = name.strip()
            except Exception:
                effective_focus_name = None
            scope_files = [effective_focus_name] if effective_focus_name else None

            rag_hits_raw.extend(
                self.raw_large_loop.build_evidence_hits(
                    user_message,
                    file_id=raw_large_fid,
                    request_id=request_id,
                )
            )
        else:
            # Stop re-embedding the same query per file: compute query embedding once per turn
            # and re-use it across dense retrieval calls (focused + secondary docs).
            #
            # Note: scope=all uses the MultiFileAgentLoop, which computes its own embedding once.
            if scope_mode != "all":
                try:
                    if (effective_focus or documents) and getattr(self.rag_store, "embedder", None):
                        query_vector = self.rag_store.embed(user_message)
                except Exception:
                    query_vector = None

            if scope_mode == "all" and scope_doc_ids:
                doc_ids_in_scope = self._uniq_file_ids(scope_doc_ids)
                if doc_ids_in_scope:
                    rag_budget_tokens = self._compute_rag_budget_tokens(
                        chat_id,
                        files_in_scope=len(doc_ids_in_scope),
                        reserved_output_tokens=max_tokens,
                    )
                    # Keep per-file retrieval stable (fair) by scaling total_k with file count.
                    total_k = max(RAG_ALL_K_TOTAL, 7 * len(doc_ids_in_scope))
                    cfg = MultiFileAgentConfig(
                        total_k=total_k,
                        dense_k=16,
                        sparse_k=16,
                        radius=2,
                        max_tokens_total=max(600, rag_budget_tokens),
                        max_tokens_per_file_floor=600,
                        repair_on_empty=True,
                        max_repairs=2000,
                    )
                    windows_map = self.agent_loop.build_evidence_windows(
                        user_message,
                        file_ids=doc_ids_in_scope,
                        config=cfg,
                        request_id=request_id,
                    )
                    rag_hits_raw.extend(self.agent_loop.windows_to_rag_hits(windows_map))

                    scope_files = []
                    for fid in doc_ids_in_scope:
                        try:
                            rec = self.metadata_store.get_file(fid)
                            name = rec.get("filename") if isinstance(rec, dict) else None
                            if isinstance(name, str) and name.strip():
                                scope_files.append(name.strip())
                        except Exception:
                            continue
                    if not scope_files:
                        scope_files = None
            elif effective_focus:
                # Focused mode: hard-scope evidence to the focused file only.
                rag_budget_tokens = self._compute_rag_budget_tokens(
                    chat_id,
                    files_in_scope=1,
                    reserved_output_tokens=max_tokens,
                )
                focus_anchors = self.agent_tools.hybrid_search(
                    user_message,
                    file_id=effective_focus,
                    top_k=primary_k,
                    query_vector=query_vector,
                    dense_k=max(primary_k, 10),
                    sparse_k=max(primary_k, 10),
                )
                focus_windows_map = self.agent_tools.windows_from_anchors(
                    focus_anchors,
                    radius=2,
                    max_tokens_per_file=max(600, rag_budget_tokens),
                )
                for w in focus_windows_map.get(effective_focus, []):
                    if not w.text:
                        continue
                    rag_hits_raw.append(
                        {
                            "doc_id": effective_focus,
                            "text": w.text,
                            "filename": w.filename,
                            "page": w.page_start,
                            "page_start": w.page_start,
                            "page_end": w.page_end,
                            "chunk_id": f"seq_window:{w.seq_start}-{w.seq_end}",
                            "score": 1.0,
                        }
                    )
            elif documents:
                doc_ids = self._uniq_file_ids(documents)
                if doc_ids:
                    rag_budget_tokens = self._compute_rag_budget_tokens(
                        chat_id,
                        files_in_scope=len(doc_ids),
                        reserved_output_tokens=max_tokens,
                    )
                    anchors: List[ChunkAnchor] = []
                    for fid in doc_ids:
                        anchors.extend(
                            self.agent_tools.hybrid_search(
                                user_message,
                                file_id=fid,
                                top_k=max(1, primary_k // max(1, len(doc_ids))),
                                query_vector=query_vector,
                                dense_k=8,
                                sparse_k=8,
                            )
                        )
                    windows_map = self.agent_tools.windows_from_anchors(
                        anchors,
                        radius=2,
                        max_tokens_per_file=max(600, int(max(600, rag_budget_tokens) // max(1, len(doc_ids)))),
                    )
                    for fid in doc_ids:
                        for w in windows_map.get(fid, []):
                            if not w.text:
                                continue
                            rag_hits_raw.append(
                                {
                                    "doc_id": fid,
                                    "text": w.text,
                                    "filename": w.filename,
                                    "page": w.page_start,
                                    "page_start": w.page_start,
                                    "page_end": w.page_end,
                                    "chunk_id": f"seq_window:{w.seq_start}-{w.seq_end}",
                                    "score": 1.0,
                                }
                            )
            else:
                rag_hits_raw.extend(self.rag_store.retrieve(user_message, chat_id=chat_id, top_k=primary_k))

        selected_rag = self._dedup_rag(rag_hits_raw)

        if scope_mode == "all" and scope_doc_ids:
            doc_ids_in_scope = self._uniq_file_ids(scope_doc_ids)
            rag_budget_tokens = self._compute_rag_budget_tokens(
                chat_id,
                files_in_scope=max(1, len(doc_ids_in_scope)),
                reserved_output_tokens=max_tokens,
            )
            total_k = max(RAG_ALL_K_TOTAL, 7 * max(1, len(doc_ids_in_scope)))
            selected_rag = self._rebalance_hits_by_file(selected_rag, max_hits=total_k)
            selected_rag = self._apply_rag_budget_balanced(
                chat_id,
                selected_rag,
                file_order=scope_doc_ids,
                budget_tokens=rag_budget_tokens,
            )
        else:
            inferred_files = len({h.get("doc_id") for h in selected_rag if isinstance(h.get("doc_id"), str) and h.get("doc_id")})
            rag_budget_tokens = self._compute_rag_budget_tokens(
                chat_id,
                files_in_scope=max(1, inferred_files),
                reserved_output_tokens=max_tokens,
            )
            selected_rag = self._apply_rag_budget_with_tokens(chat_id, selected_rag, budget_tokens=rag_budget_tokens)

        # Debug: show how many hits per file made it through budgeting (backend logs only).
        try:
            if selected_rag:
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

        context_pack = build_context_pack(
            ltm_hits=ltm_hits,
            rag_hits=selected_rag,
            selection=selection_for_prompt,
            effective_focus=effective_focus,
            effective_focus_name=effective_focus_name,
            scope_files=scope_files,
            include_selection_excerpt=not has_selection_text,
        )
        if compare_request and scope_mode == "all":
            # Keep this short to avoid burning budget; its role is to enforce balanced
            # coverage when the user prompt is underspecified ("compare").
            task_hint = (
                "TASK:\n"
                "Compare the documents in SCOPE.\n"
                "- Write a short summary for each document.\n"
                "- Then list similarities and differences.\n"
                "- Use only the EVIDENCE; do not invent details.\n"
            )
            context_pack = (task_hint + "\n" + (context_pack or "")).strip()

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

            dirty_user = (
                build_dirty_user_turn(user_message, selection=selection_for_prompt) if has_selection_text else None
            )
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

            total_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "Planner stream done chat=%s scope=%s focus=%s summary=%s docs=%d rag_hits=%d ltm=%d time_ms=%.1f request_id=%s",
                chat_id,
                scope_mode,
                effective_focus,
                summary_request,
                len(chat_doc_ids),
                len(selected_rag),
                len(ltm_hits),
                total_ms,
                request_id,
            )

        return generator()

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

        if has_docs and self._is_compare_request(query):
            target_words = max(target_words, 260)

        hint_doc = has_docs or any(k in q for k in ["document", "file", "pdf", "upload"])
        return {
            "target_words": target_words,
            "hint_doc": hint_doc,
        }


__all__ = ["InsightOrchestrator"]
