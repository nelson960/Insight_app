from __future__ import annotations

import copy
import uuid
import logging
import re
import math
import os
from pathlib import Path
from typing import Dict, Optional, List

import atexit
import json
import queue
import threading
import time
import ctypes

import llama_cpp
from llama_cpp import Llama

from backend.services.storage.sqlite_store import SQLiteMetadataStore
from backend.services.ipc_events import emit_event

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore


logger = logging.getLogger(__name__)


class LlamaSessionManager:
    _PROMPT_RENDERER_LLAMA3 = "llama3_manual_v2"
    _PROMPT_RENDERER_CHATML = "chatml_manual_v1"
    _PROMPT_RENDERER_UNKNOWN = "unsupported"
    _PROMPT_RENDERER_ID = _PROMPT_RENDERER_LLAMA3
    _PERSIST_DEBOUNCE_SEC = 0.75
    _SNAPSHOT_DEBOUNCE_SEC = 0.35
    _STATE_KIND_COMPACT = "llama_state_compact_v1"
    _KV_FILE_FORMAT = "raw_llama_state_v1"
    _OUTPUT_RESERVE_PCT = 0.10
    _CONTEXT_MARGIN_PCT = 0.02
    _MAX_INPUT_PCT = 0.65
    _COMPACT_PCT = 0.70
    """
    Single-model, multi-session manager using llama_cpp KV snapshots.

    - Loads the model once.
    - Tracks per-session KV state in memory.
    - Supports create, ask, fork, reset, delete.
    - Optimized to avoid unnecessary state loads/resets and slow KV rebuilds.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        ctx_size: int = 8192,
        gpu_layers: int = -1,
        chat_format: str = "llama-3",
        persist_dir: Optional[Path] = None,
        ltm_store=None,
        metadata_store: Optional[SQLiteMetadataStore] = None,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        self.model_path = model_path
        self.chat_format = chat_format
        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=ctx_size,
            n_gpu_layers=gpu_layers,
            chat_format=chat_format,
        )
        # Always trust the runtime context size from llama_cpp (it may differ from the
        # requested value if the backend clamps it). Using a mismatched value can cause
        # hard failures when the prompt nears the true context limit.
        actual_ctx = ctx_size
        try:
            actual_ctx = int(self.llm.n_ctx())
        except Exception:
            pass
        if actual_ctx != ctx_size:
            logger.warning("Requested ctx_size=%d but llama_cpp is using n_ctx=%d", ctx_size, actual_ctx)
        self.ctx_size = actual_ctx
        try:
            logger.info("LlamaSessionManager model ready n_ctx=%d n_batch=%d", self.ctx_size, int(self.llm.n_batch))
        except Exception:
            pass

        # Select the prompt renderer based on GGUF metadata markers.
        # This must happen after model load (so `llm.metadata` exists) and before
        # we compute model_info() (which surfaces the selected renderer).
        self._PROMPT_RENDERER_ID = self._detect_prompt_renderer_id()
        if self._PROMPT_RENDERER_ID == self._PROMPT_RENDERER_UNKNOWN:
            raise ValueError(
                "Unsupported chat template for this model. "
                "Please choose a GGUF with a known chat template (Llama-3 or Qwen/ChatML)."
            )

        self._model_info = self._build_model_info()
        try:
            info = self._model_info
            logger.info(
                "LLM loaded name=%s arch=%s ctx_train=%s ctx_runtime=%s file_type=%s renderer=%s",
                info.get("name") or "?",
                info.get("architecture") or "?",
                info.get("ctx_train") or "?",
                info.get("ctx_runtime") or "?",
                info.get("file_type") or "?",
                info.get("prompt_renderer") or "?",
            )
        except Exception:
            pass

        # session_id -> {"state": bytes, "messages": List[Dict], ...}
        self.sessions: Dict[str, Dict[str, object]] = {}
        # Per-session locks must be re-entrant because compaction uses transient model runs
        # while already inside an ask()/ask_stream() critical section.
        self._locks: Dict[str, threading.RLock] = {}
        self._model_lock = threading.Lock()

        # Track which session's KV is currently loaded in the model
        self._active_session_id: Optional[str] = None

        # Streaming cancellation (Stop button): wire llama.cpp abort callback.
        # This is the only reliable way to stop an in-flight llama_cpp generation.
        self._abort_lock = threading.Lock()
        self._abort_request_id: Optional[str] = None
        self._abort_event: Optional[threading.Event] = None
        self._abort_cb = None
        self._install_abort_callback()

        self.persist_dir = Path(persist_dir) if persist_dir else None
        self._metadata_store = metadata_store
        self.ltm_store = ltm_store
        self._persist_queue: Optional[queue.Queue[Optional[str]]] = None
        self._persist_thread: Optional[threading.Thread] = None
        self._persist_stop = threading.Event()
        self._persist_mutex = threading.Lock()
        self._persist_pending: Dict[str, Dict[str, object]] = {}
        self._persist_scheduled: set[str] = set()

        # KV snapshotting is expensive (copies hundreds of MB). To avoid blocking user-visible
        # request completion, we can defer state snapshots and persist them during idle time.
        self._snapshot_queue: Optional[queue.Queue[Optional[str]]] = None
        self._snapshot_thread: Optional[threading.Thread] = None
        self._snapshot_stop = threading.Event()
        self._snapshot_mutex = threading.Lock()
        self._snapshot_scheduled: set[str] = set()
        self._start_snapshot_worker()
        atexit.register(self._shutdown_snapshot_worker)

        if self.persist_dir:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._start_persist_worker()
            self._load_persisted_sessions()
            try:
                logger.info(
                    "LlamaSessionManager using persist_dir=%s loaded_sessions=%d",
                    self.persist_dir,
                    len(self.sessions),
                )
            except Exception:
                pass
            atexit.register(self._shutdown_persist_worker)

    # -------------------------------------------------------------------------
    # Public session API
    # -------------------------------------------------------------------------

    def model_info(self) -> Dict[str, object]:
        """
        Return a small, UI-safe snapshot of the currently loaded LLM capabilities.

        This is intentionally *not* the full GGUF metadata (which can be huge).
        """
        try:
            return dict(self._model_info)
        except Exception:
            return {}

    def _build_model_info(self) -> Dict[str, object]:
        """
        Extract a compact set of model properties for Settings/UI.

        NOTE: llama-cpp-python exposes GGUF metadata on the Llama instance as `llm.metadata`,
        but it is printed to stderr only when verbose=True. We keep our own filtered copy.
        """

        def _meta_str(key: str) -> str:
            try:
                v = (getattr(self.llm, "metadata", {}) or {}).get(key)
            except Exception:
                v = None
            if v is None:
                return ""
            try:
                return str(v)
            except Exception:
                return ""

        def _meta_int(key: str) -> Optional[int]:
            v = _meta_str(key)
            if not v:
                return None
            try:
                return int(v)
            except Exception:
                return None

        meta_arch = _meta_str("general.architecture")
        meta_name = _meta_str("general.name")
        meta_size = _meta_str("general.size_label")
        meta_file_type = _meta_int("general.file_type")
        meta_quant_ver = _meta_int("general.quantization_version")

        ctx_train: Optional[int] = None
        try:
            ctx_train = int(self.llm._model.n_ctx_train())  # type: ignore[attr-defined]
        except Exception:
            ctx_train = None
        if ctx_train is None:
            ctx_train = _meta_int("llama.context_length") or _meta_int("qwen2.context_length")

        # Chat template kind is inferred from GGUF `tokenizer.chat_template` markers.
        template = _meta_str("tokenizer.chat_template")
        if "<|start_header_id|>" in template and "<|eot_id|>" in template:
            template_kind = "llama3"
        elif "<|im_start|>" in template and "<|im_end|>" in template:
            template_kind = "chatml"
        else:
            template_kind = "unknown"

        # Tokenizer hints
        tok_model = _meta_str("tokenizer.ggml.model") or _meta_str("tokenizer.ggml.pre")
        add_bos = _meta_str("tokenizer.ggml.add_bos_token")
        bos_id = _meta_int("tokenizer.ggml.bos_token_id")
        eos_id = _meta_int("tokenizer.ggml.eos_token_id")

        return {
            "path": str(self.model_path),
            "architecture": meta_arch,
            "name": meta_name,
            "size_label": meta_size,
            "file_type": meta_file_type,
            "quantization_version": meta_quant_ver,
            "ctx_train": ctx_train,
            "ctx_runtime": int(self.ctx_size) if self.ctx_size else None,
            "chat_template_kind": template_kind,
            "chat_format": str(getattr(self, "chat_format", "") or ""),
            "prompt_renderer": str(getattr(self, "_PROMPT_RENDERER_ID", "") or ""),
            "tokenizer_model": tok_model,
            "add_bos_token": add_bos,
            "bos_token_id": bos_id,
            "eos_token_id": eos_id,
        }

    def _detect_prompt_renderer_id(self) -> str:
        """
        Infer which prompt renderer to use for the loaded model.

        We avoid llama-cpp-python "apply_chat_template" APIs (not available in our pinned
        version) and instead render prompts ourselves. To keep correctness across models,
        we detect common chat templates from GGUF metadata:
        - Llama-3 style: <|start_header_id|> ... <|eot_id|>
        - ChatML (Qwen/Qwen2.5): <|im_start|> ... <|im_end|>
        """
        try:
            meta = getattr(self.llm, "metadata", {}) or {}
        except Exception:
            meta = {}
        try:
            template = str(meta.get("tokenizer.chat_template") or "")
        except Exception:
            template = ""

        if "<|im_start|>" in template and "<|im_end|>" in template:
            return self._PROMPT_RENDERER_CHATML
        if "<|start_header_id|>" in template and "<|eot_id|>" in template:
            return self._PROMPT_RENDERER_LLAMA3
        return self._PROMPT_RENDERER_UNKNOWN

    def cancel_request(self, request_id: str) -> bool:
        """
        Best-effort cancellation of the currently running streaming request.

        Returns True if this manager accepted the cancel for the active request_id.
        """
        if not request_id:
            return False
        with self._abort_lock:
            if self._abort_request_id != request_id or self._abort_event is None:
                return False
            self._abort_event.set()
            logger.info("cancel_request accepted request_id=%s", request_id)
            return True

    def busy_state(self) -> Dict[str, object]:
        """
        Best-effort indicator for background activity.

        Used by Settings to block destructive operations (reset/clean cache) while:
        - a stream is active
        - a deferred KV snapshot is pending
        - async KV persistence is pending
        """
        with self._abort_lock:
            active_stream = self._abort_request_id is not None
            active_request_id = self._abort_request_id

        with self._snapshot_mutex:
            snapshot_scheduled = len(self._snapshot_scheduled)

        with self._persist_mutex:
            persist_scheduled = len(self._persist_scheduled)
            persist_pending = len(self._persist_pending)

        dirty_sessions = 0
        try:
            for s in self.sessions.values():
                if isinstance(s, dict) and bool(s.get("_state_dirty")):
                    dirty_sessions += 1
        except Exception:
            dirty_sessions = max(dirty_sessions, 1)

        snapshot_pending = snapshot_scheduled > 0 or dirty_sessions > 0
        persist_pending_any = persist_scheduled > 0 or persist_pending > 0
        busy = bool(active_stream or snapshot_pending or persist_pending_any)
        return {
            "busy": busy,
            "ctx_size": int(self.ctx_size) if self.ctx_size else None,
            "active_stream": active_stream,
            "active_request_id": active_request_id,
            "snapshot_pending": snapshot_pending,
            "snapshot_scheduled": snapshot_scheduled,
            "persist_pending": persist_pending_any,
            "persist_scheduled": persist_scheduled,
            "persist_queue": persist_pending,
            "dirty_sessions": dirty_sessions,
        }

    def create_session(self, system_prompt: Optional[str] = None) -> str:
        session_id = str(uuid.uuid4())
        self._locks.setdefault(session_id, threading.RLock())
        with self._model_lock:
            self.llm.reset()
            messages: List[Dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            # Ensure session KV matches our rendered prompt format (system wrapper included).
            self._prefill_chat_messages(messages)
            state_obj = self._save_state_compact()
            prompt = self._render_prompt(messages, add_generation_prompt=False)
            prompt_tokens = self._tokenize_prompt(prompt)
        session_payload = {
            "state": state_obj,
            "messages": messages,
            "compacted": False,
            "_prompt_tokens": prompt_tokens,
            "prompt_renderer": self._PROMPT_RENDERER_ID,
        }
        self.sessions[session_id] = session_payload
        self._persist_session(session_id, session_payload)
        return session_id

    def get_or_create_session(self, session_id: str, system_prompt: Optional[str] = None) -> str:
        """
        Deterministically return an existing session or create one with the given ID.
        """
        if session_id in self.sessions:
            return session_id

        self._locks.setdefault(session_id, threading.RLock())
        with self._model_lock:
            self.llm.reset()
            messages: List[Dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            self._prefill_chat_messages(messages)
            state_obj = self._save_state_compact()
            prompt = self._render_prompt(messages, add_generation_prompt=False)
            prompt_tokens = self._tokenize_prompt(prompt)
        session_payload = {
            "state": state_obj,
            "messages": messages,
            "compacted": False,
            "_prompt_tokens": prompt_tokens,
            "prompt_renderer": self._PROMPT_RENDERER_ID,
        }
        self.sessions[session_id] = session_payload
        self._persist_session(session_id, session_payload)
        return session_id

    def seed_session_messages(
        self,
        session_id: str,
        *,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
    ) -> None:
        """
        Initialize (or overwrite) a session's clean message history and KV state.

        This is used for "branching" to a new chat/card: we create a new chat_id and
        seed it with a single selected message (user or assistant) without running
        any model generation.
        """
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id is required")
        if not isinstance(messages, list):
            raise ValueError("messages must be a list")

        lock = self._locks.setdefault(session_id, threading.RLock())
        with lock:
            # Ensure the session exists and has a system message.
            self.get_or_create_session(session_id, system_prompt=system_prompt)
            session = self.sessions.get(session_id)
            if session is None:
                raise RuntimeError(f"Failed to create session {session_id}")

            # Preserve the current system prompt (first system message) unless the caller
            # explicitly provided a system message in `messages`.
            normalized: List[Dict[str, str]] = []
            if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
                normalized = messages
            else:
                base_system: List[Dict[str, str]] = [
                    m for m in (session.get("messages") or []) if isinstance(m, dict) and m.get("role") == "system"
                ]
                normalized = base_system + messages

            # Apply to KV/messages.
            with self._model_lock:
                self._ensure_session_loaded(session_id, session)
                self._commit_messages_to_session(session_id, session, normalized)

            self.sessions[session_id] = session
            self._persist_session(session_id, session)

    def ask_stream(
        self,
        session_id: str,
        message: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.2,
        request_id: Optional[str] = None,
    ):
        """
        Streaming chat entry: yields tokens and updates the KV snapshot after completion.
        """
        logger.info(
            "ask_stream start chat=%s request_id=%s prompt_chars=%d",
            session_id,
            request_id or "-",
            len(message or ""),
        )
        lock = self._locks.setdefault(session_id, threading.RLock())
        with lock:
            session = self.sessions.get(session_id)
            if session is None:
                raise ValueError(f"Unknown session: {session_id}")
            self._ensure_prompt_renderer(session_id, session)

            if self.persist_dir:
                kv_path = self.persist_dir / f"{session_id}.kv"
                if not kv_path.exists():
                    logger.warning("KV missing for %s, resetting to system-only state", session_id)
                    system_messages = [m for m in session.get("messages", []) if m.get("role") == "system"]
                    with self._model_lock:
                        self.llm.reset()
                        self._prefill_chat_messages(system_messages)
                        fresh_state = self._save_state_compact()
                        prompt = self._render_prompt(system_messages, add_generation_prompt=False)
                        prompt_tokens = self._tokenize_prompt(prompt)
                    session = {
                        "state": fresh_state,
                        "messages": system_messages,
                        "compacted": False,
                        "_prompt_tokens": prompt_tokens,
                        "prompt_renderer": self._PROMPT_RENDERER_ID,
                    }
                    self.sessions[session_id] = session
                    self._persist_session(session_id, session)

            message = self._preflight_compact_and_budget(
                session_id,
                session,
                message,
                reserved_max_tokens=max_tokens,
            )

            reply_parts: List[str] = []
            cancelled = False
            with self._model_lock:
                # Hold the global model lock for the entire streaming generation.
                # llama_cpp is not safe to interleave with other calls while a stream
                # iterator is active.
                self._ensure_session_loaded(session_id, session)
                # If the previous turn deferred the expensive KV snapshot, take it now so
                # `base_state` always matches the current clean transcript.
                if bool(session.get("_state_dirty")):
                    session["state"] = self._save_state_compact()
                    session["_state_dirty"] = False
                    session.pop("_state_dirty_at", None)
                base_state = session.get("state")
                clean_messages: List[Dict[str, str]] = list(session.get("messages", []))
                clean_tokens: List[int] = list(session.get("_prompt_tokens") or [])
                if not clean_tokens:
                    clean_prompt = self._render_prompt(clean_messages, add_generation_prompt=False)
                    clean_tokens = self._tokenize_prompt(clean_prompt)
                run_messages: List[Dict[str, str]] = list(clean_messages)
                run_messages.append({"role": "user", "content": message})

                # Avoid llama.cpp hard failures when prompt_tokens + max_tokens > ctx_size.
                run_prompt = self._render_prompt(run_messages, add_generation_prompt=True)
                run_tokens = self._tokenize_prompt(run_prompt)
                if clean_tokens and run_tokens[: len(clean_tokens)] == clean_tokens:
                    delta = run_tokens[len(clean_tokens) :]
                    if delta:
                        self._eval_tokens(delta)
                else:
                    self.llm.reset()
                    self._eval_tokens(run_tokens)

                prompt_tokens = len(run_tokens)
                max_tokens = self._clamp_max_tokens(session_id, prompt_tokens, max_tokens)

                logger.info(
                    "ask_stream llama start chat=%s msgs=%d max_tokens=%d",
                    session_id,
                    len(run_messages),
                    max_tokens,
                )
                with self._abort_lock:
                    # Activate abort only for the duration of this generation.
                    self._abort_request_id = request_id
                    self._abort_event = threading.Event() if request_id else None

                try:
                    stream = self._create_completion_from_state(
                        prompt_tokens=run_tokens,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stream=True,
                        stop=self._stop_markers(),
                    )

                    for chunk in stream:
                        token = (chunk.get("choices") or [{}])[0].get("text") or ""
                        if not token:
                            continue
                        if not reply_parts:
                            logger.info("ask_stream first token chat=%s request_id=%s", session_id, request_id or "-")
                        logger.debug(
                            "stream token (len=%d): %s",
                            len(token),
                            token[:80].replace("\n", "\\n"),
                        )
                        reply_parts.append(token)
                        yield token
                except Exception as exc:
                    # If Stop was requested, llama.cpp abort callback will terminate generation.
                    # Treat this as a clean cancellation.
                    with self._abort_lock:
                        if self._abort_event is not None and self._abort_event.is_set():
                            cancelled = True
                    if cancelled:
                        logger.info("ask_stream cancelled chat=%s request_id=%s", session_id, request_id or "-")
                    else:
                        logger.exception("ask_stream error chat=%s request_id=%s", session_id, request_id or "-")
                        raise
                finally:
                    # 🔔 fast UI signal that token streaming ended
                    try:
                        emit_event(
                            "llm_stream_end",
                            chat_id=session_id,
                            request_id=request_id or "",
                        )
                    except Exception:
                        pass
                    with self._abort_lock:
                        self._abort_request_id = None
                        self._abort_event = None

                reply = "".join(reply_parts)
                logger.info(
                    "ask_stream llama done chat=%s request_id=%s tokens=%d cancelled=%s",
                    session_id,
                    request_id or "-",
                    len(reply_parts),
                    cancelled,
                )

                # Restore clean KV before committing.
                try:
                    self.llm.reset()
                    if base_state is not None:
                        self._load_state_compact(base_state)
                    self._active_session_id = session_id
                except Exception:
                    self._active_session_id = None

                commit_messages: List[Dict[str, str]] = list(clean_messages)
                commit_messages.append({"role": "user", "content": message})
                if reply:
                    commit_messages.append({"role": "assistant", "content": reply})
                self._commit_messages_to_session(session_id, session, commit_messages)

            self.sessions[session_id] = session
            self._persist_session(session_id, session)

    # -------------------------------------------------------------------------
    # Clean KV + ephemeral context pack API (preferred for RAG/doc chat)
    # -------------------------------------------------------------------------

    def ask_with_context(
        self,
        session_id: str,
        *,
        user_text: str,
        run_user_text: Optional[str] = None,
        context_pack: str = "",
        max_tokens: int = 256,
        temperature: float = 0.2,
    ) -> str:
        """
        Chat that keeps the persistent KV clean.

        - Persistent (saved to session messages + KV): system + clean user + clean assistant
        - Ephemeral (used only for this turn, not persisted): context_pack

        Implementation: fork -> generate -> restore -> commit (append-only).
        """
        lock = self._locks.setdefault(session_id, threading.RLock())
        with lock:
            session = self.sessions.get(session_id)
            if session is None:
                raise ValueError(f"Unknown session: {session_id}")
            self._ensure_prompt_renderer(session_id, session)

            # One-time migration: older sessions persisted "turn_prompt" blobs into
            # user messages, which pollutes KV and mixes context across turns.
            # For v2, we keep KV clean and inject retrieval/doc context ephemerally.
            self._ensure_clean_kv_mode(session_id, session)

            if self.persist_dir:
                kv_path = self.persist_dir / f"{session_id}.kv"
                if not kv_path.exists():
                    logger.warning("KV missing for %s, resetting to system-only state", session_id)
                    system_messages = [m for m in session.get("messages", []) if m.get("role") == "system"]
                    with self._model_lock:
                        self.llm.reset()
                        self._prefill_chat_messages(system_messages)
                        fresh_state = self._save_state_compact()
                        prompt = self._render_prompt(system_messages, add_generation_prompt=False)
                        prompt_tokens = self._tokenize_prompt(prompt)
                    session = {
                        "state": fresh_state,
                        "messages": system_messages,
                        "compacted": False,
                        "_prompt_tokens": prompt_tokens,
                        "prompt_renderer": self._PROMPT_RENDERER_ID,
                    }
                    self.sessions[session_id] = session
                    self._persist_session(session_id, session)

            clean_user = (user_text or "").strip()
            if not clean_user:
                return ""

            # Dirty run may include selection excerpts or other per-turn guidance.
            # This must NOT be persisted into the clean session transcript/KV.
            run_user = (run_user_text or clean_user).strip()
            if not run_user:
                run_user = clean_user

            context_pack = (context_pack or "").strip()
            context_pack = self._budget_context_pack(
                session_id,
                session,
                user_text=run_user,
                context_pack=context_pack,
                reserved_max_tokens=max_tokens,
            )

            with self._model_lock:
                self._ensure_session_loaded(session_id, session)
                if bool(session.get("_state_dirty")):
                    session["state"] = self._save_state_compact()
                    session["_state_dirty"] = False
                    session.pop("_state_dirty_at", None)
                base_state = session.get("state")
                clean_messages: List[Dict[str, str]] = list(session.get("messages", []))
                clean_tokens: List[int] = list(session.get("_prompt_tokens") or [])
                if not clean_tokens:
                    clean_prompt = self._render_prompt(clean_messages, add_generation_prompt=False)
                    clean_tokens = self._tokenize_prompt(clean_prompt)

                run_messages: List[Dict[str, str]] = list(clean_messages)
                if context_pack:
                    run_messages.append({"role": "system", "content": self._format_context_pack(context_pack)})
                run_messages.append({"role": "user", "content": run_user})

                run_prompt = self._render_prompt(run_messages, add_generation_prompt=True)
                run_tokens = self._tokenize_prompt(run_prompt)
                if clean_tokens and run_tokens[: len(clean_tokens)] == clean_tokens:
                    delta = run_tokens[len(clean_tokens) :]
                    if delta:
                        self._eval_tokens(delta)
                else:
                    self.llm.reset()
                    self._eval_tokens(run_tokens)

                prompt_tokens = len(run_tokens)
                requested_max_tokens = int(max_tokens)
                max_tokens = self._clamp_max_tokens(session_id, prompt_tokens, max_tokens)

                # If the model hits max_tokens, the reply can look abruptly cut off.
                # For "normal" answers, allow a larger hard cap so the model can reach
                # an end-of-turn stop token naturally.
                hard_total = int(max_tokens)
                if int(max_tokens) >= 384 and self.ctx_size:
                    margin = 128
                    available_total = int(self.ctx_size) - int(prompt_tokens) - int(margin)
                    if available_total > 0:
                        hard_total = min(int(max_tokens) * 2, 1024, int(available_total))
                logger.info(
                    "ask_with_context output_budget chat=%s prompt_tokens=%d requested=%d clamped=%d hard_total=%d ctx=%d",
                    session_id,
                    prompt_tokens,
                    requested_max_tokens,
                    int(max_tokens),
                    int(hard_total),
                    int(self.ctx_size or 0),
                )
                session["last_output_budget"] = int(hard_total)
                session["last_reserved_output"] = int(max_tokens)

                out = self._create_completion_from_state(
                    prompt_tokens=run_tokens,
                    max_tokens=int(hard_total),
                    temperature=temperature,
                    stream=False,
                    stop=self._stop_markers(),
                )
                reply = (out.get("choices") or [{}])[0].get("text") or ""

                # Restore clean KV before committing.
                try:
                    self.llm.reset()
                    if base_state is not None:
                        self._load_state_compact(base_state)
                    self._active_session_id = session_id
                except Exception:
                    # If restore fails, fallback to a rebuild commit below.
                    self._active_session_id = None

                commit_messages: List[Dict[str, str]] = list(clean_messages)
                commit_messages.append({"role": "user", "content": clean_user})
                if reply:
                    commit_messages.append({"role": "assistant", "content": reply})

                self._commit_messages_to_session(session_id, session, commit_messages)
                self.sessions[session_id] = session
                self._persist_session(session_id, session)
                return reply

    def ask_stream_with_context(
        self,
        session_id: str,
        *,
        user_text: str,
        run_user_text: Optional[str] = None,
        context_pack: str = "",
        max_tokens: int = 256,
        temperature: float = 0.2,
        request_id: Optional[str] = None,
    ):
        """
        Streaming variant of `ask_with_context()`.

        Yields tokens from the "dirty" run (which includes ephemeral context pack),
        but commits only clean user + assistant text to persistent KV on completion.
        """
        logger.info(
            "ask_stream_with_context start chat=%s request_id=%s user_chars=%d run_user_chars=%d ctx_chars=%d",
            session_id,
            request_id or "-",
            len(user_text or ""),
            len(run_user_text or ""),
            len(context_pack or ""),
        )
        lock = self._locks.setdefault(session_id, threading.RLock())
        with lock:
            session = self.sessions.get(session_id)
            if session is None:
                raise ValueError(f"Unknown session: {session_id}")

            # One-time migration to "clean KV" mode; see ask_with_context docstring.
            self._ensure_clean_kv_mode(session_id, session)

            if self.persist_dir:
                kv_path = self.persist_dir / f"{session_id}.kv"
                if not kv_path.exists():
                    logger.warning("KV missing for %s, resetting to system-only state", session_id)
                    system_messages = [m for m in session.get("messages", []) if m.get("role") == "system"]
                    with self._model_lock:
                        self.llm.reset()
                        self._prefill_chat_messages(system_messages)
                        fresh_state = self._save_state_compact()
                        prompt = self._render_prompt(system_messages, add_generation_prompt=False)
                        prompt_tokens = self._tokenize_prompt(prompt)
                    session = {
                        "state": fresh_state,
                        "messages": system_messages,
                        "compacted": False,
                        "_prompt_tokens": prompt_tokens,
                        "prompt_renderer": self._PROMPT_RENDERER_ID,
                    }
                    self.sessions[session_id] = session
                    self._persist_session(session_id, session)

            clean_user = (user_text or "").strip()
            if not clean_user:
                return

            run_user = (run_user_text or clean_user).strip()
            if not run_user:
                run_user = clean_user

            context_pack = (context_pack or "").strip()
            context_pack = self._budget_context_pack(
                session_id,
                session,
                user_text=run_user,
                context_pack=context_pack,
                reserved_max_tokens=max_tokens,
            )

            reply_parts: List[str] = []
            cancelled = False

            with self._model_lock:
                self._ensure_session_loaded(session_id, session)
                if bool(session.get("_state_dirty")):
                    session["state"] = self._save_state_compact()
                    session["_state_dirty"] = False
                    session.pop("_state_dirty_at", None)
                base_state = session.get("state")
                clean_messages: List[Dict[str, str]] = list(session.get("messages", []))
                clean_tokens: List[int] = list(session.get("_prompt_tokens") or [])
                if not clean_tokens:
                    clean_prompt = self._render_prompt(clean_messages, add_generation_prompt=False)
                    clean_tokens = self._tokenize_prompt(clean_prompt)

                run_messages: List[Dict[str, str]] = list(clean_messages)
                if context_pack:
                    run_messages.append({"role": "system", "content": self._format_context_pack(context_pack)})
                run_messages.append({"role": "user", "content": run_user})

                run_prompt = self._render_prompt(run_messages, add_generation_prompt=True)
                run_tokens = self._tokenize_prompt(run_prompt)
                if clean_tokens and run_tokens[: len(clean_tokens)] == clean_tokens:
                    delta = run_tokens[len(clean_tokens) :]
                    if delta:
                        self._eval_tokens(delta)
                else:
                    self.llm.reset()
                    self._eval_tokens(run_tokens)
	
                prompt_tokens = len(run_tokens)
                requested_max_tokens = int(max_tokens)
                max_tokens = self._clamp_max_tokens(session_id, prompt_tokens, max_tokens)

                with self._abort_lock:
                    self._abort_request_id = request_id
                    self._abort_event = threading.Event() if request_id else None

                t_gen_start: Optional[float] = None
                t_first_token: Optional[float] = None
                t_gen_end: Optional[float] = None

                try:
                    # Allow a larger hard cap for "normal" answers so we don't cut off
                    # mid-thought when the initial dynamic output budget is small.
                    hard_total = int(max_tokens)
                    if int(max_tokens) >= 384 and self.ctx_size:
                        margin = 128
                        available_total = int(self.ctx_size) - int(prompt_tokens) - int(margin)
                        if available_total > 0:
                            hard_total = min(int(max_tokens) * 2, 1024, int(available_total))
                    logger.info(
                        "ask_stream_with_context output_budget chat=%s request_id=%s prompt_tokens=%d requested=%d clamped=%d hard_total=%d ctx=%d",
                        session_id,
                        request_id or "-",
                        prompt_tokens,
                        requested_max_tokens,
                        int(max_tokens),
                        int(hard_total),
                        int(self.ctx_size or 0),
                    )
                    session["last_output_budget"] = int(hard_total)
                    session["last_reserved_output"] = int(max_tokens)

                    stream = self._create_completion_from_state(
                        prompt_tokens=run_tokens,
                        max_tokens=int(hard_total),
                        temperature=temperature,
                        stream=True,
                        stop=self._stop_markers(),
                    )

                    # Measure decode speed excluding KV persistence and prompt build work.
                    # We start the timer immediately before consuming the generator so TTFT is
                    # meaningful and the throughput reflects user-perceived streaming.
                    t_gen_start = time.perf_counter()
                    for chunk in stream:
                        token = (chunk.get("choices") or [{}])[0].get("text") or ""
                        if not token:
                            continue
                        if t_first_token is None:
                            t_first_token = time.perf_counter()
                        reply_parts.append(token)
                        yield token
                except Exception:
                    with self._abort_lock:
                        if self._abort_event is not None and self._abort_event.is_set():
                            cancelled = True
                    if cancelled:
                        logger.info("ask_stream_with_context cancelled chat=%s request_id=%s", session_id, request_id or "-")
                    else:
                        logger.exception(
                            "ask_stream_with_context error chat=%s request_id=%s",
                            session_id,
                            request_id or "-",
                        )
                        raise
                finally:
                    if t_gen_end is None:
                        t_gen_end = time.perf_counter()
                    # 🔔 NEW: notify UI immediately when streaming stops
                    try:
                        emit_event(
                            "llm_stream_end",
                            chat_id=session_id,
                            request_id=request_id or "",
                        )
                    except Exception:
                        # Never let IPC events affect core chat flow
                        pass

                    with self._abort_lock:
                        self._abort_request_id = None
                        self._abort_event = None
                reply = "".join(reply_parts)
                gen_tokens = len(reply_parts)
                session["last_gen_tokens"] = int(gen_tokens)
                # "TTFT" = time-to-first-token; "tps" = tokens/sec after first token.
                # These are best-effort and omitted if we cannot compute them safely.
                if t_gen_start is not None and t_first_token is not None and t_gen_end is not None and gen_tokens > 0:
                    ttft_ms = max(0.0, (t_first_token - t_gen_start) * 1000.0)
                    gen_s = max(0.0, float(t_gen_end - t_first_token))
                    if gen_s > 1e-6:
                        tps = float(gen_tokens) / gen_s
                        if math.isfinite(tps):
                            session["last_gen_tps"] = round(tps, 2)
                        else:
                            session.pop("last_gen_tps", None)
                    else:
                        session.pop("last_gen_tps", None)
                    if math.isfinite(ttft_ms):
                        session["last_ttft_ms"] = int(round(ttft_ms))
                    else:
                        session.pop("last_ttft_ms", None)
                else:
                    session.pop("last_gen_tps", None)
                    session.pop("last_ttft_ms", None)
                logger.info(
                    "ask_stream_with_context llama done chat=%s request_id=%s tokens=%d cancelled=%s",
                    session_id,
                    request_id or "-",
                    gen_tokens,
                    cancelled,
                )
                try:
                    emit_event(
                        "llm_generation_done",
                        chat_id=session_id,
                        request_id=request_id or "",
                        tokens=gen_tokens,
                        cancelled=bool(cancelled),
                    )
                except Exception:
                    # Never let IPC events affect core chat flow.
                    pass

                # Restore clean KV before committing.
                try:
                    self.llm.reset()
                    if base_state is not None:
                        self._load_state_compact(base_state)
                    self._active_session_id = session_id
                except Exception:
                    self._active_session_id = None

                commit_messages: List[Dict[str, str]] = list(clean_messages)
                commit_messages.append({"role": "user", "content": clean_user})
                if reply:
                    commit_messages.append({"role": "assistant", "content": reply})

                self._commit_messages_to_session(session_id, session, commit_messages)

            self.sessions[session_id] = session
            self._persist_session(session_id, session)

    # -------------------------------------------------------------------------
    # Abort callback wiring
    # -------------------------------------------------------------------------

    def _install_abort_callback(self) -> None:
        """
        Install a llama.cpp abort callback (llama_cpp 0.3.x) that checks a threading.Event.

        This must stay alive for the lifetime of the process, so we store the CFUNCTYPE
        object on self.
        """

        def _should_abort(_userdata: ctypes.c_void_p) -> bool:
            ev = self._abort_event
            return bool(ev is not None and ev.is_set())

        try:
            self._abort_cb = llama_cpp.ggml_abort_callback(_should_abort)
            llama_cpp.llama_set_abort_callback(self.llm.ctx, self._abort_cb, None)
            logger.info("llama.cpp abort callback installed")
        except Exception as exc:
            # Streaming will still work, but Stop won't stop model compute.
            logger.warning("Failed to install llama.cpp abort callback: %s", exc)

    def _format_context_pack(self, context_pack: str) -> str:
        # This is injected as a system message for a single generation only.
        # Keep this string stable so tokenization/prefix-matching behaves predictably.
        return "CONTEXT PACK (ephemeral; do not store in history):\n" + (context_pack or "").strip()

    def _budget_context_pack(
        self,
        session_id: str,
        session: Dict[str, object],
        *,
        user_text: str,
        context_pack: str,
        reserved_max_tokens: int,
    ) -> str:
        """
        Ensure the ephemeral context pack fits in the remaining context window.

        This is a safety net: orchestrator-level budgeting is approximate; this must
        prevent llama.cpp hard failures near the context limit.
        """
        if not context_pack:
            return ""
        if not self.ctx_size:
            return context_pack

        try:
            ctx_size = int(self.ctx_size)
        except Exception:
            ctx_size = 0
        if ctx_size <= 0:
            return context_pack

        margin = int(ctx_size * self._CONTEXT_MARGIN_PCT)
        reserved_output = int(ctx_size * self._OUTPUT_RESERVE_PCT)

        # Compact if the clean session is already near full.
        self._maybe_compact(session_id, session, force=False)

        # If even the clean turn (user + reserved output) won't fit, force-compaction once.
        for attempt in range(2):
            history: List[Dict[str, str]] = list(session.get("messages", []))
            # Include an empty system message to account for chat-template overhead of the pack.
            probe = history + [{"role": "system", "content": ""}, {"role": "user", "content": user_text}]
            try:
                with self._model_lock:
                    base_tokens = self._count_chat_prompt_tokens(probe)
            except Exception:
                base_tokens = self._estimate_tokens(probe)

            available = int(ctx_size) - int(base_tokens) - int(reserved_output) - int(margin)
            if available <= 0:
                if attempt == 0:
                    self._maybe_compact(session_id, session, force=True)
                    continue
                return ""
            break

        max_ephemeral = max(1, int(available * self._MAX_INPUT_PCT))
        trimmed = self._truncate_text_head_to_tokens(context_pack, max(1, max_ephemeral))
        # Validate and shrink further if the template overhead makes it spill.
        for _ in range(3):
            history = list(session.get("messages", []))
            candidate_msgs = history + [
                {"role": "system", "content": self._format_context_pack(trimmed)},
                {"role": "user", "content": user_text},
            ]
            try:
                with self._model_lock:
                    prompt_tokens = self._count_chat_prompt_tokens(candidate_msgs)
            except Exception:
                prompt_tokens = self._estimate_tokens(candidate_msgs)
            if prompt_tokens + reserved_output + margin <= self.ctx_size:
                return trimmed
            available = max(1, int(available * 0.7))
            trimmed = self._truncate_text_head_to_tokens(trimmed, available)
        return trimmed

    def _commit_messages_to_session(
        self,
        session_id: str,
        session: Dict[str, object],
        commit_messages: List[Dict[str, str]],
    ) -> None:
        """
        Update persistent session messages + KV to match `commit_messages`.

        Must be called while holding `_model_lock`.
        """
        prev_messages: List[Dict[str, str]] = list(session.get("messages", []))
        prev_tokens = session.get("_prompt_tokens")
        if not isinstance(prev_tokens, list):
            prev_prompt = self._render_prompt(prev_messages, add_generation_prompt=False)
            prev_tokens = self._tokenize_prompt(prev_prompt)

        next_prompt = self._render_prompt(commit_messages, add_generation_prompt=False)
        next_tokens = self._tokenize_prompt(next_prompt)

        if prev_tokens and next_tokens[: len(prev_tokens)] == prev_tokens:
            delta = next_tokens[len(prev_tokens) :]
            if delta:
                if self.ctx_size and len(next_tokens) > int(self.ctx_size):
                    raise ValueError(
                        f"Requested tokens ({len(next_tokens)}) exceed context window of {self.ctx_size}"
                    )
                self._eval_tokens(delta)
        elif not prev_tokens:
            # First commit (system-only -> first turn) is still append-only.
            if self.ctx_size and len(next_tokens) > int(self.ctx_size):
                raise ValueError(f"Requested tokens ({len(next_tokens)}) exceed context window of {self.ctx_size}")
            self._eval_tokens(next_tokens)
        else:
            # Fallback: rebuild KV from scratch if we can't prove prefix relationship.
            self.llm.reset()
            self._prefill_chat_messages(commit_messages)

        session["messages"] = commit_messages
        session["kv_mode"] = "clean_v2"
        session["_prompt_tokens"] = next_tokens
        session["prompt_renderer"] = self._PROMPT_RENDERER_ID
        self._active_session_id = session_id
        # IMPORTANT: do not call `_save_state_compact()` here. Copying the full KV state can
        # take hundreds of milliseconds to seconds on large contexts, and would block the
        # user-visible request completion (especially at the end of streaming).
        #
        # Instead, mark the state as dirty and snapshot/persist from the background worker
        # during idle time. We will still snapshot synchronously when needed (e.g. before
        # switching away) to preserve correctness.
        session["_state_dirty"] = True
        session["_state_dirty_at"] = time.monotonic()
        self._schedule_snapshot(session_id)

    def _tokenize_prompt(self, prompt: str) -> List[int]:
        """
        Tokenize a prompt containing llama.cpp special tokens (e.g., <|eot_id|>).

        Must be called while holding `_model_lock`.
        """
        data = prompt.encode("utf-8")
        try:
            return list(self.llm.tokenize(data, add_bos=False, special=True))
        except TypeError:
            # Older llama-cpp-python uses positional args (text, add_bos, special).
            return list(self.llm.tokenize(data, False, True))

    def _eval_tokens(self, tokens: List[int]) -> None:
        """
        Eval token ids into the current model state, chunked to avoid llama.cpp batch issues.

        Must be called while holding `_model_lock`.
        """
        if not tokens:
            return
        n_batch = int(getattr(self.llm, "n_batch", 512) or 512)
        # Be conservative: large batches near the end of the context can trigger shape mismatches.
        step = max(1, min(n_batch, 512))
        for i in range(0, len(tokens), step):
            self.llm.eval(tokens[i : i + step])

    def _create_completion_from_state(
        self,
        *,
        prompt_tokens: Optional[List[int]] = None,
        max_tokens: int,
        temperature: float,
        stream: bool,
        stop: Optional[List[str]] = None,
    ):
        """
        Generate a completion from the *current* model KV/state.

        IMPORTANT: We must *not* let llama-cpp-python reset the model state. In
        llama_cpp 0.3.16, `create_completion()` does not expose a "reset=False"
        path, and can also re-evaluate prompt tokens internally (undoing our
        careful chunked eval and causing shape mismatch errors).

        For performance + correctness, generate directly from the current KV via
        `Llama.generate(..., reset=False)` and stop on the chat EOT markers.
        """
        stop = stop or []
        stop_token_ids: set[int] = set()
        for s in stop:
            if not isinstance(s, str) or not s:
                continue
            try:
                toks = list(self.llm.tokenize(s.encode("utf-8"), add_bos=False, special=True))
            except TypeError:
                toks = list(self.llm.tokenize(s.encode("utf-8"), False, True))
            if len(toks) == 1:
                try:
                    stop_token_ids.add(int(toks[0]))
                except Exception:
                    pass
        try:
            stop_token_ids.add(int(self.llm.token_eos()))
        except Exception:
            pass

        def _iter_tokens():
            produced = 0
            try:
                gen = self.llm.generate(
                    [],
                    temp=float(temperature),
                    reset=False,
                )
                for tok_id in gen:
                    try:
                        tok_i = int(tok_id)
                    except Exception:
                        continue
                    if tok_i in stop_token_ids:
                        break
                    piece = self.llm.detokenize([tok_i]).decode("utf-8", errors="ignore")
                    if not piece:
                        continue
                    produced += 1
                    yield piece
                    if produced >= int(max_tokens):
                        break
            except Exception:
                # If direct generate fails for any reason, fall back to create_completion.
                # This is slower and may reset/eval internally, but keeps the system usable.
                prompt = prompt_tokens if prompt_tokens else ""
                out = self.llm.create_completion(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True,
                    stop=stop,
                )
                for chunk in out:
                    text = (chunk.get("choices") or [{}])[0].get("text") or ""
                    if text:
                        yield text

        if stream:
            def _stream():
                for piece in _iter_tokens():
                    yield {"choices": [{"text": piece}]}

            return _stream()

        text = "".join(_iter_tokens())
        return {"choices": [{"text": text}]}

    def _prefill_chat_messages(self, messages: List[Dict[str, str]]) -> None:
        """
        Prefill the model KV for a full chat transcript without generating new tokens.

        This codebase targets llama_cpp versions without a public chat-template renderer,
        so we pre-render a stable prompt (renderer selected per-model) and prefill via
        `eval()` only.
        """
        prompt = self._render_prompt(messages, add_generation_prompt=False)
        tokens = self._tokenize_prompt(prompt)
        if self.ctx_size and len(tokens) > int(self.ctx_size):
            raise ValueError(f"Requested tokens ({len(tokens)}) exceed context window of {self.ctx_size}")
        self._eval_tokens(tokens)

    def _render_prompt(self, messages: List[Dict[str, str]], *, add_generation_prompt: bool) -> str:
        renderer = str(getattr(self, "_PROMPT_RENDERER_ID", "") or self._PROMPT_RENDERER_LLAMA3)
        if renderer == self._PROMPT_RENDERER_CHATML:
            return self._render_chatml_prompt(messages, add_generation_prompt=add_generation_prompt)
        if renderer == self._PROMPT_RENDERER_UNKNOWN:
            raise ValueError("Unsupported chat template for this model.")
        return self._render_llama3_prompt(messages, add_generation_prompt=add_generation_prompt)

    def _stop_markers(self) -> List[str]:
        """
        Stop markers for the current prompt renderer.

        NOTE: `_create_completion_from_state()` also always stops on the model EOS token id.
        """
        renderer = str(getattr(self, "_PROMPT_RENDERER_ID", "") or self._PROMPT_RENDERER_LLAMA3)
        if renderer == self._PROMPT_RENDERER_CHATML:
            # ChatML/Qwen-style turns end with <|im_end|>. Some models also use <|endoftext|>.
            return ["<|im_end|>", "<|endoftext|>"]
        return ["<|eot_id|>", "<|end_of_text|>"]

    def _render_chatml_prompt(self, messages: List[Dict[str, str]], *, add_generation_prompt: bool) -> str:
        """
        Minimal ChatML prompt renderer (Qwen/Qwen2.5 instruct).

        Matches GGUF chat templates of the form:
          <|im_start|>{role}\n{content}<|im_end|>\n
        """
        system_message = ""
        rest = list(messages or [])
        if rest and rest[0].get("role") == "system":
            system_message = rest[0].get("content") or ""
            rest = rest[1:]

        out: List[str] = []
        # Always include a system block for stability (even if empty).
        out.append("<|im_start|>system\n")
        out.append(system_message)
        out.append("<|im_end|>\n")

        for msg in rest:
            role = (msg.get("role") or "user").strip()
            content = msg.get("content") or ""
            out.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")

        if add_generation_prompt:
            out.append("<|im_start|>assistant\n")

        return "".join(out)

    def _render_llama3_prompt(self, messages: List[Dict[str, str]], *, add_generation_prompt: bool) -> str:
        """
        Minimal Llama-3 prompt renderer compatible with llama.cpp tokenizers.

        Keep this minimal + stable to maximize prefix-matching.
        """
        system_message = ""
        rest = list(messages or [])
        if rest and rest[0].get("role") == "system":
            system_message = rest[0].get("content") or ""
            rest = rest[1:]

        out: List[str] = []
        out.append("<|begin_of_text|>")
        out.append("<|start_header_id|>system<|end_header_id|>\n\n")
        out.append(system_message)
        out.append("<|eot_id|>")

        for msg in rest:
            role = (msg.get("role") or "user").strip()
            content = msg.get("content") or ""
            out.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>")

        if add_generation_prompt:
            out.append("<|start_header_id|>assistant<|end_header_id|>\n\n")

        return "".join(out)

    def _looks_like_legacy_turn_prompt(self, messages: List[Dict[str, str]]) -> bool:
        """
        Detect older persisted user prompts that contained injected blocks like:
        - "Documents:"
        - "Long-Term Memory:"
        - "Context:"
        """
        for m in messages or []:
            if m.get("role") != "user":
                continue
            content = (m.get("content") or "").strip()
            if not content:
                continue
            if any(
                marker in content
                for marker in (
                    "Long-Term Memory:\n",
                    "Documents:\n",
                    "Context:\n",
                    "User Query:\n",
                    "Follow-up Query:\n",
                    "Selected Excerpt",
                )
            ):
                return True
        return False

    def _ensure_prompt_renderer(self, session_id: str, session: Dict[str, object]) -> None:
        """
        Ensure the session KV/state matches our prompt renderer and tokenization.

        Delta-commit requires that the loaded KV corresponds exactly to the rendered
        prompt tokens for `session["messages"]`. If this session was created with a
        different renderer (or with an empty KV), rebuild once.
        """
        if session.get("prompt_renderer") == self._PROMPT_RENDERER_ID:
            if not isinstance(session.get("_prompt_tokens"), list):
                with self._model_lock:
                    prompt = self._render_prompt(list(session.get("messages", [])), add_generation_prompt=False)
                    session["_prompt_tokens"] = self._tokenize_prompt(prompt)
            return

        messages: List[Dict[str, str]] = list(session.get("messages", []))
        logger.info("Rebuilding KV for prompt renderer migration chat=%s", session_id)
        with self._model_lock:
            self.llm.reset()
            self._prefill_chat_messages(messages)
            session["state"] = self._save_state_compact()
            self._active_session_id = session_id
            prompt = self._render_prompt(messages, add_generation_prompt=False)
            session["_prompt_tokens"] = self._tokenize_prompt(prompt)

        session["prompt_renderer"] = self._PROMPT_RENDERER_ID
        self.sessions[session_id] = session
        self._persist_session(session_id, session)

    def _ensure_clean_kv_mode(self, session_id: str, session: Dict[str, object]) -> None:
        """
        Ensure this session is operating in "clean KV" mode.

        If we detect legacy persisted prompts, we rebuild KV once from a clean
        transcript (SQLite preferred) and keep only a small recent tail.
        """
        if session.get("kv_mode") == "clean_v2":
            self._ensure_prompt_renderer(session_id, session)
            return

        messages: List[Dict[str, str]] = list(session.get("messages", []))
        if not self._looks_like_legacy_turn_prompt(messages):
            session["kv_mode"] = "clean_v2"
            self._ensure_prompt_renderer(session_id, session)
            return

        logger.info("Migrating legacy KV session to clean_v2 chat=%s", session_id)

        system_msgs = [m for m in messages if m.get("role") == "system"]
        ui_msgs = self._fetch_ui_transcript(session_id, max_messages=40)
        convo_msgs = ui_msgs[-4:] if ui_msgs else [m for m in messages if m.get("role") != "system"][-4:]
        if not ui_msgs:
            convo_msgs = self._clean_history(convo_msgs)

        new_msgs: List[Dict[str, str]] = []
        new_msgs.extend(system_msgs)
        new_msgs.extend(convo_msgs)

        with self._model_lock:
            self.llm.reset()
            try:
                self._prefill_chat_messages(new_msgs)
            except Exception:
                pass
            session["state"] = self._save_state_compact()
            self._active_session_id = session_id
            prompt = self._render_prompt(new_msgs, add_generation_prompt=False)
            session["_prompt_tokens"] = self._tokenize_prompt(prompt)
            session["prompt_renderer"] = self._PROMPT_RENDERER_ID

        session["messages"] = new_msgs
        session["compacted"] = True
        session["kv_mode"] = "clean_v2"
        self.sessions[session_id] = session
        self._persist_session(session_id, session)

    def ask(
        self,
        session_id: str,
        message: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.2,
    ) -> str:
        """
        Main chat entry: appends user+assistant messages and updates the KV snapshot.
        """
        lock = self._locks.setdefault(session_id, threading.RLock())
        with lock:
            session = self.sessions.get(session_id)
            if session is None:
                raise ValueError(f"Unknown session: {session_id}")

            # If the persisted KV file was removed externally, reset this session.
            if self.persist_dir:
                kv_path = self.persist_dir / f"{session_id}.kv"
                if not kv_path.exists():
                    logger.warning("KV missing for %s, resetting to system-only state", session_id)
                    system_messages = [m for m in session.get("messages", []) if m.get("role") == "system"]
                    with self._model_lock:
                        self.llm.reset()
                        self._prefill_chat_messages(system_messages)
                        fresh_state = self._save_state_compact()
                        prompt = self._render_prompt(system_messages, add_generation_prompt=False)
                        prompt_tokens = self._tokenize_prompt(prompt)
                    session = {
                        "state": fresh_state,
                        "messages": system_messages,
                        "compacted": False,
                        "_prompt_tokens": prompt_tokens,
                        "prompt_renderer": self._PROMPT_RENDERER_ID,
                    }
                    self.sessions[session_id] = session
                    self._persist_session(session_id, session)

            message = self._preflight_compact_and_budget(
                session_id,
                session,
                message,
                reserved_max_tokens=max_tokens,
            )

            with self._model_lock:
                try:
                    # Load KV only if we are switching sessions
                    self._ensure_session_loaded(session_id, session)
                    if bool(session.get("_state_dirty")):
                        session["state"] = self._save_state_compact()
                        session["_state_dirty"] = False
                        session.pop("_state_dirty_at", None)
                    base_state = session.get("state")
                    clean_messages: List[Dict[str, str]] = list(session.get("messages", []))
                    clean_tokens: List[int] = list(session.get("_prompt_tokens") or [])
                    if not clean_tokens:
                        clean_prompt = self._render_prompt(clean_messages, add_generation_prompt=False)
                        clean_tokens = self._tokenize_prompt(clean_prompt)
                    run_messages: List[Dict[str, str]] = list(clean_messages)
                    run_messages.append({"role": "user", "content": message})

                    # Avoid llama.cpp hard failures when prompt_tokens + max_tokens > ctx_size.
                    run_prompt = self._render_prompt(run_messages, add_generation_prompt=True)
                    run_tokens = self._tokenize_prompt(run_prompt)
                    if clean_tokens and run_tokens[: len(clean_tokens)] == clean_tokens:
                        delta = run_tokens[len(clean_tokens) :]
                        if delta:
                            self._eval_tokens(delta)
                    else:
                        self.llm.reset()
                        self._eval_tokens(run_tokens)

                    prompt_tokens = len(run_tokens)
                    max_tokens = self._clamp_max_tokens(session_id, prompt_tokens, max_tokens)

                    out = self._create_completion_from_state(
                        prompt_tokens=run_tokens,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stream=False,
                        stop=self._stop_markers(),
                    )
                    reply = (out.get("choices") or [{}])[0].get("text") or ""

                    # Restore clean KV before committing.
                    try:
                        self.llm.reset()
                        if base_state is not None:
                            self._load_state_compact(base_state)
                        self._active_session_id = session_id
                    except Exception:
                        self._active_session_id = None

                    commit_messages: List[Dict[str, str]] = list(clean_messages)
                    commit_messages.append({"role": "user", "content": message})
                    if reply:
                        commit_messages.append({"role": "assistant", "content": reply})
                    self._commit_messages_to_session(session_id, session, commit_messages)

                    self.sessions[session_id] = session
                    self._persist_session(session_id, session)
                    return reply
                except Exception as exc:
                    logger.error("ask failed for chat=%s: %s", session_id, exc)
                    raise

    def get_context_status(self, session_id: str) -> Dict[str, object]:
        session = self.sessions.get(session_id)
        if session is None:
            raise ValueError(f"Unknown session: {session_id}")
        messages = session.get("messages", [])
        try:
            with self._model_lock:
                used_tokens = self._count_chat_prompt_tokens(messages)
        except Exception:
            used_tokens = self._estimate_tokens(messages)
        capacity = self.ctx_size
        percent = used_tokens / capacity if capacity else 0.0
        status: Dict[str, object] = {
            "chat_id": session_id,
            "used_tokens": used_tokens,
            "capacity_tokens": capacity,
            "percent": round(percent * 100, 2),
            "compacted": bool(session.get("compacted", False)),
        }
        last_tps = session.get("last_gen_tps")
        if isinstance(last_tps, (int, float)) and math.isfinite(float(last_tps)) and float(last_tps) > 0:
            status["last_gen_tps"] = float(last_tps)
        last_ttft = session.get("last_ttft_ms")
        if isinstance(last_ttft, (int, float)) and math.isfinite(float(last_ttft)) and float(last_ttft) >= 0:
            status["last_ttft_ms"] = int(float(last_ttft))
        last_tokens = session.get("last_gen_tokens")
        if isinstance(last_tokens, (int, float)) and math.isfinite(float(last_tokens)) and float(last_tokens) >= 0:
            status["last_gen_tokens"] = int(float(last_tokens))
        # Expose a best-effort "input budget" for the current context state.
        # This reflects how much ephemeral context could fit BEFORE adding a new user turn.
        try:
            reserved_output = int(capacity * self._OUTPUT_RESERVE_PCT)
            margin = int(capacity * self._CONTEXT_MARGIN_PCT)
            available = max(0, int(capacity) - int(used_tokens) - int(reserved_output) - int(margin))
            max_input = max(0, int(available * self._MAX_INPUT_PCT))
            status["input_budget_tokens"] = int(max_input)
            status["input_budget_reserved"] = int(reserved_output)
        except Exception:
            pass
        return status

    def fork_session(self, session_id: str) -> str:
        session = self.sessions.get(session_id)
        if session is None:
            raise ValueError(f"Unknown session to fork: {session_id}")
        new_session_id = str(uuid.uuid4())
        # Deep copy state to avoid shared reference (do not use pickle).
        state_copy = copy.deepcopy(session.get("state"))
        fork_payload = {
            "state": state_copy,
            "messages": list(session["messages"]),
            "compacted": session.get("compacted", False),
            "_prompt_tokens": list(session.get("_prompt_tokens") or []),
        }
        self.sessions[new_session_id] = fork_payload
        self._locks.setdefault(new_session_id, threading.RLock())
        self._persist_session(new_session_id, fork_payload)
        return new_session_id

    def reset_session(self, session_id: str) -> None:
        lock = self._locks.setdefault(session_id, threading.RLock())
        with lock:
            session = self.sessions.get(session_id)
            if session is None:
                raise ValueError(f"Unknown session: {session_id}")
            system_messages = [m for m in session.get("messages", []) if m.get("role") == "system"]
            with self._model_lock:
                self.llm.reset()
                self._prefill_chat_messages(system_messages)
                session["state"] = self._save_state_compact()
                # If we just reset this session, it's now the active one
                self._active_session_id = session_id
                prompt = self._render_prompt(system_messages, add_generation_prompt=False)
                session["_prompt_tokens"] = self._tokenize_prompt(prompt)
                session["prompt_renderer"] = self._PROMPT_RENDERER_ID
            session["messages"] = system_messages
            session["compacted"] = False
            self.sessions[session_id] = session
            self._persist_session(session_id, session)

    def delete_session(self, session_id: str) -> None:
        lock = self._locks.setdefault(session_id, threading.RLock())
        with lock:
            self.sessions.pop(session_id, None)
            with self._persist_mutex:
                self._persist_pending.pop(session_id, None)
                self._persist_scheduled.discard(session_id)
            if self._active_session_id == session_id:
                # Clear KV for safety since we reuse the same llm instance
                with self._model_lock:
                    try:
                        self.llm.reset()
                    except Exception:
                        pass
                self._active_session_id = None

            if self.persist_dir:
                kv_path = self.persist_dir / f"{session_id}.kv"
                msg_path = self.persist_dir / f"{session_id}.json"
                meta_path = self.persist_dir / f"{session_id}.meta.json"
                for p in (kv_path, msg_path, meta_path):
                    if p.exists():
                        p.unlink()

    def list_sessions(self) -> List[str]:
        return list(self.sessions.keys())

    # -------------------------------------------------------------------------
    # Raw / transient / ephemeral calls
    # -------------------------------------------------------------------------

    def ask_raw(self, session_id: str, text: str, *, max_tokens: int = 256, temperature: float = 0.2) -> str:
        """
        Invoke the model with a raw prompt (no chat template), using the KV state for the session.
        """
        lock = self._locks.setdefault(session_id, threading.RLock())
        with lock:
            session = self.sessions.get(session_id)
            if session is None:
                raise ValueError(f"Unknown session: {session_id}")
            with self._model_lock:
                self._ensure_session_loaded(session_id, session)
                out = self.llm(text, max_tokens=max_tokens, temperature=temperature)
                reply = out["choices"][0]["text"]
                session["state"] = self._save_state_compact()
                self._active_session_id = session_id
            self.sessions[session_id] = session
            self._persist_session(session_id, session)
            return reply

    def run_transient(self, session_id: str, prompt: str, *, max_tokens: int = 256, temperature: float = 0.2) -> str:
        """
        Run a one-off generation using the session state but restore the state afterward
        so the chat KV is unchanged.
        """
        lock = self._locks.setdefault(session_id, threading.RLock())
        with lock:
            session = self.sessions.get(session_id)
            if session is None:
                raise ValueError(f"Unknown session: {session_id}")

            original_state = session["state"]
            with self._model_lock:
                # We explicitly do NOT rely on current _active_session_id here,
                # since we want a clean run that won't contaminate the chat history.
                self.llm.reset()
                out = self.llm(prompt, max_tokens=max_tokens, temperature=temperature)
                reply = out["choices"][0]["text"]
                # restore original state
                self._load_state_compact(original_state)
                self._active_session_id = session_id
            return reply

    def run_ephemeral(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.2,
    ) -> str:
        """
        Run a one-off chat-completion without touching any session state.
        The model is reset before and after to avoid contaminating KV cache.
        """
        with self._model_lock:
            try:
                self.llm.reset()
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                prompt = self._render_prompt(messages, add_generation_prompt=True)
                out = self.llm(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.9,
                    repeat_penalty=1.15,
                    frequency_penalty=0.6,
                    stop=["SUMMARY_END"],
                )
                return ((out.get("choices") or [{}])[0].get("text") or "").strip()
            finally:
                try:
                    self.llm.reset()
                except Exception:
                    pass
                self._active_session_id = None

    # -------------------------------------------------------------------------
    # Persistence helpers
    # -------------------------------------------------------------------------

    def _persist_session(self, session_id: str, session: Dict[str, object]) -> None:
        """
        Schedule persistence of this session (KV + messages + metadata) to disk.

        This must not block the user-visible request path; we debounce and write from a
        background thread.
        """
        if not self.persist_dir:
            return
        if not isinstance(session_id, str) or not session_id:
            return
        # Never persist a KV snapshot that doesn't correspond to the stored messages.
        # When `_state_dirty` is True we have updated `session["messages"]` but have not yet
        # captured a new KV state. Persisting now would create an inconsistent `.kv`+`.json`
        # pair and force a rebuild on restart. Snapshot first, then persist.
        if bool(session.get("_state_dirty")):
            self._schedule_snapshot(session_id)
            return
        self._schedule_persist(session_id, session)

    def _load_persisted_sessions(self) -> None:
        if not self.persist_dir or not self.persist_dir.exists():
            return
        # Drop orphaned session metadata (e.g., crash mid-snapshot left .json/.meta without .kv).
        try:
            kv_ids = {p.stem for p in self.persist_dir.glob("*.kv")}
            msg_ids: set[str] = set()
            meta_ids: set[str] = set()
            for p in self.persist_dir.iterdir():
                if not p.is_file():
                    continue
                name = p.name
                if name.endswith(".meta.json"):
                    sid = name[: -len(".meta.json")]
                    if sid:
                        meta_ids.add(sid)
                    continue
                if name.endswith(".json"):
                    sid = name[: -len(".json")]
                    if sid:
                        msg_ids.add(sid)
            orphan_ids = (msg_ids | meta_ids) - kv_ids
            for session_id in orphan_ids:
                for suffix in (".json", ".meta.json"):
                    path = self.persist_dir / f"{session_id}{suffix}"
                    if path.exists():
                        try:
                            path.unlink()
                        except Exception:
                            pass
                logger.warning(
                    "Dropped orphan session metadata for %s (missing .kv snapshot)", session_id
                )
        except Exception as exc:
            logger.debug("Failed to reconcile persisted session files: %s", exc)
        # Load .kv state + .json messages + optional .meta.json.
        #
        # Security note: historically we pickled the state object into `.kv`. Pickle is unsafe
        # to load from disk because it can execute code. We now persist `.kv` as raw
        # llama_state bytes and keep the small metadata needed to restore it in `.meta.json`.
        #
        # For legacy `.kv` files that do not declare the new format, we intentionally avoid
        # unpickling and instead mark the session dirty so it will rebuild KV from the
        # stored clean messages on first use.
        for path in self.persist_dir.glob("*.kv"):
            try:
                session_id = path.stem
                msg_path = self.persist_dir / f"{session_id}.json"
                messages: List[Dict[str, str]] = []
                if msg_path.exists():
                    try:
                        messages = json.loads(msg_path.read_text()) or []
                    except Exception:
                        messages = []
                meta_path = self.persist_dir / f"{session_id}.meta.json"
                meta: Dict[str, object] = {}
                if meta_path.exists():
                    try:
                        meta = json.loads(meta_path.read_text()) or {}
                    except Exception:
                        meta = {}

                state_obj: object | None = None
                is_new_format = False
                if isinstance(meta, dict):
                    is_new_format = (
                        meta.get("kv_format") == self._KV_FILE_FORMAT
                        and meta.get("state_kind") == self._STATE_KIND_COMPACT
                    )

                if is_new_format:
                    llama_state = path.read_bytes()
                    size = int(meta.get("llama_state_size") or len(llama_state) or 0)
                    n_tokens = int(meta.get("n_tokens") or 0)
                    seed = int(meta.get("seed") or 0)
                    input_ids = meta.get("input_ids")
                    # Convert list → numpy array when available (matches llama_cpp internals).
                    if np is not None and isinstance(input_ids, list):
                        try:
                            input_ids = np.array(input_ids, dtype=np.int32)  # type: ignore[call-arg]
                        except Exception:
                            input_ids = None
                    state_obj = {
                        "_kind": self._STATE_KIND_COMPACT,
                        "llama_state": llama_state[:size],
                        "llama_state_size": size,
                        "n_tokens": n_tokens,
                        "input_ids": input_ids,
                        "seed": seed,
                    }
                else:
                    # Legacy persisted KV is not loaded; force a rebuild from clean messages later.
                    state_obj = None

                session_payload: Dict[str, object] = {"state": state_obj, "messages": messages}
                if isinstance(meta, dict):
                    for k in ("compacted", "kv_mode", "prompt_renderer", "_prompt_tokens", "compaction_tick", "ltm_summary"):
                        if k in meta:
                            session_payload[k] = meta.get(k)
                if "compacted" not in session_payload:
                    session_payload["compacted"] = False
                # If we couldn't load a KV snapshot, mark dirty so `_ensure_session_loaded`
                # rebuilds KV from the clean transcript on first use.
                if state_obj is None and messages:
                    session_payload["_state_dirty"] = True
                    session_payload["_state_dirty_at"] = time.monotonic()
                self.sessions[session_id] = session_payload
                self._locks.setdefault(session_id, threading.RLock())
            except Exception:
                logger.warning("Failed to load persisted session from %s", path, exc_info=True)
                continue

    def _start_persist_worker(self) -> None:
        if self._persist_thread is not None:
            return
        self._persist_queue = queue.Queue()
        self._persist_thread = threading.Thread(target=self._persist_worker_loop, daemon=True)
        self._persist_thread.start()

    def _shutdown_persist_worker(self) -> None:
        try:
            self._persist_stop.set()
            q = self._persist_queue
            if q is not None:
                q.put(None)
            t = self._persist_thread
            if t is not None and t.is_alive():
                t.join(timeout=2.0)
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Snapshot helpers (defer expensive save_state off request hot-path)
    # -------------------------------------------------------------------------

    def _start_snapshot_worker(self) -> None:
        if self._snapshot_thread is not None:
            return
        self._snapshot_queue = queue.Queue()
        self._snapshot_thread = threading.Thread(target=self._snapshot_worker_loop, daemon=True)
        self._snapshot_thread.start()

    def _shutdown_snapshot_worker(self) -> None:
        try:
            self._snapshot_stop.set()
            q = self._snapshot_queue
            if q is not None:
                q.put(None)
            t = self._snapshot_thread
            if t is not None and t.is_alive():
                t.join(timeout=2.0)
        except Exception:
            pass

    def _schedule_snapshot(self, session_id: str) -> None:
        q = self._snapshot_queue
        if q is None or not isinstance(session_id, str) or not session_id:
            return
        with self._snapshot_mutex:
            if session_id in self._snapshot_scheduled:
                return
            self._snapshot_scheduled.add(session_id)
            q.put(session_id)

    def _snapshot_worker_loop(self) -> None:
        q = self._snapshot_queue
        if q is None:
            return
        while not self._snapshot_stop.is_set():
            session_id = q.get()
            if session_id is None:
                break
            try:
                # Debounce: if session keeps changing rapidly, wait a moment for it to settle.
                while True:
                    lock = self._locks.setdefault(session_id, threading.RLock())
                    with lock:
                        session = self.sessions.get(session_id)
                        if not isinstance(session, dict):
                            break
                        if not bool(session.get("_state_dirty")):
                            break
                        dirty_at = float(session.get("_state_dirty_at") or 0.0)
                    dt = time.monotonic() - dirty_at
                    if dt < self._SNAPSHOT_DEBOUNCE_SEC:
                        time.sleep(self._SNAPSHOT_DEBOUNCE_SEC - dt)
                        continue

                    lock = self._locks.setdefault(session_id, threading.RLock())
                    with lock:
                        session = self.sessions.get(session_id)
                        if not isinstance(session, dict) or not bool(session.get("_state_dirty")):
                            break
                        # Snapshotting requires that this session is currently loaded.
                        if self._active_session_id != session_id:
                            break
                        with self._model_lock:
                            # Active session already loaded; capture a compact snapshot.
                            session["state"] = self._save_state_compact()
                            session["_state_dirty"] = False
                            session.pop("_state_dirty_at", None)
                            self._active_session_id = session_id
                        self.sessions[session_id] = session
                        self._persist_session(session_id, session)
                    break
            finally:
                with self._snapshot_mutex:
                    self._snapshot_scheduled.discard(str(session_id))

    def _schedule_persist(self, session_id: str, session: Dict[str, object]) -> None:
        if not self.persist_dir:
            return
        q = self._persist_queue
        if q is None:
            return

        # Copy only what we must (messages + small metadata). State is immutable bytes.
        messages = list(session.get("messages", []) or [])
        state_obj = session.get("state")
        meta = {
            "compacted": bool(session.get("compacted", False)),
            "kv_mode": session.get("kv_mode"),
            "prompt_renderer": session.get("prompt_renderer"),
            "_prompt_tokens": list(session.get("_prompt_tokens") or []),
            "compaction_tick": session.get("compaction_tick"),
            "ltm_summary": session.get("ltm_summary"),
        }
        now = time.monotonic()

        with self._persist_mutex:
            self._persist_pending[session_id] = {
                "state": state_obj,
                "messages": messages,
                "meta": meta,
                "updated_at": now,
            }
            if session_id not in self._persist_scheduled:
                self._persist_scheduled.add(session_id)
                q.put(session_id)

    def _persist_worker_loop(self) -> None:
        q = self._persist_queue
        if q is None or not self.persist_dir:
            return

        while not self._persist_stop.is_set():
            session_id = q.get()
            if session_id is None:
                break

            while True:
                with self._persist_mutex:
                    rec = self._persist_pending.get(session_id)
                    if not isinstance(rec, dict):
                        self._persist_scheduled.discard(session_id)
                        break
                    updated_at = float(rec.get("updated_at") or 0.0)
                dt = time.monotonic() - updated_at
                if dt < self._PERSIST_DEBOUNCE_SEC:
                    time.sleep(self._PERSIST_DEBOUNCE_SEC - dt)
                    continue

                # Lock again and ensure nothing newer replaced this record.
                with self._persist_mutex:
                    rec2 = self._persist_pending.get(session_id)
                    if not isinstance(rec2, dict):
                        self._persist_scheduled.discard(session_id)
                        break
                    if float(rec2.get("updated_at") or 0.0) != updated_at:
                        continue
                    self._persist_pending.pop(session_id, None)
                    self._persist_scheduled.discard(session_id)

                try:
                    self._persist_session_sync(
                        session_id=session_id,
                        state_obj=rec2.get("state"),
                        messages=rec2.get("messages") or [],
                        meta=rec2.get("meta") or {},
                    )
                except Exception as exc:
                    logger.warning("Failed async persist for %s: %s", session_id, exc)
                break

    def _persist_session_sync(
        self,
        *,
        session_id: str,
        state_obj: object,
        messages: List[Dict[str, str]],
        meta: Dict[str, object],
    ) -> None:
        if not self.persist_dir:
            return
        try:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.warning("Failed to ensure persist_dir %s: %s", self.persist_dir, exc)
            return

        kv_path = self.persist_dir / f"{session_id}.kv"
        msg_path = self.persist_dir / f"{session_id}.json"
        meta_path = self.persist_dir / f"{session_id}.meta.json"

        # Atomic writes: write temp then replace.
        state_obj = self._coerce_state_compact(state_obj)
        state_meta: Dict[str, object] = {}
        try:
            if isinstance(state_obj, dict) and state_obj.get("_kind") == self._STATE_KIND_COMPACT:
                llama_state = state_obj.get("llama_state") or b""
                size = int(state_obj.get("llama_state_size") or len(llama_state) or 0)
                if isinstance(llama_state, (bytes, bytearray)) and size > 0:
                    tmp = kv_path.with_suffix(".kv.tmp")
                    with tmp.open("wb") as f:
                        f.write(bytes(llama_state)[:size])
                    tmp.replace(kv_path)
                    state_meta = {
                        "kv_format": self._KV_FILE_FORMAT,
                        "state_kind": self._STATE_KIND_COMPACT,
                        "llama_state_size": size,
                        "n_tokens": int(state_obj.get("n_tokens") or 0),
                        "seed": int(state_obj.get("seed") or 0),
                    }
                    # Store input_ids as JSON-friendly list (restored on load).
                    input_ids = state_obj.get("input_ids")
                    if input_ids is not None:
                        try:
                            if np is not None and hasattr(input_ids, "tolist"):
                                input_ids = input_ids.tolist()
                            if isinstance(input_ids, list):
                                state_meta["input_ids"] = input_ids
                        except Exception:
                            pass
        except Exception as exc:
            logger.warning("Failed to persist KV for %s: %s", session_id, exc)

        try:
            tmp = msg_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(messages or [], ensure_ascii=False))
            tmp.replace(msg_path)
        except Exception as exc:
            logger.warning("Failed to persist messages for %s: %s", session_id, exc)

        try:
            merged_meta = {**(meta or {}), **state_meta}
            tmp = meta_path.parent / (meta_path.name + ".tmp")
            tmp.write_text(json.dumps(merged_meta, ensure_ascii=False))
            tmp.replace(meta_path)
        except Exception as exc:
            logger.warning("Failed to persist metadata for %s: %s", session_id, exc)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _coerce_state_compact(self, state_obj: object) -> object:
        """
        Normalize persisted KV state objects into our compact format.

        Legacy `.kv` files may contain `llama_cpp.LlamaState`, which includes a very
        large `scores` array. We drop that and keep only what we need to restore
        llama.cpp + the Python-side token counters.
        """
        if isinstance(state_obj, dict) and state_obj.get("_kind") == self._STATE_KIND_COMPACT:
            return state_obj

        # `llama_cpp.llama.LlamaState` (or compatible) from older versions.
        if hasattr(state_obj, "llama_state") and hasattr(state_obj, "llama_state_size") and hasattr(state_obj, "n_tokens"):
            try:
                return {
                    "_kind": self._STATE_KIND_COMPACT,
                    "llama_state": getattr(state_obj, "llama_state"),
                    "llama_state_size": int(getattr(state_obj, "llama_state_size")),
                    "n_tokens": int(getattr(state_obj, "n_tokens")),
                    "input_ids": getattr(state_obj, "input_ids"),
                    "seed": int(getattr(state_obj, "seed", 0)),
                }
            except Exception:
                return state_obj

        return state_obj

    def _save_state_compact(self) -> Dict[str, object]:
        """
        Save llama.cpp KV/state without copying the huge `scores` array.

        Must be called while holding `_model_lock`.
        """
        ctx = getattr(getattr(self.llm, "_ctx", None), "ctx", None)
        if ctx is None:
            raise RuntimeError("llama context not initialized")

        # Use the newer state API (size + get_data) to avoid extra memmove/copies.
        state_size = int(llama_cpp.llama_state_get_size(ctx))
        buf = (ctypes.c_uint8 * state_size)()
        n_bytes = int(llama_cpp.llama_state_get_data(ctx, buf, state_size))
        if n_bytes <= 0 or n_bytes > state_size:
            raise RuntimeError("Failed to copy llama state data")
        llama_state = ctypes.string_at(buf, n_bytes)

        seed = int(getattr(self.llm, "_seed", 0))
        try:
            n_tokens = int(getattr(self.llm, "n_tokens"))
        except Exception:
            n_tokens = 0

        try:
            input_ids = getattr(self.llm, "input_ids").copy()
        except Exception:
            input_ids = None

        return {
            "_kind": self._STATE_KIND_COMPACT,
            "llama_state": llama_state,
            "llama_state_size": n_bytes,
            "n_tokens": n_tokens,
            "input_ids": input_ids,
            "seed": seed,
        }

    def _load_state_compact(self, state_obj: object) -> None:
        """
        Restore llama.cpp KV/state from our compact (or legacy) state object.

        Must be called while holding `_model_lock`.
        """
        if state_obj is None:
            return

        state_obj = self._coerce_state_compact(state_obj)

        if isinstance(state_obj, dict) and state_obj.get("_kind") == self._STATE_KIND_COMPACT:
            llama_state = state_obj.get("llama_state") or b""
            size = int(state_obj.get("llama_state_size") or len(llama_state) or 0)
            n_tokens = int(state_obj.get("n_tokens") or 0)
            input_ids = state_obj.get("input_ids")
            seed = state_obj.get("seed")
        elif hasattr(state_obj, "llama_state") and hasattr(state_obj, "llama_state_size") and hasattr(state_obj, "n_tokens"):
            llama_state = getattr(state_obj, "llama_state") or b""
            size = int(getattr(state_obj, "llama_state_size") or len(llama_state) or 0)
            n_tokens = int(getattr(state_obj, "n_tokens") or 0)
            input_ids = getattr(state_obj, "input_ids", None)
            seed = getattr(state_obj, "seed", None)
        else:
            # Last resort fallback (will copy scores). Keep for safety.
            self.llm.load_state(state_obj)  # type: ignore[arg-type]
            return

        ctx = getattr(getattr(self.llm, "_ctx", None), "ctx", None)
        if ctx is None:
            raise RuntimeError("llama context not initialized")
        if not isinstance(llama_state, (bytes, bytearray)) or size <= 0:
            raise RuntimeError("Invalid llama state payload")

        # Restore Python-side tracking first (matches llama_cpp.Llama.load_state ordering).
        if input_ids is not None:
            try:
                self.llm.input_ids = input_ids.copy()
            except Exception:
                pass
        try:
            self.llm.n_tokens = int(n_tokens)
        except Exception:
            pass
        if seed is not None:
            try:
                self.llm._seed = int(seed)  # noqa: SLF001
            except Exception:
                pass

        StateArray = ctypes.c_uint8 * size
        raw = StateArray.from_buffer_copy(bytes(llama_state)[:size])
        read = int(llama_cpp.llama_state_set_data(ctx, raw, size))
        if read != size:
            raise RuntimeError("Failed to set llama state data")

    def _ensure_session_loaded(self, session_id: str, session: Dict[str, object]) -> None:
        """
        Ensure that the model KV corresponds to the given session.

        If the same session is already active, we skip expensive load_state().
        If we're switching sessions, we reset + load the appropriate state.
        """
        if self._active_session_id == session_id:
            return
        # Before switching away, best-effort snapshot the currently active session if it has
        # pending (dirty) KV changes. We must not block on another session lock here (deadlock
        # risk), so we only snapshot if we can acquire the lock immediately.
        prev_id = self._active_session_id
        if prev_id and prev_id != session_id:
            prev = self.sessions.get(prev_id)
            if isinstance(prev, dict) and bool(prev.get("_state_dirty")):
                prev_lock = self._locks.get(prev_id) or self._locks.setdefault(prev_id, threading.RLock())
                acquired = False
                try:
                    acquired = prev_lock.acquire(blocking=False)
                except TypeError:
                    # Py <3.2 compatibility not needed, but keep safe.
                    try:
                        acquired = prev_lock.acquire(False)
                    except Exception:
                        acquired = False
                if acquired:
                    try:
                        prev["state"] = self._save_state_compact()
                        prev["_state_dirty"] = False
                        prev.pop("_state_dirty_at", None)
                        self.sessions[prev_id] = prev
                        self._persist_session(prev_id, prev)
                    finally:
                        try:
                            prev_lock.release()
                        except Exception:
                            pass
                else:
                    logger.warning(
                        "Active session has unsnapshotted KV but lock is busy; will rebuild on next load chat=%s",
                        prev_id,
                    )

        self.llm.reset()

        # If this session has dirty messages/state (i.e., messages advanced but we couldn't
        # snapshot KV yet), do NOT load the stale `.state` snapshot. Rebuild KV from the
        # clean messages so the model state matches the transcript, then let the normal
        # snapshotting flow capture a fresh state during idle.
        if bool(session.get("_state_dirty")):
            messages = list(session.get("messages", []) or [])
            self._prefill_chat_messages(messages)
        else:
            state = session.get("state")
            if state is not None:
                self._load_state_compact(state)
        self._active_session_id = session_id

    def _estimate_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        Estimate tokens using the model tokenizer when possible; fall back to heuristic.
        """
        try:
            text = "\n".join(m.get("content", "") for m in messages)
            with self._model_lock:
                return len(self.llm.tokenize(text.encode("utf-8")))
        except Exception:
            total_words = 0
            for m in messages:
                text = m.get("content", "")
                total_words += len(text.split())
            return int(total_words * 1.3)

    def _count_chat_prompt_tokens(self, messages: List[Dict[str, str]], *, add_generation_prompt: bool = False) -> int:
        """
        Estimate prompt tokens for the chat template (more accurate than joining message content).

        Must be called while holding `_model_lock`.
        """
        try:
            prompt = self._render_prompt(messages, add_generation_prompt=add_generation_prompt)
            return len(self._tokenize_prompt(prompt))
        except Exception:
            # Fallback MUST NOT call `_estimate_tokens()` here because this method is
            # invoked while holding `_model_lock` and `_estimate_tokens()` also locks,
            # which would deadlock.
            total_words = 0
            for m in messages:
                total_words += len((m.get("content", "") or "").split())
            return int(total_words * 1.3)

    def _clamp_max_tokens(self, session_id: str, prompt_tokens: int, requested_max: int) -> int:
        """
        Clamp generation tokens so prompt_tokens + max_tokens stays within ctx_size.

        Must be called while holding `_model_lock` (because tokenization helpers may lock).
        """
        if not self.ctx_size:
            return requested_max
        # Leave a small safety margin to avoid exact-boundary errors.
        margin = 128
        available = self.ctx_size - prompt_tokens - margin
        if available <= 0:
            # No room left for generation; fail loudly with actionable info.
            raise ValueError(
                f"Prompt too large for context window (chat={session_id} prompt_tokens={prompt_tokens} ctx={self.ctx_size})"
            )
        clamped = min(int(requested_max), int(available))
        if clamped < requested_max:
            logger.info(
                "Clamped max_tokens chat=%s requested=%d clamped=%d prompt_tokens=%d ctx=%d",
                session_id,
                requested_max,
                clamped,
                prompt_tokens,
                self.ctx_size,
            )
        # Do not force a minimum: if we're tight on context, any forced minimum can overflow.
        # If the prompt leaves effectively no room, fail early instead of letting llama.cpp error.
        if clamped <= 0:
            raise ValueError(
                f"Prompt too large for context window (chat={session_id} prompt_tokens={prompt_tokens} ctx={self.ctx_size})"
            )
        return clamped

    def _clean_history(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Strip scaffolding prefixes from stored turns so the summarizer sees cleaner text.
        Also drops obvious large scaffolding blocks like 'Documents:'.
        """
        cleaned: List[Dict[str, str]] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "") or ""
            # Drop common wrappers like "User Query:" or "Follow-up Query:"
            for prefix in (
                "User Query:\n",
                "Follow-up Query:\n",
                "Follow-up Query:",
                "Query:\n",
                "Query:",
                "Documents:\n",
                "Context:\n",
            ):
                if content.startswith(prefix):
                    content = content[len(prefix) :].lstrip()
                    break
            # Drop standalone "Documents:" blocks
            if content.strip().startswith("Documents:"):
                continue
            if content.strip().startswith("Context:"):
                continue
            cleaned.append({"role": role, "content": content})
        return cleaned

    def _fetch_ui_transcript(self, chat_id: str, *, max_messages: int = 80) -> List[Dict[str, str]]:
        """
        Prefer the clean UI transcript from SQLite (user/assistant messages only).
        Falls back to session messages if SQLite is unavailable.
        """
        store = self._metadata_store
        if store is None:
            return []
        try:
            rows = store.fetch_messages_for_chat(chat_id, limit=max_messages)
            return [m for m in rows if m.get("role") in ("user", "assistant")]
        except Exception:
            return []

    def _truncate_text_tail_to_tokens(self, text: str, max_tokens: int) -> str:
        if not text or max_tokens <= 0:
            return ""
        try:
            with self._model_lock:
                tokens = self.llm.tokenize(text.encode("utf-8"))
                if len(tokens) <= max_tokens:
                    return text
                tail = tokens[-max_tokens:]
                if hasattr(self.llm, "detokenize"):
                    out = self.llm.detokenize(tail)
                    return out.decode("utf-8", errors="ignore")
        except Exception:
            pass
        # Fallback: rough char-based tail slice.
        approx_chars = max_tokens * 4
        return text[-approx_chars:]

    def _truncate_text_head_to_tokens(self, text: str, max_tokens: int) -> str:
        if not text or max_tokens <= 0:
            return ""
        try:
            with self._model_lock:
                tokens = self.llm.tokenize(text.encode("utf-8"))
                if len(tokens) <= max_tokens:
                    return text
                head = tokens[:max_tokens]
                if hasattr(self.llm, "detokenize"):
                    out = self.llm.detokenize(head)
                    return out.decode("utf-8", errors="ignore")
        except Exception:
            pass
        approx_chars = max_tokens * 4
        return text[:approx_chars]

    def _preflight_compact_and_budget(
        self,
        session_id: str,
        session: Dict[str, object],
        message: str,
        *,
        reserved_max_tokens: int,
    ) -> str:
        """
        Preflight safety for context window:

        1) Compact session history if already near-full.
        2) If adding this message would overflow the context window, force-compaction.
        3) If still too large, strip heavy prompt blocks (Context/Documents/LTM) and
           finally truncate the per-turn prompt to fit.
        """
        if not self.ctx_size:
            return message

        # First, compact if the existing session is already near full.
        self._maybe_compact(session_id, session, force=False)

        margin = 128
        attempts = 0
        forced = False
        while attempts < 4:
            history: List[Dict[str, str]] = list(session.get("messages", []))
            prospective = history + [{"role": "user", "content": message}]
            try:
                with self._model_lock:
                    prompt_tokens = self._count_chat_prompt_tokens(prospective)
            except Exception:
                prompt_tokens = self._estimate_tokens(prospective)

            # If prompt itself fits, generation tokens will be clamped later.
            if prompt_tokens + margin < self.ctx_size:
                return message

            # Prompt is too large even before generation: first try forced compaction once.
            if not forced:
                self._maybe_compact(session_id, session, force=True)
                forced = True
                attempts += 1
                continue

            # Still too large: drop the heavy sections from this turn prompt and/or truncate.
            message = self._shrink_turn_prompt_to_fit(session_id, session, message, margin=margin)
            attempts += 1

        raise ValueError(
            f"Prompt too large for context window even after compaction (chat={session_id} ctx={self.ctx_size})"
        )

    def _strip_prompt_section(self, text: str, heading: str) -> str:
        if not text:
            return text
        headings = r"(?:Long-Term Memory|Documents|Context)"
        pattern = rf"(?:\n\n|\A){re.escape(heading)}:\n.*?(?=\n\n{headings}:\n|\Z)"
        return re.sub(pattern, "", text, flags=re.DOTALL).strip()

    def _extract_query_only(self, text: str) -> str:
        if not text:
            return ""
        m = re.search(r"^(User Query|Follow-up Query):\n(.*?)(?:\n\n(?:Long-Term Memory|Documents|Context):\n|\Z)", text, flags=re.DOTALL)
        if not m:
            return text.strip()
        label = m.group(1)
        body = (m.group(2) or "").strip()
        return f"{label}:\n{body}".strip()

    def _shrink_turn_prompt_to_fit(
        self,
        session_id: str,
        session: Dict[str, object],
        message: str,
        *,
        margin: int = 128,
    ) -> str:
        """
        Reduce the per-turn prompt (message content) until the full chat prompt fits in ctx.
        """
        history: List[Dict[str, str]] = list(session.get("messages", []))

        def fits(candidate: str) -> bool:
            prospective = history + [{"role": "user", "content": candidate}]
            with self._model_lock:
                tokens = self._count_chat_prompt_tokens(prospective)
            return tokens + margin < self.ctx_size

        candidate = (message or "").strip()
        if not candidate:
            return candidate

        for heading in ("Context", "Documents", "Long-Term Memory"):
            if f"{heading}:\n" in candidate:
                candidate = self._strip_prompt_section(candidate, heading)
                try:
                    if fits(candidate):
                        return candidate
                except Exception:
                    pass

        candidate = self._extract_query_only(candidate)
        try:
            if fits(candidate):
                return candidate
        except Exception:
            pass

        # Last resort: truncate the per-turn prompt to the remaining budget.
        try:
            with self._model_lock:
                base = self._count_chat_prompt_tokens(history + [{"role": "user", "content": ""}])
        except Exception:
            base = self._estimate_tokens(history) + 8

        available = max(0, int(self.ctx_size) - int(margin) - int(base))
        if available <= 0:
            raise ValueError(
                f"Chat history too large for context window (chat={session_id} ctx={self.ctx_size})"
            )

        truncated = self._truncate_text_head_to_tokens(candidate, available)
        # Ensure the truncated version actually fits; if not, keep halving.
        while True:
            try:
                if fits(truncated):
                    return truncated
            except Exception:
                return truncated
            available = max(1, available // 2)
            truncated = self._truncate_text_head_to_tokens(candidate, available)

    def _maybe_compact(self, session_id: str, session: Dict[str, object], *, force: bool = False) -> None:
        messages: List[Dict[str, str]] = session.get("messages", [])
        if not self.ctx_size:
            return

        if not force:
            try:
                with self._model_lock:
                    used = self._count_chat_prompt_tokens(messages)
            except Exception:
                used = self._estimate_tokens(messages)
            if used < self._COMPACT_PCT * self.ctx_size:
                return

        system_msgs = [m for m in messages if m.get("role") == "system"]
        # Prefer clean UI transcript for compaction (avoids summarizing injected RAG/doc blobs).
        ui_msgs = self._fetch_ui_transcript(session_id, max_messages=120)
        convo_msgs = ui_msgs if ui_msgs else [m for m in messages if m.get("role") != "system"]

        # If conversation is too short, do a lightweight rebuild in force-mode
        # to strip injected per-turn blobs from the stored prompt history.
        if len(convo_msgs) <= 4:
            if not force:
                return
            new_msgs: List[Dict[str, str]] = []
            new_msgs.extend(system_msgs)
            new_msgs.extend(convo_msgs)
            with self._model_lock:
                self.llm.reset()
                try:
                    self._prefill_chat_messages(new_msgs)
                except Exception:
                    pass
                new_state = self._save_state_compact()
                prompt = self._render_prompt(new_msgs, add_generation_prompt=False)
                session["_prompt_tokens"] = self._tokenize_prompt(prompt)
                session["prompt_renderer"] = self._PROMPT_RENDERER_ID
            session["state"] = new_state
            session["messages"] = new_msgs
            session["compacted"] = bool(session.get("compacted", False))
            self.sessions[session_id] = session
            self._active_session_id = session_id
            self._persist_session(session_id, session)
            return

        # Build summary from older turns (all but last 2 exchanges)
        summary_source = convo_msgs[:-4]
        # Only apply cleaning when using model/session messages (UI transcript is already clean).
        if ui_msgs:
            cleaned_source = summary_source
        else:
            cleaned_source = self._clean_history(summary_source)
        summary_text = self._compact_with_summarizer(session_id, cleaned_source)
        summary_msg = (
            {"role": "assistant", "content": f"Conversation summary:\n{summary_text}".strip()}
            if summary_text
            else None
        )
        tail_msgs = convo_msgs[-4:]  # keep last 2 exchanges

        new_msgs: List[Dict[str, str]] = []
        new_msgs.extend(system_msgs)
        if summary_msg:
            new_msgs.append(summary_msg)
        new_msgs.extend(tail_msgs)

        # Reset KV and rebuild in a single pass using the chat template
        with self._model_lock:
            self.llm.reset()
            try:
                self._prefill_chat_messages(new_msgs)
            except Exception:
                # If feeding fails, fall back to empty state
                pass
            new_state = self._save_state_compact()
            prompt = self._render_prompt(new_msgs, add_generation_prompt=False)
            session["_prompt_tokens"] = self._tokenize_prompt(prompt)
            session["prompt_renderer"] = self._PROMPT_RENDERER_ID

        session["state"] = new_state
        session["messages"] = new_msgs
        session["compacted"] = True
        if summary_text:
            # Store in-session summary metadata
            session["ltm_summary"] = summary_text
            comp_tick = session.get("compaction_tick", 0)
            comp_tick = comp_tick + 1 if isinstance(comp_tick, int) else 1
            session["compaction_tick"] = comp_tick

            # Push summary into LTM store (async) if available
            if self.ltm_store:
                try:
                    threading.Thread(
                        target=self.ltm_store.save_memories,
                        args=(session_id, [summary_text]),
                        daemon=True,
                    ).start()
                except Exception as exc:
                    logging.getLogger(__name__).warning(
                        "Failed to store LTM summary for %s: %s",
                        session_id,
                        exc,
                    )


        self.sessions[session_id] = session
        self._active_session_id = session_id
        self._persist_session(session_id, session)

    def _compact_with_summarizer(self, session_id: str, summary_source: List[Dict[str, str]]) -> str:
        """
        Use a transient LLM call to summarize older turns; if it fails, fallback to heuristic.
        """
        if not summary_source:
            return ""
        try:
            convo_text = "\n".join(
                f"{m.get('role','user').upper()}: {m.get('content','')}" for m in summary_source
            )

            # Token cap on summarizer input (more stable than char caps).
            # Keep the tail of the “older conversation” if it’s too large.
            max_input_tokens = 1200
            convo_text = self._truncate_text_tail_to_tokens(convo_text, max_input_tokens)

            prompt = (
                "Summarize the earlier part of this conversation (exclude the most recent exchanges).\n"
                "Do not invent facts. Keep names, file references, and numbers accurate.\n\n"
                "Return this structure:\n"
                "1) Goals\n"
                "2) Key Facts\n"
                "3) Decisions / Conclusions\n"
                "4) Open Questions / Next Steps\n"
                "5) User Preferences (if any)\n\n"
                f"Conversation:\n{convo_text}\n\nSummary:"
            )
            return self.run_transient(session_id, prompt, max_tokens=260, temperature=0.2).strip()
        except Exception:
            # Fallback heuristic: keep a few sentences
            text = " ".join(m.get("content", "") for m in summary_source)
            if not text:
                return ""
            import re

            sentences = re.split(r"(?<=[.!?])\s+", text.strip())
            summary_parts: List[str] = []
            for s in sentences:
                if len(" ".join(summary_parts) + " " + s) > 600:
                    break
                summary_parts.append(s)
                if len(summary_parts) >= 3:
                    break
            summary = " ".join(summary_parts).strip()
            return summary or text[:600]


__all__ = ["LlamaSessionManager"]
