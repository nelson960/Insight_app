from __future__ import annotations

import uuid
import logging
import re
from pathlib import Path
from typing import Dict, Optional, List

import pickle
import threading
import ctypes

import llama_cpp
from llama_cpp import Llama

from backend.services.storage.sqlite_store import SQLiteMetadataStore


logger = logging.getLogger(__name__)


class LlamaSessionManager:
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
        if self.persist_dir:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._load_persisted_sessions()
            try:
                logger.info(
                    "LlamaSessionManager using persist_dir=%s loaded_sessions=%d",
                    self.persist_dir,
                    len(self.sessions),
                )
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # Public session API
    # -------------------------------------------------------------------------

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

    def create_session(self, system_prompt: Optional[str] = None) -> str:
        session_id = str(uuid.uuid4())
        self._locks.setdefault(session_id, threading.RLock())
        with self._model_lock:
            self.llm.reset()
            messages: List[Dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            state_obj = self.llm.save_state()
        session_payload = {
            "state": state_obj,
            "messages": messages,
            "compacted": False,
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
            state_obj = self.llm.save_state()
        session_payload = {"state": state_obj, "messages": messages, "compacted": False}
        self.sessions[session_id] = session_payload
        self._persist_session(session_id, session_payload)
        return session_id

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

            if self.persist_dir:
                kv_path = self.persist_dir / f"{session_id}.kv"
                if not kv_path.exists():
                    logger.warning("KV missing for %s, resetting to system-only state", session_id)
                    system_messages = [m for m in session.get("messages", []) if m.get("role") == "system"]
                    with self._model_lock:
                        self.llm.reset()
                        fresh_state = self.llm.save_state()
                    session = {"state": fresh_state, "messages": system_messages, "compacted": False}
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
                messages: List[Dict[str, str]] = list(session.get("messages", []))
                messages.append({"role": "user", "content": message})

                # Avoid llama.cpp hard failures when prompt_tokens + max_tokens > ctx_size.
                prompt_tokens = self._count_chat_prompt_tokens(messages)
                max_tokens = self._clamp_max_tokens(session_id, prompt_tokens, max_tokens)

                logger.info(
                    "ask_stream llama start chat=%s msgs=%d max_tokens=%d",
                    session_id,
                    len(messages),
                    max_tokens,
                )
                with self._abort_lock:
                    # Activate abort only for the duration of this generation.
                    self._abort_request_id = request_id
                    self._abort_event = threading.Event() if request_id else None

                try:
                    stream = self.llm.create_chat_completion(
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stream=True,
                    )

                    for chunk in stream:
                        delta = chunk["choices"][0].get("delta") or {}
                        token = delta.get("content") or delta.get("text") or ""
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
                    with self._abort_lock:
                        self._abort_request_id = None
                        self._abort_event = None

                reply = "".join(reply_parts)
                if reply:
                    messages.append({"role": "assistant", "content": reply})
                logger.info(
                    "ask_stream llama done chat=%s request_id=%s tokens=%d cancelled=%s",
                    session_id,
                    request_id or "-",
                    len(reply_parts),
                    cancelled,
                )
                session["state"] = self.llm.save_state()
                session["messages"] = messages
                self._active_session_id = session_id
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
                        fresh_state = self.llm.save_state()
                    session = {"state": fresh_state, "messages": system_messages, "compacted": False}
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

                    messages: List[Dict[str, str]] = list(session.get("messages", []))
                    messages.append({"role": "user", "content": message})

                    # Avoid llama.cpp hard failures when prompt_tokens + max_tokens > ctx_size.
                    prompt_tokens = self._count_chat_prompt_tokens(messages)
                    max_tokens = self._clamp_max_tokens(session_id, prompt_tokens, max_tokens)

                    out = self.llm.create_chat_completion(
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    reply = out["choices"][0]["message"]["content"]
                    messages.append({"role": "assistant", "content": reply})

                    # Save updated KV and messages
                    session["state"] = self.llm.save_state()
                    session["messages"] = messages
                    self.sessions[session_id] = session
                    self._active_session_id = session_id
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
        return {
            "chat_id": session_id,
            "used_tokens": used_tokens,
            "capacity_tokens": capacity,
            "percent": round(percent * 100, 2),
            "compacted": bool(session.get("compacted", False)),
        }

    def fork_session(self, session_id: str) -> str:
        session = self.sessions.get(session_id)
        if session is None:
            raise ValueError(f"Unknown session to fork: {session_id}")
        new_session_id = str(uuid.uuid4())
        # Deep copy state to avoid shared reference
        state_copy = pickle.loads(pickle.dumps(session["state"]))
        fork_payload = {
            "state": state_copy,
            "messages": list(session["messages"]),
            "compacted": session.get("compacted", False),
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
            with self._model_lock:
                self.llm.reset()
                session["state"] = self.llm.save_state()
                # If we just reset this session, it's now the active one
                self._active_session_id = session_id
            session["messages"] = [m for m in session.get("messages", []) if m.get("role") == "system"]
            session["compacted"] = False
            self.sessions[session_id] = session
            self._persist_session(session_id, session)

    def delete_session(self, session_id: str) -> None:
        lock = self._locks.setdefault(session_id, threading.RLock())
        with lock:
            self.sessions.pop(session_id, None)
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
                for p in (kv_path, msg_path):
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
                session["state"] = self.llm.save_state()
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
                self.llm.load_state(original_state)
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
                out = self.llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.9,
                    repeat_penalty=1.15,
                    frequency_penalty=0.6,
                    stop=["SUMMARY_END"],
                )
                return out["choices"][0]["message"]["content"].strip()
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
        if not self.persist_dir:
            return
        try:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.warning("Failed to ensure persist_dir %s: %s", self.persist_dir, exc)
            return
        kv_path = self.persist_dir / f"{session_id}.kv"
        msg_path = self.persist_dir / f"{session_id}.json"
        state_obj = session.get("state")
        try:
            if state_obj is not None:
                with kv_path.open("wb") as f:
                    pickle.dump(state_obj, f)
        except Exception as exc:
            logger.warning("Failed to persist KV for %s: %s", session_id, exc)
        try:
            import json
            msg_path.write_text(json.dumps(session.get("messages", []), ensure_ascii=False))
        except Exception as exc:
            logger.warning("Failed to persist messages for %s: %s", session_id, exc)

    def _load_persisted_sessions(self) -> None:
        if not self.persist_dir or not self.persist_dir.exists():
            return
        import json
        for path in self.persist_dir.glob("*.kv"):
            try:
                session_id = path.stem
                with path.open("rb") as f:
                    state = pickle.load(f)
                msg_path = self.persist_dir / f"{session_id}.json"
                messages: List[Dict[str, str]] = []
                if msg_path.exists():
                    try:
                        messages = json.loads(msg_path.read_text()) or []
                    except Exception:
                        messages = []
                self.sessions[session_id] = {"state": state, "messages": messages, "compacted": False}
                self._locks.setdefault(session_id, threading.RLock())
            except Exception:
                continue

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _ensure_session_loaded(self, session_id: str, session: Dict[str, object]) -> None:
        """
        Ensure that the model KV corresponds to the given session.

        If the same session is already active, we skip expensive load_state().
        If we're switching sessions, we reset + load the appropriate state.
        """
        if self._active_session_id == session_id:
            return
        self.llm.reset()
        state = session.get("state")
        if state is not None:
            self.llm.load_state(state)
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

    def _count_chat_prompt_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        Estimate prompt tokens for the chat template (more accurate than joining message content).

        Must be called while holding `_model_lock`.
        """
        try:
            prompt = self.llm.apply_chat_template(messages, tokenize=False)
            return len(self.llm.tokenize(prompt.encode("utf-8")))
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
            if used < 0.9 * self.ctx_size:
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
                    prompt = self.llm.apply_chat_template(new_msgs, tokenize=False)
                    self.llm(prompt, max_tokens=0, cache_prompt=True)
                except Exception:
                    pass
                new_state = self.llm.save_state()
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
                prompt = self.llm.apply_chat_template(new_msgs, tokenize=False)
                # max_tokens=0 + cache_prompt=True builds KV only, no generation
                self.llm(prompt, max_tokens=0, cache_prompt=True)
            except Exception:
                # If feeding fails, fall back to empty state
                pass
            new_state = self.llm.save_state()

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
                    payload = f"[chat:{session_id}] {summary_text}"
                    threading.Thread(
                        target=self.ltm_store.save_memories,
                        args=(session_id, [payload]),
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
