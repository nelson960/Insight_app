from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

_app: Optional["FastAPI"] = None
_state_lock = threading.Lock()
_chat_loaded = False
_chat_loading = False
_chat_error: Optional[str] = None


def register_app(app: "FastAPI") -> None:
    global _app
    _app = app


def chat_router_ready() -> bool:
    with _state_lock:
        return _chat_loaded


def chat_router_loading() -> bool:
    with _state_lock:
        return _chat_loading


def chat_router_error() -> Optional[str]:
    with _state_lock:
        return _chat_error


def _read_model_path_quick(db_path: Path, *, timeout_ms: int = 250) -> str:
    if not db_path.exists():
        return ""
    conn: sqlite3.Connection | None = None
    try:
        timeout = max(0.0, float(timeout_ms) / 1000.0)
        db_uri = f"{db_path.resolve().as_uri()}?mode=ro"
        conn = sqlite3.connect(db_uri, uri=True, timeout=timeout)
        conn.execute(f"PRAGMA busy_timeout={int(timeout_ms)}")
        row = conn.execute("SELECT value_json FROM app_settings WHERE key=?", ("llm_model_path",)).fetchone()
        if not row or row[0] is None:
            return ""
        try:
            raw = json.loads(row[0])
            return raw if isinstance(raw, str) else ""
        except Exception:
            return ""
    except sqlite3.Error as exc:
        err = str(exc)
        if "no such table" in err:
            return ""
        if "unable to open database file" in err:
            return ""
        return ""
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def _model_configured() -> bool:
    try:
        from backend.core.workspace import get_workspace
    except Exception:
        return False
    try:
        model_path = _read_model_path_quick(get_workspace().db)
        if not model_path:
            return False
        p = Path(model_path).expanduser()
        return p.exists() and p.is_file()
    except Exception:
        return False


def maybe_start_chat_router_load(reason: str = "startup") -> bool:
    global _chat_loading, _chat_loaded, _chat_error
    with _state_lock:
        if _chat_loaded or _chat_loading:
            return False
        app = _app
        if app is None:
            return False
        if not _model_configured():
            return False
        _chat_loading = True
        _chat_error = None

    def _load() -> None:
        nonlocal app
        global _chat_loaded, _chat_error, _chat_loading
        try:
            from backend.api.routers.chat import router as chat_router

            app.include_router(chat_router)
            with _state_lock:
                _chat_loaded = True
            try:
                from backend.services.ipc_events import emit_event

                emit_event("chat_router_ready", ok=True, reason=reason)
            except Exception:
                pass
            logger.info("Chat router loaded (%s)", reason)
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            with _state_lock:
                _chat_error = err
            try:
                from backend.services.ipc_events import emit_event

                emit_event("chat_router_ready", ok=False, error=err, reason=reason)
            except Exception:
                pass
            logger.exception("Chat router load failed (%s): %s", reason, err)
        finally:
            with _state_lock:
                _chat_loading = False

    threading.Thread(target=_load, name="insight-chat-router-loader", daemon=True).start()
    return True
