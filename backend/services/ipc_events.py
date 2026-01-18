from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

IpcEmitter = Callable[[Dict[str, Any]], None]


@dataclass
class _IpcState:
    enabled: bool = True
    broken: bool = False
    broken_logged: bool = False


_emit: Optional[IpcEmitter] = None
_state = _IpcState()
_lock = threading.Lock()


def set_ipc_emitter(fn: Optional[IpcEmitter]) -> None:
    """
    Called by backend/engine.py to enable IPC out-of-band events.
    Pass None on shutdown.
    """
    global _emit
    with _lock:
        _emit = fn
        # Reset state if emitter is reattached
        _state.enabled = True
        _state.broken = False
        _state.broken_logged = False


def is_ipc_mode() -> bool:
    """True when the engine has set an emitter and IPC is not disabled."""
    with _lock:
        return _emit is not None and _state.enabled and not _state.broken


def _disable_ipc(reason: str) -> None:
    with _lock:
        _state.enabled = False
        _state.broken = True
        if not _state.broken_logged:
            _state.broken_logged = True
            logger.warning("IPC disabled (%s). Further IPC events will be dropped.", reason)


def emit_event(name: str, **payload: Any) -> None:
    """
    Best-effort event emission. Never raises.
    """
    with _lock:
        fn = _emit
        if fn is None or not _state.enabled or _state.broken:
            return

    obj: Dict[str, Any] = {"type": "event", "name": name, **payload}

    try:
        fn(obj)

    except BrokenPipeError:
        _disable_ipc("broken_pipe")
        return

    except OSError as e:
        if getattr(e, "errno", None) == 32 or "Broken pipe" in str(e):
            _disable_ipc("broken_pipe")
            return
        logger.warning("IPC event emission failed name=%s error=%r", name, e, exc_info=True)
        return

    except Exception as e:
        logger.warning("IPC event emission failed name=%s error=%r", name, e, exc_info=True)
        return
