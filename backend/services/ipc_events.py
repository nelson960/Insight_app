from __future__ import annotations

from typing import Any, Callable, Dict, Optional

_emit: Optional[Callable[[Dict[str, Any]], None]] = None


def set_ipc_emitter(fn: Optional[Callable[[Dict[str, Any]], None]]) -> None:
    """
    Set the function used to emit IPC out-of-band events.

    In normal HTTP server mode this should remain None.
    In desktop IPC mode, backend/engine.py sets this so other modules can publish events.
    """
    global _emit
    _emit = fn


def emit_event(name: str, **payload: Any) -> None:
    """
    Best-effort event emission. Never raises.

    Example:
      emit_event("files_changed", chat_id="...", file_id="...", filename="...", status="registered")
    """
    fn = _emit
    if not fn:
        return
    try:
        obj: Dict[str, Any] = {"type": "event", "name": name, **payload}
        fn(obj)
    except Exception:
        # Never let event emission break core flows.
        return

