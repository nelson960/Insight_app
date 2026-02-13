from __future__ import annotations

import logging
import os
import secrets
from typing import Optional

logger = logging.getLogger(__name__)

_IPC_TOKEN_ENV = "INSIGHT_IPC_TOKEN"
_ALLOW_INSECURE_ENV = "INSIGHT_ALLOW_INSECURE_IPC"


def _read_bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def require_ipc_token() -> Optional[str]:
    """
    Return the mandatory IPC token.

    Default behavior is secure: if the token is missing, generate a process-local token
    and require it. `INSIGHT_ALLOW_INSECURE_IPC=1` is an explicit dev-only escape hatch.
    """
    token = (os.getenv(_IPC_TOKEN_ENV) or "").strip()
    if token:
        return token
    if _read_bool_env(_ALLOW_INSECURE_ENV, False):
        return None
    token = secrets.token_hex(16)
    os.environ[_IPC_TOKEN_ENV] = token
    logger.warning(
        "%s was missing; generated an ephemeral token and enforcing token checks.",
        _IPC_TOKEN_ENV,
    )
    return token


__all__ = ["require_ipc_token"]
