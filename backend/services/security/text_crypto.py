from __future__ import annotations

import base64
from typing import Optional

from .encryption import decrypt_bytes, encrypt_bytes

_ENC_PREFIX = "enc:v1:"


def is_encrypted_text(value: object) -> bool:
    return isinstance(value, str) and value.startswith(_ENC_PREFIX)


def encrypt_text(value: str, *, key: Optional[bytes]) -> str:
    if not isinstance(value, str):
        value = str(value or "")
    if not value:
        return ""
    if key is None:
        return value
    if value.startswith(_ENC_PREFIX):
        return value
    blob = encrypt_bytes(key, value.encode("utf-8"))
    encoded = base64.urlsafe_b64encode(blob).decode("ascii")
    return _ENC_PREFIX + encoded


def decrypt_text(value: str, *, key: Optional[bytes]) -> str:
    if not isinstance(value, str):
        return ""
    if not value:
        return ""
    if not value.startswith(_ENC_PREFIX):
        return value
    if key is None:
        return ""
    payload = value[len(_ENC_PREFIX) :]
    try:
        blob = base64.urlsafe_b64decode(payload.encode("ascii"))
        return decrypt_bytes(key, blob).decode("utf-8", errors="ignore")
    except Exception:
        return ""


__all__ = ["decrypt_text", "encrypt_text", "is_encrypted_text"]
