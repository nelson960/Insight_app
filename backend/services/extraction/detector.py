from __future__ import annotations

import logging
import mimetypes
from pathlib import Path

try:
    import magic  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    magic = None

logger = logging.getLogger(__name__)


def detect_mime_type(file_path: Path) -> str:
    """
    Best-effort MIME detection using python-magic with mimetypes fallback.
    """

    if magic:
        try:
            detected = magic.from_file(str(file_path), mime=True)
            if detected and detected != "application/octet-stream":
                logger.debug("magic detected mime %s for %s", detected, file_path)
                return detected
        except Exception as exc:
            logger.warning("magic mime detection failed for %s: %s", file_path, exc)
    guessed, _ = mimetypes.guess_type(str(file_path))
    mime = guessed or "application/octet-stream"
    logger.debug("mimetypes fallback detected mime %s for %s", mime, file_path)
    return mime


__all__ = ["detect_mime_type"]
