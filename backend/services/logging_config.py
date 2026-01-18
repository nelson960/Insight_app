from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional

from backend.core.workspace import get_workspace

LOG_DIR = get_workspace().logs
LOG_FILE = LOG_DIR / "app.log"


def configure_logging(
    *,
    level: Optional[int] = None,
    log_dir: Optional[Path] = None,
    log_file_name: str = "app.log",
) -> None:
    """
    Configure application-wide logging to write into the logs directory.

    Idempotent: safe to call multiple times.
    """

    target_dir = log_dir or LOG_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    log_path = target_dir / log_file_name

    root = logging.getLogger()
    existing_handler = next(
        (
            handler
            for handler in root.handlers
            if isinstance(handler, logging.FileHandler) and handler.baseFilename == str(log_path)
        ),
        None,
    )
    if existing_handler:
        if level is not None:
            resolved = _resolve_level(level)
            root.setLevel(resolved)
            for handler in root.handlers:
                handler.setLevel(resolved)
        return

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    resolved_level = _resolve_level(level)
    root.setLevel(resolved_level)
    file_handler.setLevel(resolved_level)
    stream_handler.setLevel(resolved_level)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)


def _resolve_level(explicit_level: Optional[int]) -> int:
    if explicit_level is not None:
        return explicit_level
    env_level = os.getenv("LOG_LEVEL") or os.getenv("INSIGHT_LOG_LEVEL")
    if env_level:
        numeric = logging.getLevelName(env_level.upper())
        if isinstance(numeric, int):
            return numeric
    return logging.INFO


__all__ = ["configure_logging", "LOG_DIR", "LOG_FILE"]
