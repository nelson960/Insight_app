"""
Persistent embedding download status.

Stores embedding download state to ~/.insight/embedding_status.json
so we don't retry failed downloads every startup.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Literal

StatusType = Literal["idle", "downloading", "ready", "error"]


class EmbeddingDownloadStatus:
    """Manage persistent embedding download status."""

    def __init__(self, workspace_dir: Path):
        self._status_path = workspace_dir / "embedding_status.json"
        self._status: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        """Load status from disk."""
        if self._status_path.exists():
            try:
                with open(self._status_path, "r") as f:
                    return json.load(f)
            except Exception:
                pass

        # Default status
        return {
            "status": "idle",
            "last_check": None,
            "error": None,
            "error_timestamp": None,
            "retry_count": 0,
        }

    def _save(self) -> None:
        """Save status to disk."""
        self._status["last_check"] = datetime.now().isoformat()

        with open(self._status_path, "w") as f:
            json.dump(self._status, f, indent=2)

    def get_status(self) -> StatusType:
        """Get current status."""
        return self._status.get("status", "idle")

    def get_error(self) -> str | None:
        """Get last error message."""
        return self._status.get("error")

    def can_retry(self, max_retries: int = 3) -> bool:
        """Check if we should retry download.

        Returns True if we should attempt/retry download:
        - If status is "idle" or "error" AND retry_count < max_retries
        - If status is "downloading" (stale from crashed run) AND retry_count < max_retries
        - If status is "ready" (allows re-download if needed)

        Returns False if:
        - Already exceeded max retries (retry_count >= max_retries)
        """
        status = self.get_status()
        retry_count = self._status.get("retry_count", 0)

        # Hard limit: don't retry if we've exceeded max retries
        if retry_count >= max_retries:
            return False

        # Can retry if status allows it
        # - "idle": haven't tried yet or reset
        # - "error": previous failure, can retry
        # - "downloading": stale state from crashed run, allow retry
        # - "ready": already downloaded, but allow re-download if needed
        return status in {"idle", "error", "downloading", "ready"}

    def set_downloading(self) -> None:
        """Set status to downloading."""
        self._status["status"] = "downloading"
        self._status["error"] = None
        self._status["error_timestamp"] = None
        self._save()

    def set_ready(self) -> None:
        """Set status to ready (success)."""
        self._status["status"] = "ready"
        self._status["error"] = None
        self._status["error_timestamp"] = None
        self._status["retry_count"] = 0  # Reset on success
        self._save()

    def set_error(self, error: str) -> None:
        """Set status to error with message."""
        self._status["status"] = "error"
        self._status["error"] = error
        self._status["error_timestamp"] = datetime.now().isoformat()
        self._status["retry_count"] = self._status.get("retry_count", 0) + 1
        self._save()

    def reset(self) -> None:
        """Reset to idle (for manual retry)."""
        self._status["status"] = "idle"
        self._status["error"] = None
        self._status["error_timestamp"] = None
        self._save()


# Singleton instance
_instance: EmbeddingDownloadStatus | None = None


def get_embedding_status(workspace_dir: Path) -> EmbeddingDownloadStatus:
    """Get the singleton embedding status instance."""
    global _instance
    if _instance is None:
        _instance = EmbeddingDownloadStatus(workspace_dir)
    return _instance
