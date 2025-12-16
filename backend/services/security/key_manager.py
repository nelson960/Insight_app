from __future__ import annotations

import os
from pathlib import Path

from backend.core.workspace import get_workspace


class KeyManager:
    """Provides access to a master encryption key stored in the workspace."""

    KEY_FILENAME = "master.key"

    def __init__(self, workspace=None) -> None:
        self.workspace = workspace or get_workspace()
        self.key_path = self.workspace.keys / self.KEY_FILENAME
        self._key: bytes | None = None

    def get_key(self) -> bytes:
        if self._key is None:
            if self.key_path.exists():
                self._key = self.key_path.read_bytes()
            else:
                self._key = os.urandom(32)
                self.key_path.parent.mkdir(parents=True, exist_ok=True)
                self.key_path.write_bytes(self._key)
                os.chmod(self.key_path, 0o600)
        return self._key
