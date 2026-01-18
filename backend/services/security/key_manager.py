from __future__ import annotations

import base64
import logging
import os
import subprocess
import sys
from pathlib import Path

from backend.core.workspace import get_workspace

logger = logging.getLogger(__name__)


class KeyManager:
    """Provides access to a master encryption key stored in the workspace."""

    KEY_FILENAME = ".insight_master.key"
    LEGACY_KEY_FILENAME = "master.key"
    KEYCHAIN_SERVICE = "insight-master-key"
    KEYCHAIN_ACCOUNT = "default"

    def __init__(self, workspace=None) -> None:
        self.workspace = workspace or get_workspace()
        self.key_path = self.workspace.base / self.KEY_FILENAME
        self.legacy_key_path = self.workspace.base / "keys" / self.LEGACY_KEY_FILENAME
        self._key: bytes | None = None

    def _read_keychain(self) -> bytes | None:
        if sys.platform != "darwin":
            return None
        try:
            proc = subprocess.run(
                [
                    "security",
                    "find-generic-password",
                    "-s",
                    self.KEYCHAIN_SERVICE,
                    "-a",
                    self.KEYCHAIN_ACCOUNT,
                    "-w",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=1.0,
            )
            if proc.returncode != 0:
                return None
            raw = (proc.stdout or "").strip()
            if not raw:
                return None
            return base64.b64decode(raw.encode("utf-8"))
        except subprocess.TimeoutExpired:
            logger.warning("Keychain read timed out")
            return None
        except Exception:
            logger.warning("Keychain read failed", exc_info=True)
            return None

    def _write_keychain(self, key: bytes) -> bool:
        if sys.platform != "darwin":
            return False
        try:
            encoded = base64.b64encode(key).decode("utf-8")
            proc = subprocess.run(
                [
                    "security",
                    "add-generic-password",
                    "-U",
                    "-s",
                    self.KEYCHAIN_SERVICE,
                    "-a",
                    self.KEYCHAIN_ACCOUNT,
                    "-w",
                    encoded,
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                logger.warning(
                    "Keychain write failed rc=%s stderr=%s",
                    proc.returncode,
                    (proc.stderr or "").strip(),
                )
                return False
            return True
        except Exception:
            logger.warning("Keychain write failed", exc_info=True)
            return False

    def get_key(self) -> bytes:
        if self._key is None:
            key = self._read_keychain()
            if key:
                self._key = key
                return self._key
            if self.key_path.exists():
                self._key = self.key_path.read_bytes()
                if self._write_keychain(self._key):
                    try:
                        self.key_path.unlink()
                        logger.info("Migrated master key to Keychain and removed %s", self.key_path)
                    except Exception:
                        logger.warning("Failed to remove key file %s", self.key_path, exc_info=True)
                else:
                    logger.warning("Using legacy key file at %s", self.key_path)
            elif self.legacy_key_path.exists():
                self._key = self.legacy_key_path.read_bytes()
                if self._write_keychain(self._key):
                    try:
                        self.legacy_key_path.unlink()
                        if self.legacy_key_path.parent.exists() and not any(self.legacy_key_path.parent.iterdir()):
                            self.legacy_key_path.parent.rmdir()
                        logger.info(
                            "Migrated legacy master key to Keychain and removed %s",
                            self.legacy_key_path,
                        )
                    except Exception:
                        logger.warning(
                            "Failed to remove legacy key file %s",
                            self.legacy_key_path,
                            exc_info=True,
                        )
                else:
                    self.key_path.write_bytes(self._key)
                    os.chmod(self.key_path, 0o600)
                    try:
                        self.legacy_key_path.unlink()
                        if self.legacy_key_path.parent.exists() and not any(self.legacy_key_path.parent.iterdir()):
                            self.legacy_key_path.parent.rmdir()
                    except Exception:
                        logger.warning(
                            "Failed to remove legacy key file %s",
                            self.legacy_key_path,
                            exc_info=True,
                        )
                    logger.warning("Migrated legacy key file to %s", self.key_path)
            else:
                self._key = os.urandom(32)
                if self._write_keychain(self._key):
                    logger.info("Stored new master key in Keychain")
                else:
                    self.key_path.parent.mkdir(parents=True, exist_ok=True)
                    self.key_path.write_bytes(self._key)
                    os.chmod(self.key_path, 0o600)
                    logger.warning("Stored master key in file %s", self.key_path)
        return self._key
