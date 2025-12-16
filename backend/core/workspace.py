from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = _PROJECT_ROOT / "storage"


class Workspace:
    """Resolves Insight storage paths under a single workspace root."""

    def __init__(self, base_dir: Optional[Path | str] = None) -> None:
        self.base = Path(base_dir or DEFAULT_DIR).expanduser()
        self._ensure_structure()
        self._ensure_permissions()

    def _ensure_structure(self) -> None:
        for sub in ["uploads", "qdrant", "cache", "logs", "keys", "config"]:
            path = self.base / sub
            path.mkdir(parents=True, exist_ok=True)
        (self.config_dir / "config.yaml").touch(exist_ok=True)

    def _ensure_permissions(self) -> None:
        try:
            os.chmod(self.base, 0o700)
        except PermissionError:
            pass

    @property
    def uploads(self) -> Path:
        return self.base / "uploads"

    @property
    def qdrant(self) -> Path:
        return self.base / "qdrant"

    @property
    def db(self) -> Path:
        return self.base / "db.sqlite"

    @property
    def cache(self) -> Path:
        return self.base / "cache"

    @property
    def logs(self) -> Path:
        return self.base / "logs"

    @property
    def keys(self) -> Path:
        return self.base / "keys"

    @property
    def config_path(self) -> Path:
        return self.config_dir / "config.yaml"

    @property
    def config_dir(self) -> Path:
        return self.base / "config"

    @property
    def runtime_config_path(self) -> Path:
        # Retained for backward compatibility; runtime config now lives in config.yaml.
        return self.config_path

    def save_config(self, cfg: dict) -> None:
        with self.config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(cfg, handle)

    def load_config(self) -> dict:
        if not self.config_path.exists():
            return {}
        with self.config_path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    def reset(self, *, confirm: bool = False) -> None:
        if not confirm:
            raise ValueError("Reset not confirmed.")
        targets = [
            self.uploads,
            self.qdrant,
            self.cache,
            self.db,
            self.logs,
        ]
        for target in targets:
            if not target.exists():
                continue
            if target.is_file():
                target.unlink()
            else:
                shutil.rmtree(target)
        self._ensure_structure()


_workspace_instance: Workspace | None = None


def get_workspace(base_dir: Optional[Path | str] = None) -> Workspace:
    global _workspace_instance
    if base_dir is not None or _workspace_instance is None:
        _workspace_instance = Workspace(base_dir=base_dir)
    return _workspace_instance


__all__ = ["Workspace", "get_workspace", "DEFAULT_DIR"]
