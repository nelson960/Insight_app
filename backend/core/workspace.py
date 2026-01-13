from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = _PROJECT_ROOT / "storage"

# Check for workspace directory override (used in bundled builds)
# The Rust sidecar sets INSIGHT_WORKSPACE_DIR=~/.insight when bundled
_WORKSPACE_FROM_ENV = os.getenv("INSIGHT_WORKSPACE_DIR")
if _WORKSPACE_FROM_ENV:
    DEFAULT_DIR = Path(_WORKSPACE_FROM_ENV)


class Workspace:
    """Resolves Insight storage paths under a single workspace root."""

    def __init__(self, base_dir: Optional[Path | str] = None) -> None:
        self.base = Path(base_dir or DEFAULT_DIR).expanduser()
        self._ensure_structure()
        self._ensure_permissions()

    def _ensure_structure(self) -> None:
        for sub in ["uploads", "qdrant", "cache", "logs"]:
            path = self.base / sub
            path.mkdir(parents=True, exist_ok=True)

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

    def reset(self, *, confirm: bool = False, keep_em_models: bool = False) -> None:
        if not confirm:
            raise ValueError("Reset not confirmed.")
        targets = [
            self.uploads,
            self.qdrant,
            self.cache,
            self.db,
            self.logs,
            self.base / "kv_sessions",
        ]
        if not keep_em_models:
            targets.append(self.base / "em_models")
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
