from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from backend.core.workspace import get_workspace
from backend.runtime_utils import is_packaged
from backend.services.storage.sqlite_store import SQLiteMetadataStore


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str) -> Optional[int]:
    val = os.getenv(name)
    if not val:
        return None
    try:
        return int(val)
    except ValueError:
        return None


def _has_required_assets(base: Path) -> bool:
    """Check if the embedding model directory has required files."""
    return bool((base / "tokenizer.json").exists() and (base / "onnx" / "model.onnx").exists())


@dataclass
class EngineConfig:
    host: str
    port: int
    model_path: Path
    ctx_size: Optional[int]
    n_threads: Optional[int]
    n_gpu_layers: Optional[int]
    default_max_tokens: int
    log_dir: Path
    log_prompts: bool
    log_completions: bool
    log_preview_chars: int
    embedding_path: Optional[Path]
    embedding_model: str
    embedding_auto_download: bool

    @staticmethod
    def _load_model_path_from_settings() -> str:
        db_path = get_workspace().db
        if not db_path.exists():
            return ""
        try:
            store = SQLiteMetadataStore(db_path)
            raw = store.get_setting("llm_model_path", "")
            if isinstance(raw, str):
                return raw.strip()
            return ""
        except Exception:
            return ""

    @staticmethod
    def _resolve_embedding_path() -> Optional[Path]:
        """
        Resolve the embedding model directory.

        Uses the same logic as AppDependencies.nomic_model_dir():
        1) Environment variable INSIGHT_ENGINE_EMBEDDING_PATH
        2) Workspace-local storage/em_models/nomic-embed-text
        3) Bundled backend/em_models/nomic-embed-text (dev mode)
        4) Falls back to workspace-local (will auto-download if enabled)
        """
        # 1) Check environment variable override
        env_path = os.getenv("INSIGHT_ENGINE_EMBEDDING_PATH")
        if env_path:
            path = Path(env_path)
            if _has_required_assets(path):
                return path

        # 2) Check workspace-local storage
        workspace = get_workspace()
        workspace_dir = workspace.base / "em_models" / "nomic-embed-text"
        if _has_required_assets(workspace_dir):
            return workspace_dir

        # 3) Check bundled repo models (dev mode only)
        try:
            project_root = Path(__file__).resolve().parents[3]
            bundled_dir = project_root / "backend" / "em_models" / "nomic-embed-text"
            if _has_required_assets(bundled_dir):
                return bundled_dir
        except Exception:
            pass

        # 4) Fall back to workspace-local for auto-download
        return workspace_dir

    @classmethod
    def from_env(cls) -> "EngineConfig":
        model_path = os.getenv("INSIGHT_ENGINE_MODEL_PATH") or ""
        if not model_path:
            model_path = cls._load_model_path_from_settings()
        if not model_path:
            raise ValueError("INSIGHT_ENGINE_MODEL_PATH is required.")
        host = os.getenv("INSIGHT_ENGINE_HOST", "127.0.0.1")
        if is_packaged() and host not in {"127.0.0.1", "localhost", "::1"}:
            host = "127.0.0.1"
        log_dir = Path(os.getenv("INSIGHT_LOG_DIR") or (Path.home() / ".insight" / "engine_logs"))
        embedding_path = cls._resolve_embedding_path()
        return cls(
            host=host,
            port=int(os.getenv("INSIGHT_ENGINE_PORT", "11435")),
            model_path=Path(model_path),
            ctx_size=_env_int("INSIGHT_ENGINE_CTX"),
            n_threads=_env_int("INSIGHT_ENGINE_THREADS"),
            n_gpu_layers=_env_int("INSIGHT_ENGINE_GPU_LAYERS"),
            default_max_tokens=int(os.getenv("INSIGHT_ENGINE_MAX_TOKENS", "1024")),
            log_dir=log_dir,
            log_prompts=_env_bool("INSIGHT_LOG_PROMPTS", False),
            log_completions=_env_bool("INSIGHT_LOG_COMPLETIONS", False),
            log_preview_chars=int(os.getenv("INSIGHT_LOG_PREVIEW_CHARS", "400")),
            embedding_path=embedding_path,
            embedding_model=os.getenv("INSIGHT_ENGINE_EMBEDDING_MODEL", "nomic-embed-text-v1.5"),
            embedding_auto_download=_env_bool("INSIGHT_ENGINE_EMBEDDING_AUTO_DOWNLOAD", False),
        )
