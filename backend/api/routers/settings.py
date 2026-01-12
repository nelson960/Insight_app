from __future__ import annotations

import os
import socket
import threading
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from fastapi import APIRouter, Body, HTTPException, Query

from backend.api.deps import AppDependencies
from backend.core.workspace import get_workspace
from backend.services.retrieval.index_maintenance import validate_file_index, repair_file_index
from backend.services.health import run_startup_health
from backend.services.gguf_metadata import detect_chat_template_kind, is_gguf
from backend.services.connectors.nomic import (
    MissingDependencyError,
    ensure_local_nomic_model_files,
    get_download_state,
    set_download_state,
)
from backend.services.connectors.nomic_onnx import NomicOnnxConfig
from backend.services.storage import SQLiteConfig, SQLiteMetadataStore, create_sqlite_store
from backend.services.raw_engine_server.manager import raw_engine_manager


router = APIRouter(prefix="/settings", tags=["Settings"])

ALLOWED_CTX_SIZES = {8192, 32768}
RAW_ENGINE_DEFAULTS = {
    "raw_engine_host": "127.0.0.1",
    "raw_engine_port": 11435,
    "raw_engine_model_path": "",
    "raw_engine_ctx": None,
    "raw_engine_threads": None,
    "raw_engine_gpu_layers": None,
    "raw_engine_max_tokens": 1024,
    "raw_engine_embedding_model": "nomic-embed-text-v1.5",
    "raw_engine_embedding_auto_download": True,
    "raw_engine_log_preview_chars": 400,
    "raw_engine_log_prompts": False,
    "raw_engine_log_completions": False,
}
RAW_ENGINE_SETTING_KEYS = set(RAW_ENGINE_DEFAULTS.keys())


def _ensure_bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    return default


def _safe_int(v: Any, default: int) -> int:
    try:
        x = int(v)
        return x
    except Exception:
        return default


def _optional_int(v: Any) -> Optional[int]:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except Exception:
        return None


def _positive_int(v: Any) -> Optional[int]:
    out = _optional_int(v)
    if out is None:
        return None
    return out if out > 0 else None


def _non_negative_int(v: Any) -> Optional[int]:
    out = _optional_int(v)
    if out is None:
        return None
    return out if out >= 0 else None


def _port_available(host: str, port: int) -> tuple[bool, Optional[str]]:
    try:
        with socket.create_server((host, port)):
            return True, None
    except OSError as exc:
        return False, str(exc)


def _normalize_ctx_size(v: Any, default: int = 32768) -> int:
    try:
        x = int(v)
    except Exception:
        return default
    return x if x in ALLOWED_CTX_SIZES else default


def _dir_usage(path: Path) -> Tuple[int, int]:
    """
    Return (bytes, files) under `path` (recursive).
    """
    total = 0
    files = 0
    if not path.exists():
        return 0, 0
    if path.is_file():
        try:
            return path.stat().st_size, 1
        except Exception:
            return 0, 0
    for root, _dirs, filenames in os.walk(path):
        for name in filenames:
            p = Path(root) / name
            try:
                total += p.stat().st_size
                files += 1
            except Exception:
                continue
    return total, files


def _settings_store() -> tuple[SQLiteMetadataStore, bool]:
    """
    Return a SQLite settings store without forcing Qdrant initialization.

    Returns (store, should_close).
    """
    existing = getattr(AppDependencies, "_sqlite_store", None)
    if isinstance(existing, SQLiteMetadataStore):
        return existing, False
    return AppDependencies.sqlite_store(), False


def _require_idle(op: str) -> None:
    state = AppDependencies.busy_state()
    if not state.get("busy"):
        return
    raise HTTPException(
        status_code=409,
        detail={
            "error": f"Cannot {op} while background work is running. Please wait for it to finish.",
            "busy": state,
        },
    )


@router.get("")
def get_settings() -> Dict[str, Any]:
    store, should_close = _settings_store()
    try:
        settings = store.list_settings()
    finally:
        if should_close:
            store.close()

    # Provide stable defaults even if unset.
    model_path = settings.get("llm_model_path")
    gpu_layers = settings.get("llm_gpu_layers")
    ctx_size = settings.get("llm_ctx_size")
    theme_mode = settings.get("theme_mode")

    if not isinstance(model_path, str) or not model_path:
        model_path = ""
    if not isinstance(gpu_layers, int):
        gpu_layers = 99
    if not isinstance(ctx_size, int) or ctx_size not in ALLOWED_CTX_SIZES:
        ctx_size = 32768
    if theme_mode not in ("system", "dark", "light"):
        theme_mode = "system"

    # Embedding model is fixed (nomic ONNX). Report presence for UX.
    embed_base = AppDependencies.nomic_model_dir()
    embed_model_path = embed_base / "onnx" / "model.onnx"
    embed_ok = bool((embed_base / "tokenizer.json").exists() and embed_model_path.exists())

    llm_loaded = False
    llm_model_info: Optional[Dict[str, Any]] = None
    try:
        mgr = getattr(AppDependencies, "_session_manager", None)
        if mgr is not None:
            llm_loaded = True
            llm_model_info = mgr.model_info()
    except Exception:
        llm_loaded = False
        llm_model_info = None

    return {
        "settings": {
            "theme_mode": theme_mode,
            "llm_model_path": model_path,
            "llm_gpu_layers": gpu_layers,
            "llm_ctx_size": ctx_size,
        },
        "llm": {
            "ctx_sizes": sorted(list(ALLOWED_CTX_SIZES)),
            "loaded": llm_loaded,
            "model_info": llm_model_info,
        },
        "embedding": {
            "fixed": True,
            "model": "nomic-embed-text (onnx)",
            "path": str(embed_model_path),
            "present": bool(embed_ok),
            "auto_download": True,
        },
    }


@router.get("/llm/info")
def llm_info() -> Dict[str, Any]:
    """
    Return details about the currently loaded LLM (if any).

    This endpoint does NOT force model load; it only reports info once the LLM is already initialized.
    """
    mgr = getattr(AppDependencies, "_session_manager", None)
    if mgr is None:
        return {"loaded": False, "model_info": None}
    try:
        return {"loaded": True, "model_info": mgr.model_info()}
    except Exception as exc:
        return {"loaded": True, "model_info": None, "error": str(exc)}


@router.get("/busy")
def busy_state() -> Dict[str, Any]:
    return AppDependencies.busy_state()


@router.get("/health")
def health_state() -> Dict[str, Any]:
    """
    Lightweight startup checks for the desktop UI.
    """
    return run_startup_health()


@router.get("/raw_engine/status")
def raw_engine_status() -> Dict[str, Any]:
    return raw_engine_manager().status()


@router.post("/raw_engine/start")
def raw_engine_start() -> Dict[str, Any]:
    store, should_close = _settings_store()
    try:
        model_path = store.get_setting("llm_model_path", "")
    finally:
        if should_close:
            store.close()
    if not isinstance(model_path, str) or not model_path:
        return {"ok": False, "error": "Set a GGUF model path in Settings → Model first."}
    p = Path(model_path).expanduser()
    if not p.exists():
        return {"ok": False, "error": f"Model file not found: {p}"}
    if not p.is_file():
        return {"ok": False, "error": f"Not a file: {p}"}
    if p.suffix.lower() != ".gguf":
        return {"ok": False, "error": "Model must be a .gguf file"}
    if not is_gguf(p):
        return {"ok": False, "error": "File does not look like a valid GGUF model"}
    template_kind = detect_chat_template_kind(p)
    if template_kind not in {"chatml", "llama3"}:
        return {"ok": False, "error": "Unsupported chat template. Use ChatML or Llama-3."}
    return raw_engine_manager().start()


@router.post("/raw_engine/stop")
def raw_engine_stop() -> Dict[str, Any]:
    return raw_engine_manager().stop()


@router.get("/raw_engine/logs")
def raw_engine_logs(limit: int = Query(250, ge=1, le=250)) -> Dict[str, Any]:
    return {"ok": True, "lines": raw_engine_manager().logs(limit)}


@router.post("/embedding/download")
def download_embedding_model() -> Dict[str, Any]:
    _require_idle("download embedding model")
    model_dir = AppDependencies.nomic_model_dir()
    model_dir.mkdir(parents=True, exist_ok=True)
    cfg = NomicOnnxConfig()
    required = [cfg.model_filename, "tokenizer.json"]
    missing = [p for p in required if not (model_dir / p).exists()]
    if not missing:
        set_download_state("ready")
        return {"ok": True, "downloaded": False, "path": str(model_dir), "status": "ready"}
    state = get_download_state()
    if state.get("status") == "downloading":
        return {"ok": True, "downloaded": False, "path": str(model_dir), "status": "downloading"}

    def _run_download() -> None:
        try:
            ensure_local_nomic_model_files(model_dir, required_paths=required)
        except MissingDependencyError as exc:
            set_download_state("error", str(exc))
        except Exception as exc:
            set_download_state("error", str(exc))

    set_download_state("downloading")
    t = threading.Thread(target=_run_download, daemon=True)
    t.start()
    return {"ok": True, "downloaded": False, "path": str(model_dir), "status": "downloading"}


@router.post("")
def set_settings(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """
    Persist app settings. Some settings require restart to take effect.
    """
    settings = payload.get("settings") if isinstance(payload, dict) else None
    if not isinstance(settings, dict):
        return {"ok": False, "error": "missing settings object"}

    store, should_close = _settings_store()

    restart_required = False

    if "theme_mode" in settings:
        mode = settings.get("theme_mode")
        if mode in ("system", "dark", "light"):
            store.set_setting("theme_mode", mode)
        else:
            if should_close:
                store.close()
            return {"ok": False, "error": "invalid theme_mode"}

    # LLM settings (model path / ctx size) are applied via /settings/llm/apply because they
    # require clearing user data (KV sessions + DB + Qdrant) to avoid inconsistent state.
    if "llm_model_path" in settings or "llm_ctx_size" in settings:
        if should_close:
            store.close()
        return {
            "ok": False,
            "error": "Use POST /settings/llm/apply to change model or context length (requires reset).",
        }

    if "llm_gpu_layers" in settings:
        gl = _safe_int(settings.get("llm_gpu_layers"), 99)
        store.set_setting("llm_gpu_layers", gl)
        restart_required = True

    if should_close:
        store.close()
    return {"ok": True, "restart_required": restart_required}


@router.post("/model/validate")
def validate_model(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    model_path = payload.get("model_path") if isinstance(payload, dict) else None
    if not isinstance(model_path, str) or not model_path:
        return {"ok": False, "error": "model_path is required"}
    p = Path(model_path).expanduser()
    if not p.exists():
        return {"ok": False, "error": f"Model file not found: {p}"}
    if not p.is_file():
        return {"ok": False, "error": f"Not a file: {p}"}
    if p.suffix.lower() != ".gguf":
        return {"ok": False, "error": "Model must be a .gguf file"}
    if not is_gguf(p):
        return {"ok": False, "error": "File does not look like a valid GGUF model"}
    template_kind = detect_chat_template_kind(p)
    if template_kind not in {"chatml", "llama3"}:
        return {
            "ok": False,
            "error": "Unsupported chat template. Use a GGUF with ChatML or Llama-3.",
        }
    try:
        size = p.stat().st_size
    except Exception:
        size = 0
    return {
        "ok": True,
        "model_path": str(p),
        "size_bytes": int(size),
        "note": "Model path looks valid. Full compatibility is verified when the engine loads the model.",
        "chat_template_kind": template_kind,
    }


@router.post("/llm/apply")
def apply_llm_settings(payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    """
    Apply model/context settings and clear user data.

    This is destructive by design:
    - deletes DB (messages/files/doc_pages/chunks/jobs), Qdrant vectors, KV sessions, uploads, cache, logs
    - preserves the updated app settings so the engine can restart/reload cleanly
    """
    confirm = payload.get("confirm") if isinstance(payload, dict) else None
    if confirm is not True:
        return {"ok": False, "error": "confirm=true required"}

    _require_idle("apply model settings")

    settings = payload.get("settings") if isinstance(payload, dict) else None
    if not isinstance(settings, dict):
        return {"ok": False, "error": "settings object is required"}

    store, should_close = _settings_store()
    try:
        merged = dict(store.list_settings())
    finally:
        if should_close:
            store.close()

    if "llm_ctx_size" in settings:
        merged["llm_ctx_size"] = _normalize_ctx_size(settings.get("llm_ctx_size"), 32768)

    if "llm_gpu_layers" in settings:
        merged["llm_gpu_layers"] = _safe_int(settings.get("llm_gpu_layers"), 99)

    if "llm_model_path" in settings:
        mp = settings.get("llm_model_path")
        if not isinstance(mp, str) or not mp:
            return {"ok": False, "error": "invalid llm_model_path"}
        p = Path(mp).expanduser()
        if not p.exists():
            return {"ok": False, "error": f"Model file not found: {p}"}
        if not p.is_file():
            return {"ok": False, "error": f"Not a file: {p}"}
        if p.suffix.lower() != ".gguf":
            return {"ok": False, "error": "Model must be a .gguf file"}
        if not is_gguf(p):
            return {"ok": False, "error": "File does not look like a valid GGUF model"}
        template_kind = detect_chat_template_kind(p)
        if template_kind not in {"chatml", "llama3"}:
            return {"ok": False, "error": "Unsupported chat template. Use ChatML or Llama-3."}
        merged["llm_model_path"] = str(p)

    # Remember workspace base (reset_all clears the workspace singleton).
    ws_base = Path(AppDependencies.workspace().base)

    # Reset all user data (closes Qdrant/SQLite first).
    AppDependencies.reset_all(confirm=True, keep_em_models=True)

    # Recreate a fresh DB and restore only app settings. Do not open local Qdrant here
    # (it is expensive and can hold locks until the engine restarts).
    ws = get_workspace(base_dir=ws_base)
    restored = create_sqlite_store(ws.db, config=SQLiteConfig())
    for k, v in merged.items():
        restored.set_setting(k, v)
    restored.close()

    return {
        "ok": True,
        "reset_done": True,
        "settings": {
            "llm_model_path": merged.get("llm_model_path"),
            "llm_gpu_layers": merged.get("llm_gpu_layers"),
            "llm_ctx_size": merged.get("llm_ctx_size"),
        },
    }


@router.get("/storage")
def storage_usage() -> Dict[str, Any]:
    ws = AppDependencies.workspace()
    base = Path(ws.base)

    # Workspace structure includes: uploads, qdrant, cache, logs, db.sqlite.
    parts = {
        "uploads": base / "uploads",
        "qdrant": base / "qdrant",
        "kv_sessions": base / "kv_sessions",
        "cache": base / "cache",
        "logs": base / "logs",
        "db": base / "db.sqlite",
    }
    breakdown: Dict[str, Any] = {}
    # Total covers *everything* under storage/, including any future folders.
    total, total_files = _dir_usage(base)
    known_total = 0
    known_files = 0
    for name, p in parts.items():
        b, f = _dir_usage(p)
        breakdown[name] = {"bytes": b, "files": f, "path": str(p)}
        known_total += b
        known_files += f

    other_bytes = max(0, total - known_total)
    other_files = max(0, total_files - known_files)
    if other_bytes or other_files:
        breakdown["other"] = {"bytes": other_bytes, "files": other_files, "path": str(base)}

    return {
        "base": str(base),
        "total_bytes": total,
        "total_files": total_files,
        "breakdown": breakdown,
    }


@router.post("/storage/clean_cache")
def clean_cache(payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    """
    Deletes only temporary cache files under storage/cache (and optionally trims logs).
    """
    _require_idle("clean cache")
    ws = AppDependencies.workspace()
    cache_dir = Path(ws.cache)
    logs_dir = Path(ws.logs)

    trim_logs = _ensure_bool(payload.get("trim_logs") if isinstance(payload, dict) else False, False)

    before_cache, _ = _dir_usage(cache_dir)
    before_logs, _ = _dir_usage(logs_dir) if trim_logs else (0, 0)

    removed_cache_files = 0
    if cache_dir.exists():
        for entry in cache_dir.iterdir():
            try:
                if entry.is_file():
                    entry.unlink()
                    removed_cache_files += 1
                else:
                    shutil.rmtree(entry)
                    removed_cache_files += 1
            except Exception:
                continue

    removed_log_files = 0
    if trim_logs and logs_dir.exists():
        for entry in logs_dir.iterdir():
            try:
                if entry.is_file():
                    entry.unlink()
                    removed_log_files += 1
                else:
                    shutil.rmtree(entry)
                    removed_log_files += 1
            except Exception:
                continue

    after_cache, _ = _dir_usage(cache_dir)
    after_logs, _ = _dir_usage(logs_dir) if trim_logs else (0, 0)

    freed = max(0, (before_cache + before_logs) - (after_cache + after_logs))
    return {
        "ok": True,
        "freed_bytes": freed,
        "removed": {
            "cache_entries": removed_cache_files,
            "log_entries": removed_log_files,
        },
    }


@router.post("/storage/reset")
def reset_storage(payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    """
    Deletes *all* user data in the workspace (SQLite, Qdrant, KV sessions, uploads, cache, logs).
    Requires restart to reinitialize clients.
    """
    confirm = payload.get("confirm") if isinstance(payload, dict) else None
    if confirm is not True:
        return {"ok": False, "error": "confirm=true required"}

    try:
        _require_idle("reset")
        AppDependencies.reset_all(confirm=True, keep_em_models=False)
        set_download_state("idle")
        return {"ok": True, "restart_required": True}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@router.get("/index/validate/{file_id}")
def validate_index(file_id: str) -> Dict[str, Any]:
    """
    Validate Qdrant↔SQLite consistency for a single file_id.

    This is a diagnostic endpoint to detect missing/orphan chunk vectors.
    """
    store, vector_index = AppDependencies.storage()
    res = validate_file_index(
        sqlite_store=store,
        qdrant_client=vector_index.client,
        collection_name=vector_index.collection_name,
        file_id=file_id,
    )
    return {"ok": True, "validation": res.to_dict()}


@router.post("/index/repair")
def repair_index(payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    """
    Best-effort index repair for a single file_id (rare corruption/partial writes).

    This re-embeds missing chunks from SQLite and upserts them into Qdrant.
    """
    _require_idle("repair index")
    file_id = payload.get("file_id") if isinstance(payload, dict) else None
    if not isinstance(file_id, str) or not file_id.strip():
        return {"ok": False, "error": "file_id is required"}

    embedder = AppDependencies.query_embedder()
    if embedder is None:
        return {"ok": False, "error": "embedder not available"}

    store, vector_index = AppDependencies.storage()
    dry_run = _ensure_bool(payload.get("dry_run") if isinstance(payload, dict) else False, False)
    delete_orphans = _ensure_bool(payload.get("delete_orphans") if isinstance(payload, dict) else False, False)
    max_repairs = _safe_int(payload.get("max_repairs") if isinstance(payload, dict) else None, 5000)

    result = repair_file_index(
        sqlite_store=store,
        qdrant_client=vector_index.client,
        collection_name=vector_index.collection_name,
        file_id=file_id.strip(),
        embedder=embedder,
        max_repairs=max_repairs,
        delete_orphans=delete_orphans,
        dry_run=dry_run,
    )
    return result


__all__ = ["router"]
