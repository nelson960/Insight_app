from __future__ import annotations

import json
import logging
import os
import socket
import sqlite3
import sys
import threading
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING, Callable

from fastapi import APIRouter, Body, HTTPException, Query

from backend.api.deps import AppDependencies, peek_session_manager, peek_sqlite_store
from backend.core.workspace import get_workspace
from backend.services.gguf_metadata import (
    is_gguf,
    read_gguf_int_kv,
    read_gguf_int_kv_suffix,
    read_gguf_string_kv,
)
from backend.runtime_utils import is_packaged
from backend.services.llama_templates import (
    MODEL_VALIDATION_VERSION,
    build_model_record,
    chat_template_error_message,
    compute_model_fingerprint,
    model_record_is_current,
    validate_chat_template_for_path,
)

if TYPE_CHECKING:
    from backend.services.storage import SQLiteMetadataStore


router = APIRouter(prefix="/settings", tags=["Settings"])
logger = logging.getLogger(__name__)

DEFAULT_CTX_SIZES = [8192, 32768]
RAW_ENGINE_DEFAULTS = {
    "raw_engine_host": "127.0.0.1",
    "raw_engine_port": 11435,
    "raw_engine_model_path": "",
    "raw_engine_ctx": 32768,
    "raw_engine_threads": None,
    "raw_engine_gpu_layers": None,
    "raw_engine_max_tokens": 1024,
    "raw_engine_embedding_model": "nomic-embed-text-v1.5",
    "raw_engine_embedding_auto_download": False,
    "raw_engine_log_preview_chars": 400,
    "raw_engine_log_prompts": False,
    "raw_engine_log_completions": False,
}
RAW_ENGINE_SETTING_KEYS = set(RAW_ENGINE_DEFAULTS.keys())


def _is_sqlite_lock_error(exc: Exception) -> bool:
    """Check if an exception is a SQLite lock/busy error."""
    if not isinstance(exc, sqlite3.OperationalError):
        return False
    error_str = str(exc).lower()
    return (
        "locked" in error_str or
        "disk i/o error" in error_str or
        "database is locked" in error_str or
        "database is busy" in error_str
    )


def _retry_on_lock[T](
    func: Callable[[], T],
    max_retries: int = 3,
    delay_ms: int = 200,
) -> T:
    """Retry a function if it fails with SQLite lock errors.

    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts
        delay_ms: Delay between retries in milliseconds

    Returns:
        The result of the function

    Raises:
        The last exception if all retries fail
    """
    last_exception = None
    for attempt in range(max_retries):
        try:
            return func()
        except sqlite3.OperationalError as exc:
            last_exception = exc
            if _is_sqlite_lock_error(exc) and attempt < max_retries - 1:
                # Wait before retrying with exponential backoff
                time.sleep(delay_ms / 1000 * (attempt + 1))
                continue
            # Re-raise if not a lock error or out of retries
            raise
    # This shouldn't be reached, but just in case
    if last_exception:
        raise last_exception
    raise RuntimeError("Retry failed with no exception")


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


def _validate_chat_template_path(p: Path) -> Tuple[bool, Optional[str], Optional[str]]:
    try:
        validation = validate_chat_template_for_path(p)
    except Exception as exc:
        logger.warning("Chat template validation failed path=%s err=%s", p, exc)
        return False, chat_template_error_message("llama_cpp_unavailable"), None
    if not validation.ok:
        return False, chat_template_error_message(validation.reason), None
    return True, None, validation.template_name


def _validate_chat_template_record(p: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        validation = validate_chat_template_for_path(p)
    except Exception as exc:
        logger.warning("Chat template validation failed path=%s err=%s", p, exc)
        return None, chat_template_error_message("llama_cpp_unavailable")
    if not validation.ok:
        return None, chat_template_error_message(validation.reason)
    record = build_model_record(p, validation)
    return record, None


def _persist_model_record(
    store: "SQLiteMetadataStore",
    *,
    prefix: str,
    record: Dict[str, Any],
) -> None:
    fingerprint = record.get("model_id") or compute_model_fingerprint(Path(record.get("path", "")))
    store.set_setting(f"{prefix}_model_record_json", record)
    store.set_setting(f"{prefix}_model_fingerprint", fingerprint)
    store.set_setting(f"{prefix}_model_validation_version", MODEL_VALIDATION_VERSION)


def _load_model_record_if_current(
    store: "SQLiteMetadataStore",
    *,
    prefix: str,
    path: Path,
) -> Optional[Dict[str, Any]]:
    try:
        record = store.get_setting(f"{prefix}_model_record_json")
    except Exception:
        record = None
    if model_record_is_current(record, path):
        return record
    return None


def _port_available(host: str, port: int) -> tuple[bool, Optional[str]]:
    try:
        with socket.create_server((host, port)):
            return True, None
    except OSError as exc:
        return False, str(exc)


def _normalize_raw_engine_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, default in RAW_ENGINE_DEFAULTS.items():
        val = settings.get(key, default)
        if key in {"raw_engine_host", "raw_engine_model_path", "raw_engine_embedding_model"}:
            if not isinstance(val, str):
                val = default if isinstance(default, str) else ""
            if key != "raw_engine_model_path":
                val = val.strip()
            out[key] = val
            continue
        if key in {"raw_engine_port", "raw_engine_max_tokens", "raw_engine_log_preview_chars"}:
            if not isinstance(val, int):
                val = _safe_int(val, default if isinstance(default, int) else 0)
            if key == "raw_engine_port" and not (1 <= val <= 65535):
                val = default
            if key in {"raw_engine_max_tokens", "raw_engine_log_preview_chars"} and val < 0:
                val = default
            out[key] = val
            continue
        if key in {"raw_engine_ctx", "raw_engine_threads", "raw_engine_gpu_layers"}:
            if val is None:
                out[key] = None
                continue
            if not isinstance(val, int):
                try:
                    val = int(val)
                except Exception:
                    val = default
            if key == "raw_engine_gpu_layers" and isinstance(val, int) and val < -1:
                val = default
            if key in {"raw_engine_ctx", "raw_engine_threads"} and isinstance(val, int) and val <= 0:
                val = default
            out[key] = val
            continue
        if key in {"raw_engine_embedding_auto_download", "raw_engine_log_prompts", "raw_engine_log_completions"}:
            if isinstance(val, bool):
                out[key] = val
            else:
                out[key] = bool(default)
            continue
        out[key] = val
    return out


def _raw_engine_mgr():
    try:
        from backend.services.raw_engine_server.manager import raw_engine_manager
        return raw_engine_manager()
    except ImportError:
        # Raw engine server not available in production builds
        raise RuntimeError(
            "Raw engine mode is not available in this build. "
            "This feature requires development mode with Python source files."
        )


def _read_ctx_max(path: Optional[Path]) -> Optional[int]:
    if not path or not path.exists() or not path.is_file():
        return None
    try:
        val = read_gguf_int_kv_suffix(path, ["context_length"])
    except Exception:
        return None
    if not isinstance(val, int) or val <= 0:
        return None
    return val


def _build_model_info_preview(path: Path) -> Dict[str, Any]:
    try:
        arch = read_gguf_string_kv(path, "general.architecture") or ""
        name = read_gguf_string_kv(path, "general.name") or ""
        size_label = read_gguf_string_kv(path, "general.size_label") or ""
        file_type = read_gguf_int_kv(path, "general.file_type")
        quant_ver = read_gguf_int_kv(path, "general.quantization_version")
        n_layer = read_gguf_int_kv_suffix(path, [".block_count"])
        n_head = read_gguf_int_kv_suffix(path, [".attention.head_count"])
        n_head_kv = read_gguf_int_kv_suffix(path, [".attention.head_count_kv"])
        n_embd = read_gguf_int_kv_suffix(path, [".embedding_length"])
        ctx_train = read_gguf_int_kv_suffix(path, [".context_length"])
        tok_model = read_gguf_string_kv(path, "tokenizer.ggml.model") or read_gguf_string_kv(
            path, "tokenizer.ggml.pre"
        )
        vocab_size = read_gguf_int_kv(path, "tokenizer.ggml.tokens")
    except Exception:
        arch = ""
        name = ""
        size_label = ""
        file_type = None
        quant_ver = None
        n_layer = None
        n_head = None
        n_head_kv = None
        n_embd = None
        ctx_train = None
        tok_model = ""
        vocab_size = None
    return {
        "path": str(path),
        "architecture": arch or None,
        "name": name or None,
        "size_label": size_label or None,
        "file_type": file_type,
        "quantization_version": quant_ver,
        "n_layer": n_layer,
        "n_head": n_head,
        "n_head_kv": n_head_kv,
        "n_embd": n_embd,
        "ctx_train": ctx_train,
        "tokenizer_model": tok_model or None,
        "vocab_size": vocab_size,
    }


def _ctx_options(ctx_max: Optional[int]) -> list[int]:
    opts = list(DEFAULT_CTX_SIZES)
    if isinstance(ctx_max, int) and ctx_max > 0:
        opts = [v for v in opts if v <= ctx_max]
        if ctx_max not in opts:
            opts.append(ctx_max)
    if not opts and isinstance(ctx_max, int) and ctx_max > 0:
        opts = [ctx_max]
    return sorted(set(opts))


def _normalize_ctx_size(v: Any, *, ctx_max: Optional[int], default: int = 32768) -> int:
    try:
        x = int(v)
    except Exception:
        x = default
    if x <= 0:
        x = default
    if isinstance(ctx_max, int) and ctx_max > 0:
        if x > ctx_max:
            x = ctx_max
        if x <= 0:
            x = ctx_max
    return x


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
    existing = peek_sqlite_store()
    if existing is not None:
        return existing, False
    return AppDependencies.sqlite_store(), False


def _read_settings_fast(db_path: Path, *, timeout_ms: int = 250) -> tuple[Dict[str, Any], Optional[str]]:
    """
    Lightweight settings read that avoids SQLiteMetadataStore initialization.

    Returns (settings, error). Missing DB or table returns empty settings with no error.
    """
    if not db_path.exists():
        return {}, None
    conn: sqlite3.Connection | None = None
    try:
        timeout = max(0.0, float(timeout_ms) / 1000.0)
        db_uri = f"{db_path.resolve().as_uri()}?mode=ro"
        conn = sqlite3.connect(db_uri, uri=True, timeout=timeout)
        conn.execute(f"PRAGMA busy_timeout={int(timeout_ms)}")
        rows = conn.execute("SELECT key, value_json FROM app_settings").fetchall()
        settings: Dict[str, Any] = {}
        for key, raw in rows:
            if raw is None:
                continue
            try:
                settings[str(key)] = json.loads(raw)
            except Exception:
                continue
        return settings, None
    except sqlite3.Error as exc:
        err = str(exc)
        if "no such table" in err:
            return {}, None
        if "unable to open database file" in err:
            return {}, None
        return {}, err
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def _resolve_embed_dir_fast() -> Path:
    ws = get_workspace()
    workspace_dir = Path(ws.base) / "em_models" / "nomic-embed-text"
    if not getattr(sys, "frozen", False):
        bundled_dir = Path(__file__).resolve().parents[3] / "backend" / "em_models" / "nomic-embed-text"
        if (bundled_dir / "tokenizer.json").exists() and (bundled_dir / "onnx" / "model.onnx").exists():
            return bundled_dir
    return workspace_dir


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
    settings: Dict[str, Any] = {}
    settings_error: Optional[str] = None
    existing = peek_sqlite_store()
    if existing is not None:
        store, should_close = _settings_store()
        try:
            settings = store.list_settings()
        finally:
            if should_close:
                store.close()
    else:
        settings, settings_error = _read_settings_fast(get_workspace().db)
        if settings_error:
            logger.warning("Fast settings read failed: %s", settings_error)

    # Provide stable defaults even if unset.
    model_path = settings.get("llm_model_path")
    gpu_layers = settings.get("llm_gpu_layers")
    ctx_size = settings.get("llm_ctx_size")
    theme_mode = settings.get("theme_mode")
    raw_engine_settings = _normalize_raw_engine_settings(settings)
    raw_model_path = raw_engine_settings.get("raw_engine_model_path") if isinstance(raw_engine_settings, dict) else ""
    raw_model_path = raw_model_path or model_path
    raw_ctx_max = None
    if isinstance(raw_model_path, str) and raw_model_path:
        raw_ctx_max = _read_ctx_max(Path(raw_model_path).expanduser())
    raw_ctx_sizes = _ctx_options(raw_ctx_max)
    raw_engine_settings["raw_engine_ctx"] = _normalize_ctx_size(
        raw_engine_settings.get("raw_engine_ctx"),
        ctx_max=raw_ctx_max,
        default=ctx_size,
    )
    rag_default_mode = settings.get("rag_default_mode")
    rag_default_detail = settings.get("rag_default_detail")

    if not isinstance(model_path, str) or not model_path:
        model_path = ""
    if not isinstance(gpu_layers, int):
        gpu_layers = 99
    ctx_max = None
    if isinstance(model_path, str) and model_path:
        ctx_max = _read_ctx_max(Path(model_path).expanduser())
    ctx_sizes = _ctx_options(ctx_max)
    ctx_size = _normalize_ctx_size(ctx_size, ctx_max=ctx_max, default=32768)
    if theme_mode not in ("system", "dark", "light"):
        theme_mode = "system"
    if rag_default_mode not in ("small_doc", "rag"):
        rag_default_mode = "small_doc"
    if not isinstance(rag_default_detail, int) or rag_default_detail < 1 or rag_default_detail > 5:
        rag_default_detail = 3

    # Embedding model is fixed (nomic ONNX). Report presence for UX.
    embed_base = _resolve_embed_dir_fast()
    embed_model_path = embed_base / "onnx" / "model.onnx"
    embed_ok = bool((embed_base / "tokenizer.json").exists() and embed_model_path.exists())

    llm_loaded = False
    llm_model_info: Optional[Dict[str, Any]] = None
    try:
        mgr = peek_session_manager()
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
            "rag_default_mode": rag_default_mode,
            "rag_default_detail": rag_default_detail,
            **raw_engine_settings,
        },
        "llm": {
            "ctx_sizes": ctx_sizes,
            "ctx_max": ctx_max,
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
        "raw_engine": {
            "defaults": RAW_ENGINE_DEFAULTS,
            "ctx_sizes": raw_ctx_sizes,
            "ctx_max": raw_ctx_max,
        },
    }


@router.get("/llm/info")
def llm_info(load: bool = Query(False)) -> Dict[str, Any]:
    """
    Return details about the currently loaded LLM (if any).

    This will attempt to initialize the session manager if it doesn't exist yet,
    which triggers model load. Returns info once the LLM is initialized.
    """
    try:
        if load:
            # Try to get the session manager - this will trigger initialization if needed
            mgr = AppDependencies.session_manager()
            return {"loaded": True, "model_info": mgr.model_info()}
        mgr = peek_session_manager()
        if mgr is None:
            return {"loaded": False, "model_info": None}
        return {"loaded": True, "model_info": mgr.model_info()}
    except FileNotFoundError as exc:
        # Model not configured or file not found
        logger.debug(f"LLM not loaded: {exc}")
        return {"loaded": False, "model_info": None, "error": "Model not configured or not found"}
    except Exception as exc:
        logger.error(f"Error getting LLM info: {exc}")
        return {"loaded": False, "model_info": None, "error": str(exc)}


@router.get("/busy")
def busy_state() -> Dict[str, Any]:
    return AppDependencies.busy_state()


@router.get("/health")
def health_state(full: bool = Query(False)) -> Dict[str, Any]:
    """
    Lightweight startup checks for the desktop UI.
    Set full=1 for slower, comprehensive checks.
    """
    if full:
        from backend.services.health import run_startup_health

        return run_startup_health()

    from backend.services.health import get_startup_health_report

    return get_startup_health_report()


@router.get("/raw_engine/status")
def raw_engine_status() -> Dict[str, Any]:
    return _raw_engine_mgr().status()


@router.post("/raw_engine/start")
def raw_engine_start() -> Dict[str, Any]:
    status = _raw_engine_mgr().status()
    if status.get("running") or status.get("starting"):
        return status
    store, should_close = _settings_store()
    try:
        settings = store.list_settings()
    finally:
        if should_close:
            store.close()
    raw_engine_settings = _normalize_raw_engine_settings(settings)
    model_override = raw_engine_settings.get("raw_engine_model_path") if isinstance(raw_engine_settings, dict) else ""
    model_path = model_override or settings.get("llm_model_path", "")
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
    record = None
    store, should_close = _settings_store()
    try:
        record = _load_model_record_if_current(store, prefix="raw_engine", path=p)
    finally:
        if should_close:
            store.close()
    if record is None:
        record, err = _validate_chat_template_record(p)
        if err:
            return {"ok": False, "error": err}
        if record:
            store, should_close = _settings_store()
            try:
                _persist_model_record(store, prefix="raw_engine", record=record)
            finally:
                if should_close:
                    store.close()

    host = raw_engine_settings.get("raw_engine_host", RAW_ENGINE_DEFAULTS["raw_engine_host"])
    port = raw_engine_settings.get("raw_engine_port", RAW_ENGINE_DEFAULTS["raw_engine_port"])
    if not isinstance(host, str) or not host.strip():
        return {"ok": False, "error": "Raw server host is required."}
    try:
        port = int(port)
    except Exception:
        return {"ok": False, "error": "Raw server port must be a number."}
    if port < 1 or port > 65535:
        return {"ok": False, "error": "Raw server port must be between 1 and 65535."}
    try:
        socket.getaddrinfo(host, port)
    except Exception as exc:
        return {"ok": False, "error": f"Raw server host invalid: {exc}"}
    ok, err = _port_available(host, port)
    if not ok:
        return {"ok": False, "error": f"Port {port} unavailable on {host}: {err}"}

    env_overrides: Dict[str, str] = {
        "INSIGHT_ENGINE_HOST": host,
        "INSIGHT_ENGINE_PORT": str(port),
        "INSIGHT_ENGINE_MODEL_PATH": str(p),
        "INSIGHT_ENGINE_MAX_TOKENS": str(raw_engine_settings.get("raw_engine_max_tokens")),
        "INSIGHT_ENGINE_EMBEDDING_MODEL": str(raw_engine_settings.get("raw_engine_embedding_model")),
        "INSIGHT_ENGINE_EMBEDDING_AUTO_DOWNLOAD": "1"
        if raw_engine_settings.get("raw_engine_embedding_auto_download")
        else "0",
        "INSIGHT_LOG_PREVIEW_CHARS": str(raw_engine_settings.get("raw_engine_log_preview_chars")),
        "INSIGHT_LOG_PROMPTS": "1" if raw_engine_settings.get("raw_engine_log_prompts") else "0",
        "INSIGHT_LOG_COMPLETIONS": "1" if raw_engine_settings.get("raw_engine_log_completions") else "0",
    }
    if raw_engine_settings.get("raw_engine_ctx"):
        env_overrides["INSIGHT_ENGINE_CTX"] = str(raw_engine_settings.get("raw_engine_ctx"))
    if raw_engine_settings.get("raw_engine_threads"):
        env_overrides["INSIGHT_ENGINE_THREADS"] = str(raw_engine_settings.get("raw_engine_threads"))
    if raw_engine_settings.get("raw_engine_gpu_layers") is not None:
        env_overrides["INSIGHT_ENGINE_GPU_LAYERS"] = str(raw_engine_settings.get("raw_engine_gpu_layers"))

    return _raw_engine_mgr().start(env_overrides)


@router.post("/raw_engine/stop")
def raw_engine_stop() -> Dict[str, Any]:
    return _raw_engine_mgr().stop()


@router.get("/raw_engine/logs")
def raw_engine_logs(limit: int = Query(250, ge=1, le=250)) -> Dict[str, Any]:
    return {"ok": True, "lines": _raw_engine_mgr().logs(limit)}


@router.post("/embedding/download")
def download_embedding_model() -> Dict[str, Any]:
    from backend.services.connectors.nomic import (
        start_embedding_download_process,
        clear_download_cancel,
        get_download_state,
        set_download_state,
    )
    from backend.services.connectors.nomic_onnx import NomicOnnxConfig
    from backend.services.health import clear_startup_health_cache

    _require_idle("download embedding model")
    clear_download_cancel()
    model_dir = AppDependencies.nomic_model_dir()
    model_dir.mkdir(parents=True, exist_ok=True)
    cfg = NomicOnnxConfig()
    required = [cfg.model_filename, "tokenizer.json"]
    missing = [p for p in required if not (model_dir / p).exists()]
    if not missing:
        set_download_state("ready")
        return {"ok": True, "downloaded": False, "path": str(model_dir), "status": "ready"}
    try:
        result = start_embedding_download_process(model_dir, required_paths=required)
    except Exception as exc:
        set_download_state("error", str(exc))
        return {"ok": False, "error": str(exc)}
    clear_startup_health_cache()
    return {
        "ok": True,
        "downloaded": False,
        "path": str(model_dir),
        "status": result.get("status", "downloading"),
    }


@router.post("/embedding/cancel")
def cancel_embedding_download() -> Dict[str, Any]:
    from backend.services.connectors.nomic import (
        request_download_cancel,
        cancel_embedding_download_process,
        get_download_state,
        set_download_state,
    )
    from backend.services.health import clear_startup_health_cache

    state = get_download_state()
    if state.get("status") != "downloading":
        return {"ok": True, "cancelled": False, "status": state.get("status")}
    request_download_cancel()
    try:
        cancel_embedding_download_process(AppDependencies.nomic_model_dir())
    except Exception:
        pass
    set_download_state("error", "Download cancelled.")
    clear_startup_health_cache()
    return {"ok": True, "cancelled": True, "status": "cancelled"}


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
    raw_keys = [k for k in settings.keys() if k in RAW_ENGINE_SETTING_KEYS]
    if raw_keys:
        status = _raw_engine_mgr().status()
        if status.get("running") or status.get("starting"):
            if should_close:
                store.close()
            return {"ok": False, "error": "Stop the raw server before changing its settings."}

        if "raw_engine_host" in settings:
            host = settings.get("raw_engine_host")
            if not isinstance(host, str) or not host.strip():
                if should_close:
                    store.close()
                return {"ok": False, "error": "raw_engine_host is required"}
            if is_packaged() and host.strip() not in {"127.0.0.1", "localhost", "::1"}:
                if should_close:
                    store.close()
                return {"ok": False, "error": "raw_engine_host must be localhost in bundled builds"}
            store.set_setting("raw_engine_host", host.strip())

        if "raw_engine_port" in settings:
            port = _safe_int(settings.get("raw_engine_port"), -1)
            if port < 1 or port > 65535:
                if should_close:
                    store.close()
                return {"ok": False, "error": "raw_engine_port must be between 1 and 65535"}
            store.set_setting("raw_engine_port", port)

        if "raw_engine_model_path" in settings:
            mp = settings.get("raw_engine_model_path")
            if not isinstance(mp, str):
                if should_close:
                    store.close()
                return {"ok": False, "error": "raw_engine_model_path must be a string"}
            mp = mp.strip()
            if not mp:
                store.set_setting("raw_engine_model_path", "")
            else:
                p = Path(mp).expanduser()
                if not p.exists():
                    if should_close:
                        store.close()
                    return {"ok": False, "error": f"Model file not found: {p}"}
                if not p.is_file():
                    if should_close:
                        store.close()
                    return {"ok": False, "error": f"Not a file: {p}"}
                if p.suffix.lower() != ".gguf":
                    if should_close:
                        store.close()
                    return {"ok": False, "error": "Model must be a .gguf file"}
                if not is_gguf(p):
                    if should_close:
                        store.close()
                    return {"ok": False, "error": "File does not look like a valid GGUF model"}
                record, err = _validate_chat_template_record(p)
                if err:
                    if should_close:
                        store.close()
                    return {"ok": False, "error": err}
                store.set_setting("raw_engine_model_path", str(p))
                if record:
                    _persist_model_record(store, prefix="raw_engine", record=record)

        if "raw_engine_ctx" in settings:
            ctx = _positive_int(settings.get("raw_engine_ctx"))
            raw_ctx_max = None
            raw_mp = settings.get("raw_engine_model_path")
            if not raw_mp:
                try:
                    raw_mp = store.get_setting("raw_engine_model_path")
                except Exception:
                    raw_mp = None
            if not raw_mp:
                try:
                    raw_mp = store.get_setting("llm_model_path")
                except Exception:
                    raw_mp = None
            if isinstance(raw_mp, str) and raw_mp:
                raw_ctx_max = _read_ctx_max(Path(raw_mp).expanduser())
            if ctx is not None:
                ctx = _normalize_ctx_size(ctx, ctx_max=raw_ctx_max, default=ctx)
            store.set_setting("raw_engine_ctx", ctx)

        if "raw_engine_threads" in settings:
            threads = _positive_int(settings.get("raw_engine_threads"))
            store.set_setting("raw_engine_threads", threads)

        if "raw_engine_gpu_layers" in settings:
            gl_raw = settings.get("raw_engine_gpu_layers")
            if gl_raw is None or gl_raw == "":
                store.set_setting("raw_engine_gpu_layers", None)
            else:
                gl = _safe_int(gl_raw, -999)
                if gl < -1:
                    if should_close:
                        store.close()
                    return {"ok": False, "error": "raw_engine_gpu_layers must be -1 or >= 0"}
                store.set_setting("raw_engine_gpu_layers", gl)

        if "raw_engine_max_tokens" in settings:
            max_tokens = _positive_int(settings.get("raw_engine_max_tokens"))
            if max_tokens is None:
                if should_close:
                    store.close()
                return {"ok": False, "error": "raw_engine_max_tokens must be > 0"}
            store.set_setting("raw_engine_max_tokens", max_tokens)

        if "raw_engine_embedding_model" in settings:
            em = settings.get("raw_engine_embedding_model")
            if not isinstance(em, str) or not em.strip():
                if should_close:
                    store.close()
                return {"ok": False, "error": "raw_engine_embedding_model is required"}
            store.set_setting("raw_engine_embedding_model", em.strip())

        if "raw_engine_embedding_auto_download" in settings:
            ad = settings.get("raw_engine_embedding_auto_download")
            if not isinstance(ad, bool):
                if should_close:
                    store.close()
                return {"ok": False, "error": "raw_engine_embedding_auto_download must be true/false"}
            store.set_setting("raw_engine_embedding_auto_download", ad)

        if "raw_engine_log_preview_chars" in settings:
            lp = _non_negative_int(settings.get("raw_engine_log_preview_chars"))
            if lp is None:
                if should_close:
                    store.close()
                return {"ok": False, "error": "raw_engine_log_preview_chars must be >= 0"}
            store.set_setting("raw_engine_log_preview_chars", lp)

        if "raw_engine_log_prompts" in settings:
            lp = settings.get("raw_engine_log_prompts")
            if not isinstance(lp, bool):
                if should_close:
                    store.close()
                return {"ok": False, "error": "raw_engine_log_prompts must be true/false"}
            store.set_setting("raw_engine_log_prompts", lp)

        if "raw_engine_log_completions" in settings:
            lc = settings.get("raw_engine_log_completions")
            if not isinstance(lc, bool):
                if should_close:
                    store.close()
                return {"ok": False, "error": "raw_engine_log_completions must be true/false"}
            store.set_setting("raw_engine_log_completions", lc)

    if "theme_mode" in settings:
        mode = settings.get("theme_mode")
        if mode in ("system", "dark", "light"):
            store.set_setting("theme_mode", mode)
        else:
            if should_close:
                store.close()
            return {"ok": False, "error": "invalid theme_mode"}

    if "rag_default_mode" in settings:
        mode = settings.get("rag_default_mode")
        if mode in ("small_doc", "rag"):
            store.set_setting("rag_default_mode", mode)
        else:
            if should_close:
                store.close()
            return {"ok": False, "error": "invalid rag_default_mode"}

    if "rag_default_detail" in settings:
        detail = _safe_int(settings.get("rag_default_detail"), 3)
        if detail < 1 or detail > 5:
            if should_close:
                store.close()
            return {"ok": False, "error": "rag_default_detail must be between 1 and 5"}
        store.set_setting("rag_default_detail", detail)

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
    try:
        from backend.services.health import clear_startup_health_cache

        clear_startup_health_cache()
    except Exception:
        pass
    return {"ok": True, "restart_required": restart_required}


@router.post("/model/validate")
def validate_model(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    try:
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

        logger.debug("validate_model: Starting validation for %s", p)

        # Try to load cached record from database with a short timeout
        # If database is busy, skip the cache and validate the file directly
        record: Optional[Dict[str, Any]] = None
        try:
            store, should_close = _settings_store()
            try:
                record = _load_model_record_if_current(store, prefix="llm", path=p)
                logger.debug("validate_model: Got cached record for %s: %s", p, "found" if record else "None")
            except sqlite3.OperationalError as exc:
                if _is_sqlite_lock_error(exc):
                    logger.debug("validate_model: Database busy, skipping cache for %s", p)
                    record = None
                else:
                    raise
            finally:
                if should_close:
                    store.close()
        except Exception as exc:
            logger.warning("validate_model: Error loading cached record for %s: %s", p, exc)
            record = None

        # If no cached record, validate the file directly (no database access)
        if record is None:
            record, err = _validate_chat_template_record(p)
            if err:
                return {"ok": False, "error": err}
        try:
            size = p.stat().st_size
        except Exception:
            size = 0
        template_name = None
        fingerprint = None
        if record:
            template_name = record.get("template_selected_name")
            fingerprint = record.get("model_id")
        ctx_max = _read_ctx_max(p)
        ctx_sizes = _ctx_options(ctx_max)
        model_info = _build_model_info_preview(p)
        logger.debug("validate_model: Validation complete for %s", p)
        return {
            "ok": True,
            "model_path": str(p),
            "size_bytes": int(size),
            "note": "Model path looks valid. Full compatibility is verified when the engine loads the model.",
            "chat_template_kind": "embedded",
            "chat_template_name": template_name,
            "model_fingerprint": fingerprint,
            "ctx_max": ctx_max,
            "ctx_sizes": ctx_sizes,
            "model_info": model_info,
        }
    except sqlite3.OperationalError as exc:
        if _is_sqlite_lock_error(exc):
            logger.error("validate_model: Lock error in outer handler: %s", exc)
            return {"ok": False, "error": "Database is busy. Please try again."}
        raise
    except Exception as exc:
        logger.exception("Unexpected error in validate_model: %s", exc)
        return {"ok": False, "error": f"Validation error: {str(exc)}"}


@router.post("/llm/apply")
def apply_llm_settings(payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    """
    Apply model/context settings and clear user data.

    This is destructive by design:
    - deletes DB (messages/files/doc_pages/chunks/jobs), Qdrant vectors, KV sessions, uploads, cache, logs
    - preserves the updated app settings so the engine can restart/reload cleanly
    """
    try:
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
        except sqlite3.OperationalError as exc:
            error_str = str(exc).lower()
            if "locked" in error_str or "disk i/o" in error_str or "database is locked" in error_str:
                return {"ok": False, "error": "Database is busy. Please try again."}
            raise
        finally:
            if should_close:
                store.close()

        if "llm_gpu_layers" in settings:
            merged["llm_gpu_layers"] = _safe_int(settings.get("llm_gpu_layers"), 99)

        ctx_path: Optional[Path] = None
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
            record: Optional[Dict[str, Any]] = None
            # Need to get a fresh store reference since we closed it above
            store2, should_close2 = _settings_store()
            try:
                record = _load_model_record_if_current(store2, prefix="llm", path=p)
            except sqlite3.OperationalError as exc:
                error_str = str(exc).lower()
                if "locked" in error_str or "disk i/o" in error_str or "database is locked" in error_str:
                    record = None  # Continue without cached record
                else:
                    raise
            finally:
                if should_close2:
                    store2.close()

            if record is None:
                record, err = _validate_chat_template_record(p)
                if err:
                    return {"ok": False, "error": err}
            merged["llm_model_path"] = str(p)
            ctx_path = p
            if record:
                merged["llm_model_record_json"] = record
                merged["llm_model_fingerprint"] = record.get("model_id")
                merged["llm_model_validation_version"] = MODEL_VALIDATION_VERSION
        else:
            mp_existing = merged.get("llm_model_path")
            if isinstance(mp_existing, str) and mp_existing:
                ctx_path = Path(mp_existing).expanduser()

        ctx_max = _read_ctx_max(ctx_path)
        if "llm_ctx_size" in settings:
            merged["llm_ctx_size"] = _normalize_ctx_size(
                settings.get("llm_ctx_size"),
                ctx_max=ctx_max,
                default=32768,
            )
        elif ctx_max:
            merged["llm_ctx_size"] = _normalize_ctx_size(
                merged.get("llm_ctx_size"),
                ctx_max=ctx_max,
                default=32768,
            )

        # Remember workspace base (reset_all clears the workspace singleton).
        ws_base = Path(AppDependencies.workspace().base)

        # Reset all user data (closes Qdrant/SQLite first).
        AppDependencies.reset_all(confirm=True, keep_em_models=True)

        # Recreate a fresh DB and restore only app settings. Do not open local Qdrant here
        # (it is expensive and can hold locks until the engine restarts).
        ws = get_workspace(base_dir=ws_base)
        from backend.services.storage.sqlite_store import SQLiteConfig, create_sqlite_store

        restored = create_sqlite_store(ws.db, config=SQLiteConfig())
        for k, v in merged.items():
            restored.set_setting(k, v)
        restored.close()

        return {"ok": True, "restart_required": True}
    except sqlite3.OperationalError as exc:
        error_str = str(exc).lower()
        if "locked" in error_str or "disk i/o" in error_str or "database is locked" in error_str:
            return {"ok": False, "error": "Database is busy. Please try again."}
        raise
    except Exception as exc:
        logger.exception("Unexpected error in apply_llm_settings: %s", exc)
        return {"ok": False, "error": f"Apply error: {str(exc)}"}


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
    from backend.services.connectors.nomic import set_download_state

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
    from backend.services.retrieval.index_maintenance import validate_file_index

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
    from backend.services.retrieval.index_maintenance import repair_file_index

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
