from __future__ import annotations

import json
import logging
import sqlite3
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.core.workspace import Workspace, get_workspace
from backend.api.chat_router_loader import (
    chat_router_error,
    chat_router_loading,
    chat_router_ready,
)

logger = logging.getLogger(__name__)
_startup_health_lock = threading.Lock()
_startup_health_cache: Optional[Dict[str, Any]] = None
_startup_health_ready = False
_startup_health_cached_at: Optional[float] = None
_STARTUP_HEALTH_TTL_SECS = 5.0


def _issue(
    code: str,
    severity: str,
    message: str,
    fix: str,
    action: Optional[str] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "code": code,
        "severity": severity,
        "message": message,
        "fix": fix,
    }
    if action:
        out["action"] = action
    return out


def _set_startup_health_cache(report: Dict[str, Any], *, ready: bool) -> None:
    cached = dict(report)
    checks = dict(cached.get("checks") or {})
    cached_at = time.time()
    checks["startup_health_ready"] = ready
    checks["startup_health_cached_at"] = cached_at
    cached["checks"] = checks

    with _startup_health_lock:
        global _startup_health_cache, _startup_health_ready, _startup_health_cached_at
        if _startup_health_ready and not ready:
            return
        _startup_health_cache = cached
        _startup_health_ready = ready
        _startup_health_cached_at = cached_at


def get_startup_health_cached() -> Optional[Dict[str, Any]]:
    with _startup_health_lock:
        if _startup_health_cache is None:
            return None
        return dict(_startup_health_cache)


def clear_startup_health_cache() -> None:
    with _startup_health_lock:
        global _startup_health_cache, _startup_health_ready, _startup_health_cached_at
        _startup_health_cache = None
        _startup_health_ready = False
        _startup_health_cached_at = None


def _read_settings_quick(db_path: Path, *, timeout_ms: int = 250) -> tuple[str, Any, Any, Optional[str]]:
    model_path = ""
    ctx_size = None
    gpu_layers = None
    if not db_path.exists():
        return model_path, ctx_size, gpu_layers, None

    conn: sqlite3.Connection | None = None
    try:
        timeout = max(0.0, float(timeout_ms) / 1000.0)
        db_uri = f"{db_path.resolve().as_uri()}?mode=ro"
        conn = sqlite3.connect(db_uri, uri=True, timeout=timeout)
        conn.execute(f"PRAGMA busy_timeout={int(timeout_ms)}")
        rows = conn.execute(
            "SELECT key, value_json FROM app_settings WHERE key IN (?,?,?)",
            ("llm_model_path", "llm_ctx_size", "llm_gpu_layers"),
        ).fetchall()
        for key, raw in rows:
            if raw is None:
                continue
            try:
                parsed = json.loads(raw)
            except Exception:
                continue
            if key == "llm_model_path":
                model_path = parsed if isinstance(parsed, str) else ""
            elif key == "llm_ctx_size":
                ctx_size = parsed
            elif key == "llm_gpu_layers":
                gpu_layers = parsed

        return model_path, ctx_size, gpu_layers, None
    except sqlite3.Error as exc:
        err = str(exc)
        if "no such table" in err:
            return model_path, ctx_size, gpu_layers, None
        if "unable to open database file" in err:
            return model_path, ctx_size, gpu_layers, None
        return model_path, ctx_size, gpu_layers, err
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def _resolve_embed_dir_fast(ws: Workspace) -> Path:
    """
    Resolve the embedding model directory without boot-trace logging or heavy deps.
    """
    workspace_dir = ws.base / "em_models" / "nomic-embed-text"
    if not getattr(sys, "frozen", False):
        bundled_dir = Path(__file__).resolve().parents[2] / "backend" / "em_models" / "nomic-embed-text"
        if (bundled_dir / "tokenizer.json").exists() and (bundled_dir / "onnx" / "model.onnx").exists():
            return bundled_dir
    return workspace_dir


def run_startup_health_fast() -> Dict[str, Any]:
    """
    Fast startup checks for UI polling.
    Skips heavy validation (GGUF parsing, keychain) and uses a short SQLite timeout.
    """
    issues: List[Dict[str, Any]] = []
    checks: Dict[str, Any] = {}

    ws = get_workspace()
    from backend.services.connectors.nomic import get_download_state

    # Storage writability check.
    try:
        probe = ws.base / ".health_check.tmp"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        checks["storage_writable"] = True
    except Exception as exc:
        checks["storage_writable"] = False
        issues.append(
            _issue(
                "storage_not_writable",
                "error",
                "The storage folder is not writable.",
                "Close Insight, then make sure you have write access to your Insight data folder.",
            )
        )
        logger.warning("Health check storage write failed: %s", exc)

    model_path, ctx_size, gpu_layers, settings_error = _read_settings_quick(ws.db)
    if settings_error:
        if "locked" not in settings_error.lower():
            issues.append(
                _issue(
                    "settings_unreadable",
                    "warning",
                    "Settings could not be read yet.",
                    "Restart Insight. If the problem persists, use Settings → Reset to reinitialize.",
                )
            )
        logger.warning("Health check settings read failed: %s", settings_error)

    checks["llm_ctx_size"] = ctx_size
    checks["llm_gpu_layers"] = gpu_layers

    model_path = (model_path or "").strip()
    model_ready = False
    if not model_path:
        issues.append(
            _issue(
                "model_not_configured",
                "error",
                "No model is configured yet.",
                "Open Settings → Model and choose a GGUF model file.",
                action="open_settings",
            )
        )
        checks["model_path"] = None
    else:
        p = Path(model_path).expanduser()
        checks["model_path"] = str(p)
        if not p.exists():
            issues.append(
                _issue(
                    "model_missing",
                    "error",
                    "The configured model file cannot be found.",
                    "Open Settings → Model and choose a valid GGUF file.",
                    action="open_settings",
                )
            )
        elif not p.is_file():
            issues.append(
                _issue(
                    "model_not_file",
                    "error",
                    "The configured model path is not a file.",
                    "Open Settings → Model and choose a GGUF file.",
                    action="open_settings",
                )
            )
        elif p.suffix.lower() != ".gguf":
            issues.append(
                _issue(
                    "model_wrong_extension",
                    "error",
                    "The configured model is not a .gguf file.",
                    "Open Settings → Model and choose a GGUF file.",
                    action="open_settings",
                )
            )
        else:
            model_ready = True

    try:
        checks["chat_router_ready"] = chat_router_ready()
        checks["chat_router_loading"] = chat_router_loading()
        chat_err = chat_router_error()
        if chat_err:
            checks["chat_router_error"] = chat_err
    except Exception:
        checks["chat_router_ready"] = False
    if model_ready:
        if checks.get("chat_router_error"):
            issues.append(
                _issue(
                    "chat_router_failed",
                    "error",
                    "Chat engine failed to start.",
                    "Restart Insight. If it persists, re-apply the model in Settings.",
                    action="open_settings",
                )
            )
        elif checks.get("chat_router_loading"):
            issues.append(
                _issue(
                    "chat_router_loading",
                    "warning",
                    "Chat engine is warming up.",
                    "Keep Insight open and try again in a few seconds.",
                )
            )
        elif checks.get("chat_router_ready") is False:
            issues.append(
                _issue(
                    "chat_router_not_ready",
                    "warning",
                    "Chat engine is not ready yet.",
                    "Restart Insight. If it persists, re-apply the model in Settings.",
                    action="open_settings",
                )
            )

    embed_base = _resolve_embed_dir_fast(ws)
    embed_ok = bool((embed_base / "tokenizer.json").exists() and (embed_base / "onnx" / "model.onnx").exists())
    checks["embedding_present"] = embed_ok
    checks["embedding_path"] = str(embed_base)
    download_state = get_download_state()
    download_status = download_state.get("status")
    if not embed_ok and download_status == "ready":
        download_status = "idle"
    checks["embedding_download_status"] = download_status
    checks["embedding_download_error"] = download_state.get("error")
    if not embed_ok and download_status != "downloading":
        issues.append(
            _issue(
                "embedding_missing",
                "warning",
                "Embedding model files are missing.",
                "Open Settings → Embeddings to download the model.",
                action="open_settings",
            )
        )
    if not embed_ok and download_state.get("status") == "downloading":
        issues.append(
            _issue(
                "embedding_downloading",
                "warning",
                "Embedding model is downloading.",
                "Keep Insight open until the download completes.",
                action="open_settings",
            )
        )
    if download_state.get("status") == "error":
        issues.append(
            _issue(
                "embedding_download_failed",
                "warning",
                "Embedding download failed.",
                "Click “Download embeddings” in Settings and keep Insight open during download.",
                action="open_settings",
            )
        )

    ok = not any(issue.get("severity") == "error" for issue in issues)
    report = {"ok": ok, "issues": issues, "checks": checks}
    _set_startup_health_cache(report, ready=False)
    return report


def get_startup_health_report() -> Dict[str, Any]:
    cached = get_startup_health_cached()
    if cached is not None:
        checks = cached.get("checks") or {}
        cached_at = checks.get("startup_health_cached_at")
        if isinstance(cached_at, (int, float)):
            age = time.time() - float(cached_at)
            if age <= _STARTUP_HEALTH_TTL_SECS:
                return cached
        # Cached report is stale; force a fast refresh.
        clear_startup_health_cache()
    return run_startup_health_fast()


def run_startup_health() -> Dict[str, Any]:
    """
    Lightweight startup checks for desktop UX.

    Returns:
      {
        "ok": bool,
        "issues": [ {code,severity,message,fix,action?}, ... ],
        "checks": { ... }  # non-fatal context for UI/debug
      }
    """
    issues: List[Dict[str, Any]] = []
    checks: Dict[str, Any] = {}

    ws = get_workspace()
    from backend.api.deps import AppDependencies
    from backend.services.connectors.nomic import get_download_state
    from backend.services.gguf_metadata import detect_chat_template_kind, is_gguf
    from backend.services.security.key_manager import KeyManager

    # Storage writability check.
    try:
        probe = ws.base / ".health_check.tmp"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        checks["storage_writable"] = True
    except Exception as exc:
        checks["storage_writable"] = False
        issues.append(
            _issue(
                "storage_not_writable",
                "error",
                "The storage folder is not writable.",
                "Close Insight, then make sure you have write access to your Insight data folder.",
            )
        )
        logger.warning("Health check storage write failed: %s", exc)

    # Read settings without opening Qdrant or initializing full SQLite metadata.
    model_path = ""
    ctx_size = None
    gpu_layers = None
    settings_error: Optional[str] = None
    try:
        model_path, ctx_size, gpu_layers, settings_error = _read_settings_quick(ws.db, timeout_ms=500)
    except Exception as exc:
        settings_error = str(exc)
    if settings_error:
        issues.append(
            _issue(
                "settings_unreadable",
                "warning",
                "Settings could not be read yet.",
                "Restart Insight. If the problem persists, use Settings → Reset to reinitialize.",
            )
        )
        logger.warning("Health check settings read failed: %s", settings_error)

    checks["llm_ctx_size"] = ctx_size
    checks["llm_gpu_layers"] = gpu_layers

    # Model path validation.
    model_path = (model_path or "").strip()
    model_ready = False
    if not model_path:
        issues.append(
            _issue(
                "model_not_configured",
                "error",
                "No model is configured yet.",
                "Open Settings → Model and choose a GGUF model file.",
                action="open_settings",
            )
        )
        checks["model_path"] = None
    else:
        p = Path(model_path).expanduser()
        checks["model_path"] = str(p)
        if not p.exists():
            issues.append(
                _issue(
                    "model_missing",
                    "error",
                    "The configured model file cannot be found.",
                    "Open Settings → Model and choose a valid GGUF file.",
                    action="open_settings",
                )
            )
        elif not p.is_file():
            issues.append(
                _issue(
                    "model_not_file",
                    "error",
                    "The configured model path is not a file.",
                    "Open Settings → Model and choose a GGUF file.",
                    action="open_settings",
                )
            )
        elif p.suffix.lower() != ".gguf":
            issues.append(
                _issue(
                    "model_wrong_extension",
                    "error",
                    "The configured model is not a .gguf file.",
                    "Open Settings → Model and choose a GGUF file.",
                    action="open_settings",
                )
            )
        elif not is_gguf(p):
            issues.append(
                _issue(
                    "model_invalid",
                    "error",
                    "The model file does not look like a valid GGUF model.",
                    "Open Settings → Model and choose a different GGUF file.",
                    action="open_settings",
                )
            )
        else:
            template_kind = detect_chat_template_kind(p)
            checks["model_chat_template_kind"] = template_kind
            if template_kind not in {"chatml", "llama3"}:
                issues.append(
                    _issue(
                        "model_template_unsupported",
                        "error",
                        "The model chat template is not supported.",
                        "Choose a GGUF with a ChatML or Llama-3 chat template.",
                        action="open_settings",
                    )
                    )
            else:
                model_ready = True

    try:
        checks["chat_router_ready"] = chat_router_ready()
        checks["chat_router_loading"] = chat_router_loading()
        chat_err = chat_router_error()
        if chat_err:
            checks["chat_router_error"] = chat_err
    except Exception:
        checks["chat_router_ready"] = False
    if model_ready:
        if checks.get("chat_router_error"):
            issues.append(
                _issue(
                    "chat_router_failed",
                    "error",
                    "Chat engine failed to start.",
                    "Restart Insight. If it persists, re-apply the model in Settings.",
                    action="open_settings",
                )
            )
        elif checks.get("chat_router_loading"):
            issues.append(
                _issue(
                    "chat_router_loading",
                    "warning",
                    "Chat engine is warming up.",
                    "Keep Insight open and try again in a few seconds.",
                )
            )
        elif checks.get("chat_router_ready") is False:
            issues.append(
                _issue(
                    "chat_router_not_ready",
                    "warning",
                    "Chat engine is not ready yet.",
                    "Restart Insight. If it persists, re-apply the model in Settings.",
                    action="open_settings",
                )
            )

    # Embedding assets presence (documents will not ingest without them).
    embed_base = AppDependencies.nomic_model_dir()
    embed_ok = bool((embed_base / "tokenizer.json").exists() and (embed_base / "onnx" / "model.onnx").exists())
    checks["embedding_present"] = embed_ok
    checks["embedding_path"] = str(embed_base)
    download_state = get_download_state()
    download_status = download_state.get("status")
    if not embed_ok and download_status == "ready":
        download_status = "idle"
    checks["embedding_download_status"] = download_status
    checks["embedding_download_error"] = download_state.get("error")
    if not embed_ok and download_status != "downloading":
        issues.append(
            _issue(
                "embedding_missing",
                "warning",
                "Embedding model files are missing.",
                "Open Settings → Embeddings to download the model.",
                action="open_settings",
            )
        )
    if not embed_ok and download_state.get("status") == "downloading":
        issues.append(
            _issue(
                "embedding_downloading",
                "warning",
                "Embedding model is downloading.",
                "Keep Insight open until the download completes.",
                action="open_settings",
            )
        )
    if download_state.get("status") == "error":
        issues.append(
            _issue(
                "embedding_download_failed",
                "warning",
                "Embedding download failed.",
                "Click “Download embeddings” in Settings and keep Insight open during download.",
                action="open_settings",
            )
        )

    # Keychain / encryption status (warning only).
    try:
        km = KeyManager(ws)
        keychain_key = km._read_keychain()
        if keychain_key:
            checks["key_storage"] = "keychain"
        elif km.key_path.exists() or km.legacy_key_path.exists():
            checks["key_storage"] = "file"
            issues.append(
                _issue(
                    "keychain_fallback",
                    "warning",
                    "Encryption key is stored locally instead of the macOS Keychain.",
                    "Allow Keychain access for Insight and restart the app.",
                )
            )
        else:
            checks["key_storage"] = "pending"
    except Exception as exc:
        checks["key_storage"] = "unknown"
        issues.append(
            _issue(
                "keychain_unavailable",
                "warning",
                "Keychain access could not be verified.",
                "Restart Insight. If it persists, re-open Settings and try again.",
            )
        )
        logger.warning("Health check keychain failed: %s", exc)

    ok = not any(issue.get("severity") == "error" for issue in issues)
    report = {"ok": ok, "issues": issues, "checks": checks}
    _set_startup_health_cache(report, ready=True)
    return report


__all__ = [
    "run_startup_health",
    "run_startup_health_fast",
    "get_startup_health_report",
    "get_startup_health_cached",
    "clear_startup_health_cache",
]
