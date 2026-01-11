from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.core.workspace import get_workspace
from backend.services.security import KeyManager
from backend.services.gguf_metadata import detect_chat_template_kind, is_gguf
from backend.services.storage import SQLiteConfig, create_sqlite_store
from backend.api.deps import AppDependencies
from backend.services.connectors.nomic import get_download_state, maybe_start_auto_download
from backend.services.connectors.nomic_onnx import NomicOnnxConfig

logger = logging.getLogger(__name__)


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

    # Read settings without opening Qdrant.
    model_path = ""
    ctx_size = None
    gpu_layers = None
    try:
        store = create_sqlite_store(ws.db, config=SQLiteConfig())
        try:
            mp = store.get_setting("llm_model_path", "")
            model_path = mp if isinstance(mp, str) else ""
            ctx_size = store.get_setting("llm_ctx_size", None)
            gpu_layers = store.get_setting("llm_gpu_layers", None)
        finally:
            store.close()
    except Exception as exc:
        issues.append(
            _issue(
                "settings_unreadable",
                "warning",
                "Settings could not be read yet.",
                "Restart Insight. If the problem persists, use Settings → Reset to reinitialize.",
            )
        )
        logger.warning("Health check settings read failed: %s", exc)

    checks["llm_ctx_size"] = ctx_size
    checks["llm_gpu_layers"] = gpu_layers

    # Model path validation.
    model_path = (model_path or "").strip()
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
    if not embed_ok and download_status in {"idle", "error"}:
        cfg = NomicOnnxConfig()
        required = [cfg.model_filename, "tokenizer.json"]
        try:
            maybe_start_auto_download(embed_base, required_paths=required)
        except Exception as exc:
            logger.debug("Auto-download kickoff failed: %s", exc)
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
    return {"ok": ok, "issues": issues, "checks": checks}


__all__ = ["run_startup_health"]
