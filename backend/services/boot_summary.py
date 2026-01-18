"""
Backend self-report JSON summary for user support.

Writes ~/.insight/logs/boot_summary_latest.json on startup containing:
- Versions (Python, platform)
- Packaged status (sys.frozen, _MEIPASS)
- Resolved workspace
- Certificate path and validation
- Dependency status (onnx, tokenizers, llama)
- Last error (if any)

This is 10x easier for user support than asking users to paste long logs.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from backend.core.workspace import get_workspace


def generate_boot_summary() -> Dict[str, Any]:
    """
    Generate a comprehensive boot summary.

    Returns:
        Dict with all boot information
    """
    summary = {
        "timestamp": datetime.now().isoformat(),
        "versions": {},
        "packaged": {},
        "workspace": {},
        "certificates": {},
        "dependencies": {},
        "errors": [],
    }

    # Versions
    summary["versions"]["python"] = sys.version
    summary["versions"]["python_executable"] = sys.executable
    summary["versions"]["platform"] = sys.platform

    # Packaged status
    is_frozen = getattr(sys, 'frozen', False)
    meipass = getattr(sys, '_MEIPASS', None)

    summary["packaged"]["is_frozen"] = is_frozen
    summary["packaged"]["meipass"] = meipass
    summary["packaged"]["meipass_exists"] = meipass and Path(meipass).exists() if meipass else False

    # Workspace
    try:
        ws = get_workspace()
        summary["workspace"]["base"] = str(ws.base)
        summary["workspace"]["exists"] = ws.base.exists()
        summary["workspace"]["writable"] = _check_writable(ws.base)

        # Check subdirectories
        for name in ["uploads", "qdrant", "cache", "logs", "kv_sessions", "em_models"]:
            subdir = getattr(ws, name, None)
            if subdir:
                summary["workspace"][f"{name}_exists"] = subdir.exists()
    except Exception as e:
        summary["errors"].append({
            "category": "workspace",
            "error": str(e),
        })

    # Certificates
    try:
        import certifi
        cert_path = certifi.where()
        cert_exists = Path(cert_path).exists()

        summary["certificates"]["certifi_bundle"] = cert_path
        summary["certificates"]["exists"] = cert_exists
        summary["certificates"]["env_ca_bundle"] = os.environ.get("REQUESTS_CA_BUNDLE")
        summary["certificates"]["env_cert_file"] = os.environ.get("SSL_CERT_FILE")

        if cert_exists:
            summary["certificates"]["size"] = Path(cert_path).stat().st_size
    except ImportError:
        summary["certificates"]["error"] = "certifi not installed"
        summary["errors"].append({
            "category": "certificates",
            "error": "certifi not installed",
        })
    except Exception as e:
        summary["certificates"]["error"] = str(e)
        summary["errors"].append({
            "category": "certificates",
            "error": str(e),
        })

    # Dependencies
    dependencies = {
        "onnxruntime": _check_onnxruntime,
        "llama_cpp": _check_llama_cpp,
        "tokenizers": _check_tokenizers,
        "sqlite": _check_sqlite,
        "qdrant": _check_qdrant,
    }

    for name, check_func in dependencies.items():
        try:
            result = check_func()
            summary["dependencies"][name] = result
            if result.get("error"):
                summary["errors"].append({
                    "category": "dependency",
                    "dependency": name,
                    "error": result.get("error"),
                })
        except Exception as e:
            summary["dependencies"][name] = {
                "available": False,
                "error": str(e),
            }
            summary["errors"].append({
                "category": "dependency",
                "dependency": name,
                "error": str(e),
            })

    return summary


def _check_writable(path: Path) -> bool:
    """Check if a path is writable."""
    try:
        probe = path / ".write_test.tmp"
        probe.write_text("test")
        probe.unlink()
        return True
    except Exception:
        return False


def _check_onnxruntime() -> Dict[str, Any]:
    """Check ONNX Runtime."""
    result = {
        "available": False,
        "version": None,
        "providers": None,
    }

    try:
        import onnxruntime as ort
        result["available"] = True
        result["version"] = ort.__version__
        result["providers"] = ort.get_available_providers()
    except Exception as e:
        result["error"] = str(e)

    return result


def _check_llama_cpp() -> Dict[str, Any]:
    """Check llama.cpp."""
    result = {
        "available": False,
        "version": None,
        "native_lib": False,
    }

    try:
        import llama_cpp
        result["available"] = True
        result["version"] = getattr(llama_cpp, "__version__", "unknown")

        try:
            from llama_cpp import llama_cpp_lib
            result["native_lib"] = True
        except ImportError as e:
            result["native_lib_error"] = str(e)
    except Exception as e:
        result["error"] = str(e)

    return result


def _check_tokenizers() -> Dict[str, Any]:
    """Check tokenizers."""
    result = {
        "available": False,
        "version": None,
    }

    try:
        from tokenizers import Tokenizer
        result["available"] = True

        try:
            import tokenizers
            result["version"] = getattr(tokenizers, "__version__", "unknown")
        except Exception:
            pass
    except Exception as e:
        result["error"] = str(e)

    return result


def _check_sqlite() -> Dict[str, Any]:
    """Check SQLite."""
    result = {
        "available": True,
        "version": None,
        "thread_safety": None,
    }

    try:
        import sqlite3
        result["version"] = sqlite3.sqlite_version
        result["thread_safety"] = sqlite3.threadsafety
    except Exception as e:
        result["error"] = str(e)

    return result


def _check_qdrant() -> Dict[str, Any]:
    """Check Qdrant."""
    result = {
        "available": False,
    }

    try:
        from qdrant_client import QdrantClient
        result["available"] = True
    except Exception as e:
        result["error"] = str(e)

    return result


def write_boot_summary() -> Path:
    """
    Write boot summary to JSON file.

    Returns:
        Path to the written summary file
    """
    try:
        ws = get_workspace()
        log_dir = ws.logs
    except Exception:
        # Fallback if workspace not ready
        log_dir = Path.home() / ".insight" / "logs"

    log_dir.mkdir(parents=True, exist_ok=True)

    # Generate summary
    summary = generate_boot_summary()

    # Write to boot_summary_latest.json (overwrites)
    summary_path = log_dir / "boot_summary_latest.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Also write to timestamped file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_path = log_dir / f"boot_summary_{timestamp}.json"
    with open(timestamped_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary_path


def get_boot_summary() -> Dict[str, Any]:
    """
    Get the current boot summary.

    Returns:
        Boot summary dict, or empty dict if not available
    """
    try:
        ws = get_workspace()
        summary_path = ws.logs / "boot_summary_latest.json"

        if summary_path.exists():
            with open(summary_path, "r") as f:
                return json.load(f)
    except Exception:
        pass

    return {}
