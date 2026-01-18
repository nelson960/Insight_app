"""
Comprehensive diagnostics endpoint for packaging issues.

This endpoint provides detailed information about the runtime environment,
including SSL certificates, native libraries, and dependencies.

Usage:
    GET /diagnostics/packaging

Returns:
    JSON with diagnostic information about the packaged environment.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter

from backend.core.workspace import get_workspace
from backend.runtime_utils import get_bundle_info, verify_ssl_connection

router = APIRouter(prefix="/diagnostics", tags=["Diagnostics"])


def _check_certificates() -> Dict[str, Any]:
    """Check SSL certificate configuration."""
    result = {
        "certifi_available": False,
        "certifi_bundle": None,
        "certifi_exists": False,
        "env_request_ca_bundle": os.environ.get("REQUESTS_CA_BUNDLE"),
        "env_ssl_cert_file": os.environ.get("SSL_CERT_FILE"),
        "env_cert_dir": os.environ.get("SSL_CERT_DIR"),
        "env_curl_ca_bundle": os.environ.get("CURL_CA_BUNDLE"),
    }

    try:
        import certifi
        result["certifi_available"] = True
        bundle = certifi.where()
        result["certifi_bundle"] = bundle
        result["certifi_exists"] = Path(bundle).exists()

        # Try to read the bundle
        if result["certifi_exists"]:
            try:
                bundle_path = Path(bundle)
                result["certifi_size"] = bundle_path.stat().st_size
                result["certifi_readable"] = bundle_path.is_file()
                result["certifi_absolute"] = str(bundle_path.absolute())
            except Exception as e:
                result["certifi_error"] = str(e)

        # Verify which cert path is actually being used
        # Priority: REQUESTS_CA_BUNDLE > SSL_CERT_FILE > certifi.where()
        active_cert = os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("SSL_CERT_FILE") or bundle
        result["active_cert_bundle"] = active_cert
        result["active_cert_exists"] = Path(active_cert).exists() if active_cert else False
    except ImportError:
        result["certifi_error"] = "certifi module not installed"

    return result


def _check_onnxruntime() -> Dict[str, Any]:
    """Check ONNX Runtime installation."""
    result = {
        "available": False,
        "version": None,
        "providers": None,
        "device": None,
        "error": None,
    }

    try:
        import onnxruntime as ort
        result["available"] = True
        result["version"] = ort.__version__

        try:
            result["providers"] = ort.get_available_providers()
        except Exception as e:
            result["providers_error"] = str(e)

        try:
            result["device"] = ort.get_device() if hasattr(ort, 'get_device') else "unknown"
        except Exception as e:
            result["device_error"] = str(e)

        # Try to create a dummy session
        try:
            # This tests if the native libraries are properly loaded
            from onnxruntime import backend
            result["backend"] = backend.name
        except Exception as e:
            result["backend_error"] = str(e)

    except ImportError as e:
        result["error"] = f"ImportError: {e}"
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    return result


def _check_llama_cpp() -> Dict[str, Any]:
    """Check llama.cpp installation."""
    result = {
        "available": False,
        "version": None,
        "module_path": None,
        "native_lib_loaded": False,
        "error": None,
    }

    try:
        import llama_cpp
        result["available"] = True
        result["version"] = getattr(llama_cpp, "__version__", "unknown")
        result["module_path"] = llama_cpp.__file__

        # Check if native library is accessible
        try:
            from llama_cpp import llama_cpp_lib
            result["native_lib_loaded"] = True
            result["native_lib"] = str(llama_cpp_lib)
        except ImportError as e:
            result["native_lib_error"] = str(e)

        # Check for GGUF support
        try:
            from llama_cpp import llama_gguf_file
            result["gguf_support"] = True
        except ImportError:
            result["gguf_support"] = False

    except ImportError as e:
        result["error"] = f"ImportError: {e}"
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    return result


def _check_tokenizers() -> Dict[str, Any]:
    """Check tokenizers library installation."""
    result = {
        "available": False,
        "version": None,
        "error": None,
    }

    try:
        from tokenizers import Tokenizer
        result["available"] = True

        try:
            import tokenizers
            result["version"] = getattr(tokenizers, "__version__", "unknown")
        except Exception:
            pass

        # Try to create a simple tokenizer
        try:
            tok = Tokenizer.from_pretrained("bert-base-uncased")
            result["test_tokenizer_loaded"] = True
        except Exception as e:
            result["test_tokenizer_error"] = str(e)

    except ImportError as e:
        result["error"] = f"ImportError: {e}"
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    return result


def _check_sqlite() -> Dict[str, Any]:
    """Check SQLite database."""
    result = {
        "version": None,
        "thread_safety": None,
        "compile_options": None,
        "database_path": None,
        "database_exists": False,
        "database_writable": False,
    }

    try:
        import sqlite3
        result["version"] = sqlite3.sqlite_version
        result["thread_safety"] = sqlite3.threadsafety

        try:
            conn = sqlite3.connect(":memory:")
            result["compile_options"] = [row[0] for row in conn.execute("PRAGMA compile_options").fetchall()]
            conn.close()
        except Exception as e:
            result["compile_options_error"] = str(e)

        # Check workspace database
        try:
            ws = get_workspace()
            db_path = ws.db
            result["database_path"] = str(db_path)
            result["database_exists"] = db_path.exists()

            if result["database_exists"]:
                # Test writability
                test_path = db_path.parent / ".write_test.tmp"
                try:
                    test_path.write_text("test")
                    test_path.unlink()
                    result["database_writable"] = True
                except Exception as e:
                    result["database_write_error"] = str(e)
        except Exception as e:
            result["database_check_error"] = str(e)

    except ImportError as e:
        result["error"] = f"ImportError: {e}"
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    return result


def _check_workspace() -> Dict[str, Any]:
    """Check workspace configuration."""
    result = {}

    try:
        ws = get_workspace()
        result["workspace_base"] = str(ws.base)
        result["workspace_exists"] = ws.base.exists()

        if result["workspace_exists"]:
            # Check subdirectories
            result["subdirectories"] = {}
            for name in ["uploads", "qdrant", "cache", "logs", "kv_sessions", "em_models"]:
                subdir = getattr(ws, name, None)
                if subdir:
                    exists = subdir.exists()
                    result["subdirectories"][name] = {
                        "path": str(subdir),
                        "exists": exists,
                    }

        # Check environment variable
        result["env_workspace_dir"] = os.getenv("INSIGHT_WORKSPACE_DIR")

    except Exception as e:
        result["error"] = str(e)

    return result


def _check_ssl_connection() -> Dict[str, Any]:
    """Test SSL connection to HuggingFace."""
    result = {
        "test_url": "https://huggingface.co",
        "success": False,
        "error": None,
        "cert_used": None,
    }

    try:
        import urllib.request
        import ssl

        cert_bundle = os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("SSL_CERT_FILE")
        result["cert_used"] = cert_bundle

        context = ssl.create_default_context()
        with urllib.request.urlopen("https://huggingface.co", context=context, timeout=5) as response:
            result["success"] = response.status == 200
    except Exception as e:
        result["error"] = str(e)

    return result


def _check_embedding_models() -> Dict[str, Any]:
    """Check embedding model availability."""
    result = {
        "model_dir": None,
        "tokenizer_exists": False,
        "model_exists": False,
        "complete": False,
    }

    try:
        from backend.api.deps import AppDependencies
        model_dir = AppDependencies.nomic_model_dir()
        result["model_dir"] = str(model_dir)

        tokenizer_path = model_dir / "tokenizer.json"
        model_path = model_dir / "onnx" / "model.onnx"

        result["tokenizer_exists"] = tokenizer_path.exists()
        result["model_exists"] = model_path.exists()
        result["complete"] = result["tokenizer_exists"] and result["model_exists"]

        if result["tokenizer_exists"]:
            result["tokenizer_size"] = tokenizer_path.stat().st_size

        if result["model_exists"]:
            result["model_size"] = model_path.stat().st_size

    except Exception as e:
        result["error"] = str(e)

    return result


def _check_boot_trace() -> Dict[str, Any]:
    """Check boot trace log."""
    result = {
        "log_dir": None,
        "log_files": [],
        "latest_log": None,
    }

    try:
        ws = get_workspace()
        log_dir = ws.logs
        result["log_dir"] = str(log_dir)

        if log_dir.exists():
            boot_trace_files = sorted(log_dir.glob("boot_trace_*.log"), reverse=True)
            result["log_files"] = [f.name for f in boot_trace_files[:10]]  # Last 10

            if boot_trace_files:
                result["latest_log"] = str(boot_trace_files[0])
                # Read last few lines
                try:
                    with open(boot_trace_files[0], 'r') as f:
                        lines = f.readlines()
                        result["latest_log_lines"] = len(lines)
                        result["latest_log_tail"] = ''.join(lines[-20:])  # Last 20 lines
                except Exception as e:
                    result["latest_log_error"] = str(e)

    except Exception as e:
        result["error"] = str(e)

    return result


@router.get("/packaging")
def get_packaging_diagnostics() -> Dict[str, Any]:
    """
    Get comprehensive packaging diagnostics.

    This endpoint checks:
    - Runtime environment (packaged vs dev)
    - SSL certificates
    - Native libraries (ONNX, llama.cpp, tokenizers)
    - Database connectivity
    - Workspace configuration
    - Embedding model availability
    - Boot trace logs

    Returns:
        JSON with all diagnostic information
    """
    bundle_info = get_bundle_info()

    return {
        "timestamp": None,  # Will be set by response
        "runtime": {
            "packaged": bundle_info.get('packaged', False),
            "meipass": bundle_info.get('meipass'),
            "executable": bundle_info.get('executable'),
            "python_version": sys.version,
            "platform": sys.platform,
        },
        "certificates": _check_certificates(),
        "ssl_connection": _check_ssl_connection(),
        "onnxruntime": _check_onnxruntime(),
        "llama_cpp": _check_llama_cpp(),
        "tokenizers": _check_tokenizers(),
        "sqlite": _check_sqlite(),
        "workspace": _check_workspace(),
        "embedding_models": _check_embedding_models(),
        "boot_trace": _check_boot_trace(),
    }


@router.get("/smoketest")
def run_smoke_test() -> Dict[str, Any]:
    """
    Run a smoke test to verify critical functionality.

    Tests each component in sequence and reports pass/fail.

    Returns:
        JSON with test results
    """
    from datetime import datetime

    tests = []
    errors = []

    def test(name: str, func) -> None:
        """Run a test and record results."""
        try:
            func()
            tests.append({"name": name, "status": "pass"})
        except Exception as e:
            tests.append({"name": name, "status": "fail", "error": str(e)})
            errors.append({"name": name, "error": str(e)})

    # Test 1: Workspace
    def test_workspace():
        ws = get_workspace()
        assert ws.base.exists(), f"Workspace base doesn't exist: {ws.base}"

    test("workspace", test_workspace)

    # Test 2: SQLite
    def test_sqlite():
        ws = get_workspace()
        from backend.services.storage.sqlite_store import create_sqlite_store, SQLiteConfig
        store = create_sqlite_store(ws.db, config=SQLiteConfig())
        store.close()

    test("sqlite", test_sqlite)

    # Test 3: SSL Certificates
    def test_certificates():
        import certifi
        bundle = certifi.where()
        assert Path(bundle).exists(), f"Certificate bundle not found: {bundle}"

    test("certificates", test_certificates)

    # Test 4: ONNX Runtime
    def test_onnx():
        import onnxruntime as ort
        providers = ort.get_available_providers()
        assert providers, "No ONNX providers available"

    test("onnxruntime", test_onnx)

    # Test 5: tokenizers
    def test_tokenizers():
        from tokenizers import Tokenizer
        # Just verify import works
        assert Tokenizer is not None

    test("tokenizers", test_tokenizers)

    # Test 6: llama_cpp
    def test_llama_cpp():
        import llama_cpp
        assert llama_cpp.__version__, "llama_cpp has no version"

    test("llama_cpp", test_llama_cpp)

    # Test 7: Embedding connector (requires models)
    def test_embedding():
        from backend.api.deps import AppDependencies
        from backend.services.connectors.nomic_onnx import NomicOnnxEmbedTextConnector, NomicOnnxConfig

        model_dir = AppDependencies.nomic_model_dir()
        connector = NomicOnnxEmbedTextConnector(
            model_dir=model_dir,
            config=NomicOnnxConfig(),
            auto_download=True,
        )
        result = connector.embed("test", ["hello"])
        assert result and len(result) == 1, "Embedding failed"

    test("embedding", test_embedding)

    return {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total": len(tests),
            "passed": sum(1 for t in tests if t["status"] == "pass"),
            "failed": sum(1 for t in tests if t["status"] == "fail"),
        },
        "tests": tests,
        "success": len(errors) == 0,
    }
