"""Background warm-up for heavy modules (non-blocking startup)."""
import asyncio
import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Warm-up status tracking
_warmup_status: dict[str, dict] = {}
_lock = threading.Lock()


def get_warmup_status(component: str) -> dict:
    """Get warm-up status for a component."""
    with _lock:
        return _warmup_status.get(component, {"status": "not_started"})


def set_warmup_status(component: str, status: str, error: Optional[str] = None):
    """Set warm-up status for a component."""
    with _lock:
        _warmup_status[component] = {
            "status": status,
            "error": error,
            "timestamp": time.time()
        }


def warmup_onnxruntime():
    """Warm up ONNX Runtime (load native libs)."""
    try:
        set_warmup_status("onnxruntime", "loading")
        # Direct import - no backend.runtime wrapper
        import onnxruntime  # noqa
        # Force loading by checking version
        _ = onnxruntime.__version__
        set_warmup_status("onnxruntime", "ready")
        logger.info("ONNX Runtime warm-up complete")
    except Exception as e:
        set_warmup_status("onnxruntime", "failed", str(e))
        logger.warning(f"ONNX Runtime warm-up failed: {e}")


def warmup_tokenizers():
    """Warm up tokenizers library."""
    try:
        set_warmup_status("tokenizers", "loading")
        # Direct import - no backend.runtime wrapper
        import tokenizers  # noqa
        # Force loading by checking version
        _ = tokenizers.__version__
        set_warmup_status("tokenizers", "ready")
        logger.info("Tokenizers warm-up complete")
    except Exception as e:
        set_warmup_status("tokenizers", "failed", str(e))
        logger.warning(f"Tokenizers warm-up failed: {e}")


def warmup_llama_cpp():
    """Warm up llama.cpp (load native lib)."""
    try:
        set_warmup_status("llama_cpp", "loading")
        # Direct import - no backend.runtime wrapper
        import llama_cpp  # noqa
        # Force loading by checking version
        _ = llama_cpp.__version__
        set_warmup_status("llama_cpp", "ready")
        logger.info("llama.cpp warm-up complete")
    except Exception as e:
        set_warmup_status("llama_cpp", "failed", str(e))
        logger.warning(f"llama.cpp warm-up failed: {e}")


def warmup_all_background(delay: float = 0.5):
    """
    Run warm-up in a background thread (non-blocking).

    Args:
        delay: Seconds to wait before starting warm-up (lets FastAPI become ready first)
    """
    def _warmup_worker():
        time.sleep(delay)

        # Warm up in order of dependency
        warmup_tokenizers()
        warmup_onnxruntime()
        # llama_cpp is heaviest - do it last
        # warmup_llama_cpp()  # Skip for now - only load when user selects model

    thread = threading.Thread(target=_warmup_worker, daemon=True, name="warmup-worker")
    thread.start()


async def warmup_all_background_async(delay: float = 0.5):
    """
    Async version of warm-up (for use with FastAPI lifespan).

    Args:
        delay: Seconds to wait before starting warm-up
    """
    await asyncio.sleep(delay)

    # Run warm-up in thread pool to avoid blocking event loop
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, lambda: warmup_all_background(0))


# Component status for UI
def get_all_warmup_status() -> dict:
    """Get status of all warm-up components."""
    with _lock:
        return dict(_warmup_status)
