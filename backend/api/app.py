from __future__ import annotations

import logging
import sys
import threading
import time
from fastapi import FastAPI, Request
from starlette.responses import JSONResponse

# Boot trace (optional import)
try:
    from backend.services.boot_trace import log_boot_step
    _boot_trace_available = True
except Exception:
    _boot_trace_available = False

logger = logging.getLogger(__name__)

_embedding_prefetch_started = False

_IPC_HEADER = "x-insight-ipc"
_IPC_VALUE = "1"


def _prefetch_embedding_assets() -> None:
    """
    Best-effort prefetch of embedding assets so the first upload/ingestion doesn't block.

    The download is BOUNDED and NON-BLOCKING:
    - Runs in background daemon thread
    - Times out after 30 seconds if download hangs
    - App becomes READY even if embeddings are still downloading

    Note: engine.py invokes the FastAPI app via raw ASGI without lifespan events, so
    we trigger this from create_app() (not via FastAPI startup hooks).
    """

    def worker() -> None:
        if _boot_trace_available:
            try:
                log_boot_step("embedding_prefetch_start")
            except Exception:
                pass

        from backend.api.deps import AppDependencies  # local import to avoid early heavy imports
        from backend.services.embedding_status import get_embedding_status
        from backend.services.connectors.nomic import (
            MissingDependencyError,
            ensure_local_nomic_model_files,
        )
        from backend.services.connectors.nomic_onnx import NomicOnnxConfig

        ws = AppDependencies.workspace()
        embed_status = get_embedding_status(ws.base)

        # Check if we should even try downloading
        current_status = embed_status.get_status()
        if current_status == "ready":
            logger.info("Embedding models already ready, skipping prefetch")
            if _boot_trace_available:
                try:
                    log_boot_step("embedding_prefetch_skipped", status="ALREADY_READY")
                except Exception:
                    pass
            return

        # Check if we've failed too many times
        current_status = embed_status.get_status()
        retry_count = embed_status._status.get("retry_count", 0)

        if _boot_trace_available:
            try:
                log_boot_step("embedding_prefetch_retry_check",
                              status=current_status,
                              retry_count=str(retry_count),
                              max_retries="3")
            except Exception:
                pass

        if not embed_status.can_retry(max_retries=3):
            logger.warning(
                "Embedding download skipped: status=%s, retry_count=%s (max=3). Use Settings to retry.",
                current_status,
                retry_count
            )
            if _boot_trace_available:
                try:
                    log_boot_step("embedding_prefetch_skipped",
                                  status="TOO_MANY_RETRIES",
                                  current_status=current_status,
                                  retry_count=str(retry_count),
                                  max_retries="3")
                except Exception:
                    pass
            return

        model_dir = AppDependencies.nomic_model_dir()
        cfg = NomicOnnxConfig()
        required = [cfg.model_filename, "tokenizer.json"]

        if _boot_trace_available:
            try:
                log_boot_step("embedding_prefetch_checking",
                              model_dir=str(model_dir),
                              required_files=str(required),
                              current_status=current_status)
            except Exception:
                pass

        # Set status to downloading
        embed_status.set_downloading()

        # Run download with timeout using threading.Timer
        import threading
        import time

        download_complete = threading.Event()
        download_error = [None]  # Use list to allow modification in nested function

        def download_worker():
            try:
                ensure_local_nomic_model_files(model_dir, required_paths=required)
                download_complete.set()
            except Exception as e:
                download_error[0] = e
                download_complete.set()

        # Start download thread
        download_thread = threading.Thread(target=download_worker, daemon=True)
        download_thread.start()

        # Wait for completion with timeout
        download_thread.join(timeout=30)

        if download_thread.is_alive():
            # Timeout - thread is still running
            logger.warning("Embedding prefetch timeout (will continue in background)")
            embed_status.set_error("Timeout: download took longer than 30 seconds")
            if _boot_trace_available:
                try:
                    log_boot_step("embedding_prefetch_timeout",
                                  status="TIMEOUT",
                                  note="Download continues in background")
                except Exception:
                    pass
        elif download_error[0]:
            # Download failed
            exc = download_error[0]
            logger.warning("Embedding prefetch failed: %s", exc)
            if isinstance(exc, MissingDependencyError):
                embed_status.set_error(f"Missing dependency: {exc}")
                if _boot_trace_available:
                    try:
                        log_boot_step("embedding_prefetch_skipped", status="MISSING_DEP", error=str(exc))
                    except Exception:
                        pass
            else:
                embed_status.set_error(f"{type(exc).__name__}: {exc}")
                if _boot_trace_available:
                    try:
                        from backend.services.boot_trace import log_boot_error
                        log_boot_error("embedding_prefetch", exc)
                    except Exception:
                        pass
        else:
            # Success
            logger.info("Embedding assets ready at %s", model_dir)
            embed_status.set_ready()
            if _boot_trace_available:
                try:
                    log_boot_step("embedding_prefetch_complete", status="OK", model_dir=str(model_dir))
                except Exception:
                    pass

    global _embedding_prefetch_started
    if _embedding_prefetch_started:
        return
    _embedding_prefetch_started = True
    threading.Thread(target=worker, name="insight-prefetch-embeddings", daemon=True).start()


def create_app() -> FastAPI:
    if _boot_trace_available:
        try:
            log_boot_step("create_app_start")
        except Exception:
            pass

    _start = time.time()
    from backend.api.routers import docs, files, search, settings, diagnostics
    _routers_elapsed = time.time() - _start
    if _routers_elapsed > 0.1:
        print(f"[APP IMPORT] backend.api.routers: {_routers_elapsed:.2f}s", file=sys.stderr)

    _start = time.time()
    from backend.services.logging_config import configure_logging
    _logging_elapsed = time.time() - _start
    if _logging_elapsed > 0.1:
        print(f"[APP IMPORT] backend.services.logging_config: {_logging_elapsed:.2f}s", file=sys.stderr)

    app = FastAPI(title="Insight Backend")

    if _boot_trace_available:
        try:
            log_boot_step("configure_logging_start")
        except Exception:
            pass

    configure_logging()

    if _boot_trace_available:
        try:
            log_boot_step("configure_logging_done")
        except Exception:
            pass

    @app.middleware("http")
    async def _ipc_only(request: Request, call_next):
        """
        IPC-only backend: reject all non-IPC HTTP requests.

        The UI talks to Python via Tauri IPC (engine.py), which injects the header.
        """
        is_ipc = request.headers.get(_IPC_HEADER) == _IPC_VALUE
        if not is_ipc:
            return JSONResponse(status_code=403, content={"ok": False, "error": "ipc_required"})
        return await call_next(request)

    app.include_router(docs.router)
    app.include_router(files.router)
    app.include_router(search.router)
    app.include_router(settings.router)
    app.include_router(diagnostics.router)  # Packaging diagnostics

    # Register the app for background chat-router loading (no startup import).
    try:
        from backend.api.chat_router_loader import register_app

        register_app(app)
    except Exception:
        pass

    # NOTE: Embedding auto-download DISABLED to avoid blocking startup
    # Users must manually install embeddings via Settings → Embeddings
    # This prevents:
    # - 53-second startup stall
    # - Random network hangs at boot
    # - "Infinite loading" perception
    #
    # To re-enable auto-download (not recommended), uncomment:
    # _prefetch_embedding_assets()

    return app
