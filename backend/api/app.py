from __future__ import annotations

import logging
import os
import sys
import threading
import time
from fastapi import FastAPI, Request, HTTPException
from starlette.responses import JSONResponse
from backend.services.ipc_auth import require_ipc_token

# Boot trace (optional import)
try:
    from backend.services.boot_trace import log_boot_step
    _boot_trace_available = True
except Exception:
    _boot_trace_available = False

logger = logging.getLogger(__name__)

_embedding_prefetch_started = False
_sqlite_warmup_started = False

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


def _warmup_sqlite_store() -> None:
    """Best-effort background warmup for SQLite + encryption key path."""

    def worker() -> None:
        start = time.time()
        try:
            from backend.api.deps import AppDependencies

            store = AppDependencies.sqlite_store()
            try:
                store._connection.execute("SELECT 1").fetchone()
            except Exception:
                pass
            elapsed_ms = (time.time() - start) * 1000.0
            logger.info("SQLite warmup ready in %.1fms", elapsed_ms)
            if _boot_trace_available:
                try:
                    log_boot_step("sqlite_warmup_done", status="OK", elapsed_ms=f"{elapsed_ms:.1f}")
                except Exception:
                    pass
        except Exception as exc:
            logger.warning("SQLite warmup failed: %s", exc)
            if _boot_trace_available:
                try:
                    from backend.services.boot_trace import log_boot_error

                    log_boot_error("sqlite_warmup", exc)
                except Exception:
                    pass

    global _sqlite_warmup_started
    if _sqlite_warmup_started:
        return
    _sqlite_warmup_started = True
    threading.Thread(target=worker, name="insight-warmup-sqlite", daemon=True).start()


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
    # Initialize per-app dependency container and bind it as the default for background work.
    try:
        from backend.api.deps import bind_container, create_container

        app.state.deps = create_container()
        bind_container(app.state.deps)
    except Exception:
        app.state.deps = None

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
        ipc_token = require_ipc_token()
        if ipc_token and request.headers.get("x-insight-ipc-token") != ipc_token:
            return JSONResponse(status_code=403, content={"ok": False, "error": "ipc_token_invalid"})
        token = None
        try:
            if getattr(app.state, "deps", None) is not None:
                from backend.api.deps import set_request_container

                token = set_request_container(app.state.deps)
            return await call_next(request)
        finally:
            if token is not None:
                from backend.api.deps import reset_request_container

                reset_request_container(token)

    def _normalize_detail(detail):
        if isinstance(detail, dict):
            if "error" in detail or "message" in detail:
                return detail
            # Preserve existing structure but add a generic error code.
            return {"error": "http_error", "message": detail.get("detail") or str(detail)}
        if isinstance(detail, str):
            return {"error": "http_error", "message": detail}
        return {"error": "http_error", "message": str(detail)}

    @app.exception_handler(HTTPException)
    async def _http_exception_handler(_request: Request, exc: HTTPException):
        detail = _normalize_detail(exc.detail)
        return JSONResponse(status_code=exc.status_code, content={"detail": detail})

    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(_request: Request, exc: Exception):
        logger.exception("Unhandled API error")
        return JSONResponse(
            status_code=500,
            content={"detail": {"error": "internal_error", "message": "Internal server error"}},
        )

    @app.on_event("shutdown")
    async def _shutdown_deps() -> None:
        try:
            from backend.api.deps import AppDependencies

            AppDependencies.close_all()
        except Exception:
            pass

    from backend.api.routers import chat

    app.include_router(docs.router)
    app.include_router(files.router)
    app.include_router(search.router)
    app.include_router(settings.router)
    app.include_router(chat.router)
    app.include_router(diagnostics.router)  # Packaging diagnostics

    # NOTE: Embedding auto-download DISABLED to avoid blocking startup
    # Users must manually install embeddings via Settings → Embeddings
    # This prevents:
    # - 53-second startup stall
    # - Random network hangs at boot
    # - "Infinite loading" perception
    #
    # To re-enable auto-download (not recommended), uncomment:
    # _prefetch_embedding_assets()

    # Warm SQLite/encryption path in the background so first chat open is instant.
    _warmup_sqlite_store()

    return app
