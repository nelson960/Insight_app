from __future__ import annotations

import logging
import os
import threading
from fastapi import FastAPI, Request
from starlette.responses import JSONResponse

from backend.api.routers import chat, docs, files, search, settings
from backend.services.connectors.nomic import MissingDependencyError, ensure_local_nomic_model_files
from backend.services.connectors.nomic_onnx import NomicOnnxConfig
from backend.services.logging_config import configure_logging

logger = logging.getLogger(__name__)

_embedding_prefetch_started = False

_IPC_HEADER = "x-insight-ipc"
_IPC_VALUE = "1"


def _prefetch_embedding_assets() -> None:
    """
    Best-effort prefetch of embedding assets so the first upload/ingestion doesn't block.

    Note: engine.py invokes the FastAPI app via raw ASGI without lifespan events, so
    we trigger this from create_app() (not via FastAPI startup hooks).
    """

    def worker() -> None:
        from backend.api.deps import AppDependencies  # local import to avoid early heavy imports

        model_dir = AppDependencies.nomic_model_dir()
        cfg = NomicOnnxConfig()
        required = [cfg.model_filename, "tokenizer.json"]
        try:
            ensure_local_nomic_model_files(model_dir, required_paths=required)
            logger.info("Embedding assets ready at %s", model_dir)
        except MissingDependencyError as exc:
            logger.warning("Embedding prefetch skipped (missing dependency): %s", exc)
        except Exception as exc:
            logger.warning("Embedding prefetch failed: %s", exc)

    global _embedding_prefetch_started
    if _embedding_prefetch_started:
        return
    _embedding_prefetch_started = True
    threading.Thread(target=worker, name="insight-prefetch-embeddings", daemon=True).start()


def create_app() -> FastAPI:
    app = FastAPI(title="Insight Backend")
    configure_logging()

    @app.middleware("http")
    async def _ipc_only(request: Request, call_next):
        """
        Desktop mode default: reject non-IPC HTTP requests.

        The UI talks to Python via Tauri IPC (engine.py), which injects the header.
        If you explicitly want to run a localhost HTTP server for debugging, set:
            INSIGHT_ALLOW_HTTP=1
        """
        allow_http = os.getenv("INSIGHT_ALLOW_HTTP", "").strip().lower() in ("1", "true", "yes")
        is_ipc = request.headers.get(_IPC_HEADER) == _IPC_VALUE

        # Hardening: destructive settings endpoints are IPC-only, even when running
        # with INSIGHT_ALLOW_HTTP=1 for debugging.
        destructive_settings_paths = {
            "/settings/llm/apply",
            "/settings/storage/clean_cache",
            "/settings/storage/reset",
            "/settings/index/repair",
        }
        if request.url.path in destructive_settings_paths and not is_ipc:
            return JSONResponse(status_code=403, content={"ok": False, "error": "ipc_required"})

        if allow_http:
            return await call_next(request)
        if is_ipc:
            return await call_next(request)
        return JSONResponse(status_code=403, content={"ok": False, "error": "http_disabled"})

    app.include_router(chat.router)
    app.include_router(docs.router)
    app.include_router(files.router)
    app.include_router(search.router)
    app.include_router(settings.router)

    # Prefetch embedding assets on app creation so desktop installs download once up-front.
    _prefetch_embedding_assets()

    return app


app = create_app()
