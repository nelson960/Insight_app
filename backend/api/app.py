from __future__ import annotations

from fastapi import FastAPI

from backend.api.routers import chat, docs, files, jobs, search, session, settings
from backend.services.logging_config import configure_logging


def create_app() -> FastAPI:
    app = FastAPI(title="Insight Backend")
    configure_logging()
    app.include_router(chat.router)
    app.include_router(docs.router)
    app.include_router(files.router)
    app.include_router(jobs.router)
    app.include_router(search.router)
    app.include_router(session.router)
    app.include_router(settings.router)
    return app


app = create_app()
