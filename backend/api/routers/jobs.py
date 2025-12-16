from __future__ import annotations

from fastapi import APIRouter, Query

from backend.api.deps import AppDependencies

router = APIRouter(prefix="/jobs", tags=["Jobs"])


@router.get("/")
def list_jobs(limit: int = Query(50, ge=1, le=500)):
    store, _ = AppDependencies.storage()
    jobs = store.list_jobs(limit)
    return {"jobs": jobs}
