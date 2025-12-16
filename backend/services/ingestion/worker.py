from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Protocol

from .models import IngestionErrorCode, IngestionRequest
from .pipeline import IngestionPipeline, IngestionPipelineError

logger = logging.getLogger(__name__)


@dataclass
class IngestionJob:
    """Represents a queued ingestion job."""

    job_id: str
    request: IngestionRequest


class JobQueue(Protocol):
    """Abstract queue backing the ingestion worker."""

    def reserve(self, *, timeout: Optional[float] = None) -> Optional[IngestionJob]:
        ...

    def ack_success(self, job_id: str) -> None:
        ...

    def ack_failure(
        self,
        job_id: str,
        *,
        error_code: IngestionErrorCode,
        retry_at: Optional[datetime],
        error_message: str,
    ) -> None:
        ...


class IngestionWorker:
    """Consumes queued ingestion jobs and drives the pipeline."""

    def __init__(
        self,
        *,
        pipeline: IngestionPipeline,
        queue: JobQueue,
        idle_sleep_seconds: float = 1.0,
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        if idle_sleep_seconds <= 0:
            raise ValueError("idle_sleep_seconds must be positive")
        self._pipeline = pipeline
        self._queue = queue
        self._idle_sleep_seconds = idle_sleep_seconds
        self._stop_event = stop_event or threading.Event()

    @property
    def stop_event(self) -> threading.Event:
        return self._stop_event

    def run_forever(self) -> None:
        logger.info("Ingestion worker started")
        while not self._stop_event.is_set():
            job = self._queue.reserve(timeout=self._idle_sleep_seconds)
            if job is None:
                continue

            logger.info("Processing ingestion job %s", job.job_id)
            try:
                context = self._pipeline.run(job.request)
            except IngestionPipelineError as err:
                retry_at = err.context.retry.next_retry_at if err.context.retry else None
                logger.warning(
                    "Ingestion pipeline error for job %s (file=%s, retry_at=%s)",
                    job.job_id,
                    err.file_id,
                    retry_at,
                )
                self._queue.ack_failure(
                    job.job_id,
                    error_code=err.error_code,
                    retry_at=retry_at,
                    error_message=str(err),
                )
            except Exception as exc:
                logger.exception("Unexpected failure in ingestion job %s", job.job_id)
                self._queue.ack_failure(
                    job.job_id,
                    error_code=IngestionErrorCode.UNKNOWN,
                    retry_at=None,
                    error_message=str(exc),
                )
            else:
                self._queue.ack_success(job.job_id)

        logger.info("Ingestion worker stopped")


__all__ = ["IngestionWorker", "IngestionJob", "JobQueue"]
