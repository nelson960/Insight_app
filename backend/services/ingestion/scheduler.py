from __future__ import annotations

import logging
import threading
import os
from datetime import datetime, timezone
from queue import Queue, Empty
from uuid import uuid4
from typing import Dict, Set

from backend.services.ingestion import IngestionPipeline
from backend.services.ingestion.models import FileIngestionStatus, IngestionRequest
from backend.services.ingestion.pipeline import IngestionCancelled
from backend.services.storage.sqlite_store import SQLiteMetadataStore

logger = logging.getLogger(__name__)


class IngestionScheduler:
    """Background worker that processes ingestion requests asynchronously."""

    def __init__(self, pipeline: IngestionPipeline, metadata_store: SQLiteMetadataStore) -> None:
        self._pipeline = pipeline
        self._metadata_store = metadata_store
        self._reconcile_stale_jobs()
        self._queue: Queue[tuple[str, IngestionRequest]] = Queue()
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._known_jobs: Dict[str, IngestionRequest] = {}
        self._cancelled: Set[str] = set()
        self._timed_out: Set[str] = set()
        self._running_cancel: Dict[str, threading.Event] = {}
        self._watchdog_interval_seconds = self._safe_env_int("INSIGHT_INGEST_WATCHDOG_SEC", 15)
        self._max_running_age_seconds = self._safe_env_int("INSIGHT_INGEST_MAX_RUNNING_SEC", 20 * 60)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        self._watchdog_thread = threading.Thread(target=self._watchdog, daemon=True)
        self._watchdog_thread.start()

    @staticmethod
    def _safe_env_int(key: str, default: int) -> int:
        raw = os.getenv(key)
        if raw is None:
            return int(default)
        text = str(raw).strip()
        if not text:
            return int(default)
        try:
            return int(text)
        except Exception:
            pass
        # Allow simple duration suffixes (e.g., "3m", "3minutes", "45s").
        lowered = text.lower()
        units = {
            "seconds": 1,
            "second": 1,
            "secs": 1,
            "sec": 1,
            "s": 1,
            "minutes": 60,
            "minute": 60,
            "mins": 60,
            "min": 60,
            "m": 60,
            "hours": 3600,
            "hour": 3600,
            "hrs": 3600,
            "hr": 3600,
            "h": 3600,
        }
        for suffix, scale in sorted(units.items(), key=lambda item: -len(item[0])):
            if lowered.endswith(suffix):
                num = lowered[: -len(suffix)].strip()
                if not num:
                    break
                try:
                    return max(1, int(float(num) * scale))
                except Exception:
                    break
        return int(default)

    @staticmethod
    def _parse_created_at(value: object) -> datetime | None:
        if not isinstance(value, str) or not value:
            return None
        # SQLite datetime('now') returns "YYYY-MM-DD HH:MM:SS" (UTC).
        try:
            dt = datetime.fromisoformat(value)
        except Exception:
            return None
        if dt.tzinfo is None:
            # Treat naive SQLite timestamps as UTC.
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    def _watchdog(self) -> None:
        """
        Periodically mark long-running jobs as failed to avoid "busy forever" states.

        This is best-effort: it cannot forcibly kill a hung Python thread, but it can:
        - request cancellation via the pipeline's cancel_check
        - mark the job/file failed in SQLite so the UI can recover and offer retry.
        """
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=max(1, self._watchdog_interval_seconds))
            if self._stop_event.is_set():
                return
            if self._max_running_age_seconds <= 0:
                continue
            try:
                running = self._metadata_store.list_jobs_with_status(("running",), limit=1000)
            except Exception:
                continue
            if not running:
                continue

            now = datetime.now(timezone.utc)
            for job in running:
                job_id = str(job.get("id") or "")
                file_id = str(job.get("file_id") or "")
                created_at = self._parse_created_at(job.get("created_at"))
                if not job_id or created_at is None:
                    continue
                age = (now - created_at).total_seconds()
                if age < float(self._max_running_age_seconds):
                    continue

                with self._lock:
                    if job_id in self._timed_out:
                        continue
                    self._timed_out.add(job_id)
                    ev = self._running_cancel.get(job_id)
                    if ev is not None:
                        ev.set()

                logger.warning(
                    "Ingestion watchdog timing out job %s (file=%s age=%.1fs)",
                    job_id,
                    file_id,
                    age,
                )
                try:
                    self._metadata_store.update_job_status(job_id, "failed", error="watchdog timeout")
                except Exception:
                    pass
                if file_id:
                    try:
                        self._metadata_store.mark_file_status(file_id, FileIngestionStatus.FAILED)
                    except Exception:
                        pass

    def _reconcile_stale_jobs(self) -> None:
        """
        Best-effort cleanup for queued/running jobs from a previous app instance.

        The current desktop architecture runs ingestion in-process; if the app is
        restarted mid-ingestion, the worker thread is gone but the job row remains
        stuck in `queued`/`running`, causing `/settings/busy` to show "ingestion"
        forever and blocking reset/clean operations.
        """
        try:
            stale = self._metadata_store.list_jobs_with_status(("queued", "running"), limit=1000)
        except Exception as exc:
            logger.debug("Unable to list active jobs for reconciliation: %s", exc)
            return
        if not stale:
            return
        for job in stale:
            job_id = str(job.get("id") or "")
            file_id = str(job.get("file_id") or "")
            if not job_id:
                continue
            try:
                self._metadata_store.update_job_status(job_id, "failed", error="stale job from previous run")
            except Exception:
                pass
            if file_id:
                try:
                    self._metadata_store.mark_file_status(file_id, FileIngestionStatus.FAILED)
                except Exception:
                    pass
        logger.warning("Marked %d stale ingestion jobs as failed", len(stale))

    def schedule(self, request: IngestionRequest, job_id: str | None = None) -> str:
        job_id = job_id or f"job_{uuid4().hex}"
        self._metadata_store.create_job(job_id, "ingestion", request.file_id)
        with self._lock:
            self._known_jobs[job_id] = request
        self._queue.put((job_id, request))
        logger.info("Scheduled ingestion job %s for %s", job_id, request.file_id)
        return job_id

    def cancel_chat(self, chat_id: str) -> int:
        """
        Best-effort cancellation for all queued/running ingestion jobs for a chat_id.

        - Queued jobs will be skipped when dequeued.
        - Running jobs will stop at the next pipeline cancellation checkpoint.
        """
        if not chat_id:
            return 0
        cancelled = 0
        with self._lock:
            for job_id, req in list(self._known_jobs.items()):
                if getattr(req, "chat_id", None) != chat_id:
                    continue
                # If the underlying file is shared by other chats, keep ingestion running so
                # the remaining chats don't lose their indexing progress when a parent card
                # is deleted mid-ingestion.
                try:
                    other_chats = self._metadata_store.list_chat_ids_for_file(req.file_id)
                    if any(cid != chat_id for cid in other_chats):
                        continue
                except Exception:
                    # Best-effort: if we can't determine sharing, proceed with cancellation.
                    pass
                if job_id in self._cancelled:
                    continue
                self._cancelled.add(job_id)
                ev = self._running_cancel.get(job_id)
                if ev is not None:
                    ev.set()
                cancelled += 1
                try:
                    self._metadata_store.update_job_status(job_id, "cancelled")
                except Exception:
                    pass
        if cancelled:
            logger.info("Cancelled %d ingestion jobs for chat_id=%s", cancelled, chat_id)
        return cancelled

    def cancel_file(self, file_id: str, *, chat_id: str | None = None) -> int:
        """
        Best-effort cancellation for queued/running ingestion jobs for a single file_id.

        If chat_id is provided, only cancels jobs scheduled for that chat.
        """
        if not file_id:
            return 0
        cancelled = 0
        with self._lock:
            for job_id, req in list(self._known_jobs.items()):
                if getattr(req, "file_id", None) != file_id:
                    continue
                if chat_id and getattr(req, "chat_id", None) != chat_id:
                    continue
                if job_id in self._cancelled:
                    continue
                self._cancelled.add(job_id)
                ev = self._running_cancel.get(job_id)
                if ev is not None:
                    ev.set()
                cancelled += 1
                try:
                    self._metadata_store.update_job_status(job_id, "cancelled")
                except Exception:
                    pass
        if cancelled:
            logger.info("Cancelled %d ingestion jobs for file_id=%s", cancelled, file_id)
        return cancelled

    def busy_state(self) -> dict[str, int]:
        """
        Return a cheap snapshot of in-memory ingestion activity.

        Note: SQLite job rows can become stale when the process is restarted or a job
        is force-failed by the watchdog. The UI and settings safety checks should
        consider both SQLite and in-memory activity.
        """
        try:
            queued = int(self._queue.qsize())
        except Exception:
            queued = 0
        with self._lock:
            running = len(self._running_cancel)
            timed_out = len(self._timed_out)
        return {"queued": queued, "running": running, "timed_out": timed_out}

    def _worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                job_id, request = self._queue.get(timeout=0.5)
            except Empty:
                continue
            cleanup_path = request.cleanup_path
            try:
                with self._lock:
                    if job_id in self._cancelled:
                        logger.info("Skipping cancelled ingestion job %s", job_id)
                        self._metadata_store.update_job_status(job_id, "cancelled")
                        continue
                    cancel_event = threading.Event()
                    self._running_cancel[job_id] = cancel_event
                logger.info("Starting async ingestion for job %s (file %s)", job_id, request.file_id)
                self._metadata_store.update_job_status(job_id, "running")
                self._pipeline.run(request, cancel_check=cancel_event.is_set)
                logger.info("Async ingestion completed for job %s", job_id)
                self._metadata_store.update_job_status(job_id, "completed")
            except IngestionCancelled:
                logger.info("Async ingestion cancelled for job %s", job_id)
                with self._lock:
                    timed_out = job_id in self._timed_out
                if timed_out:
                    self._metadata_store.update_job_status(job_id, "failed", error="watchdog timeout")
                else:
                    self._metadata_store.update_job_status(job_id, "cancelled")
            except Exception as exc:
                logger.exception("Async ingestion failed for job %s", job_id)
                self._metadata_store.update_job_status(job_id, "failed", error=str(exc))
            finally:
                with self._lock:
                    self._running_cancel.pop(job_id, None)
                    # Keep known_jobs for bookkeeping; caller may inspect jobs later.
                self._queue.task_done()
                if cleanup_path:
                    try:
                        os.remove(cleanup_path)
                        logger.info("Removed temporary file %s", cleanup_path)
                    except FileNotFoundError:
                        pass
                    except Exception:
                        logger.exception("Failed removing temp file %s", cleanup_path)

    def shutdown(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=2)
        self._watchdog_thread.join(timeout=2)
