"""
ASGI-only Engine IPC wrapper (integrated) — SAFE FOR PyInstaller + macOS spawn.

Key fixes vs previous version:
- NO heavy imports / NO create_app() at module import time.
- multiprocessing.freeze_support() is called early.
- App is created lazily inside main_async() via get_app().
- INSIGHT_SMOKETEST=1 (or --smoketest) runs checks and EXITS (no IPC loop, no downloads/warmups).
- Optional import profiling via INSIGHT_PROFILE_IMPORTS=1 (enabled only when we actually import backend).

IPC request schema (from Rust):
{
  "request_id": "uuid-...",        # REQUIRED for concurrency/cancel; if missing, we generate one
  "endpoint": "/chat?debug=true",  # REQUIRED
  "method": "POST",                # optional, default POST
  "payload": { ... },              # JSON body forwarded to FastAPI
  "headers": { ... },              # optional headers forwarded to FastAPI
  "stream": true/false,            # if true -> stream tokens
  "session_id": "chat_123"         # optional convenience; forwarded as x-session-id header
}

Commands:
{"cmd": "shutdown"}
{"cmd": "cancel", "request_id": "uuid-..."}

Non-stream response (single line):
{
  "request_id": "...",
  "ok": true/false,
  "status": 200,
  "data": {...} | "raw text",
  "error": "message if any"
}

Streaming response (multiple lines):
{"request_id":"...","stream_start":{"status":200,"headers":{...}}}
{"request_id":"...","stream_token":"<chunk>"}
{"request_id":"...","stream_end":true}

Streaming errors:
{"request_id":"...","stream_error":"..."}
{"request_id":"...","stream_end":true}
"""

from __future__ import annotations

import anyio
import codecs
import json
import posixpath
import logging
import multiprocessing as mp
import os
import sys
import threading
import time
import uuid
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Multiprocessing safety for PyInstaller
# ---------------------------------------------------------------------------

# Must be safe to run in spawned child contexts.
mp.freeze_support()


# ---------------------------------------------------------------------------
# Minimal boot trace helpers (lazy import so children don't boot the world)
# ---------------------------------------------------------------------------

_BOOT_TRACE_AVAILABLE = False


def _try_enable_boot_trace() -> None:
    global _BOOT_TRACE_AVAILABLE
    if _BOOT_TRACE_AVAILABLE:
        return
    try:
        # Lazy import: should be lightweight; if it isn't, it will still only happen in real main path.
        from backend.services.boot_trace import init_boot_trace  # type: ignore

        init_boot_trace()
        _BOOT_TRACE_AVAILABLE = True
    except Exception:
        _BOOT_TRACE_AVAILABLE = False


def boot_step(name: str, *, status: str | None = None, **fields: Any) -> None:
    """
    Best-effort boot trace logging.
    Never raises. Prints to stderr if boot_trace isn't importable.
    """
    try:
        _try_enable_boot_trace()
        if _BOOT_TRACE_AVAILABLE:
            from backend.services.boot_trace import log_boot_step  # type: ignore

            log_boot_step(name, status=status, **fields)
            return
    except Exception:
        pass

    # Fallback: stderr breadcrumb (kept minimal)
    try:
        msg = f"[BOOT] {name}"
        if status:
            msg += f" [{status}]"
        if fields:
            msg += " " + " ".join(f"{k}={fields[k]}" for k in fields.keys())
        print(msg, file=sys.stderr)
    except Exception:
        pass


def boot_error(stage: str, exc: BaseException) -> None:
    try:
        _try_enable_boot_trace()
        if _BOOT_TRACE_AVAILABLE:
            from backend.services.boot_trace import log_boot_error  # type: ignore

            log_boot_error(stage, exc)
            return
    except Exception:
        pass
    try:
        print(f"[BOOT ERROR] {stage}: {exc}", file=sys.stderr)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] engine %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# IPC allowlist
# ---------------------------------------------------------------------------

_IPC_ALLOWED_ENDPOINTS: Dict[str, List[str]] = {
    "GET": [
        "/settings",
        "/settings/health",
        "/settings/llm/info",
        "/settings/busy",
        "/settings/storage",
        "/settings/raw_engine/status",
        "/settings/raw_engine/logs",
        "/settings/index/validate/",
        "/chat/sessions",
        "/chat/context/",
        "/files/chat/",
        "/files/progress/",
        "/files/extracted/",
        "/docs/page/",
        "/search/doc/",
        # Diagnostics endpoints
        "/diagnostics/packaging",
        "/diagnostics/smoketest",
    ],
    "POST": [
        "/chat",
        "/chat/branch",
        "/files/upload",
        "/files/ingest_path",
        "/settings",
        "/settings/model/validate",
        "/settings/llm/apply",
        "/settings/embedding/download",
        "/settings/embedding/cancel",
        "/settings/raw_engine/start",
        "/settings/raw_engine/stop",
        "/settings/storage/clean_cache",
        "/settings/storage/reset",
        "/settings/index/repair",
    ],
    "PUT": [
        "/docs/page/",
    ],
    "DELETE": [
        "/chat/sessions/",
        "/files/chat/",
    ],
}


def _is_allowed_ipc_endpoint(method: str, endpoint: str) -> bool:
    parsed = urlparse(endpoint)
    path = parsed.path or ""
    if not path.startswith("/"):
        return False
    if ".." in path:
        return False
    try:
        path = posixpath.normpath(path)
    except Exception:
        return False
    if not path.startswith("/"):
        return False
    if "/../" in path or path.startswith("../"):
        return False

    patterns = _IPC_ALLOWED_ENDPOINTS.get(method.upper(), [])
    for pattern in patterns:
        if pattern.endswith("/"):
            if path.startswith(pattern):
                return True
        else:
            if path == pattern:
                return True
    return False


# ---------------------------------------------------------------------------
# stdout helpers
# ---------------------------------------------------------------------------

class StdoutWriter:
    """Serialize writes to stdout to avoid mixed JSON lines."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def emit(self, obj: Dict[str, Any]) -> None:
        line = json.dumps(obj, ensure_ascii=False) + "\n"
        with self._lock:
            try:
                sys.stdout.write(line)
                sys.stdout.flush()
            except BrokenPipeError:
                return
            except OSError as e:
                if getattr(e, "errno", None) == 32 or "Broken pipe" in str(e):
                    return
                raise


# ---------------------------------------------------------------------------
# Lazy app creation (NO heavy imports at module import time)
# ---------------------------------------------------------------------------

_APP = None


def _enable_import_profiling_if_requested() -> bool:
    """
    Enables your backend import profiler only if requested.
    This is called immediately before importing backend.api.app.
    """
    if os.environ.get("INSIGHT_PROFILE_IMPORTS") != "1":
        return False
    try:
        from backend import import_profiler  # type: ignore

        import_profiler.enable_import_profiling()
        boot_step("import_profiler_enabled", status="OK")
        return True
    except Exception as e:
        boot_step("import_profiler_enabled", status="FAILED", error=str(e))
        return False


def get_app():
    """
    Lazily import/create the FastAPI app.
    Safe for PyInstaller + spawn: nothing heavy happens until this is called.
    """
    global _APP
    if _APP is not None:
        return _APP

    boot_step("engine_import", status="START")
    boot_step("before_backend_import")

    prof_enabled = _enable_import_profiling_if_requested()

    t0 = time.time()
    try:
        # HEAVY import happens only here
        from backend.api.app import create_app  # type: ignore
    except Exception as e:
        boot_error("backend_import", e)
        raise

    dt = time.time() - t0
    print(f"[IMPORT TIMING] backend.api.app import took {dt:.2f} seconds", file=sys.stderr)
    boot_step("after_backend_import", status="SUCCESS", elapsed_seconds=f"{dt:.2f}")

    if prof_enabled:
        try:
            from backend import import_profiler  # type: ignore

            print(import_profiler.get_import_report(), file=sys.stderr)
        except Exception:
            pass

    boot_step("before_app_creation")
    try:
        _APP = create_app()
        boot_step("after_app_creation", status="OK")
    except Exception as e:
        boot_error("app_creation", e)
        raise

    return _APP


# ---------------------------------------------------------------------------
# ASGI invocation
# ---------------------------------------------------------------------------

def _build_headers(headers: Dict[str, Any]) -> List[Tuple[bytes, bytes]]:
    out: List[Tuple[bytes, bytes]] = []

    if not any(str(k).lower() == "content-type" for k in headers.keys()):
        out.append((b"content-type", b"application/json"))

    if not any(str(k).lower() == "x-insight-ipc" for k in headers.keys()):
        out.append((b"x-insight-ipc", b"1"))

    for k, v in headers.items():
        out.append((str(k).lower().encode("utf-8"), str(v).encode("utf-8")))
    return out


async def _invoke_asgi(
    *,
    app,
    method: str,
    endpoint: str,
    payload: Dict[str, Any],
    headers: Dict[str, Any],
    stream: bool,
    request_id: str,
    writer: StdoutWriter,
) -> Dict[str, Any] | None:
    parsed = urlparse(endpoint)
    body_bytes = json.dumps(payload or {}).encode("utf-8")

    sent_body = False
    disconnect_event = anyio.Event()

    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": method,
        "path": parsed.path,
        "raw_path": parsed.path.encode("utf-8"),
        "query_string": parsed.query.encode("utf-8"),
        "headers": _build_headers(headers),
    }

    response_status: int | None = None
    response_headers: Dict[str, str] = {}
    response_body = bytearray()
    error_body = bytearray()
    is_error_response = False

    decoder = codecs.getincrementaldecoder("utf-8")()

    stream_send = None
    stream_recv = None
    stream_end_emitted = False

    if stream:
        stream_send, stream_recv = anyio.create_memory_object_stream(200)

    async def receive() -> Dict[str, Any]:
        nonlocal sent_body
        if not sent_body:
            sent_body = True
            return {"type": "http.request", "body": body_bytes, "more_body": False}
        await disconnect_event.wait()
        return {"type": "http.disconnect"}

    async def send(message: Dict[str, Any]) -> None:
        nonlocal response_status, stream_end_emitted, is_error_response

        if message["type"] == "http.response.start":
            response_status = int(message.get("status", 200))
            is_error_response = response_status >= 400
            for k, v in message.get("headers", []):
                try:
                    response_headers[k.decode("utf-8")] = v.decode("utf-8")
                except Exception:
                    response_headers[str(k)] = str(v)

            if stream and stream_send is not None:
                await stream_send.send(
                    {
                        "request_id": request_id,
                        "stream_start": {"status": response_status, "headers": response_headers},
                    }
                )
            return

        if message["type"] != "http.response.body":
            return

        chunk = message.get("body", b"") or b""
        more_body = bool(message.get("more_body", False))

        if stream:
            if stream_send is None:
                return

            if is_error_response:
                if chunk:
                    error_body.extend(chunk)

                if not more_body:
                    if not stream_end_emitted:
                        stream_end_emitted = True
                        raw = bytes(error_body).decode("utf-8", errors="replace").strip()
                        msg = raw or f"HTTP {response_status or 500}"

                        # best-effort: improve a common error shape
                        try:
                            parsed_err = json.loads(raw) if raw else None
                            if isinstance(parsed_err, dict):
                                detail = parsed_err.get("detail")
                                if isinstance(detail, dict):
                                    if detail.get("error") == "ingestion_not_ready":
                                        progress = detail.get("progress") if isinstance(detail.get("progress"), dict) else None
                                        if progress:
                                            done = progress.get("done", 0)
                                            total = progress.get("total", 0)
                                            percent = progress.get("percent", None)
                                            if percent is not None:
                                                msg = f"Indexing documents… {done}/{total} ({percent}%)."
                                            else:
                                                msg = f"Indexing documents… {done}/{total}."
                                        else:
                                            msg = "Indexing documents… please wait."
                                    else:
                                        msg = str(detail.get("error") or detail.get("message") or msg)
                                elif isinstance(detail, str) and detail.strip():
                                    msg = detail.strip()
                        except Exception:
                            pass

                        await stream_send.send({"request_id": request_id, "stream_error": msg})
                        await stream_send.send({"request_id": request_id, "stream_end": True})
                        await stream_send.aclose()
                        disconnect_event.set()
                return

            if chunk:
                text = decoder.decode(chunk)
                if text:
                    await stream_send.send({"request_id": request_id, "stream_token": text})

            if not more_body:
                if not stream_end_emitted:
                    stream_end_emitted = True
                    tail = decoder.decode(b"", final=True)
                    if tail:
                        await stream_send.send({"request_id": request_id, "stream_token": tail})
                    await stream_send.send({"request_id": request_id, "stream_end": True})
                    await stream_send.aclose()
                    disconnect_event.set()
        else:
            response_body.extend(chunk)

    async def _stream_writer_task(recv_stream) -> None:
        async with recv_stream:
            async for item in recv_stream:
                writer.emit(item)

    if stream:
        assert stream_recv is not None
        assert stream_send is not None
        async with anyio.create_task_group() as tg:
            tg.start_soon(_stream_writer_task, stream_recv)
            try:
                await app(scope, receive, send)
            finally:
                # Ensure terminal stream_end is always emitted
                if not stream_end_emitted:
                    try:
                        tail = decoder.decode(b"", final=True)
                        if tail:
                            await stream_send.send({"request_id": request_id, "stream_token": tail})
                    except Exception:
                        pass
                    try:
                        await stream_send.send({"request_id": request_id, "stream_end": True})
                    except Exception:
                        pass
                    try:
                        await stream_send.aclose()
                    except Exception:
                        pass
                disconnect_event.set()
        return None

    await app(scope, receive, send)

    status = response_status or 500
    try:
        text = response_body.decode("utf-8")
        try:
            data = json.loads(text)
        except Exception:
            data = text
    except Exception:
        data = bytes(response_body)

    return {
        "request_id": request_id,
        "ok": status < 400,
        "status": status,
        "data": data,
        "headers": response_headers,
    }


# ---------------------------------------------------------------------------
# Dispatcher with concurrency + cancellation
# ---------------------------------------------------------------------------

class RequestManager:
    def __init__(self, app) -> None:
        self.app = app
        self.writer = StdoutWriter()
        self._tg: anyio.abc.TaskGroup | None = None
        self._active: Dict[str, anyio.CancelScope] = {}
        self._active_lock = anyio.Lock()

    async def start(self) -> None:
        self._tg = await anyio.create_task_group().__aenter__()

    async def stop(self) -> None:
        if self._tg is not None:
            await self._tg.__aexit__(None, None, None)
            self._tg = None

    async def cancel(self, request_id: str) -> bool:
        async with self._active_lock:
            scope = self._active.get(request_id)
            if scope is None:
                return False
            scope.cancel()
            return True

    async def submit(self, req: Dict[str, Any]) -> None:
        if self._tg is None:
            raise RuntimeError("RequestManager not started")

        request_id = str(req.get("request_id") or uuid.uuid4())
        endpoint = req.get("endpoint")
        method = str(req.get("method", "POST")).upper()
        payload = req.get("payload") or {}
        headers = req.get("headers") or {}
        stream = bool(req.get("stream", False))

        session_id = req.get("session_id")
        if session_id and not any(str(k).lower() == "x-session-id" for k in headers.keys()):
            headers["x-session-id"] = str(session_id)

        if not endpoint or not isinstance(endpoint, str) or not endpoint.startswith("/"):
            self.writer.emit(
                {
                    "request_id": request_id,
                    "ok": False,
                    "status": 400,
                    "error": "Invalid or missing endpoint (must start with '/')",
                }
            )
            return

        if not _is_allowed_ipc_endpoint(method, endpoint):
            self.writer.emit(
                {"request_id": request_id, "ok": False, "status": 403, "error": "endpoint_not_allowed"}
            )
            return

        if not isinstance(payload, dict):
            self.writer.emit(
                {"request_id": request_id, "ok": False, "status": 400, "error": "payload must be a JSON object"}
            )
            return

        async def _run_one() -> None:
            scope = anyio.CancelScope()
            async with self._active_lock:
                self._active[request_id] = scope

            try:
                with scope:
                    if stream:
                        await _invoke_asgi(
                            app=self.app,
                            method=method,
                            endpoint=endpoint,
                            payload=payload,
                            headers=headers,
                            stream=True,
                            request_id=request_id,
                            writer=self.writer,
                        )
                    else:
                        result = await _invoke_asgi(
                            app=self.app,
                            method=method,
                            endpoint=endpoint,
                            payload=payload,
                            headers=headers,
                            stream=False,
                            request_id=request_id,
                            writer=self.writer,
                        )
                        assert result is not None
                        self.writer.emit(result)

            except anyio.get_cancelled_exc_class():
                if stream:
                    self.writer.emit({"request_id": request_id, "stream_error": "cancelled"})
                    self.writer.emit({"request_id": request_id, "stream_end": True})
                else:
                    self.writer.emit({"request_id": request_id, "ok": False, "status": 499, "error": "cancelled"})
            except Exception as exc:
                logger.exception("Request failed request_id=%s", request_id)
                if stream:
                    self.writer.emit({"request_id": request_id, "stream_error": str(exc)})
                    self.writer.emit({"request_id": request_id, "stream_end": True})
                else:
                    self.writer.emit({"request_id": request_id, "ok": False, "status": 500, "error": str(exc)})
            finally:
                async with self._active_lock:
                    self._active.pop(request_id, None)

        self._tg.start_soon(_run_one)


# ---------------------------------------------------------------------------
# stdin reader -> async dispatcher bridge
# ---------------------------------------------------------------------------

async def _stdin_producer(send: anyio.abc.ObjectSendStream[str]) -> None:
    def _read_loop() -> None:
        for line in sys.stdin:
            try:
                anyio.from_thread.run(send.send, line)
            except Exception:
                break
        try:
            anyio.from_thread.run(send.aclose)
        except Exception:
            pass

    await anyio.to_thread.run_sync(_read_loop)


def _is_smoketest() -> bool:
    if os.environ.get("INSIGHT_SMOKETEST") == "1":
        return True
    if "--smoketest" in sys.argv:
        return True
    return False


async def _run_smoketest_and_exit() -> None:
    """
    Smoketest should NOT start IPC loop and should NOT trigger downloads/warmups.
    It should run checks and exit.
    """
    boot_step("smoketest_start", status="START")

    exit_code = 0
    try:
        # Keep this import inside smoketest path.
        from backend.smoke_test import run_smoke_tests  # type: ignore

        exit_code = int(run_smoke_tests())
        boot_step("smoketest_done", status="OK" if exit_code == 0 else "FAILED", exit_code=exit_code)
    except Exception as e:
        boot_error("smoketest", e)
        exit_code = 2

    # Ensure process exits even if background threads exist
    os._exit(exit_code)


async def main_async() -> None:
    boot_step("main_async_start", status="START")

    if _is_smoketest():
        await _run_smoketest_and_exit()
        return  # unreachable, but keeps type checkers happy

    # Create app lazily here (NOT at module import)
    app = get_app()

    manager = RequestManager(app)
    await manager.start()
    boot_step("manager_started", status="OK")

    logger.info("ASGI engine ready; waiting for IPC on stdin")

    # Enable out-of-band events in IPC mode.
    try:
        from backend.services.ipc_events import set_ipc_emitter  # type: ignore

        set_ipc_emitter(manager.writer.emit)
    except Exception:
        pass

    # Bridge stdin -> async dispatcher.
    send, recv = anyio.create_memory_object_stream(200)

    async def _startup_tasks() -> None:
        def _run_sync() -> None:
            # Emit startup health report once (for UI startup modal).
            try:
                boot_step("startup_health_check_start", status="START")
                from backend.services.health import run_startup_health_fast  # type: ignore
                from backend.services.ipc_events import emit_event  # type: ignore

                # Fast health first so the UI can show setup blocking quickly.
                report_fast = run_startup_health_fast()
                emit_event("startup_health", report=report_fast, phase="fast")
                boot_step(
                    "startup_health_check",
                    status="OK",
                    report_keys=str(getattr(report_fast, "keys", lambda: [])()),
                )
            except Exception as e:
                boot_error("startup_health_check", e)

            # Write boot summary JSON for user support.
            try:
                from backend.services.boot_summary import write_boot_summary  # type: ignore

                summary_path = write_boot_summary()
                logger.info("Boot summary written to: %s", summary_path)
            except Exception as e:
                logger.warning("Failed to write boot summary: %s", e)

            # Background warm-up (non-blocking) — keep it AFTER app is ready.
            def _warmup_worker() -> None:
                time.sleep(1.0)
                try:
                    import tokenizers  # noqa

                    _ = tokenizers.__version__
                    logger.info("Tokenizers warm-up complete")
                except Exception as e:
                    logger.warning("Tokenizers warm-up failed: %s", e)
                try:
                    import onnxruntime  # noqa

                    _ = onnxruntime.__version__
                    logger.info("ONNX Runtime warm-up complete")
                except Exception as e:
                    logger.warning("ONNX Runtime warm-up failed: %s", e)

            threading.Thread(target=_warmup_worker, daemon=True, name="warmup-worker").start()
            logger.info("Background warm-up started (non-blocking)")

            # Chat router is now registered at app startup (no background loader needed).

        await anyio.to_thread.run_sync(_run_sync)

    async with anyio.create_task_group() as tg:
        tg.start_soon(_stdin_producer, send)
        tg.start_soon(_startup_tasks)

        async with recv:
            async for raw_line in recv:
                line = (raw_line or "").strip()
                if not line:
                    continue

                try:
                    msg = json.loads(line)
                    if not isinstance(msg, dict):
                        raise ValueError("IPC message must be a JSON object")
                except Exception as exc:
                    rid = str(uuid.uuid4())
                    manager.writer.emit({"request_id": rid, "ok": False, "status": 400, "error": f"Invalid JSON: {exc}"})
                    continue

                cmd = msg.get("cmd")
                if cmd == "shutdown":
                    try:
                        from backend.services.raw_engine_server.manager import raw_engine_manager  # type: ignore

                        raw_engine_manager().stop()
                        from backend.services.connectors.nomic import cancel_embedding_download_process  # type: ignore

                        cancel_embedding_download_process()
                    except Exception:
                        pass
                    manager.writer.emit(
                        {"request_id": msg.get("request_id") or str(uuid.uuid4()), "ok": True, "status": 200, "data": {"detail": "shutdown"}}
                    )
                    tg.cancel_scope.cancel()
                    break

                if cmd == "cancel":
                    rid = str(msg.get("request_id") or "")
                    if not rid:
                        logger.warning("cancel missing request_id")
                        continue

                    # Best-effort: cancel model generation only (avoid ASGI mid-stream teardown races)
                    try:
                        from backend.api.deps import AppDependencies  # type: ignore

                        _ = AppDependencies.session_manager().cancel_request(rid)
                    except Exception as exc:
                        logger.warning("Model cancel failed request_id=%s err=%s", rid, exc)
                    continue

                await manager.submit(msg)

    await manager.stop()

    try:
        from backend.services.ipc_events import set_ipc_emitter  # type: ignore

        set_ipc_emitter(None)
    except Exception:
        pass


def main() -> None:
    anyio.run(main_async)


if __name__ == "__main__":
    main()
