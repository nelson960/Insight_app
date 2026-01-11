"""
ASGI-only Engine IPC wrapper (integrated).

Reads newline-delimited JSON from stdin and routes requests directly to FastAPI via ASGI.
Supports concurrent requests, streaming, cancellation, request_id passthrough, and session binding.

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

import codecs
import json
import logging
import sys
import threading
import uuid
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

import anyio

# --- Ensure project root is importable ---
try:
    import pathlib

    PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
except Exception:
    pass

from backend.api.app import create_app  # noqa: E402


# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] engine %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)

APP = create_app()

_IPC_ALLOWED_ENDPOINTS: Dict[str, List[str]] = {
    "GET": [
        "/settings",
        "/settings/health",
        "/settings/llm/info",
        "/settings/busy",
        "/settings/storage",
        "/settings/index/validate/",
        "/chat/sessions",
        "/chat/context/",
        "/files/chat/",
        "/files/progress/",
        "/files/extracted/",
        "/docs/page/",
        "/search/doc/",
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
    # Basic traversal hardening even though these are IPC-only paths.
    if ".." in path:
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


# -----------------------------------------------------------------------------
# Output (stdout) helpers
# -----------------------------------------------------------------------------

class StdoutWriter:
    """Serialize writes to stdout to avoid mixed JSON lines."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def emit(self, obj: Dict[str, Any]) -> None:
        line = json.dumps(obj, ensure_ascii=False) + "\n"
        with self._lock:
            # Keep sync write inside lock; ensures whole line is written atomically
            sys.stdout.write(line)
            sys.stdout.flush()


# -----------------------------------------------------------------------------
# ASGI invocation
# -----------------------------------------------------------------------------

def _build_headers(headers: Dict[str, Any]) -> List[Tuple[bytes, bytes]]:
    out: List[Tuple[bytes, bytes]] = []
    # Ensure JSON body is treated correctly unless caller overrides
    if not any(str(k).lower() == "content-type" for k in headers.keys()):
        out.append((b"content-type", b"application/json"))

    # Mark requests as IPC-internal so the FastAPI app can reject external HTTP.
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
    """
    Invoke FastAPI via raw ASGI.

    If stream=True:
      - Emits stream_* lines to stdout
      - Returns None

    If stream=False:
      - Buffers full response and returns dict for a single JSON line.
    """
    parsed = urlparse(endpoint)
    body_bytes = json.dumps(payload or {}).encode("utf-8")
    sent_body = False
    # Starlette's StreamingResponse runs a disconnect listener that awaits receive()
    # until it gets an "http.disconnect". If receive() returns immediately in a loop,
    # it can starve the streaming task; if it never yields control, the call can hang.
    # We block on this event after the first body, and signal disconnect at stream end.
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
    logger.debug(
        "ASGI scope request_id=%s stream=%s method=%s path=%s query=%s",
        request_id,
        stream,
        method,
        parsed.path,
        parsed.query,
    )

    response_status: int | None = None
    response_headers: Dict[str, str] = {}
    response_body = bytearray()
    error_body = bytearray()
    is_error_response = False

    # Robust against multi-byte splits across chunks
    decoder = codecs.getincrementaldecoder("utf-8")()

    # For streaming, use a bounded queue to prevent unbounded buffering if consumer is slow.
    # This also provides natural backpressure.
    stream_queue: anyio.abc.ObjectSendStream[Dict[str, Any]] | None = None
    stream_recv: anyio.abc.ObjectReceiveStream[Dict[str, Any]] | None = None
    stream_end_emitted = False

    if stream:
        stream_send, stream_recv = anyio.create_memory_object_stream(200)
        stream_queue = stream_send

    async def receive() -> Dict[str, Any]:
        nonlocal sent_body
        if not sent_body:
            sent_body = True
            return {"type": "http.request", "body": body_bytes, "more_body": False}
        # After request body is delivered, block until we decide to "disconnect".
        # This yields control to the event loop so StreamingResponse can run.
        await disconnect_event.wait()
        return {"type": "http.disconnect"}

    async def send(message: Dict[str, Any]) -> None:
        nonlocal response_status
        nonlocal stream_end_emitted
        nonlocal is_error_response

        if message["type"] == "http.response.start":
            response_status = int(message.get("status", 200))
            is_error_response = response_status >= 400
            for k, v in message.get("headers", []):
                try:
                    response_headers[k.decode("utf-8")] = v.decode("utf-8")
                except Exception:
                    # best-effort decoding
                    response_headers[str(k)] = str(v)

            logger.info(
                "ASGI response.start request_id=%s status=%s stream=%s",
                request_id,
                response_status,
                stream,
            )
            if stream and stream_queue is not None:
                await stream_queue.send(
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
            if stream_queue is None:
                return

            # If the ASGI app returns a non-2xx response in "stream mode" (e.g. 409 ingestion_not_ready),
            # do NOT pass the body through as model tokens. Instead, buffer the body and emit a single
            # stream_error so the frontend can handle it cleanly.
            if is_error_response:
                if chunk:
                    error_body.extend(chunk)

                if not more_body:
                    if not stream_end_emitted:
                        stream_end_emitted = True
                        try:
                            raw = bytes(error_body).decode("utf-8", errors="replace").strip()
                            msg = raw or f"HTTP {response_status or 500}"
                            try:
                                parsed = json.loads(raw) if raw else None
                                if isinstance(parsed, dict):
                                    detail = parsed.get("detail")
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

                            await stream_queue.send({"request_id": request_id, "stream_error": msg})
                        except Exception as exc:
                            await stream_queue.send({"request_id": request_id, "stream_error": str(exc)})

                        logger.info("ASGI stream_end request_id=%s (error status=%s)", request_id, response_status)
                        await stream_queue.send({"request_id": request_id, "stream_end": True})
                        await stream_queue.aclose()
                        disconnect_event.set()
                return

            if chunk:
                logger.debug(
                    "ASGI response.body request_id=%s bytes=%d more_body=%s",
                    request_id,
                    len(chunk),
                    more_body,
                )
                text = decoder.decode(chunk)
                if text:
                    await stream_queue.send({"request_id": request_id, "stream_token": text})

            if not more_body:
                # Some Starlette paths emit the final body chunk and then return; this is the
                # canonical end-of-stream. Ensure we only emit once.
                if not stream_end_emitted:
                    stream_end_emitted = True
                    # Flush any remaining decoder state
                    tail = decoder.decode(b"", final=True)
                    if tail:
                        await stream_queue.send({"request_id": request_id, "stream_token": tail})
                    logger.info("ASGI stream_end request_id=%s", request_id)
                    await stream_queue.send({"request_id": request_id, "stream_end": True})
                    await stream_queue.aclose()
                    # Let Starlette's disconnect watcher exit cleanly.
                    disconnect_event.set()
        else:
            response_body.extend(chunk)

    async def _stream_writer_task(recv_stream: anyio.abc.ObjectReceiveStream[Dict[str, Any]]) -> None:
        async with recv_stream:
            async for item in recv_stream:
                rid = item.get("request_id", "-")
                if "stream_start" in item:
                    logger.info("Emit stream_start request_id=%s", rid)
                elif "stream_token" in item:
                    tok = item.get("stream_token") or ""
                    logger.debug(
                        "Emit stream_token request_id=%s chars=%d",
                        rid,
                        len(str(tok)),
                    )
                elif item.get("stream_error"):
                    logger.warning(
                        "Emit stream_error request_id=%s error=%s",
                        rid,
                        item.get("stream_error"),
                    )
                elif item.get("stream_end"):
                    logger.info("Emit stream_end request_id=%s", rid)
                writer.emit(item)

    if stream:
        assert stream_recv is not None
        async with anyio.create_task_group() as tg:
            tg.start_soon(_stream_writer_task, stream_recv)
            logger.info(
                "ASGI call begin request_id=%s stream=true %s %s",
                request_id,
                method,
                endpoint,
            )
            try:
                await app(scope, receive, send)
                logger.info("ASGI call done request_id=%s stream=true", request_id)
            finally:
                # Defensive: some disconnect/cancel paths can cause the ASGI app to return
                # without ever sending the terminal `http.response.body` with more_body=False.
                # If that happens, the Rust side will block forever waiting for stream_end.
                if stream_queue is not None and not stream_end_emitted:
                    stream_end_emitted = True
                    try:
                        tail = decoder.decode(b"", final=True)
                        if tail:
                            await stream_queue.send({"request_id": request_id, "stream_token": tail})
                    except Exception:
                        pass
                    logger.warning("ASGI forcing stream_end request_id=%s (missing final body)", request_id)
                    try:
                        await stream_queue.send({"request_id": request_id, "stream_end": True})
                    except Exception:
                        pass
                    try:
                        await stream_queue.aclose()
                    except Exception:
                        pass
                disconnect_event.set()
        return None

    # Non-streaming
    logger.info(
        "ASGI call begin request_id=%s stream=false %s %s",
        request_id,
        method,
        endpoint,
    )
    await app(scope, receive, send)
    logger.info("ASGI call done request_id=%s stream=false", request_id)

    # Decode and parse
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


# -----------------------------------------------------------------------------
# Dispatcher with concurrency + cancellation
# -----------------------------------------------------------------------------

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

        logger.info(
            "Dispatch request_id=%s stream=%s %s %s payload_keys=%d",
            request_id,
            stream,
            method,
            endpoint,
            len(payload) if isinstance(payload, dict) else 0,
        )

        # Session binding convenience: forward session_id as header if provided
        session_id = req.get("session_id")
        if session_id and not any(str(k).lower() == "x-session-id" for k in headers.keys()):
            headers["x-session-id"] = str(session_id)

        # Basic validation
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
            logger.warning("IPC rejected endpoint=%s method=%s request_id=%s", endpoint, method, request_id)
            self.writer.emit(
                {
                    "request_id": request_id,
                    "ok": False,
                    "status": 403,
                    "error": "endpoint_not_allowed",
                }
            )
            return

        if not isinstance(payload, dict):
            self.writer.emit(
                {
                    "request_id": request_id,
                    "ok": False,
                    "status": 400,
                    "error": "payload must be a JSON object",
                }
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
                # Cancellation: always produce a clean terminal message
                if stream:
                    self.writer.emit(
                        {"request_id": request_id, "stream_error": "cancelled"}
                    )
                    self.writer.emit({"request_id": request_id, "stream_end": True})
                else:
                    self.writer.emit(
                        {
                            "request_id": request_id,
                            "ok": False,
                            "status": 499,
                            "error": "cancelled",
                        }
                    )
            except Exception as exc:
                logger.exception("Request failed request_id=%s", request_id)
                if stream:
                    self.writer.emit(
                        {"request_id": request_id, "stream_error": str(exc)}
                    )
                    self.writer.emit({"request_id": request_id, "stream_end": True})
                else:
                    self.writer.emit(
                        {
                            "request_id": request_id,
                            "ok": False,
                            "status": 500,
                            "error": str(exc),
                        }
                    )
            finally:
                async with self._active_lock:
                    self._active.pop(request_id, None)

        self._tg.start_soon(_run_one)


# -----------------------------------------------------------------------------
# stdin reader -> async dispatcher bridge
# -----------------------------------------------------------------------------

async def _stdin_producer(send: anyio.abc.ObjectSendStream[str]) -> None:
    """
    Reads lines from sys.stdin in a worker thread and pushes them to an async channel.
    """
    def _read_loop() -> None:
        for line in sys.stdin:
            try:
                logger.debug("stdin recv bytes=%d", len(line))
            except Exception:
                pass
            anyio.from_thread.run(send.send, line)
        anyio.from_thread.run(send.aclose)

    await anyio.to_thread.run_sync(_read_loop)


async def main_async() -> None:
    manager = RequestManager(APP)
    await manager.start()

    logger.info("ASGI engine ready; waiting for IPC on stdin")

    # Enable out-of-band events (files_changed, file_text_ready, etc.) in IPC mode.
    # This is intentionally a no-op in HTTP server mode.
    try:
        from backend.services.ipc_events import set_ipc_emitter  # local import

        set_ipc_emitter(manager.writer.emit)
    except Exception:
        pass

    # Emit startup health report once (for UI startup modal).
    try:
        from backend.services.health import run_startup_health  # local import
        from backend.services.ipc_events import emit_event  # local import

        report = run_startup_health()
        emit_event("startup_health", report=report)
    except Exception:
        logger.warning("Startup health check failed", exc_info=True)

    # Bridge stdin (thread) -> async dispatcher.
    # Use a bounded channel to avoid unbounded buffering if Rust sends fast.
    # Note: don't subscript at runtime; keep typing via variable annotations if needed.
    send, recv = anyio.create_memory_object_stream(200)

    async with anyio.create_task_group() as tg:
        tg.start_soon(_stdin_producer, send)

        async with recv:
            async for raw_line in recv:
                line = (raw_line or "").strip()
                if not line:
                    continue

                # Parse JSON
                try:
                    msg = json.loads(line)
                    if not isinstance(msg, dict):
                        raise ValueError("IPC message must be a JSON object")
                except Exception as exc:
                    # request_id may not exist; include a generated id
                    rid = str(uuid.uuid4())
                    manager.writer.emit(
                        {"request_id": rid, "ok": False, "status": 400, "error": f"Invalid JSON: {exc}"}
                    )
                    continue

                # Commands
                cmd = msg.get("cmd")
                logger.info(
                    "IPC recv request_id=%s cmd=%s endpoint=%s stream=%s",
                    msg.get("request_id") or "-",
                    cmd or "-",
                    msg.get("endpoint") or "-",
                    bool(msg.get("stream", False)),
                )
                if cmd == "shutdown":
                    manager.writer.emit(
                        {"request_id": msg.get("request_id") or str(uuid.uuid4()), "ok": True, "status": 200, "data": {"detail": "shutdown"}}
                    )
                    tg.cancel_scope.cancel()
                    break

                if cmd == "cancel":
                    rid = str(msg.get("request_id") or "")
                    if not rid:
                        # Cancel is best-effort; don't emit on stdout to avoid polluting
                        # the request/response stream (especially while a stream thread is reading).
                        logger.warning("cancel missing request_id")
                        continue
                    # IMPORTANT:
                    # Do NOT cancel the ASGI request scope here.
                    #
                    # If we cancel the ASGI scope while a StreamingResponse is mid-flight,
                    # Starlette may return without emitting the terminal response body
                    # (`more_body=False`). That would prevent us from emitting `stream_end`,
                    # leaving the Rust side blocked forever waiting for stream termination.
                    #
                    # We only cancel llama.cpp compute; the stream will then naturally
                    # complete and emit `stream_end` (or `_invoke_asgi` will force it).
                    ok_scope = False
                    # Cancel llama.cpp compute (stops model generation)
                    ok_model = False
                    try:
                        from backend.api.deps import AppDependencies  # local import to avoid eager init

                        ok_model = AppDependencies.session_manager().cancel_request(rid)
                    except Exception as exc:
                        logger.warning("Model cancel failed request_id=%s err=%s", rid, exc)
                    logger.info("Cancel request_id=%s ok_scope=%s ok_model=%s", rid, ok_scope, ok_model)
                    continue

                # Normal request
                await manager.submit(msg)

    await manager.stop()
    try:
        from backend.services.ipc_events import set_ipc_emitter  # local import

        set_ipc_emitter(None)
    except Exception:
        pass


def main() -> None:
    anyio.run(main_async)


if __name__ == "__main__":
    main()
