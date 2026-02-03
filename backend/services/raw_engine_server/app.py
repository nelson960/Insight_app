from __future__ import annotations

import functools
import queue
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

import anyio
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .config import EngineConfig
from .engine import RawEngine
from .logging_store import LogStore
from .schemas import ChatCompletionRequest, EmbeddingsRequest
from .sse import sse_data, sse_json


def create_app() -> FastAPI:
    config = EngineConfig.from_env()
    engine = RawEngine(config)
    log_store = LogStore(config.log_dir)
    model_name = engine.model_info.get("name") or engine.model_info.get("path") or "insight-raw"

    app = FastAPI(title="Insight Raw Engine API", version="0.1.0")

    def _preview(text: str) -> str:
        limit = max(0, int(config.log_preview_chars))
        if limit <= 0:
            return ""
        if len(text) <= limit:
            return text
        return text[:limit] + "…"

    def _usage(prompt_tokens: int, completion_text: str) -> Dict[str, int]:
        completion_tokens = engine.count_tokens(completion_text)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    def _embeddings_ready() -> bool:
        if config.embedding_path is None:
            return False
        base = config.embedding_path
        try:
            return bool((base / "tokenizer.json").exists() and (base / "onnx" / "model.onnx").exists())
        except Exception:
            return False

    def _log_entry(entry: Dict[str, Any]) -> None:
        log_store.record(entry)

    def _build_prompt(request: ChatCompletionRequest) -> Dict[str, Any]:
        messages = [msg.model_dump() for msg in request.messages]
        prompt = engine.render_prompt(messages, add_generation_prompt=True)
        prompt_tokens = engine.count_tokens(prompt)
        stop = list(request.stop or [])
        stop.extend(engine.stop_markers())
        max_tokens = engine.clamp_max_tokens(prompt_tokens, request.max_tokens)
        return {
            "prompt": prompt,
            "prompt_tokens": prompt_tokens,
            "stop": stop,
            "max_tokens": max_tokens,
        }

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        chat_busy = not engine.acquire_chat()
        if not chat_busy:
            engine.release_chat()
        embeddings_busy = not engine.acquire_embeddings()
        if not embeddings_busy:
            engine.release_embeddings()
        return {
            "ok": True,
            "status": "ok",
            "model_loaded": True,
            "model": model_name,
            "uptime_sec": round(engine.uptime(), 2),
            "ctx_size": engine.ctx_size,
            "prompt_renderer": engine.model_info.get("prompt_renderer"),
            "embeddings_ready": _embeddings_ready(),
            "embedding_model": config.embedding_model,
            "embedding_path_present": bool(config.embedding_path),
            "chat_busy": chat_busy,
            "embeddings_busy": embeddings_busy,
        }

    @app.get("/v1/model/info")
    async def model_info() -> Dict[str, Any]:
        return {"ok": True, "model": engine.model_info}


    @app.post("/v1/embeddings")
    async def embeddings(request: EmbeddingsRequest) -> Dict[str, Any]:
        if not engine.acquire_embeddings():
            raise HTTPException(status_code=429, detail="engine_busy")
        started = time.time()
        request_id = f"emb-{uuid.uuid4().hex}"
        try:
            inputs = request.input
            if isinstance(inputs, str):
                inputs = [inputs]
            embeddings = await anyio.to_thread.run_sync(engine.embed, inputs)
            data = [
                {"object": "embedding", "index": idx, "embedding": vector}
                for idx, vector in enumerate(embeddings)
            ]
            entry = {
                "id": request_id,
                "endpoint": "/v1/embeddings",
                "created": int(started),
                "model": request.model or config.embedding_model,
                "status": 200,
                "latency_ms": int((time.time() - started) * 1000),
                "items": len(data),
            }
            _log_entry(entry)
            return {"object": "list", "data": data, "model": request.model or config.embedding_model}
        finally:
            engine.release_embeddings()

    @app.get("/v1/logs/recent")
    async def logs_recent(limit: int = Query(50, ge=1, le=500)) -> Dict[str, Any]:
        return {"data": log_store.recent(limit)}

    @app.get("/v1/logs/{entry_id}")
    async def logs_get(entry_id: str) -> Dict[str, Any]:
        entry = log_store.get(entry_id)
        if not entry:
            raise HTTPException(status_code=404, detail="log_not_found")
        return entry

    @app.delete("/v1/logs")
    async def logs_clear() -> Dict[str, Any]:
        ok = log_store.clear()
        if not ok:
            raise HTTPException(status_code=500, detail="log_clear_failed")
        return {"ok": True}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest, raw_request: Request) -> Any:
        if not engine.acquire_chat():
            raise HTTPException(status_code=429, detail="engine_busy")
        started = time.time()
        request_id = f"chat-{uuid.uuid4().hex}"
        try:
            prompt_data = _build_prompt(request)
            prompt = prompt_data["prompt"]
            prompt_tokens = prompt_data["prompt_tokens"]
            stop = prompt_data["stop"]
            max_tokens = prompt_data["max_tokens"]

            if prompt_tokens >= engine.ctx_size:
                raise HTTPException(status_code=400, detail="prompt_too_long")
            if max_tokens <= 0:
                raise HTTPException(status_code=400, detail="max_tokens_exhausted")

            if not request.stream:
                response = await anyio.to_thread.run_sync(
                    functools.partial(
                        engine.create_completion,
                        prompt=prompt,
                        stop=stop,
                        max_tokens=max_tokens,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        top_k=request.top_k,
                        repeat_penalty=request.repeat_penalty,
                    )
                )
                choice = response.get("choices", [{}])[0]
                completion_text = choice.get("text") or ""
                finish_reason = choice.get("finish_reason") or "stop"
                usage = _usage(prompt_tokens, completion_text)
                payload = {
                    "id": request_id,
                    "object": "chat.completion",
                    "created": int(started),
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": completion_text},
                            "finish_reason": finish_reason,
                        }
                    ],
                    "usage": usage,
                }
                entry = {
                    "id": request_id,
                    "endpoint": "/v1/chat/completions",
                    "created": int(started),
                    "model": model_name,
                    "stream": False,
                    "status": 200,
                    "latency_ms": int((time.time() - started) * 1000),
                    "prompt_tokens": usage["prompt_tokens"],
                    "completion_tokens": usage["completion_tokens"],
                    "total_tokens": usage["total_tokens"],
                    "finish_reason": finish_reason,
                }
                if config.log_prompts:
                    entry["prompt_preview"] = _preview(prompt)
                if config.log_completions:
                    entry["completion_preview"] = _preview(completion_text)
                _log_entry(entry)
                engine.release_chat()
                return JSONResponse(payload)

            async def event_stream() -> Any:
                completion_parts: List[str] = []
                finish_reason: Optional[str] = None
                status_code = 200
                error_text: Optional[str] = None
                stop_event = threading.Event()
                stream_queue: "queue.Queue[object]" = queue.Queue()
                queue_empty = object()
                stream_done = object()

                def _queue_get() -> object:
                    try:
                        return stream_queue.get(timeout=0.1)
                    except queue.Empty:
                        return queue_empty

                def _stream_worker() -> None:
                    try:
                        stream = engine.stream_completion(
                            prompt=prompt,
                            stop=stop,
                            max_tokens=max_tokens,
                            temperature=request.temperature,
                            top_p=request.top_p,
                            top_k=request.top_k,
                            repeat_penalty=request.repeat_penalty,
                        )
                        for chunk in stream:
                            if stop_event.is_set():
                                break
                            stream_queue.put(chunk)
                        stream_queue.put(stream_done)
                    except Exception as exc:
                        stream_queue.put(exc)

                worker = threading.Thread(
                    target=_stream_worker,
                    name="raw-engine-stream",
                    daemon=True,
                )
                worker.start()
                try:
                    while True:
                        item = await anyio.to_thread.run_sync(_queue_get)
                        if item is queue_empty:
                            if await raw_request.is_disconnected():
                                finish_reason = "cancelled"
                                stop_event.set()
                                break
                            continue
                        if item is stream_done:
                            break
                        if isinstance(item, Exception):
                            status_code = 500
                            error_text = str(item)
                            finish_reason = "error"
                            break
                        if await raw_request.is_disconnected():
                            finish_reason = "cancelled"
                            stop_event.set()
                            break
                        chunk = item
                        choice = chunk.get("choices", [{}])[0]
                        delta_text = choice.get("text") or ""
                        finish_reason = choice.get("finish_reason") or finish_reason
                        if delta_text:
                            completion_parts.append(delta_text)
                            payload = {
                                "id": request_id,
                                "object": "chat.completion.chunk",
                                "created": int(started),
                                "model": model_name,
                                "choices": [{"index": 0, "delta": {"content": delta_text}}],
                            }
                            yield sse_json(payload)
                    if status_code == 200:
                        yield sse_data("[DONE]")
                except Exception as exc:
                    status_code = 500
                    error_text = str(exc)
                finally:
                    stop_event.set()
                    await anyio.to_thread.run_sync(worker.join)
                    completion_text = "".join(completion_parts)
                    usage = _usage(prompt_tokens, completion_text)
                    entry = {
                        "id": request_id,
                        "endpoint": "/v1/chat/completions",
                        "created": int(started),
                        "model": model_name,
                        "stream": True,
                        "status": status_code,
                        "latency_ms": int((time.time() - started) * 1000),
                        "prompt_tokens": usage["prompt_tokens"],
                        "completion_tokens": usage["completion_tokens"],
                        "total_tokens": usage["total_tokens"],
                        "finish_reason": finish_reason or ("error" if status_code != 200 else "stop"),
                    }
                    if error_text:
                        entry["error"] = error_text
                    if config.log_prompts:
                        entry["prompt_preview"] = _preview(prompt)
                    if config.log_completions:
                        entry["completion_preview"] = _preview(completion_text)
                    _log_entry(entry)
                    engine.release_chat()

            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        except HTTPException:
            engine.release_chat()
            raise
        except Exception as exc:
            engine.release_chat()
            entry = {
                "id": request_id,
                "endpoint": "/v1/chat/completions",
                "created": int(started),
                "model": model_name,
                "stream": request.stream,
                "status": 500,
                "latency_ms": int((time.time() - started) * 1000),
                "error": str(exc),
            }
            _log_entry(entry)
            raise

    return app


__all__ = ["create_app"]
