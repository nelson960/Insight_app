from __future__ import annotations

import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

if "llama_cpp" not in sys.modules:
    sys.modules["llama_cpp"] = types.SimpleNamespace(Llama=object)

from fastapi.testclient import TestClient

from backend.services.raw_engine_server import app as raw_app


def _make_config(tmp_dir: str, *, ctx_size: int = 64, default_max_tokens: int = 16) -> SimpleNamespace:
    return SimpleNamespace(
        host="127.0.0.1",
        port=11435,
        model_path=Path("stub.gguf"),
        auth_token=None,
        ctx_size=ctx_size,
        n_threads=None,
        n_gpu_layers=None,
        default_max_tokens=default_max_tokens,
        log_dir=Path(tmp_dir),
        log_prompts=False,
        log_completions=False,
        log_preview_chars=200,
        embedding_path=Path(tmp_dir) / "embeddings",
        embedding_model="nomic-embed-text-v1.5",
        embedding_auto_download=False,
    )


class StubEngine:
    def __init__(
        self,
        config: SimpleNamespace,
        *,
        prompt_tokens: int = 1,
        clamp_override: int | None = None,
        completion_text: str = "ok",
        stream_chunks: list[dict] | None = None,
        stream_exc: Exception | None = None,
        ctx_size: int | None = None,
    ) -> None:
        self.config = config
        self.ctx_size = int(ctx_size if ctx_size is not None else (config.ctx_size or 0))
        self.model_info = {"name": "stub-model", "prompt_renderer": "chatml"}
        self.prompt_renderer = "chatml"
        self.started_at = time.time()
        self._chat_lock = threading.Lock()
        self._embeddings_lock = threading.Lock()
        self._prompt_tokens = prompt_tokens
        self._clamp_override = clamp_override
        self._completion_text = completion_text
        self._stream_chunks = stream_chunks
        self._stream_exc = stream_exc
        self._last_prompt: str | None = None

    def acquire(self) -> bool:
        return self.acquire_chat()

    def release(self) -> None:
        self.release_chat()

    def acquire_chat(self) -> bool:
        return self._chat_lock.acquire(blocking=False)

    def release_chat(self) -> None:
        try:
            self._chat_lock.release()
        except RuntimeError:
            pass

    def acquire_embeddings(self) -> bool:
        return self._embeddings_lock.acquire(blocking=False)

    def release_embeddings(self) -> None:
        try:
            self._embeddings_lock.release()
        except RuntimeError:
            pass

    def uptime(self) -> float:
        return 12.34

    def render_prompt(self, messages: list[dict], *, add_generation_prompt: bool) -> str:
        self._last_prompt = "PROMPT"
        return self._last_prompt

    def stop_markers(self) -> list[str]:
        return ["<STOP>"]

    def count_tokens(self, text: str) -> int:
        if text == self._last_prompt:
            return int(self._prompt_tokens)
        return max(1, len(text.split()))

    def clamp_max_tokens(self, prompt_tokens: int, requested: int | None) -> int:
        if self._clamp_override is not None:
            return int(self._clamp_override)
        if requested is None:
            requested = int(self.config.default_max_tokens)
        max_available = max(0, int(self.ctx_size) - int(prompt_tokens))
        if max_available <= 0:
            return 0
        return max(1, min(int(requested), max_available))

    def create_completion(
        self,
        *,
        prompt: str,
        stop: list[str] | None,
        max_tokens: int,
        temperature: float | None,
        top_p: float | None,
        top_k: int | None,
        repeat_penalty: float | None,
    ) -> dict:
        return {"choices": [{"text": self._completion_text, "finish_reason": "stop"}]}

    def stream_completion(
        self,
        *,
        prompt: str,
        stop: list[str] | None,
        max_tokens: int,
        temperature: float | None,
        top_p: float | None,
        top_k: int | None,
        repeat_penalty: float | None,
    ):
        if self._stream_exc is not None:
            raise self._stream_exc
        chunks = self._stream_chunks or [
            {"choices": [{"text": "He"}]},
            {"choices": [{"text": "llo", "finish_reason": "stop"}]},
        ]
        for chunk in chunks:
            yield chunk

    def embed(self, inputs: list[str]) -> list[list[float]]:
        return [[float(idx), float(idx) + 0.5] for idx, _ in enumerate(inputs)]


class RawEngineServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

    def _build_client(self, engine: StubEngine, config: SimpleNamespace) -> TestClient:
        with patch.object(raw_app.EngineConfig, "from_env", return_value=config), patch.object(
            raw_app, "RawEngine", return_value=engine
        ):
            app = raw_app.create_app()
        return TestClient(app)

    def _make_client(self, **engine_kwargs):
        config = _make_config(self.tmpdir.name, ctx_size=engine_kwargs.get("ctx_size", 64))
        engine = StubEngine(config, **engine_kwargs)
        return self._build_client(engine, config), engine

    def test_health(self) -> None:
        client, _engine = self._make_client()
        resp = client.get("/health")
        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["model"], "stub-model")
        self.assertEqual(payload["prompt_renderer"], "chatml")

    def test_chat_non_stream_success(self) -> None:
        client, _engine = self._make_client(prompt_tokens=2, completion_text="hello")
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": False},
        )
        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertEqual(payload["choices"][0]["message"]["content"], "hello")
        self.assertEqual(payload["usage"]["prompt_tokens"], 2)
        self.assertEqual(payload["usage"]["completion_tokens"], 1)
        self.assertEqual(payload["usage"]["total_tokens"], 3)

    def test_chat_prompt_too_long(self) -> None:
        client, _engine = self._make_client(ctx_size=4, prompt_tokens=4)
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": False},
        )
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.json()["detail"], "prompt_too_long")

    def test_chat_max_tokens_exhausted(self) -> None:
        client, _engine = self._make_client(ctx_size=10, prompt_tokens=1, clamp_override=0)
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": False},
        )
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.json()["detail"], "max_tokens_exhausted")

    def test_chat_stream_success(self) -> None:
        client, _engine = self._make_client()
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
        ) as resp:
            self.assertEqual(resp.status_code, 200)
            body = b"".join(resp.iter_bytes()).decode("utf-8")
        self.assertIn("chat.completion.chunk", body)
        self.assertIn("data: [DONE]", body)
        self.assertIn("He", body)

    def test_chat_stream_error_logs(self) -> None:
        client, _engine = self._make_client(stream_exc=RuntimeError("boom"))
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
        ) as resp:
            self.assertEqual(resp.status_code, 200)
            _ = b"".join(resp.iter_bytes())
        logs = client.get("/v1/logs/recent?limit=1").json()["data"]
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]["status"], 500)
        self.assertIn("boom", logs[0]["error"])

    def test_embeddings_success(self) -> None:
        client, _engine = self._make_client()
        resp = client.post("/v1/embeddings", json={"input": ["a", "b"]})
        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertEqual(len(payload["data"]), 2)
        self.assertEqual(payload["data"][0]["embedding"], [0.0, 0.5])

    def test_embeddings_busy(self) -> None:
        client, engine = self._make_client()
        self.assertTrue(engine.acquire_embeddings())
        resp = client.post("/v1/embeddings", json={"input": "hi"})
        self.assertEqual(resp.status_code, 429)
        engine.release_embeddings()

    def test_embeddings_can_run_during_chat(self) -> None:
        client, engine = self._make_client()
        self.assertTrue(engine.acquire_chat())
        resp = client.post("/v1/embeddings", json={"input": "hi"})
        self.assertEqual(resp.status_code, 200)
        engine.release_chat()

    def test_chat_busy(self) -> None:
        client, engine = self._make_client()
        self.assertTrue(engine.acquire_chat())
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": False},
        )
        self.assertEqual(resp.status_code, 429)
        engine.release_chat()

    def test_logs_get(self) -> None:
        client, _engine = self._make_client()
        resp = client.post("/v1/embeddings", json={"input": "hi"})
        self.assertEqual(resp.status_code, 200)
        entry = client.get("/v1/logs/recent?limit=1").json()["data"][0]
        fetched = client.get(f"/v1/logs/{entry['id']}").json()
        self.assertEqual(fetched["id"], entry["id"])


if __name__ == "__main__":
    unittest.main()
