from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .config import EngineConfig

try:
    from llama_cpp import Llama  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("llama_cpp is required for raw engine.") from exc

from backend.services.connectors.nomic_onnx import NomicOnnxConfig, NomicOnnxEmbedTextConnector

logger = logging.getLogger(__name__)


class RawEngine:
    _PROMPT_RENDERER_LLAMA3 = "llama3"
    _PROMPT_RENDERER_CHATML = "chatml"
    _PROMPT_RENDERER_UNKNOWN = "unknown"
    _DEFAULT_SYSTEM_PROMPT = "You are Insight, a local privacy-first AI assistant."
    _DEFAULT_TEMPERATURE = 0.7
    _DEFAULT_TOP_P = 0.9
    _DEFAULT_TOP_K = 40
    _DEFAULT_REPEAT_PENALTY = 1.1

    def __init__(self, config: EngineConfig) -> None:
        if not config.model_path.exists():
            raise FileNotFoundError(f"Model not found at {config.model_path}")
        self.config = config
        self._chat_lock = threading.Lock()
        self._embeddings_lock = threading.Lock()
        self._embedder_lock = threading.Lock()
        self._embedder: Optional[NomicOnnxEmbedTextConnector] = None
        self.started_at = time.time()

        kwargs: Dict[str, Any] = {"model_path": str(config.model_path)}
        if config.ctx_size is not None:
            kwargs["n_ctx"] = int(config.ctx_size)
        if config.n_gpu_layers is not None:
            kwargs["n_gpu_layers"] = int(config.n_gpu_layers)
        if config.n_threads is not None:
            kwargs["n_threads"] = int(config.n_threads)

        self.llm = Llama(**kwargs)
        self.ctx_size = self._resolve_ctx_size(config.ctx_size)
        self.prompt_renderer = self._detect_prompt_renderer_id()
        if self.prompt_renderer == self._PROMPT_RENDERER_UNKNOWN:
            raise ValueError("Unsupported chat template for this model.")
        self.model_info = self._build_model_info()

    def _resolve_ctx_size(self, requested: Optional[int]) -> int:
        try:
            actual = int(self.llm.n_ctx())
            if requested is not None and actual != requested:
                logger.warning("Requested ctx_size=%d but llama_cpp is using n_ctx=%d", requested, actual)
            return actual
        except Exception:
            return int(requested or 0)

    def _detect_prompt_renderer_id(self) -> str:
        try:
            meta = getattr(self.llm, "metadata", {}) or {}
        except Exception:
            meta = {}
        try:
            template = str(meta.get("tokenizer.chat_template") or "")
        except Exception:
            template = ""

        if "<|im_start|>" in template and "<|im_end|>" in template:
            return self._PROMPT_RENDERER_CHATML
        if "<|start_header_id|>" in template and "<|eot_id|>" in template:
            return self._PROMPT_RENDERER_LLAMA3
        return self._PROMPT_RENDERER_UNKNOWN

    def _build_model_info(self) -> Dict[str, Any]:
        try:
            meta = getattr(self.llm, "metadata", {}) or {}
        except Exception:
            meta = {}
        return {
            "path": str(self.config.model_path),
            "name": meta.get("general.name") or meta.get("general.basename") or self.config.model_path.name,
            "architecture": meta.get("general.architecture"),
            "file_type": meta.get("general.file_type"),
            "quantization_version": meta.get("general.quantization_version"),
            "ctx_train": meta.get("llama.context_length") or meta.get("qwen2.context_length"),
            "ctx_runtime": self.ctx_size,
            "prompt_renderer": self.prompt_renderer,
        }

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
        return max(0.0, time.time() - self.started_at)

    def _normalize_messages(self, messages: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
        normalized = list(messages or [])
        if not normalized or normalized[0].get("role") != "system":
            normalized.insert(0, {"role": "system", "content": self._DEFAULT_SYSTEM_PROMPT})
        return normalized

    def _render_chatml_prompt(self, messages: Sequence[Dict[str, str]], *, add_generation_prompt: bool) -> str:
        system_message = ""
        has_system = False
        rest = list(messages or [])
        if rest and rest[0].get("role") == "system":
            has_system = True
            system_message = rest[0].get("content") or ""
            rest = rest[1:]

        out: List[str] = []
        out.append("<|im_start|>system\n")
        out.append(system_message)
        out.append("<|im_end|>\n")

        for msg in rest:
            role = (msg.get("role") or "user").strip()
            content = msg.get("content") or ""
            out.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")

        if add_generation_prompt:
            out.append("<|im_start|>assistant\n")
        return "".join(out)

    def _render_llama3_prompt(self, messages: Sequence[Dict[str, str]], *, add_generation_prompt: bool) -> str:
        system_message = ""
        rest = list(messages or [])
        if rest and rest[0].get("role") == "system":
            system_message = rest[0].get("content") or ""
            rest = rest[1:]

        out: List[str] = []
        out.append("<|begin_of_text|>")
        out.append("<|start_header_id|>system<|end_header_id|>\n\n")
        out.append(system_message)
        out.append("<|eot_id|>")

        for msg in rest:
            role = (msg.get("role") or "user").strip()
            content = msg.get("content") or ""
            out.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>")

        if add_generation_prompt:
            out.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        return "".join(out)

    def render_prompt(self, messages: Sequence[Dict[str, str]], *, add_generation_prompt: bool) -> str:
        normalized = self._normalize_messages(messages)
        if self.prompt_renderer == self._PROMPT_RENDERER_CHATML:
            return self._render_chatml_prompt(normalized, add_generation_prompt=add_generation_prompt)
        if self.prompt_renderer == self._PROMPT_RENDERER_LLAMA3:
            return self._render_llama3_prompt(normalized, add_generation_prompt=add_generation_prompt)
        raise ValueError("Unsupported chat template for this model.")

    def stop_markers(self) -> List[str]:
        if self.prompt_renderer == self._PROMPT_RENDERER_CHATML:
            return ["<|im_end|>", "<|endoftext|>"]
        return ["<|eot_id|>", "<|end_of_text|>"]

    def tokenize(self, text: str) -> List[int]:
        return list(self.llm.tokenize(text.encode("utf-8")))

    def count_tokens(self, text: str) -> int:
        return len(self.tokenize(text))

    def clamp_max_tokens(self, prompt_tokens: int, requested: Optional[int]) -> int:
        if requested is None:
            requested = int(self.config.default_max_tokens)
        max_available = max(0, int(self.ctx_size) - int(prompt_tokens))
        if max_available <= 0:
            return 0
        return max(1, min(int(requested), max_available))

    def _normalize_sampling(
        self,
        temperature: Optional[float],
        top_p: Optional[float],
        top_k: Optional[int],
        repeat_penalty: Optional[float],
    ) -> Tuple[float, float, int, float]:
        temp = float(temperature) if temperature is not None else self._DEFAULT_TEMPERATURE
        p = float(top_p) if top_p is not None else self._DEFAULT_TOP_P
        k = int(top_k) if top_k is not None else self._DEFAULT_TOP_K
        penalty = float(repeat_penalty) if repeat_penalty is not None else self._DEFAULT_REPEAT_PENALTY
        return temp, p, max(0, k), penalty

    def create_completion(
        self,
        *,
        prompt: str,
        stop: Optional[List[str]],
        max_tokens: int,
        temperature: Optional[float],
        top_p: Optional[float],
        top_k: Optional[int],
        repeat_penalty: Optional[float],
    ) -> Dict[str, Any]:
        self.llm.reset()
        temp, p, k, penalty = self._normalize_sampling(temperature, top_p, top_k, repeat_penalty)
        response = self.llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temp,
            top_p=p,
            top_k=k,
            repeat_penalty=penalty,
            stop=stop,
            stream=False,
        )
        return response

    def stream_completion(
        self,
        *,
        prompt: str,
        stop: Optional[List[str]],
        max_tokens: int,
        temperature: Optional[float],
        top_p: Optional[float],
        top_k: Optional[int],
        repeat_penalty: Optional[float],
    ) -> Iterable[Dict[str, Any]]:
        self.llm.reset()
        temp, p, k, penalty = self._normalize_sampling(temperature, top_p, top_k, repeat_penalty)
        return self.llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temp,
            top_p=p,
            top_k=k,
            repeat_penalty=penalty,
            stop=stop,
            stream=True,
        )

    def embed(self, inputs: Sequence[str]) -> List[List[float]]:
        if self.config.embedding_path is None:
            raise RuntimeError("Embeddings not configured.")
        with self._embedder_lock:
            if self._embedder is None:
                self._embedder = NomicOnnxEmbedTextConnector(
                    model_dir=self.config.embedding_path,
                    config=NomicOnnxConfig(),
                    auto_download=self.config.embedding_auto_download,
                    repo_id=self.config.embedding_model,
                )
        return [list(vec) for vec in self._embedder.embed(self.config.embedding_model, inputs)]


__all__ = ["RawEngine"]
