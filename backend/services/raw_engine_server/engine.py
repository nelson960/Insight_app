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
from backend.services.llama_templates import (
    apply_chat_template_minja,
    chat_template_error_message,
    normalize_messages_for_template,
    validate_chat_template_for_llm,
)

logger = logging.getLogger(__name__)


class RawEngine:
    _PROMPT_RENDERER_LLAMA3 = "llama3"
    _PROMPT_RENDERER_CHATML = "chatml"
    _PROMPT_RENDERER_UNKNOWN = "unknown"
    _DEFAULT_SYSTEM_PROMPT = ""
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
        tpl_result = validate_chat_template_for_llm(
            self.llm,
            model_path=self.config.model_path,
            require_chat_template=True,
            reject_multimodal=True,
        )
        if not tpl_result.ok:
            raise ValueError(chat_template_error_message(tpl_result.reason))
        self._chat_template = tpl_result.template
        self._chat_template_name = tpl_result.template_name or "default"
        self._stop_token_texts = list(tpl_result.stop_token_texts or [])
        self._stop_token_ids = list(tpl_result.stop_token_ids or [])
        self.prompt_renderer = f"minja:{self._chat_template_name}"
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
        def _meta_str(key: str) -> str:
            try:
                val = meta.get(key)
                return "" if val is None else str(val)
            except Exception:
                return ""

        def _meta_int(key: str) -> Optional[int]:
            try:
                val = meta.get(key)
                if val is None:
                    return None
                return int(val)
            except Exception:
                return None

        def _meta_int_suffix(suffixes: List[str]) -> Optional[int]:
            for key, value in meta.items():
                if not isinstance(key, str):
                    continue
                if not any(key.endswith(sfx) for sfx in suffixes):
                    continue
                if value is None:
                    continue
                try:
                    return int(value)
                except Exception:
                    continue
            return None

        def _meta_str_suffix(suffixes: List[str]) -> str:
            for key, value in meta.items():
                if not isinstance(key, str):
                    continue
                if not any(key.endswith(sfx) for sfx in suffixes):
                    continue
                if value is None:
                    continue
                try:
                    return str(value)
                except Exception:
                    continue
            return ""

        def _meta_float_suffix(suffixes: List[str]) -> Optional[float]:
            for key, value in meta.items():
                if not isinstance(key, str):
                    continue
                if not any(key.endswith(sfx) for sfx in suffixes):
                    continue
                try:
                    return float(value)
                except Exception:
                    continue
            return None

        meta_arch = _meta_str("general.architecture")
        meta_name = _meta_str("general.name")
        meta_basename = _meta_str("general.basename")
        meta_size = _meta_str("general.size_label")
        meta_file_type = _meta_int("general.file_type")
        meta_quant_ver = _meta_int("general.quantization_version")

        n_layer = _meta_int_suffix([".block_count"])
        n_head = _meta_int_suffix([".attention.head_count"])
        n_head_kv = _meta_int_suffix([".attention.head_count_kv"])
        n_embd = _meta_int_suffix([".embedding_length"])
        rope_type = _meta_str_suffix([".rope.type", ".rope.scaling.type"])
        rope_freq_base = _meta_float_suffix([".rope.freq_base"])

        ctx_train: Optional[int] = None
        try:
            ctx_train = int(self.llm._model.n_ctx_train())  # type: ignore[attr-defined]
        except Exception:
            ctx_train = _meta_int_suffix([".context_length"])

        tok_model = _meta_str("tokenizer.ggml.model") or _meta_str("tokenizer.ggml.pre")
        add_bos = _meta_str("tokenizer.ggml.add_bos_token")
        bos_id = _meta_int("tokenizer.ggml.bos_token_id")
        eos_id = _meta_int("tokenizer.ggml.eos_token_id")
        vocab_size: Optional[int] = None
        try:
            vocab_size = int(self.llm.n_vocab())  # type: ignore[attr-defined]
        except Exception:
            vocab_size = _meta_int("tokenizer.ggml.tokens")

        kv_cache_gib: Optional[float] = None
        try:
            if n_layer and n_head and n_head_kv and n_embd and self.ctx_size:
                head_dim = max(1, int(n_embd // max(1, n_head)))
                kv_bytes = 4 * n_layer * int(self.ctx_size) * n_head_kv * head_dim
                kv_cache_gib = kv_bytes / (1024 ** 3)
        except Exception:
            kv_cache_gib = None

        return {
            "path": str(self.config.model_path),
            "name": meta_name or meta_basename or self.config.model_path.name,
            "architecture": meta_arch or None,
            "size_label": meta_size or None,
            "file_type": meta_file_type,
            "quantization_version": meta_quant_ver,
            "ctx_train": ctx_train,
            "ctx_runtime": self.ctx_size,
            "n_layer": n_layer,
            "n_head": n_head,
            "n_head_kv": n_head_kv,
            "n_embd": n_embd,
            "rope_type": rope_type or None,
            "rope_freq_base": rope_freq_base,
            "vocab_size": vocab_size,
            "tokenizer_model": tok_model or None,
            "add_bos_token": add_bos or None,
            "bos_token_id": bos_id,
            "eos_token_id": eos_id,
            "kv_cache_gib": kv_cache_gib,
            "prompt_renderer": self.prompt_renderer,
            "chat_template_name": str(getattr(self, "_chat_template_name", "") or "default"),
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
        if not getattr(self, "_chat_template", None):
            raise ValueError("Missing chat template for this model.")
        normalized = normalize_messages_for_template(self._normalize_messages(messages))
        return apply_chat_template_minja(
            self._chat_template,
            normalized,
            add_generation_prompt=add_generation_prompt,
        )

    def stop_markers(self) -> List[str]:
        if getattr(self, "_stop_token_texts", None):
            return list(self._stop_token_texts)
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
