from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, List

import requests

logger = logging.getLogger(__name__)


class LlamaCppConnector:
    """Connector for llama.cpp's OpenAI-compatible server."""

    def __init__(self, config: Dict[str, Any]):
        models = config.get("models", {})
        mode_cfg = models.get(config.get("mode", "local"), {})
        if not mode_cfg:
            raise ValueError("Missing llama.cpp model configuration")
        self.base_url = mode_cfg.get("base_url", "http://127.0.0.1:8080")
        self.model = mode_cfg.get("model")
        if not self.model:
            raise ValueError("Model name/path must be provided for llama.cpp connector")

    def set_mode(self, mode: str) -> None:
        # Single-mode connector; no-op for compatibility.
        return

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        resp = requests.post(f"{self.base_url}/v1/chat/completions", json=payload, timeout=180)
        resp.raise_for_status()
        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        text = choice.get("message", {}).get("content", "")
        logger.info("llama.cpp chat completed model=%s", self.model)
        return {
            "text": text,
            "model": self.model,
            "provider": "llamacpp",
        }

    def generate_stream(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> Iterator[Dict[str, Any]]:
        raise NotImplementedError("Streaming not implemented for llama.cpp connector in this setup.")

    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError("Embeddings via llama.cpp are not wired; use local embedder instead.")


__all__ = ["LlamaCppConnector"]

