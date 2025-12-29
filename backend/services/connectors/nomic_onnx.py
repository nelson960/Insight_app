from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from .nomic import DEFAULT_MODEL_REPO_ID, MissingDependencyError, ensure_local_nomic_model_files

logger = logging.getLogger(__name__)


def _lazy_import_onnxruntime():
    try:
        import onnxruntime as ort  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise MissingDependencyError(
            "onnxruntime is required for ONNX embeddings. Install with `pip install onnxruntime` "
            "(or `onnxruntime-silicon` on Apple Silicon)."
        ) from exc
    return ort


def _lazy_import_tokenizers():
    try:
        from tokenizers import Tokenizer  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise MissingDependencyError(
            "tokenizers library is required to load tokenizer.json. Install with `pip install tokenizers`."
        ) from exc
    return Tokenizer


@dataclass
class NomicOnnxConfig:
    model_filename: str = "onnx/model.onnx"
    max_length: int = 2048
    normalize_embeddings: bool = True
    providers: Optional[Tuple[str, ...]] = ("CPUExecutionProvider",)


class NomicOnnxEmbedTextConnector:
    is_local = True

    def __init__(
        self,
        model_dir: Path,
        *,
        config: Optional[NomicOnnxConfig] = None,
        auto_download: bool = False,
        repo_id: str = DEFAULT_MODEL_REPO_ID,
        revision: Optional[str] = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.config = config or NomicOnnxConfig()
        self.model_path = self.model_dir / self.config.model_filename
        self.tokenizer_path = self.model_dir / "tokenizer.json"
        if auto_download:
            # Only download the minimal required assets for the ONNX runtime:
            # tokenizer.json + the chosen ONNX model file.
            ensure_local_nomic_model_files(
                self.model_dir,
                required_paths=[self.config.model_filename, "tokenizer.json"],
                repo_id=repo_id,
                revision=revision,
            )
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found at {self.model_path}")
        if not self.tokenizer_path.exists():
            raise FileNotFoundError(f"tokenizer.json not found at {self.tokenizer_path}")

        self._ort = _lazy_import_onnxruntime()
        Tokenizer = _lazy_import_tokenizers()
        self._tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        self._tokenizer.enable_truncation(max_length=self.config.max_length)
        pad_token = "[PAD]"
        pad_id = self._tokenizer.token_to_id(pad_token)
        if pad_id is None:
            pad_id = 0
        # Important: do NOT pad to a fixed max_length. Padding to 2048 forces the ONNX model
        # to process the full sequence length even for short chunks, which can make ingestion
        # appear "hung" (minutes+) on CPU. Dynamic padding keeps shapes consistent per-batch,
        # but only pads to the longest sequence in that batch.
        self._tokenizer.enable_padding(pad_id=pad_id, pad_token=pad_token)

        providers = list(self.config.providers) if self.config.providers else None
        logger.info("Loading ONNX runtime session from %s (providers=%s)", self.model_path, providers)
        self._session = self._ort.InferenceSession(
            str(self.model_path),
            providers=providers or self._ort.get_available_providers(),
        )
        self._output_name = self._session.get_outputs()[0].name
        self._lock = threading.Lock()

    def supports(self, model: str) -> bool:
        return model in {
            "nomic-embed-text",
            "nomic-embed-text-v1.5",
            "nomic-embed-text-onnx",
        }

    def embed(self, model: str, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        if not texts:
            return []
        # The Tokenizer + ORT session are not guaranteed to be thread-safe across all builds.
        # We serialize calls to avoid rare hangs under concurrent ingestion + retrieval load.
        with self._lock:
            encodings = self._tokenizer.encode_batch(list(texts))
            input_ids = np.asarray([encoding.ids for encoding in encodings], dtype=np.int64)
            attention_mask = np.asarray([encoding.attention_mask for encoding in encodings], dtype=np.int64)
            type_ids = np.asarray([encoding.type_ids for encoding in encodings], dtype=np.int64)

            outputs = self._session.run(
                [self._output_name],
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": type_ids,
                },
            )
        hidden = outputs[0]
        embeddings = self._mean_pool(hidden, attention_mask)
        if self.config.normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.clip(norms, 1e-6, None)
            embeddings = embeddings / norms
        return embeddings.tolist()

    @staticmethod
    def _mean_pool(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        mask = attention_mask[..., None]
        summed = (last_hidden_state * mask).sum(axis=1)
        counts = np.clip(mask.sum(axis=1), 1e-6, None)
        return summed / counts


__all__ = ["NomicOnnxEmbedTextConnector", "NomicOnnxConfig"]
