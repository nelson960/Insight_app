from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

logger = logging.getLogger(__name__)


class MissingDependencyError(ImportError):
    """Raised when the required embedding runtime is missing."""


def _lazy_import_sentence_transformer():
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as exc:  # pragma: no cover - import guard
        raise MissingDependencyError(
            "sentence-transformers package not installed. "
            "Install with `pip install sentence-transformers`."
        ) from exc
    return SentenceTransformer


DEFAULT_MODEL_REPO_ID = "nomic-ai/nomic-embed-text-v1.5"

_download_lock = threading.Lock()


def _lazy_import_snapshot_download():
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except ImportError as exc:  # pragma: no cover - import guard
        raise MissingDependencyError(
            "huggingface_hub package not installed. Install with `pip install huggingface_hub`."
        ) from exc
    return snapshot_download


def ensure_local_nomic_model(
    target_dir: Path,
    *,
    repo_id: str = DEFAULT_MODEL_REPO_ID,
    revision: Optional[str] = None,
) -> Path:
    target_dir = Path(target_dir)
    if target_dir.exists() and any(target_dir.iterdir()):
        return target_dir

    snapshot_download = _lazy_import_snapshot_download()
    logger.info("Downloading Nomic embedding model (%s) into %s", repo_id, target_dir)
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        allow_patterns=["*.json", "*.txt", "*.bin", "*.onnx", "*.pt", "*.model", "*.safetensors"],
    )
    if not any(target_dir.iterdir()):
        raise RuntimeError(f"Model download for {repo_id} produced no files in {target_dir}")
    return target_dir


def ensure_local_nomic_model_files(
    target_dir: Path,
    *,
    required_paths: Sequence[str],
    repo_id: str = DEFAULT_MODEL_REPO_ID,
    revision: Optional[str] = None,
) -> Path:
    """
    Ensure specific model assets exist under `target_dir`, downloading them if missing.

    This is useful for the ONNX embedding connector which only needs a small subset
    of the repo (e.g. `tokenizer.json` + `onnx/model.onnx`) and should not require
    bundling the full embedding repository inside the app.
    """
    target_dir = Path(target_dir)
    req = [str(p) for p in required_paths if isinstance(p, str) and p.strip()]
    missing: list[str] = []
    for rel in req:
        rel_path = Path(rel)
        if rel_path.is_absolute():
            raise ValueError(f"required_paths must be relative, got: {rel}")
        if not (target_dir / rel_path).exists():
            missing.append(rel)

    if not missing:
        return target_dir

    # Avoid concurrent downloads from multiple ingestion workers / threads.
    with _download_lock:
        # Re-check after acquiring the lock (another thread may have completed it).
        still_missing: list[str] = []
        for rel in missing:
            if not (target_dir / rel).exists():
                still_missing.append(rel)
        if not still_missing:
            return target_dir

        snapshot_download = _lazy_import_snapshot_download()
        logger.info(
            "Downloading Nomic embedding model assets (%s) into %s (missing=%s)",
            repo_id,
            target_dir,
            ", ".join(still_missing),
        )
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            allow_patterns=req,
        )

        final_missing = [rel for rel in still_missing if not (target_dir / rel).exists()]
        if final_missing:
            raise RuntimeError(
                f"Model download for {repo_id} did not produce required files in {target_dir}: {final_missing}"
            )

    return target_dir


@dataclass
class NomicEmbedTextConnector:
    model_dir: Path
    model_name: str = "nomic-embed-text-v1.5"
    device: Optional[str] = None
    normalize_embeddings: bool = True
    auto_download: bool = False
    repo_id: str = DEFAULT_MODEL_REPO_ID
    revision: Optional[str] = None

    def __post_init__(self) -> None:
        self.model_dir = Path(self.model_dir)
        if self.auto_download:
            ensure_local_nomic_model(self.model_dir, repo_id=self.repo_id, revision=self.revision)
        if not self.model_dir.exists() or not any(self.model_dir.iterdir()):
            raise FileNotFoundError(
                f"Nomic embedding model not found at {self.model_dir}. "
                "Bundle the model with your application (preferred) or enable `auto_download=True` "
                "and allow a one-time download using Hugging Face Hub utilities."
            )
        self._model = None
        self.is_local = True

    def supports(self, model: str) -> bool:
        aliases = {self.model_name, "nomic-embed-text", "nomic-embed-text-v1.5"}
        return model in aliases

    def embed(self, model: str, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        if not self.supports(model):
            raise ValueError(f"Connector does not support model {model!r}")
        if not texts:
            return []
        sentence_transformer = self._ensure_model()
        logger.debug("Encoding %d texts with Nomic embedding model", len(texts))
        embeddings = sentence_transformer.encode(
            list(texts),
            batch_size=len(texts),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )
        return embeddings.tolist()

    def _ensure_model(self):
        if self._model is None:
            SentenceTransformer = _lazy_import_sentence_transformer()
            logger.info("Loading Nomic embedding model from %s", self.model_dir)
            self._model = SentenceTransformer(
                str(self.model_dir),
                device=self.device,
                trust_remote_code=True,
            )
        return self._model


__all__ = [
    "NomicEmbedTextConnector",
    "MissingDependencyError",
    "ensure_local_nomic_model",
    "ensure_local_nomic_model_files",
]
