from __future__ import annotations

import hashlib
import logging
import threading
import multiprocessing
import queue
import time
import os
import signal
import urllib.request
from urllib.parse import quote
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Dict

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

_EMBEDDING_FILE_HASHES: dict[str, dict[str, str]] = {
    DEFAULT_MODEL_REPO_ID: {
        "tokenizer.json": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
        "onnx/model.onnx": "147d5aa88c2101237358e17796cf3a227cead1ec304ec34b465bb08e9d952965",
    },
}

_download_lock = threading.Lock()
_download_state_lock = threading.Lock()
_download_cancel_lock = threading.Lock()
_download_state = {
    "status": "idle",
    "error": None,
    "last_attempt_ts": None,
    "last_success_ts": None,
}
_download_cancel_requested = False
_download_proc_lock = threading.Lock()
_download_proc: Optional[multiprocessing.Process] = None
_download_proc_queue: Optional[multiprocessing.Queue] = None
_download_pid_path: Optional[Path] = None
_download_thread_lock = threading.Lock()
_download_thread: Optional[threading.Thread] = None


def request_download_cancel() -> None:
    global _download_cancel_requested
    with _download_cancel_lock:
        _download_cancel_requested = True


def clear_download_cancel() -> None:
    global _download_cancel_requested
    with _download_cancel_lock:
        _download_cancel_requested = False


def _consume_download_cancel() -> bool:
    global _download_cancel_requested
    with _download_cancel_lock:
        if _download_cancel_requested:
            _download_cancel_requested = False
            return True
    return False


def _check_download_cancel(stage: str) -> bool:
    if _consume_download_cancel():
        _set_download_state("error", "Download cancelled.")
        logger.info("Embedding download cancelled (%s)", stage)
        return True
    return False


def _hf_resolve_url(repo_id: str, rel_path: str, revision: Optional[str]) -> str:
    rev = revision or "main"
    safe_path = "/".join(quote(part) for part in rel_path.split("/"))
    return f"https://huggingface.co/{repo_id}/resolve/{rev}/{safe_path}"


def _download_file_stream(url: str, dest: Path, *, cancel_stage: str) -> None:
    tmp_path = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url) as resp:
            with tmp_path.open("wb") as handle:
                while True:
                    if _check_download_cancel(cancel_stage):
                        raise RuntimeError("Download cancelled.")
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
        tmp_path.replace(dest)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def download_embedding_files_sync(
    target_dir: Path,
    *,
    required_paths: Sequence[str],
    repo_id: str = DEFAULT_MODEL_REPO_ID,
    revision: Optional[str] = None,
) -> None:
    """
    Synchronously download embedding assets with cancel support.
    """
    target_dir = Path(target_dir)
    req = [str(p) for p in required_paths if isinstance(p, str) and p.strip()]
    missing = [rel for rel in req if not (target_dir / rel).exists()]
    if not missing:
        _set_download_state("ready")
        return
    _set_download_state("downloading")
    for rel in missing:
        if _check_download_cancel("before_file"):
            raise RuntimeError("Download cancelled.")
        url = _hf_resolve_url(repo_id, rel, revision)
        dest = target_dir / rel
        _download_file_stream(url, dest, cancel_stage="during_file")
    _verify_embedding_files(target_dir, req, repo_id)
    _set_download_state("ready")


def start_embedding_download_thread(
    target_dir: Path,
    *,
    required_paths: Sequence[str],
    repo_id: str = DEFAULT_MODEL_REPO_ID,
    revision: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Start a foreground (UI-blocking) download in a background thread so cancel can be handled.
    """
    global _download_thread
    with _download_thread_lock:
        if _download_thread is not None and _download_thread.is_alive():
            return {"started": False, "status": "downloading"}

        clear_download_cancel()
        _set_download_state("downloading")

        def _worker() -> None:
            try:
                download_embedding_files_sync(
                    Path(target_dir),
                    required_paths=required_paths,
                    repo_id=repo_id,
                    revision=revision,
                )
            except Exception as exc:
                # download_embedding_files_sync already sets state for cancel/errors
                if get_download_state().get("status") != "error":
                    _set_download_state("error", str(exc))
            finally:
                with _download_thread_lock:
                    global _download_thread
                    _download_thread = None

        _download_thread = threading.Thread(
            target=_worker,
            daemon=True,
            name="insight-embed-download-thread",
        )
        _download_thread.start()
        return {"started": True, "status": "downloading"}


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "posix":
        try:
            os.kill(pid, 0)
            return True
        except Exception:
            return False
    return False


def _kill_pid(pid: int) -> None:
    if os.name != "posix":
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except Exception:
        return
    time.sleep(0.2)
    try:
        os.kill(pid, 0)
    except Exception:
        return
    try:
        os.kill(pid, signal.SIGKILL)
    except Exception:
        return


def _write_pidfile(path: Path, pid: int) -> None:
    try:
        path.write_text(str(pid), encoding="utf-8")
    except Exception:
        logger.warning("Failed to write embedding download pid file: %s", path)


def _clear_pidfile(path: Optional[Path]) -> None:
    if not path:
        return
    try:
        path.unlink(missing_ok=True)
    except Exception:
        logger.warning("Failed to remove embedding download pid file: %s", path)


def _download_worker_proc(
    model_dir: str,
    required_paths: Sequence[str],
    repo_id: str,
    revision: Optional[str],
    result_queue: "multiprocessing.Queue",
) -> None:
    try:
        download_embedding_files_sync(
            Path(model_dir),
            required_paths=required_paths,
            repo_id=repo_id,
            revision=revision,
        )
        result_queue.put({"ok": True})
    except Exception as exc:
        result_queue.put({"ok": False, "error": str(exc)})
        raise


def _watch_download_process(proc: multiprocessing.Process, result_queue: "multiprocessing.Queue") -> None:
    proc.join()
    with _download_proc_lock:
        global _download_proc, _download_proc_queue, _download_pid_path
        _download_proc = None
        _download_proc_queue = None
        _clear_pidfile(_download_pid_path)
        _download_pid_path = None
    # If a cancel was already recorded, do not overwrite the state.
    state = get_download_state()
    if state.get("status") == "error" and state.get("error") == "Download cancelled.":
        return
    result: Optional[dict] = None
    try:
        result = result_queue.get_nowait()
    except queue.Empty:
        result = None
    if proc.exitcode == 0 and result and result.get("ok") is True:
        _set_download_state("ready")
    else:
        msg = None
        if isinstance(result, dict):
            msg = result.get("error")
        if not msg:
            msg = f"Download process exited with code {proc.exitcode}"
        _set_download_state("error", msg)


def start_embedding_download_process(
    model_dir: Path,
    *,
    required_paths: Sequence[str],
    repo_id: str = DEFAULT_MODEL_REPO_ID,
    revision: Optional[str] = None,
) -> Dict[str, Any]:
    global _download_proc, _download_proc_queue, _download_pid_path
    with _download_proc_lock:
        pid_path = Path(model_dir) / ".embedding_download.pid"
        _download_pid_path = pid_path
        if pid_path.exists():
            pid = None
            try:
                pid = int(pid_path.read_text(encoding="utf-8").strip())
            except Exception:
                pid = None
            if pid and _pid_is_alive(pid):
                _set_download_state("downloading")
                return {"started": False, "status": "downloading"}
            _clear_pidfile(pid_path)
        if _download_proc is not None and _download_proc.is_alive():
            return {"started": False, "status": "downloading"}
        clear_download_cancel()
        ctx = multiprocessing.get_context("spawn")
        q: multiprocessing.Queue = ctx.Queue()
        proc = ctx.Process(
            target=_download_worker_proc,
            args=(str(model_dir), list(required_paths), repo_id, revision, q),
            daemon=True,
            name="insight-embed-download",
        )
        _download_proc = proc
        _download_proc_queue = q
        _set_download_state("downloading")
        proc.start()
        _write_pidfile(pid_path, proc.pid or -1)
        watcher = threading.Thread(
            target=_watch_download_process,
            args=(proc, q),
            daemon=True,
            name="insight-embed-download-watch",
        )
        watcher.start()
        return {"started": True, "status": "downloading"}


def cancel_embedding_download_process(model_dir: Optional[Path] = None) -> bool:
    global _download_proc, _download_proc_queue, _download_pid_path
    with _download_proc_lock:
        proc = _download_proc
        pid_path = _download_pid_path
        if pid_path is None and model_dir is not None:
            pid_path = Path(model_dir) / ".embedding_download.pid"
        killed = False
        if proc is not None and proc.is_alive():
            killed = True
            request_download_cancel()
            try:
                proc.terminate()
                proc.join(timeout=2.0)
                if proc.is_alive():
                    proc.kill()
            except Exception:
                pass
        else:
            if pid_path and pid_path.exists():
                pid = None
                try:
                    pid = int(pid_path.read_text(encoding="utf-8").strip())
                except Exception:
                    pid = None
                if pid and _pid_is_alive(pid):
                    killed = True
                    _kill_pid(pid)
        _clear_pidfile(pid_path)
        _download_pid_path = None
        request_download_cancel()
        _download_proc = None
        _download_proc_queue = None
        _set_download_state("error", "Download cancelled.")
        return killed


def _set_download_state(status: str, error: Optional[str] = None) -> None:
    now = time.time()
    with _download_state_lock:
        _download_state["status"] = status
        _download_state["error"] = error
        if status == "downloading":
            _download_state["last_attempt_ts"] = now
        elif status == "ready":
            _download_state["last_success_ts"] = now


def set_download_state(status: str, error: Optional[str] = None) -> None:
    _set_download_state(status, error)


def get_download_state() -> dict:
    with _download_state_lock:
        return dict(_download_state)


def _should_retry(min_interval_sec: float) -> bool:
    with _download_state_lock:
        last_attempt = _download_state.get("last_attempt_ts")
    if last_attempt is None:
        return True
    return (time.time() - float(last_attempt)) >= float(min_interval_sec)


def _lazy_import_snapshot_download():
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except ImportError as exc:  # pragma: no cover - import guard
        raise MissingDependencyError(
            "huggingface_hub package not installed. Install with `pip install huggingface_hub`."
        ) from exc
    return snapshot_download


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_embedding_files(target_dir: Path, required_paths: Sequence[str], repo_id: str) -> None:
    expected = _EMBEDDING_FILE_HASHES.get(repo_id)
    if not expected:
        return
    mismatches: list[str] = []
    for rel in required_paths:
        rel_path = str(rel)
        expected_hash = expected.get(rel_path)
        if not expected_hash:
            continue
        file_path = target_dir / rel_path
        if not file_path.exists():
            mismatches.append(rel_path)
            continue
        try:
            actual = _sha256_file(file_path)
        except Exception:
            mismatches.append(rel_path)
            continue
        if actual.lower() != expected_hash.lower():
            mismatches.append(rel_path)
    if mismatches:
        raise RuntimeError(
            "Embedding model integrity check failed for: "
            + ", ".join(mismatches)
            + ". Re-download the embedding model."
        )


def ensure_local_nomic_model(
    target_dir: Path,
    *,
    repo_id: str = DEFAULT_MODEL_REPO_ID,
    revision: Optional[str] = None,
) -> Path:
    target_dir = Path(target_dir)
    if target_dir.exists() and any(target_dir.iterdir()):
        _set_download_state("ready")
        return target_dir

    snapshot_download = _lazy_import_snapshot_download()
    logger.info("Downloading Nomic embedding model (%s) into %s", repo_id, target_dir)
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    _set_download_state("downloading")
    try:
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            allow_patterns=["*.json", "*.txt", "*.bin", "*.onnx", "*.pt", "*.model", "*.safetensors"],
        )
        if not any(target_dir.iterdir()):
            _set_download_state("error", "Model download produced no files.")
            raise RuntimeError(f"Model download for {repo_id} produced no files in {target_dir}")
        try:
            _verify_embedding_files(target_dir, ["tokenizer.json", "onnx/model.onnx"], repo_id)
        except Exception as exc:
            _set_download_state("error", str(exc))
            raise
        _set_download_state("ready")
        return target_dir
    except Exception as exc:
        _set_download_state("error", str(exc))
        raise


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
        _set_download_state("ready")
        return target_dir

    # Avoid concurrent downloads from multiple ingestion workers / threads.
    with _download_lock:
        # Re-check after acquiring the lock (another thread may have completed it).
        still_missing: list[str] = []
        for rel in missing:
            if not (target_dir / rel).exists():
                still_missing.append(rel)
        if not still_missing:
            _set_download_state("ready")
            return target_dir

        snapshot_download = _lazy_import_snapshot_download()
        logger.info(
            "Downloading Nomic embedding model assets (%s) into %s (missing=%s)",
            repo_id,
            target_dir,
            ", ".join(still_missing),
        )
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        _set_download_state("downloading")
        if _check_download_cancel("before_download"):
            return target_dir
        try:
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                local_dir=str(target_dir),
                local_dir_use_symlinks=False,
                allow_patterns=req,
            )
            if _check_download_cancel("after_download"):
                return target_dir
            final_missing = [rel for rel in still_missing if not (target_dir / rel).exists()]
            if final_missing:
                _set_download_state("error", f"Missing required files: {', '.join(final_missing)}")
                raise RuntimeError(
                    f"Model download for {repo_id} did not produce required files in {target_dir}: {final_missing}"
                )
            try:
                _verify_embedding_files(target_dir, req, repo_id)
            except Exception as exc:
                _set_download_state("error", str(exc))
                raise
        except Exception as exc:
            _set_download_state("error", str(exc))
            raise

    if _check_download_cancel("before_ready"):
        return target_dir
    _set_download_state("ready")
    return target_dir


def maybe_start_auto_download(
    target_dir: Path,
    *,
    required_paths: Sequence[str],
    repo_id: str = DEFAULT_MODEL_REPO_ID,
    revision: Optional[str] = None,
    min_interval_sec: float = 60.0,
) -> bool:
    """
    Start a background download if assets are missing and we're not already downloading.
    Returns True if a download was started.
    """
    target_dir = Path(target_dir)
    req = [str(p) for p in required_paths if isinstance(p, str) and p.strip()]
    missing = [rel for rel in req if not (target_dir / rel).exists()]
    if not missing:
        return False

    state = get_download_state()
    if state.get("status") == "downloading":
        return False
    if not _should_retry(min_interval_sec):
        return False

    def _worker() -> None:
        try:
            ensure_local_nomic_model_files(
                target_dir,
                required_paths=req,
                repo_id=repo_id,
                revision=revision,
            )
        except Exception:
            # ensure_local_nomic_model_files already sets state to error
            return

    _set_download_state("downloading")
    threading.Thread(target=_worker, name="insight-auto-embed-download", daemon=True).start()
    return True


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
    "set_download_state",
    "clear_download_cancel",
    "request_download_cancel",
    "download_embedding_files_sync",
    "start_embedding_download_thread",
    "ensure_local_nomic_model",
    "ensure_local_nomic_model_files",
    "maybe_start_auto_download",
]
