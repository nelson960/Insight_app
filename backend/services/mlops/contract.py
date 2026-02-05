from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_VAR_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")
_SHA_CACHE: Dict[str, Tuple[int, int, str]] = {}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _resolve_vars(value: Any, variables: Dict[str, str]) -> Any:
    if isinstance(value, str):
        def _replace(match: re.Match[str]) -> str:
            return variables.get(match.group(1), match.group(0))

        return _VAR_PATTERN.sub(_replace, value)
    if isinstance(value, list):
        return [_resolve_vars(item, variables) for item in value]
    if isinstance(value, dict):
        return {k: _resolve_vars(v, variables) for k, v in value.items()}
    return value


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - runtime dependency check
        raise RuntimeError("PyYAML is required to load artifact contract files") from exc
    with path.open("r", encoding="utf-8") as handle:
        obj = yaml.safe_load(handle) or {}
    if not isinstance(obj, dict):
        raise ValueError(f"Expected object at top-level in {path}")
    return obj


def _to_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except Exception:
        return default


def _find_by_id(items: List[Dict[str, Any]], item_id: str) -> Optional[Dict[str, Any]]:
    for item in items:
        if str(item.get("id") or "") == item_id:
            return item
    return None


def _normalize_path(raw: str) -> str:
    if not raw:
        return ""
    try:
        return str(Path(raw).expanduser().resolve())
    except Exception:
        return str(Path(raw).expanduser())


def _find_llm_by_path(items: List[Dict[str, Any]], path: str) -> Optional[Dict[str, Any]]:
    target = _normalize_path(path)
    if not target:
        return None
    for item in items:
        p = _normalize_path(str(item.get("path") or ""))
        if p and p == target:
            return item
    return None


def _sha256_file(path: Path) -> str:
    key = str(path)
    st = path.stat()
    cached = _SHA_CACHE.get(key)
    if cached and cached[0] == int(st.st_size) and cached[1] == int(st.st_mtime_ns):
        return cached[2]

    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    _SHA_CACHE[key] = (int(st.st_size), int(st.st_mtime_ns), digest)
    return digest


def _file_check(path: str, expected_sha: str, *, verify_hashes: bool) -> Dict[str, Any]:
    resolved = _normalize_path(path)
    out: Dict[str, Any] = {
        "path": resolved or path,
        "exists": False,
        "expected_sha256": expected_sha or None,
        "actual_sha256": None,
        "sha256_match": None,
    }
    p = Path(resolved)
    if not p.exists() or not p.is_file():
        return out

    out["exists"] = True
    if verify_hashes and expected_sha:
        actual = _sha256_file(p).lower()
        out["actual_sha256"] = actual
        out["sha256_match"] = actual == str(expected_sha).lower()
    return out


def _build_repo_paths(repo_root: Path) -> Dict[str, Path]:
    return {
        "app": repo_root / "configs" / "app.yaml",
        "models": repo_root / "configs" / "models.yaml",
        "profiles_dir": repo_root / "configs" / "profiles",
    }


def build_effective_contract_snapshot(
    *,
    settings: Dict[str, Any],
    workspace: Path,
    verify_hashes: bool = False,
    allow_missing_artifacts: bool = False,
    profile: Optional[str] = None,
    repo_root: Optional[Path] = None,
) -> Dict[str, Any]:
    repo = (repo_root or Path(__file__).resolve().parents[3]).resolve()
    workspace = workspace.expanduser().resolve()
    paths = _build_repo_paths(repo)

    errors: List[str] = []
    warnings: List[str] = []

    if not paths["app"].exists():
        raise FileNotFoundError(f"Missing app contract file: {paths['app']}")
    if not paths["models"].exists():
        raise FileNotFoundError(f"Missing model registry file: {paths['models']}")

    base_cfg = _load_yaml(paths["app"])
    models_cfg = _load_yaml(paths["models"])

    profile_name = (
        (profile or "").strip()
        or os.getenv("INSIGHT_PROFILE", "").strip()
        or str(base_cfg.get("runtime", {}).get("default_profile") or "dev")
    )
    profile_path = paths["profiles_dir"] / f"{profile_name}.yaml"

    profile_cfg: Dict[str, Any] = {}
    if profile_path.exists():
        profile_cfg = _load_yaml(profile_path)
    else:
        warnings.append(f"Profile file not found for {profile_name!r}: {profile_path}")

    merged_cfg = _deep_merge(base_cfg, profile_cfg)
    variables = {"REPO_ROOT": str(repo), "WORKSPACE": str(workspace)}
    app_cfg = _resolve_vars(merged_cfg, variables)
    model_cfg = _resolve_vars(models_cfg, variables)

    embedding_cfg = app_cfg.get("embedding") if isinstance(app_cfg.get("embedding"), dict) else {}
    vector_index_cfg = app_cfg.get("vector_index") if isinstance(app_cfg.get("vector_index"), dict) else {}
    chunks_cfg = vector_index_cfg.get("chunks") if isinstance(vector_index_cfg.get("chunks"), dict) else {}
    llm_cfg = app_cfg.get("llm") if isinstance(app_cfg.get("llm"), dict) else {}
    desktop_cfg = llm_cfg.get("desktop") if isinstance(llm_cfg.get("desktop"), dict) else {}
    raw_cfg = llm_cfg.get("raw_engine") if isinstance(llm_cfg.get("raw_engine"), dict) else {}

    emb_dim = _to_int(embedding_cfg.get("vector_dim"))
    qdrant_dim = _to_int(chunks_cfg.get("vector_size"))
    if emb_dim != qdrant_dim:
        errors.append(
            f"Embedding vector dim mismatch: embedding.vector_dim={emb_dim} vs vector_index.chunks.vector_size={qdrant_dim}"
        )

    llm_path = str(settings.get("llm_model_path") or "").strip()
    llm_ctx_size = _to_int(settings.get("llm_ctx_size"), _to_int(desktop_cfg.get("ctx_size_default")))
    llm_gpu_layers = _to_int(settings.get("llm_gpu_layers"), _to_int(desktop_cfg.get("gpu_layers_default")))

    raw_model_path = str(settings.get("raw_engine_model_path") or "").strip() or llm_path
    raw_host = str(settings.get("raw_engine_host") or raw_cfg.get("host_default") or "127.0.0.1").strip()
    raw_port = _to_int(settings.get("raw_engine_port"), _to_int(raw_cfg.get("port_default"), 11435))
    raw_ctx = _to_int(settings.get("raw_engine_ctx"), _to_int(raw_cfg.get("ctx_size_default")))
    raw_max_tokens = _to_int(settings.get("raw_engine_max_tokens"), _to_int(raw_cfg.get("max_tokens_default"), 1024))
    raw_gpu_layers = _to_int(settings.get("raw_engine_gpu_layers"))
    raw_threads = _to_int(settings.get("raw_engine_threads"))

    llm_models = [m for m in (model_cfg.get("llm_models") or []) if isinstance(m, dict)]
    emb_models = [m for m in (model_cfg.get("embedding_models") or []) if isinstance(m, dict)]

    desktop_model_id = str(desktop_cfg.get("model_id") or "")
    raw_model_id = str(raw_cfg.get("model_id") or "")
    embedding_model_id = str(embedding_cfg.get("model_id") or "")

    desktop_model_entry = _find_llm_by_path(llm_models, llm_path) or _find_by_id(llm_models, desktop_model_id)
    raw_model_entry = _find_llm_by_path(llm_models, raw_model_path) or _find_by_id(llm_models, raw_model_id)
    embedding_model_entry = _find_by_id(emb_models, embedding_model_id)

    if llm_path and desktop_model_entry is None:
        warnings.append(f"Desktop model path is not registered in configs/models.yaml: {llm_path}")
    if raw_model_path and raw_model_entry is None:
        warnings.append(f"Raw engine model path is not registered in configs/models.yaml: {raw_model_path}")
    if embedding_model_entry is None:
        errors.append(f"Embedding model id not found in configs/models.yaml: {embedding_model_id}")

    def _artifact_issue(message: str) -> None:
        if allow_missing_artifacts:
            warnings.append(message)
        else:
            errors.append(message)

    desktop_model_check = _file_check(
        llm_path or str((desktop_model_entry or {}).get("path") or ""),
        str((desktop_model_entry or {}).get("sha256") or ""),
        verify_hashes=verify_hashes,
    )
    raw_model_check = _file_check(
        raw_model_path or str((raw_model_entry or {}).get("path") or ""),
        str((raw_model_entry or {}).get("sha256") or ""),
        verify_hashes=verify_hashes,
    )

    embed_file_checks: List[Dict[str, Any]] = []
    if embedding_model_entry and isinstance(embedding_model_entry.get("files"), list):
        for item in embedding_model_entry.get("files") or []:
            if not isinstance(item, dict):
                continue
            embed_file_checks.append(
                _file_check(
                    str(item.get("path") or ""),
                    str(item.get("sha256") or ""),
                    verify_hashes=verify_hashes,
                )
            )

    if not desktop_model_check.get("exists"):
        _artifact_issue(f"Desktop model file missing: {desktop_model_check.get('path')}")
    if raw_model_path and not raw_model_check.get("exists"):
        _artifact_issue(f"Raw engine model file missing: {raw_model_check.get('path')}")

    if verify_hashes:
        if desktop_model_check.get("sha256_match") is False:
            errors.append(f"Desktop model checksum mismatch: {desktop_model_check.get('path')}")
        if raw_model_path and raw_model_check.get("sha256_match") is False:
            errors.append(f"Raw engine model checksum mismatch: {raw_model_check.get('path')}")
        for check in embed_file_checks:
            if check.get("sha256_match") is False:
                errors.append(f"Embedding artifact checksum mismatch: {check.get('path')}")

    for check in embed_file_checks:
        if not check.get("exists"):
            _artifact_issue(f"Embedding artifact missing: {check.get('path')}")

    effective = {
        "profile": profile_name,
        "embedding": {
            "model_id": embedding_model_id,
            "vector_dim": emb_dim,
            "normalize_embeddings": bool(embedding_cfg.get("normalize_embeddings", True)),
            "max_length": _to_int(embedding_cfg.get("max_length")),
        },
        "vector_index": {
            "collection_name": str(chunks_cfg.get("collection_name") or "insight_chunks"),
            "distance": str(chunks_cfg.get("distance") or "Cosine"),
            "vector_size": qdrant_dim,
            "storage_path": str(chunks_cfg.get("storage_path") or ""),
        },
        "retrieval": app_cfg.get("retrieval") or {},
        "prompt_policy": app_cfg.get("prompt_policy") or {},
        "llm": {
            "desktop": {
                "model_id": str((desktop_model_entry or {}).get("id") or desktop_model_id),
                "model_path": desktop_model_check.get("path"),
                "ctx_size": llm_ctx_size,
                "gpu_layers": llm_gpu_layers,
            },
            "raw_engine": {
                "model_id": str((raw_model_entry or {}).get("id") or raw_model_id),
                "model_path": raw_model_check.get("path"),
                "host": raw_host,
                "port": raw_port,
                "ctx_size": raw_ctx,
                "max_tokens": raw_max_tokens,
                "gpu_layers": raw_gpu_layers,
                "threads": raw_threads,
            },
        },
    }

    contract_input = {
        "artifact_contract": app_cfg.get("artifact_contract") or {},
        "effective": effective,
        "models": {
            "desktop_model": {
                "path": desktop_model_check.get("path"),
                "expected_sha256": desktop_model_check.get("expected_sha256"),
            },
            "raw_model": {
                "path": raw_model_check.get("path"),
                "expected_sha256": raw_model_check.get("expected_sha256"),
            },
            "embedding_files": [
                {"path": c.get("path"), "expected_sha256": c.get("expected_sha256")}
                for c in embed_file_checks
            ],
        },
    }
    contract_id = hashlib.sha256(
        json.dumps(contract_input, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    snapshot = {
        "contract_id": contract_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "repo_root": str(repo),
            "workspace": str(workspace),
            "app_config": str(paths["app"]),
            "models_config": str(paths["models"]),
            "profile_config": str(profile_path),
        },
        "effective": effective,
        "artifacts": {
            "desktop_model": desktop_model_check,
            "raw_model": raw_model_check,
            "embedding_files": embed_file_checks,
            "verify_hashes": bool(verify_hashes),
            "allow_missing_artifacts": bool(allow_missing_artifacts),
        },
        "validation": {
            "ok": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        },
    }
    return snapshot


def write_effective_contract_snapshot(snapshot: Dict[str, Any], *, workspace: Path) -> Path:
    out_dir = workspace.expanduser().resolve() / "contracts"
    out_dir.mkdir(parents=True, exist_ok=True)
    latest_path = out_dir / "effective_contract_latest.json"
    with latest_path.open("w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, indent=2, ensure_ascii=False)
    return latest_path


__all__ = ["build_effective_contract_snapshot", "write_effective_contract_snapshot"]
