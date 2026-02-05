#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"PyYAML is required: {exc}")


_VAR_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Insight artifact contract files.")
    parser.add_argument("--app", default="configs/app.yaml", help="Path to app contract YAML")
    parser.add_argument("--models", default="configs/models.yaml", help="Path to model registry YAML")
    parser.add_argument(
        "--workspace",
        default=None,
        help="Workspace root. Defaults to INSIGHT_WORKSPACE_DIR or ~/.insight",
    )
    parser.add_argument(
        "--skip-hashes",
        action="store_true",
        help="Skip SHA256 verification (existence/schema checks still run)",
    )
    parser.add_argument(
        "--allow-missing-artifacts",
        action="store_true",
        help="Treat missing model files as warnings instead of hard errors",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        obj = yaml.safe_load(handle) or {}
    if not isinstance(obj, dict):
        raise ValueError(f"Expected object at top-level in {path}")
    return obj


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def resolve_vars(value: Any, variables: Dict[str, str]) -> Any:
    if isinstance(value, str):
        def _replace(match: re.Match[str]) -> str:
            key = match.group(1)
            return variables.get(key, match.group(0))

        return _VAR_PATTERN.sub(_replace, value)
    if isinstance(value, list):
        return [resolve_vars(item, variables) for item in value]
    if isinstance(value, dict):
        return {k: resolve_vars(v, variables) for k, v in value.items()}
    return value


def _find_by_id(items: List[Dict[str, Any]], item_id: str) -> Dict[str, Any] | None:
    for item in items:
        if str(item.get("id")) == item_id:
            return item
    return None


def _check_file(
    path_str: str,
    expected_sha: str | None,
    *,
    verify_hashes: bool,
    allow_missing_artifacts: bool,
    errors: List[str],
    warnings: List[str],
) -> None:
    path = Path(path_str).expanduser()
    if not path.exists() or not path.is_file():
        message = f"Missing file: {path}"
        if allow_missing_artifacts:
            warnings.append(message)
        else:
            errors.append(message)
        return
    if verify_hashes and expected_sha:
        actual = sha256_file(path).lower()
        if actual != str(expected_sha).lower():
            errors.append(f"SHA mismatch for {path}: expected {expected_sha}, got {actual}")


def validate_contract(
    app_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    *,
    variables: Dict[str, str],
    verify_hashes: bool,
    allow_missing_artifacts: bool,
) -> tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    app = resolve_vars(app_cfg, variables)
    models = resolve_vars(model_cfg, variables)

    embedding = app.get("embedding") if isinstance(app.get("embedding"), dict) else {}
    vector_index = app.get("vector_index") if isinstance(app.get("vector_index"), dict) else {}
    chunks_cfg = vector_index.get("chunks") if isinstance(vector_index.get("chunks"), dict) else {}

    emb_dim = embedding.get("vector_dim")
    qdrant_dim = chunks_cfg.get("vector_size")
    if emb_dim != qdrant_dim:
        errors.append(
            f"Vector dim mismatch: embedding.vector_dim={emb_dim} vs vector_index.chunks.vector_size={qdrant_dim}"
        )

    llm_models = models.get("llm_models") if isinstance(models.get("llm_models"), list) else []
    emb_models = models.get("embedding_models") if isinstance(models.get("embedding_models"), list) else []

    emb_id = str(embedding.get("model_id") or "")
    emb_model = _find_by_id([m for m in emb_models if isinstance(m, dict)], emb_id)
    if not emb_id:
        errors.append("embedding.model_id is required")
    elif emb_model is None:
        errors.append(f"embedding.model_id={emb_id!r} not found in configs/models.yaml")
    else:
        emb_model_dim = emb_model.get("vector_dim")
        if emb_model_dim != emb_dim:
            errors.append(f"Embedding model dim mismatch: model={emb_model_dim} app={emb_dim}")

    llm = app.get("llm") if isinstance(app.get("llm"), dict) else {}
    desktop = llm.get("desktop") if isinstance(llm.get("desktop"), dict) else {}
    raw_engine = llm.get("raw_engine") if isinstance(llm.get("raw_engine"), dict) else {}

    for source, section in [("llm.desktop.model_id", desktop), ("llm.raw_engine.model_id", raw_engine)]:
        model_id = str(section.get("model_id") or "")
        if not model_id:
            errors.append(f"{source} is required")
            continue
        if _find_by_id([m for m in llm_models if isinstance(m, dict)], model_id) is None:
            errors.append(f"{source}={model_id!r} not found in configs/models.yaml")

    for model in llm_models:
        if not isinstance(model, dict):
            continue
        if not bool(model.get("required")):
            continue
        model_path = str(model.get("path") or "")
        if not model_path:
            errors.append(f"Required LLM model missing path: {model.get('id')}")
            continue
        _check_file(
            model_path,
            str(model.get("sha256") or "") or None,
            verify_hashes=verify_hashes,
            allow_missing_artifacts=allow_missing_artifacts,
            errors=errors,
            warnings=warnings,
        )

    for model in emb_models:
        if not isinstance(model, dict):
            continue
        if not bool(model.get("required")):
            continue
        files = model.get("files") if isinstance(model.get("files"), list) else []
        if not files:
            errors.append(f"Required embedding model has no files: {model.get('id')}")
            continue
        for file_entry in files:
            if not isinstance(file_entry, dict):
                continue
            file_path = str(file_entry.get("path") or "")
            if not file_path:
                errors.append(f"Embedding file path missing in model: {model.get('id')}")
                continue
            _check_file(
                file_path,
                str(file_entry.get("sha256") or "") or None,
                verify_hashes=verify_hashes,
                allow_missing_artifacts=allow_missing_artifacts,
                errors=errors,
                warnings=warnings,
            )

    return errors, warnings


def contract_id(app_cfg: Dict[str, Any], model_cfg: Dict[str, Any], variables: Dict[str, str]) -> str:
    resolved = {
        "app": resolve_vars(app_cfg, variables),
        "models": resolve_vars(model_cfg, variables),
    }
    payload = json.dumps(resolved, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    workspace_final = (
        Path(args.workspace).expanduser().resolve()
        if args.workspace
        else Path(os.getenv("INSIGHT_WORKSPACE_DIR", "~/.insight")).expanduser().resolve()
    )

    variables = {
        "REPO_ROOT": str(repo_root),
        "WORKSPACE": str(workspace_final),
    }

    app_path = Path(args.app).expanduser()
    models_path = Path(args.models).expanduser()
    if not app_path.exists():
        print(f"ERROR: missing app config: {app_path}", file=sys.stderr)
        return 1
    if not models_path.exists():
        print(f"ERROR: missing model config: {models_path}", file=sys.stderr)
        return 1

    app_cfg = load_yaml(app_path)
    model_cfg = load_yaml(models_path)
    errors, warnings = validate_contract(
        app_cfg,
        model_cfg,
        variables=variables,
        verify_hashes=not args.skip_hashes,
        allow_missing_artifacts=bool(args.allow_missing_artifacts),
    )

    cid = contract_id(app_cfg, model_cfg, variables)
    print(f"Contract ID: {cid}")
    print(f"Repo root  : {repo_root}")
    print(f"Workspace  : {workspace_final}")

    if warnings:
        print("\nValidation warnings:")
        for warning in warnings:
            print(f"- {warning}")

    if errors:
        print("\nValidation failed:", file=sys.stderr)
        for err in errors:
            print(f"- {err}", file=sys.stderr)
        return 1

    print("\nValidation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
