#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

WORKSPACE="${HOME}/.insight"
ALLOW_MISSING_ARTIFACTS=false
VERIFY_HASHES=false
PROFILE=""
DATASET="eval/datasets/smoke.jsonl"

usage() {
  cat <<'EOF'
Run core MLOps checks for Insight.

Usage:
  scripts/mlops/run_core.sh [options]

Options:
  --workspace <path>             Workspace root (default: ~/.insight)
  --allow-missing-artifacts      Treat missing model files as warnings
  --verify-hashes                Enable strict SHA256 verification
  --profile <name>               Optional profile override for export (dev/prod)
  --dataset <path>               Eval dataset path (default: eval/datasets/smoke.jsonl)
  -h, --help                     Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace)
      WORKSPACE="${2:-}"
      shift 2
      ;;
    --allow-missing-artifacts)
      ALLOW_MISSING_ARTIFACTS=true
      shift
      ;;
    --verify-hashes)
      VERIFY_HASHES=true
      shift
      ;;
    --profile)
      PROFILE="${2:-}"
      shift 2
      ;;
    --dataset)
      DATASET="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

VALIDATE_ARGS=(--workspace "$WORKSPACE")
EXPORT_ARGS=(--workspace "$WORKSPACE")
EVAL_ARGS=(
  --dataset "$DATASET"
  --min-recall-at-k 0.66
  --min-grounding-pass-rate 0.66
)

if [[ "$ALLOW_MISSING_ARTIFACTS" == true ]]; then
  VALIDATE_ARGS+=(--allow-missing-artifacts)
  EXPORT_ARGS+=(--allow-missing-artifacts)
fi

if [[ "$VERIFY_HASHES" == true ]]; then
  EXPORT_ARGS+=(--verify-hashes)
else
  VALIDATE_ARGS+=(--skip-hashes)
fi

if [[ -n "$PROFILE" ]]; then
  EXPORT_ARGS+=(--profile "$PROFILE")
fi

echo "[1/3] Validating contract..."
python3 scripts/mlops/validate_contract.py "${VALIDATE_ARGS[@]}"

echo "[2/3] Exporting effective contract..."
python3 scripts/mlops/export_effective_contract.py "${EXPORT_ARGS[@]}"

echo "[3/3] Running eval harness..."
python3 eval/run.py "${EVAL_ARGS[@]}"

echo "Core MLOps checks completed."
