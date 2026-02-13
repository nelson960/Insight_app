#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

APP_PATH=""
DURATION="180"
INTERVAL="5"
KEEP_ARTIFACTS="false"
ARTIFACTS_DIR=""

show_usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --app PATH            Path to Insight.app (auto-detect if omitted)
  --duration SEC        Monitoring duration in seconds (default: 180)
  --interval SEC        Sample interval in seconds (default: 5)
  --artifacts-dir DIR   Keep artifacts in this directory
  --keep-artifacts      Keep temporary artifacts directory
  -h, --help            Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --app)
            APP_PATH="$2"
            shift 2
            ;;
        --duration)
            DURATION="$2"
            shift 2
            ;;
        --interval)
            INTERVAL="$2"
            shift 2
            ;;
        --artifacts-dir)
            ARTIFACTS_DIR="$2"
            shift 2
            ;;
        --keep-artifacts)
            KEEP_ARTIFACTS="true"
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            show_usage
            exit 1
            ;;
    esac
done

cmd=("$PYTHON_BIN" "$SCRIPT_DIR/engine_test_runner.py" --suite memory --duration "$DURATION" --interval "$INTERVAL")
if [[ -n "$APP_PATH" ]]; then
    cmd+=(--app "$APP_PATH")
fi
if [[ -n "$ARTIFACTS_DIR" ]]; then
    cmd+=(--artifacts-dir "$ARTIFACTS_DIR")
fi
if [[ "$KEEP_ARTIFACTS" == "true" ]]; then
    cmd+=(--keep-artifacts)
fi

"${cmd[@]}"
