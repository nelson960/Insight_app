#!/usr/bin/env bash
#
# Insight Comprehensive Test Runner
#
# Runs sidecar integration suites against a built Insight.app.
#

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
GRAY='\033[0;90m'
NC='\033[0m'

bold() { echo -e "${BLUE}➜${NC} $*"; }
info() { echo -e "  ${GRAY}$*${NC}"; }
success() { echo -e "${GREEN}✓${NC} $*"; }
warning() { echo -e "${YELLOW}⚠${NC} $*"; }
error() { echo -e "${RED}✗${NC} $*" >&2; }
header() { echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n${BLUE}$*${NC}\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

APP_PATH=""
RUN_ALL="false"
RUN_QUICK="false"
RUN_SMOKE="false"
RUN_STRESS="false"
RUN_CONCURRENT="false"
RUN_MEMORY="false"
RUN_EDGE="false"

# Tunables
STRESS_ITERATIONS="40"
CONCURRENT_WORKERS="8"
CONCURRENT_ITERATIONS="30"
MEMORY_DURATION="180"
MEMORY_INTERVAL="5"
KEEP_ARTIFACTS="false"
ARTIFACTS_BASE=""

TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

RESULTS_FILE="/tmp/insight_test_results_$$"
: > "$RESULTS_FILE"

show_usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Suite selection:
  -a, --all                     Run all suites: smoke + stress + concurrent + memory + edge
  -q, --quick                   Run smoke suite only
      --stress                  Run stress suite
      --concurrent              Run concurrent suite
      --memory                  Run memory suite
      --edge-case               Run edge suite

General options:
      --app PATH                Path to Insight.app (auto-detect if omitted)
      --keep-artifacts          Keep suite artifact directories
      --artifacts-dir DIR       Base directory for suite artifacts

Suite tuning:
      --stress-iterations N     Stress loop count (default: $STRESS_ITERATIONS)
      --workers N               Concurrent workers (default: $CONCURRENT_WORKERS)
      --concurrent-iterations N Concurrent loops per worker (default: $CONCURRENT_ITERATIONS)
      --memory-duration SEC     Memory test duration (default: $MEMORY_DURATION)
      --memory-interval SEC     Memory sample interval (default: $MEMORY_INTERVAL)

Examples:
  $(basename "$0") --quick
  $(basename "$0") --all
  $(basename "$0") --stress --concurrent --workers 12
  $(basename "$0") --app /path/to/Insight.app --memory --memory-duration 300
EOF
}

run_suite() {
    local suite_name="$1"
    shift

    header "Running: $suite_name"
    local start
    start=$(date +%s)

    if "$@"; then
        local duration=$(( $(date +%s) - start ))
        success "PASSED: $suite_name (${duration}s)"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        echo "$suite_name:PASSED (${duration}s)" >> "$RESULTS_FILE"
    else
        local duration=$(( $(date +%s) - start ))
        error "FAILED: $suite_name (${duration}s)"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        echo "$suite_name:FAILED (${duration}s)" >> "$RESULTS_FILE"
    fi

    TESTS_RUN=$((TESTS_RUN + 1))
    echo ""
}

suite_cmd() {
    local suite="$1"
    local cmd=("$PYTHON_BIN" "$SCRIPT_DIR/engine_test_runner.py" --suite "$suite")

    if [[ -n "$APP_PATH" ]]; then
        cmd+=(--app "$APP_PATH")
    fi
    if [[ "$KEEP_ARTIFACTS" == "true" ]]; then
        cmd+=(--keep-artifacts)
    fi
    if [[ -n "$ARTIFACTS_BASE" ]]; then
        cmd+=(--artifacts-dir "$ARTIFACTS_BASE/$suite")
    fi

    case "$suite" in
        stress)
            cmd+=(--iterations "$STRESS_ITERATIONS")
            ;;
        concurrent)
            cmd+=(--workers "$CONCURRENT_WORKERS" --iterations "$CONCURRENT_ITERATIONS")
            ;;
        memory)
            cmd+=(--duration "$MEMORY_DURATION" --interval "$MEMORY_INTERVAL")
            ;;
    esac

    "${cmd[@]}"
}

generate_report() {
    header "Master Test Report"

    local ts
    ts="$(date +%Y%m%d_%H%M%S)"
    local report_tmp="/tmp/insight_test_report_${ts}.txt"
    local report_project="$REPO_ROOT/TEST_REPORT_${ts}.txt"

    {
        echo "Insight Comprehensive Test Report"
        echo "================================="
        echo ""
        echo "Date: $(date)"
        echo "App: ${APP_PATH:-auto-detect}"
        echo "Repository: $REPO_ROOT"
        echo "Platform: $(uname -s) $(uname -m)"
        echo ""
        echo "Summary"
        echo "-------"
        echo "Suites run: $TESTS_RUN"
        echo "Passed: $TESTS_PASSED"
        echo "Failed: $TESTS_FAILED"
        echo ""
        echo "Detailed Results"
        echo "----------------"
        cat "$RESULTS_FILE"
        echo ""
        if [[ $TESTS_FAILED -eq 0 ]]; then
            echo "Outcome: PASS"
        else
            echo "Outcome: FAIL"
        fi
    } > "$report_tmp"

    cp "$report_tmp" "$report_project"
    cat "$report_tmp"
    success "Report saved to: $report_project"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -a|--all)
            RUN_ALL="true"
            shift
            ;;
        -q|--quick)
            RUN_QUICK="true"
            shift
            ;;
        --stress)
            RUN_STRESS="true"
            shift
            ;;
        --concurrent)
            RUN_CONCURRENT="true"
            shift
            ;;
        --memory)
            RUN_MEMORY="true"
            shift
            ;;
        --edge-case)
            RUN_EDGE="true"
            shift
            ;;
        --app)
            APP_PATH="$2"
            shift 2
            ;;
        --keep-artifacts)
            KEEP_ARTIFACTS="true"
            shift
            ;;
        --artifacts-dir)
            ARTIFACTS_BASE="$2"
            shift 2
            ;;
        --stress-iterations)
            STRESS_ITERATIONS="$2"
            shift 2
            ;;
        --workers)
            CONCURRENT_WORKERS="$2"
            shift 2
            ;;
        --concurrent-iterations)
            CONCURRENT_ITERATIONS="$2"
            shift 2
            ;;
        --memory-duration)
            MEMORY_DURATION="$2"
            shift 2
            ;;
        --memory-interval)
            MEMORY_INTERVAL="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

header "Insight Comprehensive Test Suite"

if [[ "$RUN_ALL" == "true" ]]; then
    RUN_SMOKE="true"
    RUN_STRESS="true"
    RUN_CONCURRENT="true"
    RUN_MEMORY="true"
    RUN_EDGE="true"
fi

if [[ "$RUN_QUICK" == "true" ]]; then
    # Quick means smoke-only.
    RUN_SMOKE="true"
    RUN_STRESS="false"
    RUN_CONCURRENT="false"
    RUN_MEMORY="false"
    RUN_EDGE="false"
fi

# If no explicit selection, default to smoke.
if [[ "$RUN_QUICK" != "true" && "$RUN_STRESS" != "true" && "$RUN_CONCURRENT" != "true" && "$RUN_MEMORY" != "true" && "$RUN_EDGE" != "true" ]]; then
    RUN_SMOKE="true"
fi

echo -e "${BLUE}Configuration:${NC}"
info "App: ${APP_PATH:-auto-detect}"
info "Smoke: $RUN_SMOKE"
info "Quick: $RUN_QUICK"
info "Stress: $RUN_STRESS"
info "Concurrent: $RUN_CONCURRENT"
info "Memory: $RUN_MEMORY"
info "Edge: $RUN_EDGE"
info "Artifacts: ${ARTIFACTS_BASE:-temporary}"
info "Python: $PYTHON_BIN"
echo ""

if [[ "$RUN_SMOKE" == "true" ]]; then
    run_suite "Smoke" suite_cmd smoke
fi
if [[ "$RUN_STRESS" == "true" ]]; then
    run_suite "Stress" suite_cmd stress
fi
if [[ "$RUN_CONCURRENT" == "true" ]]; then
    run_suite "Concurrent" suite_cmd concurrent
fi
if [[ "$RUN_MEMORY" == "true" ]]; then
    run_suite "Memory" suite_cmd memory
fi
if [[ "$RUN_EDGE" == "true" ]]; then
    run_suite "Edge" suite_cmd edge
fi

generate_report

header "Test Suite Complete"
if [[ $TESTS_FAILED -eq 0 ]]; then
    success "All selected suites passed"
    rm -f "$RESULTS_FILE"
    exit 0
fi

error "Some suites failed"
rm -f "$RESULTS_FILE"
exit 1
