#!/usr/bin/env bash
#
# Insight Comprehensive Test Runner
#
# Runs all test suites and generates a master report
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

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_PATH="${1:-/Users/nelson/py/insight/insight_app/dist/onedir/Insight.app}"
RUN_ALL="${2:-false}"

# Test results tracking (bash 3.2 compatible - use temp file)
TEST_RESULTS_FILE="/tmp/insight_test_results_$$"
echo "" > "$TEST_RESULTS_FILE"

TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0

run_test_suite() {
    local suite_name="$1"
    local script_path="$2"
    local should_run="${3:-false}"

    header "Running: $suite_name"

    if [[ "$should_run" != "true" && "$RUN_ALL" != "true" ]]; then
        warning "SKIPPED: $suite_name (use --all or -a to run)"
        ((TESTS_SKIPPED++))
        echo "$suite_name:SKIPPED" >> "$TEST_RESULTS_FILE"
        return 0
    fi

    if [[ ! -x "$script_path" ]]; then
        chmod +x "$script_path"
    fi

    local start=$(date +%s)

    if "$script_path" "$APP_PATH"; then
        local duration=$(($(date +%s) - start))
        success "PASSED: $suite_name (${duration}s)"
        ((TESTS_PASSED++))
        echo "$suite_name:PASSED (${duration}s)" >> "$TEST_RESULTS_FILE"
    else
        local duration=$(($(date +%s) - start))
        error "FAILED: $suite_name (${duration}s)"
        ((TESTS_FAILED++))
        echo "$suite_name:FAILED (${duration}s)" >> "$TEST_RESULTS_FILE"
    fi

    ((TESTS_RUN++))
    echo ""
}

check_prerequisites() {
    header "Checking Prerequisites"

    local missing=0

    # Check if app exists
    if [[ ! -d "$APP_PATH" ]]; then
        error "App not found at: $APP_PATH"
        ((missing++))
    else
        success "App found: $APP_PATH"
    fi

    # Check if app is running
    if pgrep -f "Insight.app/Contents/MacOS/insight" >/dev/null 2>&1; then
        success "App is running"
    else
        warning "App is not running"
        info "Start the app with: open $APP_PATH"
    fi

    # Check test scripts
    local scripts=(
        "stress_test.sh"
        "concurrent_test.sh"
        "memory_leak_test.sh"
        "edge_case_test.sh"
    )

    for script in "${scripts[@]}"; do
        local path="$SCRIPT_DIR/$script"
        if [[ -f "$path" ]]; then
            success "Test script found: $script"
        else
            error "Test script missing: $script"
            ((missing++))
        fi
    done

    if [[ $missing -gt 0 ]]; then
        error "Missing $missing prerequisite(s)"
        exit 1
    fi

    echo ""
}

quick_smoke_test() {
    header "Quick Smoke Test"

    bold "Checking basic app functionality..."

    local tests_failed=0

    # Test 1: App process
    if pgrep -f "Insight.app/Contents/MacOS/insight" >/dev/null 2>&1; then
        success "App process is running"
    else
        error "App process not found"
        ((tests_failed++))
    fi

    # Test 2: Workspace directory
    if [[ -d "$HOME/.insight" ]]; then
        success "Workspace directory exists"
    else
        warning "Workspace directory not found (will be created on first run)"
    fi

    # Test 3: Log directory
    if [[ -d "$HOME/.insight/logs" ]]; then
        success "Log directory exists"

        # Check for recent errors
        local error_count=$(grep -hi "error\|exception" "$HOME/.insight/logs"/*.log 2>/dev/null | wc -l | tr -d ' ')
        if [[ $error_count -gt 0 ]]; then
            warning "Found $error_count error(s) in recent logs"
        else
            success "No errors in recent logs"
        fi
    else
        warning "Log directory not found"
    fi

    # Test 4: Memory usage
    local pid=$(pgrep -f "Insight.app/Contents/MacOS/insight" | head -1)
    if [[ -n "$pid" ]]; then
        local rss=$(ps -p "$pid" -o rss= 2>/dev/null || echo "0")
        local mb=$((rss / 1024))
        info "Memory usage: ${mb} MB"

        if [[ $mb -gt 2000 ]]; then
            warning "High memory usage: ${mb} MB"
        else
            success "Memory usage normal: ${mb} MB"
        fi
    fi

    if [[ $tests_failed -eq 0 ]]; then
        success "Smoke test passed"
        return 0
    else
        error "Smoke test failed: $tests_failed check(s)"
        return 1
    fi
}

generate_master_report() {
    header "Master Test Report"

    local report_file="/tmp/insight_test_report_$(date +%Y%m%d_%H%M%S).txt"

    cat > "$report_file" <<EOF
Insight Comprehensive Test Report
==================================

Date: $(date)
App: $APP_PATH
Hostname: $(hostname)
Platform: $(uname -s) $(uname -m)

EXECUTIVE SUMMARY
-----------------
Total Test Suites: $TESTS_RUN
Passed: $TESTS_PASSED
Failed: $TESTS_FAILED
Skipped: $TESTS_SKIPPED

DETAILED RESULTS
----------------
EOF

    if [[ -f "$TEST_RESULTS_FILE" ]]; then
        while IFS= read -r line; do
            if [[ -n "$line" ]]; then
                local suite_name=$(echo "$line" | cut -d: -f1)
                local result=$(echo "$line" | cut -d: -f2-)
                printf "%-30s %s\n" "$suite_name:" "$result" >> "$report_file"
            fi
        done < "$TEST_RESULTS_FILE"
    fi

    cat >> "$report_file" <<EOF

RECOMMENDATIONS
---------------
EOF

    if [[ $TESTS_FAILED -eq 0 ]]; then
        cat >> "$report_file" <<EOF
✓ All automated tests passed
✓ App is ready for distribution
✓ No critical issues detected

Optional:
- Run manual UI testing
- Test on fresh macOS installation
- Verify offline functionality
EOF
    else
        cat >> "$report_file" <<EOF
✗ Some tests failed - review detailed logs
✗ Fix critical issues before distribution
✗ Re-run tests after fixes

Next Steps:
1. Review failed test logs
2. Fix identified issues
3. Re-run this test suite
4. Perform manual testing
EOF
    fi

    cat >> "$report_file" <<EOF

TEST ARTIFACTS
--------------
Test logs and reports are preserved in:
- /tmp/insight_*_test_*/

For detailed analysis of each test suite, refer to individual test reports.

---
Generated by Insight Test Runner
Version: 1.0.0
EOF

    cat "$report_file"

    # Save report to project root too
    local project_report="/Users/nelson/py/insight/insight_app/TEST_REPORT_$(date +%Y%m%d_%H%M%S).txt"
    cp "$report_file" "$project_report"
    success "Report saved to: $project_report"

    # Cleanup temp file
    rm -f "$TEST_RESULTS_FILE"

    echo ""
}

show_usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] [APP_PATH]

Options:
  -a, --all              Run all test suites (including long-running tests)
  -q, --quick            Run quick smoke test only
  -h, --help             Show this help message

Arguments:
  APP_PATH               Path to Insight.app (default: auto-detect)

Examples:
  # Run quick smoke test
  $(basename "$0") --quick

  # Run all tests
  $(basename "$0") --all

  # Run specific tests
  $(basename "$0") /path/to/Insight.app

Test Suites:
  --stress               Run stress tests (document operations)
  --concurrent           Run concurrent operations tests
  --memory               Run memory leak detection
  --edge-case            Run edge case tests

Notes:
  - Stress and memory tests can take 5-10 minutes each
  - Quick smoke test completes in <30 seconds
  - Test results are preserved in /tmp/insight_*_test_*/

EOF
}

main() {
    header "Insight Comprehensive Test Suite"

    # Parse arguments
    local run_quick=false
    local run_stress=false
    local run_concurrent=false
    local run_memory=false
    local run_edge=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            -a|--all)
                RUN_ALL=true
                shift
                ;;
            -q|--quick)
                run_quick=true
                shift
                ;;
            --stress)
                run_stress=true
                shift
                ;;
            --concurrent)
                run_concurrent=true
                shift
                ;;
            --memory)
                run_memory=true
                shift
                ;;
            --edge-case)
                run_edge=true
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            -*)
                error "Unknown option: $1"
                show_usage
                exit 1
                ;;
            *)
                APP_PATH="$1"
                shift
                ;;
        esac
    done

    # Auto-detect app if not specified
    if [[ ! -d "$APP_PATH" ]]; then
        local detected=$(find /Users/nelson/py/insight/insight_app/dist -name "Insight.app" -type d 2>/dev/null | head -1)
        if [[ -n "$detected" ]]; then
            APP_PATH="$detected"
            info "Auto-detected app: $APP_PATH"
        fi
    fi

    echo -e "${BLUE}Configuration:${NC}"
    info "App: $APP_PATH"
    info "Run All: $RUN_ALL"
    echo ""

    # Check prerequisites
    check_prerequisites

    # Run quick smoke test
    if [[ "$run_quick" == "true" ]]; then
        quick_smoke_test
        exit $?
    fi

    # Always run smoke test first
    quick_smoke_test || warning "Smoke test had issues, continuing..."

    # Run specific or all test suites
    if [[ "$RUN_ALL" == "true" ]] || [[ "$run_stress" == "true" ]]; then
        run_test_suite "Stress Test" "$SCRIPT_DIR/stress_test.sh" true
    fi

    if [[ "$RUN_ALL" == "true" ]] || [[ "$run_concurrent" == "true" ]]; then
        run_test_suite "Concurrent Operations" "$SCRIPT_DIR/concurrent_test.sh" true
    fi

    if [[ "$RUN_ALL" == "true" ]] || [[ "$run_memory" == "true" ]]; then
        run_test_suite "Memory Leak Detection" "$SCRIPT_DIR/memory_leak_test.sh" true
    fi

    if [[ "$RUN_ALL" == "true" ]] || [[ "$run_edge" == "true" ]]; then
        run_test_suite "Edge Case Tests" "$SCRIPT_DIR/edge_case_test.sh" true
    fi

    # Generate master report
    generate_master_report

    # Final summary
    header "Test Suite Complete"

    if [[ $TESTS_RUN -eq 0 ]]; then
        warning "No tests were run"
        info "Use --all to run all test suites, or --help for options"
    else
        echo "Test Suites Run: $TESTS_RUN"
        success "Passed: $TESTS_PASSED"
        if [[ $TESTS_FAILED -gt 0 ]]; then
            error "Failed: $TESTS_FAILED"
        fi
        if [[ $TESTS_SKIPPED -gt 0 ]]; then
            info "Skipped: $TESTS_SKIPPED"
        fi
        echo ""

        if [[ $TESTS_FAILED -eq 0 ]]; then
            success "All tests passed! ✓"
            exit 0
        else
            error "Some tests failed ✗"
            exit 1
        fi
    fi
}

main "$@"
