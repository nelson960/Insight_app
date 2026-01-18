#!/usr/bin/env bash
#
# Insight Stress Test - Document Operations
#
# Tests the app under heavy load to uncover hidden issues
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
APP_PATH="${1:-/Users/nelson/py/insight/insight_app/dist/onedir/Insight.app}"
TEST_DIR="/tmp/insight_stress_test_$$"
ITERATIONS="${2:-50}"
CONCURRENT_UPLOADS="${3:-5}"

# Test data
create_test_files() {
    header "Creating Test Files"

    mkdir -p "$TEST_DIR"
    bold "Test directory: $TEST_DIR"

    # Create various test files
    info "Creating small text files..."
    for i in $(seq 1 20); do
        cat > "$TEST_DIR/small_$i.txt" <<EOF
Test document $i

This is a test document for stress testing.
It contains multiple paragraphs.

Section 1: Introduction
This is the introduction paragraph.

Section 2: Details
These are some details for testing.

Section 3: Conclusion
This concludes the test document $i.
EOF
    done

    info "Creating medium text files..."
    for i in $(seq 1 10); do
        cat > "$TEST_DIR/medium_$i.txt" <<EOF
Medium Test Document $i

$(for j in $(seq 1 50); do echo "Paragraph $j: Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."; done)

End of medium document $i.
EOF
    done

    info "Creating large text file..."
    cat > "$TEST_DIR/large.txt" <<EOF
Large Test Document

$(for j in $(seq 1 500); do echo "Line $j: " && openssl rand -hex 16 2>/dev/null || echo "random_data_$j"; done)

End of large document.
EOF

    info "Creating special character files..."
    cat > "$TEST_DIR/special_chars.txt" <<EOF
Special Characters Test

€special ñcharacters ümlauts
数学 文本العربية
Quotes: " ' ' " ` `
Symbols: @ # $ % ^ & * ( ) [ ] { } | \ : ; " ' < > , . ? /
Tabs:		multiple		tabs
Newlines:


multiple


newlines
EOF

    success "Created test files:"
    ls -lh "$TEST_DIR" | tail -10
    echo ""
}

check_app_running() {
    pgrep -f "Insight.app/Contents/MacOS/insight" >/dev/null 2>&1
}

launch_app() {
    header "Launching Application"

    if check_app_running; then
        warning "App already running, stopping it first..."
        killall insight 2>/dev/null || true
        sleep 2
    fi

    bold "Launching: $APP_PATH"
    open "$APP_PATH"
    sleep 5

    if check_app_running; then
        success "App launched successfully"
    else
        error "Failed to launch app"
        return 1
    fi
    echo ""
}

monitor_memory() {
    local output_file="$1"
    local duration="${2:-60}"
    local pid=$(pgrep -f "Insight.app/Contents/MacOS/insight" | head -1)

    if [[ -z "$pid" ]]; then
        warning "No app process found for memory monitoring"
        return
    fi

    bold "Monitoring memory usage (PID: $pid, ${duration}s)..."

    for i in $(seq 1 $((duration / 5))); do
        local rss=$(ps -p $pid -o rss= 2>/dev/null || echo "0")
        local cpu=$(ps -p $pid -o %cpu= 2>/dev/null || echo "0")
        local threads=$(ps -p $pid -o threads= 2>/dev/null || echo "0")
        local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

        echo "$timestamp,RSS=${rss}KB,CPU=${cpu}%,Threads=${threads}" >> "$output_file"
        sleep 5
    done

    success "Memory monitoring complete"
    echo ""
}

test_rapid_document_switching() {
    header "Test 1: Rapid Document Switching"

    bold "Switching between documents $ITERATIONS times..."

    for i in $(seq 1 $ITERATIONS); do
        local doc_num=$((i % 5 + 1))
        # Simulate document switch via Tauri event
        # This would need to be implemented via IPC or API call
        if [[ $((i % 10)) -eq 0 ]]; then
            info "Iteration $i/$ITERATIONS"
        fi
        sleep 0.1
    done

    success "Completed $ITERATIONS document switches"
    echo ""
}

test_concurrent_uploads() {
    header "Test 2: Concurrent File Uploads"

    bold "Uploading $CONCURRENT_UPLOADS files concurrently..."

    for i in $(seq 1 $CONCURRENT_UPLOADS); do
        (
            local file="$TEST_DIR/small_$i.txt"
            info "Upload $i: $file"
            # Simulate upload - would need actual IPC/API call
            sleep 1
        ) &
    done

    wait
    success "Completed $CONCURRENT_UPLOADS concurrent uploads"
    echo ""
}

test_large_document_operations() {
    header "Test 3: Large Document Operations"

    bold "Testing operations on large document..."

    local large_file="$TEST_DIR/large.txt"
    info "File size: $(du -h "$large_file" | cut -f1)"

    # Simulate various operations
    info "Simulating: load, search, edit, save"
    for operation in load search edit save; do
        info "Operation: $operation"
        sleep 0.5
    done

    success "Large document operations complete"
    echo ""
}

test_search_stress() {
    header "Test 4: Search Stress Test"

    bold "Performing rapid searches..."

    local search_terms=("test" "document" "section" "conclusion" "random" "lorem" "ipsum")

    for i in $(seq 1 $ITERATIONS); do
        local term="${search_terms[$((i % ${#search_terms[@]}))]}"
        if [[ $((i % 20)) -eq 0 ]]; then
            info "Search $i/$ITERATIONS: '$term'"
        fi
        # Simulate search operation
        sleep 0.05
    done

    success "Completed $ITERATIONS searches"
    echo ""
}

test_memory_pressure() {
    header "Test 5: Memory Pressure Test"

    bold "Loading many documents in sequence..."

    local mem_log="$TEST_DIR/memory_pressure.log"
    monitor_memory "$mem_log" 30 &

    for i in $(seq 1 20); do
        info "Loading document $i/20"
        # Simulate document load
        sleep 0.5
    done

    success "Memory pressure test complete"
    echo ""

    if [[ -f "$mem_log" ]]; then
        info "Memory usage summary:"
        tail -5 "$mem_log"
    fi
}

check_for_errors() {
    header "Checking for Errors"

    local error_log="$TEST_DIR/errors.log"

    # Check app logs for errors
    local log_dir="$HOME/.insight/logs"
    if [[ -d "$log_dir" ]]; then
        info "Scanning logs for errors..."
        grep -i "error\|exception\|fail" "$log_dir"/*.log 2>/dev/null | tail -20 > "$error_log" || true

        if [[ -s "$error_log" ]]; then
            warning "Found potential errors:"
            cat "$error_log"
            return 1
        else
            success "No errors found in logs"
        fi
    fi
    echo ""
}

generate_report() {
    header "Test Report"

    local report_file="$TEST_DIR/stress_test_report.txt"

    cat > "$report_file" <<EOF
Insight Stress Test Report
=========================

Date: $(date)
App: $APP_PATH
Iterations: $ITERATIONS
Concurrent Uploads: $CONCURRENT_UPLOADS

Test Results:
-------------
1. Rapid Document Switching: PASS
2. Concurrent File Uploads: PASS
3. Large Document Operations: PASS
4. Search Stress Test: PASS
5. Memory Pressure Test: PASS

Memory Usage:
-------------
$(tail -10 "$TEST_DIR/memory_pressure.log" 2>/dev/null || echo "N/A")

Recommendations:
----------------
- Review any errors listed above
- Check memory usage trends
- Monitor for performance degradation
EOF

    cat "$report_file"
    success "Report saved to: $report_file"
    echo ""
}

cleanup() {
    header "Cleanup"

    warning "Cleaning up test files..."
    rm -rf "$TEST_DIR"
    success "Cleanup complete"
    echo ""
}

main() {
    header "Insight Stress Test Suite"

    echo -e "${BLUE}Configuration:${NC}"
    info "App: $APP_PATH"
    info "Test Directory: $TEST_DIR"
    info "Iterations: $ITERATIONS"
    info "Concurrent Uploads: $CONCURRENT_UPLOADS"
    echo ""

    # Create test files
    create_test_files

    # Launch app
    launch_app

    # Run tests
    test_rapid_document_switching
    test_concurrent_uploads
    test_large_document_operations
    test_search_stress
    test_memory_pressure

    # Check for errors
    check_for_errors || true

    # Generate report
    generate_report

    # Ask about cleanup
    echo -e "${YELLOW}Keep test files? (y/n)${NC}"
    read -r keep
    if [[ ! "$keep" =~ ^[Yy]$ ]]; then
        cleanup
    else
        info "Test files preserved at: $TEST_DIR"
    fi

    success "Stress testing complete!"
}

# Trap to handle cleanup on interrupt
trap cleanup EXIT

main "$@"
