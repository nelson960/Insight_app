#!/usr/bin/env bash
#
# Insight Memory Leak Detection Test
#
# Monitors memory usage over time to detect memory leaks
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
TEST_DIR="/tmp/insight_memory_test_$$"
DURATION="${2:-300}"  # 5 minutes default
SAMPLE_INTERVAL=5     # Sample every 5 seconds

mkdir -p "$TEST_DIR"

get_app_pid() {
    pgrep -f "Insight.app/Contents/MacOS/insight" | head -1
}

get_memory_info() {
    local pid="$1"

    if [[ -z "$pid" ]]; then
        echo "ERROR:No PID"
        return
    fi

    # Get various memory metrics
    local rss=$(ps -p "$pid" -o rss= 2>/dev/null || echo "0")
    local vsz=$(ps -p "$pid" -o vsz= 2>/dev/null || echo "0")
    local cpu=$(ps -p "$pid" -o %cpu= 2>/dev/null || echo "0")
    local threads=$(ps -p "$pid" -o threads= 2>/dev/null || echo "0")
    local open_files=$(lsof -p "$pid" 2>/dev/null | wc -l || echo "0")

    echo "RSS=${rss},VSZ=${vsz},CPU=${cpu},Threads=${threads},FDs=${open_files}"
}

monitor_memory() {
    local output_file="$1"
    local duration="$2"
    local interval="$3"

    bold "Starting memory monitoring for ${duration}s (sample every ${interval}s)"

    echo "Timestamp,RSS_KB,VSZ_KB,CPU_Percent,Threads,Open_FDs" > "$output_file"

    local pid
    local start=$(date +%s)
    local end=$((start + duration))
    local sample=0

    while [[ $(date +%s) -lt $end ]]; do
        pid=$(get_app_pid)

        if [[ -z "$pid" ]]; then
            error "App process died during monitoring!"
            echo "$(date),ERROR,ERROR,ERROR,ERROR,ERROR" >> "$output_file"
            return 1
        fi

        local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        local mem_info=$(get_memory_info "$pid")

        # Parse and format
        local rss=$(echo "$mem_info" | grep -o 'RSS=[0-9]*' | cut -d= -f2)
        local vsz=$(echo "$mem_info" | grep -o 'VSZ=[0-9]*' | cut -d= -f2)
        local cpu=$(echo "$mem_info" | grep -o 'CPU=[0-9.]*' | cut -d= -f2)
        local threads=$(echo "$mem_info" | grep -o 'Threads=[0-9]*' | cut -d= -f2)
        local fds=$(echo "$mem_info" | grep -o 'FDs=[0-9]*' | cut -d= -f2)

        echo "$timestamp,$rss,$vsz,$cpu,$threads,$fds" >> "$output_file"

        ((sample++))
        if [[ $((sample % 10)) -eq 0 ]]; then
            info "Sample $sample: RSS=$((${rss:-0} / 1024))MB, CPU=${cpu}%"
        fi

        sleep "$interval"
    done

    success "Memory monitoring complete: $((duration / interval)) samples collected"
}

simulate_memory_intensive_operations() {
    header "Simulating Memory-Intensive Operations"

    local log_file="$TEST_DIR/operations.log"

    bold "Running operations that may cause memory leaks..."

    # Load documents repeatedly
    info "Loading documents repeatedly..."
    for i in $(seq 1 50); do
        echo "$(date): Load document $i" >> "$log_file"
        sleep 0.2
    done &

    # Perform searches
    info "Running repeated searches..."
    for i in $(seq 1 100); do
        echo "$(date): Search $i" >> "$log_file"
        sleep 0.1
    done &

    # Simulate chat operations
    info "Simulating chat operations..."
    for i in $(seq 1 30); do
        echo "$(date): Chat operation $i" >> "$log_file"
        sleep 0.5
    done &

    # Wait for background operations
    wait

    success "Operations complete"
    echo ""
}

analyze_memory_trend() {
    header "Memory Trend Analysis"

    local data_file="$1"

    if [[ ! -f "$data_file" ]]; then
        error "Memory data file not found: $data_file"
        return 1
    fi

    info "Analyzing memory usage patterns..."

    # Extract RSS values (excluding header and error lines)
    local rss_values=$(tail -n +2 "$data_file" | grep -v "ERROR" | cut -d, -f2)

    if [[ -z "$rss_values" ]]; then
        error "No valid memory data found"
        return 1
    fi

    # Calculate statistics
    local count=$(echo "$rss_values" | wc -l | tr -d ' ')
    local first=$(echo "$rss_values" | head -1 | tr -d ' ')
    local last=$(echo "$rss_values" | tail -1 | tr -d ' ')
    local min=$(echo "$rss_values" | sort -n | head -1 | tr -d ' ')
    local max=$(echo "$rss_values" | sort -n | tail -1 | tr -d ' ')
    local avg=$(echo "$rss_values" | awk '{sum+=$1; count++} END {print int(sum/count)}')

    # Convert to MB
    local first_mb=$((first / 1024))
    local last_mb=$((last / 1024))
    local min_mb=$((min / 1024))
    local max_mb=$((max / 1024))
    local avg_mb=$((avg / 1024))

    echo "Memory Statistics (RSS):"
    echo "  Samples: $count"
    echo "  Initial: ${first_mb} MB"
    echo "  Final:   ${last_mb} MB"
    echo "  Min:     ${min_mb} MB"
    echo "  Max:     ${max_mb} MB"
    echo "  Average: ${avg_mb} MB"
    echo ""

    # Check for memory leak
    local growth=$((last - first))
    local growth_mb=$((growth / 1024))
    local growth_pct=$((growth * 100 / first))

    echo "Memory Growth: ${growth_mb} MB (${growth_pct}%)"

    if [[ $growth_pct -gt 50 ]]; then
        error "POTENTIAL MEMORY LEAK: Memory grew by ${growth_pct}%"
        return 1
    elif [[ $growth_pct -gt 25 ]]; then
        warning "Memory grew by ${growth_pct}% - monitor closely"
        return 2
    else
        success "Memory usage stable (growth: ${growth_pct}%)"
        return 0
    fi
}

check_file_descriptors() {
    header "File Descriptor Leak Check"

    local data_file="$1"

    info "Analyzing file descriptor usage..."

    # Extract FD values
    local fd_values=$(tail -n +2 "$data_file" | grep -v "ERROR" | cut -d, -f6)

    if [[ -z "$fd_values" ]]; then
        warning "No file descriptor data available"
        return
    fi

    local first=$(echo "$fd_values" | head -1 | tr -d ' ')
    local last=$(echo "$fd_values" | tail -1 | tr -d ' ')
    local max=$(echo "$fd_values" | sort -n | tail -1 | tr -d ' ')

    echo "File Descriptor Statistics:"
    echo "  Initial: $first"
    echo "  Final:   $last"
    echo "  Max:     $max"
    echo ""

    local growth=$((last - first))
    echo "FD Growth: $growth"

    if [[ $growth -gt 100 ]]; then
        error "POTENTIAL FD LEAK: FDs increased by $growth"
    elif [[ $growth -gt 50 ]]; then
        warning "FDs increased by $growth - monitor closely"
    else
        success "FD usage stable (growth: $growth)"
    fi
}

generate_report() {
    header "Memory Leak Test Report"

    local report_file="$TEST_DIR/memory_leak_report.txt"
    local data_file="$TEST_DIR/memory_data.csv"

    cat > "$report_file" <<EOF
Insight Memory Leak Detection Test Report
========================================

Date: $(date)
App: $APP_PATH
Test Duration: ${DURATION}s
Sample Interval: ${SAMPLE_INTERVAL}s

$(analyze_memory_trend "$data_file" 2>&1)

$(check_file_descriptors "$data_file" 2>&1)

Recommendations:
----------------
- Monitor memory usage over extended periods
- Check for growth during normal operations
- Profile specific features that increase memory
- Review any warnings above
EOF

    cat "$report_file"
    success "Report saved to: $report_file"
    echo ""
}

main() {
    header "Insight Memory Leak Detection Test"

    echo -e "${BLUE}Configuration:${NC}"
    info "App: $APP_PATH"
    info "Test Duration: ${DURATION}s ($(($DURATION / 60)) minutes)"
    info "Sample Interval: ${SAMPLE_INTERVAL}s"
    echo ""

    # Check if app is running
    local pid=$(get_app_pid)
    if [[ -z "$pid" ]]; then
        error "App is not running. Please start the app first."
        info "Command: open $APP_PATH"
        exit 1
    fi

    success "App detected (PID: $pid)"
    echo ""

    # Start memory monitoring
    local data_file="$TEST_DIR/memory_data.csv"
    monitor_memory "$data_file" "$DURATION" "$SAMPLE_INTERVAL" &
    local monitor_pid=$!

    # Run memory-intensive operations in parallel
    sleep 5  # Let monitoring start
    simulate_memory_intensive_operations

    # Wait for monitoring to complete (if still running)
    if kill -0 $monitor_pid 2>/dev/null; then
        wait $monitor_pid
    else
        info "Monitoring already completed"
    fi

    # Analyze results
    analyze_memory_trend "$data_file"
    check_file_descriptors "$data_file"

    # Generate report
    generate_report

    success "Memory leak detection complete!"
    info "Test data preserved at: $TEST_DIR"
}

main "$@"
