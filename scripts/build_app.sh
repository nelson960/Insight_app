#!/bin/bash
#
# Insight Production Build Script
#
# This script orchestrates the complete build process for the Insight Tauri app:
# 1. Builds the Python backend with PyInstaller
# 2. Runs smoke tests on the backend
# 3. Detects system architecture
# 4. Copies the backend binary to Tauri resources
# 5. Builds the Tauri app
#
# Usage:
#   ./scripts/build_app.sh                    # Build everything
#   ./scripts/build_app.sh --skip-pyinstaller # Skip PyInstaller build
#   ./scripts/build_app.sh --skip-smoketest   # Skip smoke tests
#   ./scripts/build_app.sh --skip-tauri       # Skip Tauri build
#   ./scripts/build_app.sh --arch=arm64       # Force ARM64 architecture
#   ./scripts/build_app.sh --arch=x86_64      # Force x86_64 architecture
#
# Exit codes:
#   0 - Success
#   1 - General error
#   2 - PyInstaller build failed
#   3 - Smoke test failed
#   4 - Architecture detection failed
#   5 - Binary copy failed
#   6 - Tauri build failed
#   7 - Prerequisites not met

set -euo pipefail  # Exit on error, undefined vars, pipe failures

#-------------------------------------------------------------------------------
# Colors and formatting
#-------------------------------------------------------------------------------

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
GRAY='\033[0;90m'
NC='\033[0m' # No Color

bold() { echo -e "${BLUE}➜${NC} $*"; }
info() { echo -e "  ${GRAY}$*${NC}"; }
success() { echo -e "${GREEN}✓${NC} $*"; }
warning() { echo -e "${YELLOW}⚠${NC} $*"; }
error() { echo -e "${RED}✗${NC} $*" >&2; }
header() { echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n${BLUE}$*${NC}\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"; }

#-------------------------------------------------------------------------------
# Configuration
#-------------------------------------------------------------------------------

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="${REPO_ROOT}/backend"
INSIGHT_DIR="${REPO_ROOT}/insight"
TAURI_BIN_DIR="${INSIGHT_DIR}/src-tauri/bin"
BACKEND_DIST="${BACKEND_DIR}/dist/insight-engine"

# Flags
SKIP_PYINSTALLER=false
SKIP_SMOKETEST=false
SKIP_TAURI=false
FORCE_ARCH=""

#-------------------------------------------------------------------------------
# Helper functions
#-------------------------------------------------------------------------------

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --skip-pyinstaller    Skip PyInstaller build (use existing dist/insight-engine)
  --skip-smoketest      Skip smoke tests
  --skip-tauri          Skip Tauri app build
  --arch=arm64|x86_64   Force architecture (default: auto-detect)
  -h, --help            Show this help message

Examples:
  $(basename "$0")                          # Build everything
  $(basename "$0") --skip-pyinstaller       # Use existing backend binary
  $(basename "$0") --skip-tauri             # Only build backend
  $(basename "$0") --arch=arm64             # Force ARM64 build

EOF
    exit 0
}

detect_architecture() {
    if [[ -n "$FORCE_ARCH" ]]; then
        echo "$FORCE_ARCH"
        return 0
    fi

    local arch
    arch=$(uname -m)

    case "$arch" in
        arm64|aarch64)
            echo "arm64"
            ;;
        x86_64|amd64|i386|i686)
            echo "x86_64"
            ;;
        *)
            error "Unsupported architecture: $arch"
            return 1
            ;;
    esac
}

get_binary_name() {
    local arch="$1"
    case "$arch" in
        arm64)
            echo "insight-engine-aarch64-apple-darwin"
            ;;
        x86_64)
            echo "insight-engine-x86_64-apple-darwin"
            ;;
        *)
            error "Unknown architecture: $arch"
            return 1
            ;;
    esac
}

verify_binary() {
    local binary="$1"
    local exe_path="$binary"

    if [[ -d "$binary" ]]; then
        exe_path="${binary}/insight-engine"
    fi

    if [[ ! -f "$exe_path" ]]; then
        error "Binary not found: $exe_path"
        return 1
    fi

    if [[ ! -x "$exe_path" ]]; then
        error "Binary is not executable: $exe_path"
        return 1
    fi

    # Verify it's a Mach-O binary
    if ! file "$exe_path" | grep -q "Mach-O"; then
        error "Binary is not a valid Mach-O executable: $exe_path"
        return 1
    fi

    return 0
}

print_logs_on_failure() {
    local step="$1"

    error "=========================================="
    error "Build failed at: $step"
    error "=========================================="
    echo ""

    # Print boot trace if available
    local boot_trace
    boot_trace="$(find ~/.insight/logs -name "boot_trace_*.log" -type f 2>/dev/null | sort -r | head -1)"
    if [[ -n "$boot_trace" && -f "$boot_trace" ]]; then
        error "Last 50 lines from boot trace:"
        error "File: $boot_trace"
        echo ""
        tail -50 "$boot_trace" | while IFS= read -r line; do
            error "  $line"
        done
        echo ""
    fi

    # Print backend stderr if available
    local stderr_log
    stderr_log="$(find ~/.insight/logs -name "backend_stderr_*.log" -type f 2>/dev/null | sort -r | head -1)"
    if [[ -n "$stderr_log" && -f "$stderr_log" ]]; then
        error "Last 50 lines from backend stderr:"
        error "File: $stderr_log"
        echo ""
        tail -50 "$stderr_log" | while IFS= read -r line; do
            error "  $line"
        done
        echo ""
    fi

    # Print app.log last 30 lines
    if [[ -f ~/.insight/logs/app.log ]]; then
        error "Last 30 lines from app.log:"
        echo ""
        tail -30 ~/.insight/logs/app.log | while IFS= read -r line; do
            error "  $line"
        done
        echo ""
    fi
}

check_prerequisites() {
    header "Checking Prerequisites"

    local missing=0

    # Check Python
    if ! command -v python3 &> /dev/null; then
        error "python3 not found in PATH"
        ((missing++))
    else
        success "python3: $(python3 --version)"
    fi

    # Check PyInstaller
    if ! command -v pyinstaller &> /dev/null; then
        warning "pyinstaller not found in PATH"
        info "Install with: pip install pyinstaller"
        ((missing++))
    else
        success "pyinstaller: $(pyinstaller --version)"
    fi

    # Check Node.js
    if ! command -v node &> /dev/null; then
        error "node not found in PATH"
        ((missing++))
    else
        success "node: $(node --version)"
    fi

    # Check npm/pnpm
    if command -v pnpm &> /dev/null; then
        success "pnpm: $(pnpm --version)"
    elif command -v npm &> /dev/null; then
        success "npm: $(npm --version)"
    else
        error "Neither pnpm nor npm found in PATH"
        ((missing++))
    fi

    # Check if we're in macOS
    if [[ "$(uname)" != "Darwin" ]]; then
        error "This script only supports macOS"
        ((missing++))
    else
        success "macOS: $(sw_vers -productVersion) ($(uname -m))"
    fi

    if [[ $missing -gt 0 ]]; then
        error ""
        error "Missing $missing prerequisite(s). Please install them and try again."
        exit 7
    fi

    echo ""
}

#-------------------------------------------------------------------------------
# Build steps
#-------------------------------------------------------------------------------

build_backend() {
    header "Building Python Backend (PyInstaller)"

    cd "$BACKEND_DIR"

    bold "Running PyInstaller..."
    info "Spec file: insight_engine_portable.spec"
    info "Output: dist/insight-engine"

    if ! pyinstaller --noconfirm --clean insight_engine_portable.spec 2>&1; then
        error "PyInstaller build failed"
        print_logs_on_failure "PyInstaller build"
        exit 2
    fi

    # Verify the binary was created
    if ! verify_binary "$BACKEND_DIST"; then
        error "PyInstaller did not produce a valid binary"
        print_logs_on_failure "PyInstaller verification"
        exit 2
    fi

    local size
    size=$(du -h -d 0 "$BACKEND_DIST" | cut -f1)
    success "Backend build created: $BACKEND_DIST ($size)"

    echo ""
}

run_smoke_tests() {
    header "Running Backend Smoke Tests"

    local runner="$BACKEND_DIST"
    if [[ -d "$BACKEND_DIST" ]]; then
        runner="${BACKEND_DIST}/insight-engine"
    fi

    bold "Executing: ${runner} --smoketest"
    echo ""

    local output
    if ! output=$("$runner" --smoketest 2>&1); then
        error "Smoke test failed with exit code $?"
        echo ""
        echo "$output"
        print_logs_on_failure "Smoke test"
        exit 3
    fi

    # Check if all tests passed
    if echo "$output" | grep -q "All tests passed"; then
        success "All smoke tests passed"
    else
        error "Some smoke tests failed"
        echo ""
        echo "$output"
        print_logs_on_failure "Smoke test verification"
        exit 3
    fi

    echo ""
}

copy_binary_to_tauri() {
    header "Copying Backend Binary to Tauri Resources"

    # Detect architecture
    local arch
    if ! arch=$(detect_architecture); then
        error "Failed to detect architecture"
        exit 4
    fi
    success "Detected architecture: $arch"

    # Get target binary name
    local target_name
    if ! target_name=$(get_binary_name "$arch"); then
        error "Failed to determine target binary name"
        exit 4
    fi

    local target_path="${TAURI_BIN_DIR}/${target_name}"

    bold "Source: $BACKEND_DIST"
    bold "Target: $target_path"

    # Create bin directory if it doesn't exist
    mkdir -p "$TAURI_BIN_DIR"

    # Copy binary
    info "Copying backend..."
    rm -rf "$target_path"
    if [[ -d "$BACKEND_DIST" ]]; then
        if ! cp -R "$BACKEND_DIST" "$target_path"; then
            error "Failed to copy backend directory"
            exit 5
        fi
        chmod +x "${target_path}/insight-engine"
    else
        if ! cp "$BACKEND_DIST" "$target_path"; then
            error "Failed to copy backend binary"
            exit 5
        fi
        chmod +x "$target_path"
    fi

    # Verify the copy
    if ! verify_binary "$target_path"; then
        error "Copied binary is not valid"
        exit 5
    fi

    local size
    size=$(du -h -d 0 "$target_path" | cut -f1)
    success "Backend copied successfully: $target_path ($size)"

    echo ""
}

build_tauri_app() {
    header "Building Tauri App"

    cd "$INSIGHT_DIR"

    # Check if node_modules exists
    if [[ ! -d "node_modules" ]]; then
        bold "Installing dependencies..."
        if command -v pnpm &> /dev/null; then
            pnpm install || {
                error "Failed to install dependencies with pnpm"
                exit 6
            }
        else
            npm install || {
                error "Failed to install dependencies with npm"
                exit 6
            }
        fi
        success "Dependencies installed"
    fi

    bold "Building Tauri app..."
    info "This may take a few minutes..."
    echo ""

    if command -v pnpm &> /dev/null; then
        if ! pnpm tauri build 2>&1; then
            error "Tauri build failed"
            exit 6
        fi
    else
        if ! npm run tauri build 2>&1; then
            error "Tauri build failed"
            exit 6
        fi
    fi

    # Find the built bundle
    local bundle_path
    bundle_path="$(find "$INSIGHT_DIR/src-tauri/target/release/bundle" -name "*.app" -o -name "*.dmg" 2>/dev/null | head -1)"

    if [[ -n "$bundle_path" && -e "$bundle_path" ]]; then
        success "Tauri app built successfully!"
        info "Bundle: $bundle_path"
    else
        success "Tauri app built successfully!"
        info "Check: $INSIGHT_DIR/src-tauri/target/release/bundle/"
    fi

    echo ""
}

#-------------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------------

main() {
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --skip-pyinstaller)
                SKIP_PYINSTALLER=true
                shift
                ;;
            --skip-smoketest)
                SKIP_SMOKETEST=true
                shift
                ;;
            --skip-tauri)
                SKIP_TAURI=true
                shift
                ;;
            --arch=*)
                FORCE_ARCH="${1#*=}"
                if [[ "$FORCE_ARCH" != "arm64" && "$FORCE_ARCH" != "x86_64" ]]; then
                    error "Invalid architecture: $FORCE_ARCH (must be arm64 or x86_64)"
                    exit 1
                fi
                shift
                ;;
            -h|--help)
                usage
                ;;
            *)
                error "Unknown option: $1"
                echo "Run '$(basename "$0") --help' for usage"
                exit 1
                ;;
        esac
    done

    header "Insight Production Build"
    info "Repo root: $REPO_ROOT"
    echo ""

    # Check prerequisites
    check_prerequisites

    # Build backend
    if [[ "$SKIP_PYINSTALLER" == false ]]; then
        build_backend
    else
        header "Skipping PyInstaller Build"
        warning "Using existing backend binary"
        if ! verify_binary "$BACKEND_DIST"; then
            error "Existing binary not found or invalid: $BACKEND_DIST"
            error "Remove --skip-pyinstaller to build from scratch"
            exit 2
        fi
        success "Using existing binary: $BACKEND_DIST"
        echo ""
    fi

    # Run smoke tests
    if [[ "$SKIP_SMOKETEST" == false ]]; then
        run_smoke_tests
    else
        header "Skipping Smoke Tests"
        warning "Smoke tests disabled"
        echo ""
    fi

    # Copy binary to Tauri
    copy_binary_to_tauri

    # Build Tauri app
    if [[ "$SKIP_TAURI" == false ]]; then
        build_tauri_app
    else
        header "Skipping Tauri Build"
        warning "Tauri build disabled"
        echo ""
    fi

    # Success!
    header "Build Complete!"
    success "All steps completed successfully"
    echo ""

    # Print summary
    info "Build artifacts:"
    if [[ -e "$BACKEND_DIST" ]]; then
        info "  Backend: $BACKEND_DIST"
    fi
    if [[ -e "$TAURI_BIN_DIR/insight-engine-aarch64-apple-darwin" ]]; then
        info "  Tauri ARM64: $TAURI_BIN_DIR/insight-engine-aarch64-apple-darwin"
    fi
    if [[ -e "$TAURI_BIN_DIR/insight-engine-x86_64-apple-darwin" ]]; then
        info "  Tauri x86_64: $TAURI_BIN_DIR/insight-engine-x86_64-apple-darwin"
    fi

    echo ""
}

main "$@"
