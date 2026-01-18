#!/bin/bash
#
# Insight Onedir Production Build Script (macOS)
#
# This script creates an onedir production build for macOS:
# 1. Builds the Python backend with PyInstaller
# 2. Runs smoke tests on the backend
# 3. Builds the Tauri app (.app bundle)
# 4. Creates a distributable onedir package
# 5. Optionally code signs the binaries
#
# Usage:
#   ./scripts/build_onedir_prod.sh                 # Build everything
#   ./scripts/build_onedir_prod.sh --no-sign      # Skip code signing
#   ./scripts/build_onedir_prod.sh --arch=arm64   # Force ARM64 architecture
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
#   8 - Code signing failed

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
TAURI_DIR="${INSIGHT_DIR}/src-tauri"
BACKEND_DIST="${BACKEND_DIR}/dist/insight-engine"
DIST_DIR="${REPO_ROOT}/dist/onedir"

# Flags
SKIP_SIGN=false
FORCE_ARCH=""
SKIP_PYINSTALLER=false
SKIP_SMOKETEST=false

#-------------------------------------------------------------------------------
# Helper functions
#-------------------------------------------------------------------------------

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --no-sign             Skip code signing (for testing)
  --no-pyinstaller      Skip PyInstaller build (use existing dist/)
  --no-smoketest        Skip smoke tests
  --arch=arm64|x86_64   Force architecture (default: auto-detect)
  -h, --help            Show this help message

Examples:
  $(basename "$0")                          # Build everything with signing
  $(basename "$0") --no-sign               # Skip code signing
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
        x86_64|amd64)
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

    # Check pnpm
    if ! command -v pnpm &> /dev/null; then
        error "pnpm not found in PATH"
        info "Install with: npm install -g pnpm"
        ((missing++))
    else
        success "pnpm: $(pnpm --version)"
    fi

    # Check Cargo
    if ! command -v cargo &> /dev/null; then
        error "cargo not found in PATH"
        ((missing++))
    else
        success "cargo: $(cargo --version)"
    fi

    # Check if we're in macOS
    if [[ "$(uname)" != "Darwin" ]]; then
        error "This script only supports macOS"
        ((missing++))
    else
        success "macOS: $(sw_vers -productVersion) ($(uname -m))"
    fi

    # Check for codesign (macOS only)
    if ! command -v codesign &> /dev/null; then
        warning "codesign not found (cannot sign binaries)"
    else
        success "codesign: available"
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

    if [[ "$SKIP_PYINSTALLER" == "true" ]]; then
        warning "Skipping PyInstaller build, using existing dist/"
        if ! verify_binary "$BACKEND_DIST"; then
            error "Existing binary is invalid, cannot skip PyInstaller"
            exit 2
        fi
        return 0
    fi

    bold "Running PyInstaller..."
    info "Spec file: insight_engine_portable.spec"
    info "Output: dist/insight-engine"

    if ! pyinstaller --noconfirm --clean insight_engine_portable.spec 2>&1; then
        error "PyInstaller build failed"
        exit 2
    fi

    # Verify the binary was created
    if ! verify_binary "$BACKEND_DIST"; then
        error "PyInstaller did not produce a valid binary"
        exit 2
    fi

    local size
    size=$(du -h -d 0 "$BACKEND_DIST" | cut -f1)
    success "Backend build created: $BACKEND_DIST ($size)"

    echo ""
}

run_smoke_tests() {
    if [[ "$SKIP_SMOKETEST" == "true" ]]; then
        warning "Skipping smoke tests"
        return 0
    fi

    header "Running Backend Smoke Tests"

    local runner="$BACKEND_DIST"
    if [[ -d "$BACKEND_DIST" ]]; then
        runner="${BACKEND_DIST}/insight-engine"
    fi

    bold "Running smoketest..."
    if ! INSIGHT_SMOKETEST=1 "$runner"; then
        error "Smoke test failed"
        exit 3
    fi

    success "Smoke tests passed"
    echo ""
}

build_tauri_app() {
    header "Building Tauri App"

    cd "$INSIGHT_DIR"

    bold "Installing frontend dependencies..."
    if ! pnpm install; then
        error "Failed to install frontend dependencies"
        exit 6
    fi

    bold "Building frontend..."
    if ! pnpm build; then
        error "Failed to build frontend"
        exit 6
    fi

    bold "Building Tauri app (.app bundle)..."
    cd "$TAURI_DIR"
    if ! cargo tauri build --bundles app; then
        error "Tauri build failed"
        exit 6
    fi

    success "Tauri app built successfully"

    # Copy backend engine into the .app bundle
    bold "Copying backend engine into .app bundle..."

    local arch
    arch=$(detect_architecture)
    local binary_name
    binary_name=$(get_binary_name "$arch")

    local app_bundle="${TAURI_DIR}/target/release/bundle/macos/Insight.app"
    local backend_source="${TAURI_DIR}/bin/${binary_name}"
    local backend_dest_dir="${app_bundle}/Contents/Resources/bin"
    local backend_dest="${backend_dest_dir}/${binary_name}"

    if [[ ! -d "$backend_source" ]]; then
        error "Backend binary not found at: $backend_source"
        error "Did you run the build_backend step first?"
        exit 5
    fi

    # Create destination directory (full path)
    mkdir -p "$backend_dest"

    # Copy backend
    if ! cp -R "$backend_source"/* "$backend_dest/"; then
        error "Failed to copy backend to .app bundle"
        exit 5
    fi

    local size
    size=$(du -h -d 0 "$backend_dest" | cut -f1)
    success "Backend copied to .app bundle ($size)"

    echo ""
}

code_sign_binaries() {
    if [[ "$SKIP_SIGN" == "true" ]]; then
        warning "Skipping code signing"
        return 0
    fi

    header "Code Signing Binaries"

    if ! command -v codesign &> /dev/null; then
        warning "codesign not available, skipping"
        return 0
    fi

    local app_path="${TAURI_DIR}/target/release/bundle/macos/Insight.app"

    if [[ ! -d "$app_path" ]]; then
        error "App bundle not found: $app_path"
        return 0
    fi

    bold "Signing app bundle with ad-hoc signature..."
    info "Path: $app_path"

    # Sign with ad-hoc signature (for offline/distribution without Apple Developer account)
    # This allows the app to run on macOS without Gatekeeper issues
    if ! codesign --force --deep --sign - "$app_path" 2>&1; then
        error "Code signing failed"
        warning "App may still work but will show Gatekeeper warning"
        return 0
    fi

    # Verify signature
    if codesign -v "$app_path" 2>&1; then
        success "Code signature verified"
    else
        warning "Signature verification failed"
    fi

    echo ""
}

create_onedir_package() {
    header "Creating Onedir Package"

    local app_path="${TAURI_DIR}/target/release/bundle/macos/Insight.app"

    if [[ ! -d "$app_path" ]]; then
        error "App bundle not found: $app_path"
        exit 1
    fi

    # Create distribution directory
    rm -rf "$DIST_DIR"
    mkdir -p "$DIST_DIR"

    # Copy the .app bundle
    bold "Copying app bundle..."
    cp -R "$app_path" "$DIST_DIR/"

    # Create a README
    bold "Creating README..."
    cat > "$DIST_DIR/README.txt" << 'EOF'
INSIGHT - Onedir Production Build
================================

To run the application:
1. Double-click "Insight.app"
2. Or run: open Insight.app

To run from command line:
  ./Insight.app/Contents/MacOS/insight

Application data is stored in:
  ~/.insight/

Logs are stored in:
  ~/.insight/logs/

Version: 0.1.0
Platform: macOS (onedir)
EOF

    success "Onedir package created: $DIST_DIR"

    local size
    size=$(du -h -d 0 "$DIST_DIR" | cut -f1)
    info "Total size: $size"

    echo ""
}

#-------------------------------------------------------------------------------
# Parse arguments
#-------------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-sign)
            SKIP_SIGN=true
            shift
            ;;
        --no-pyinstaller)
            SKIP_PYINSTALLER=true
            shift
            ;;
        --no-smoketest)
            SKIP_SMOKETEST=true
            shift
            ;;
        --arch=*)
            FORCE_ARCH="${1#*=}"
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            error "Unknown option: $1"
            usage
            ;;
    esac
done

#-------------------------------------------------------------------------------
# Main build process
#-------------------------------------------------------------------------------

main() {
    header "Insight Onedir Production Build"

    echo -e "${BLUE}Configuration:${NC}"
    info "Repository: $REPO_ROOT"
    info "Distribution: $DIST_DIR"
    info "Skip signing: $SKIP_SIGN"
    info "Architecture: ${FORCE_ARCH:-auto-detect}"
    echo ""

    # Check prerequisites
    check_prerequisites

    # Build backend
    build_backend

    # Run smoke tests
    run_smoke_tests

    # Build Tauri app (includes copying backend)
    build_tauri_app

    # Code sign binaries
    code_sign_binaries

    # Create onedir package
    create_onedir_package

    # Success!
    header "Build Complete!"
    success "Onedir package ready at: $DIST_DIR"

    echo -e "\nTo run the app:"
    echo "  open ${DIST_DIR}/Insight.app"
    echo ""
}

main "$@"
