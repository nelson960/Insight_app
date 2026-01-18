# Build Instructions - Insight (macOS)

This guide covers building the Insight Tauri application with the embedded Python backend.

## Prerequisites

### Required Tools

1. **Python 3.10+** with PyInstaller
   ```bash
   pip install pyinstaller
   ```

2. **Node.js 18+** and npm/pnpm
   ```bash
   # Using pnpm (recommended)
   npm install -g pnpm

   # Or npm
   npm install
   ```

3. **macOS** (Apple Silicon or Intel)
   - Xcode Command Line Tools
   - Rust and Cargo (installed by Tauri CLI if needed)

### Verify Prerequisites

```bash
python3 --version
pyinstaller --version
node --version
pnpm --version  # or npm --version
```

## Quick Start

### Apple Silicon (ARM64)

```bash
cd /path/to/insight_app
./scripts/build_app.sh
```

### Intel (x86_64)

```bash
cd /path/to/insight_app
./scripts/build_app.sh --arch=x86_64
```

## Build Script Options

The `scripts/build_app.sh` script orchestrates the complete build process:

```bash
# Build everything (default)
./scripts/build_app.sh

# Skip PyInstaller (use existing backend binary)
./scripts/build_app.sh --skip-pyinstaller

# Skip smoke tests
./scripts/build_app.sh --skip-smoketest

# Skip Tauri build (backend only)
./scripts/build_app.sh --skip-tauri

# Force specific architecture
./scripts/build_app.sh --arch=arm64
./scripts/build_app.sh --arch=x86_64

# Combine options
./scripts/build_app.sh --skip-smoketest --arch=arm64
```

## Build Process

The script performs these steps:

1. **Check Prerequisites** - Verifies all required tools are installed
2. **Build Backend** - Runs PyInstaller to create `backend/dist/insight-engine/`
3. **Smoke Tests** - Runs `--smoketest` to verify backend functionality
4. **Detect Architecture** - Auto-detects ARM64 vs x86_64
5. **Copy Binary** - Copies engine to `insight/src-tauri/bin/<arch>/`
6. **Build Tauri** - Runs `pnpm tauri build` to create final app bundle

## Build Artifacts

After successful build:

```
backend/dist/
└── insight-engine/                   # PyInstaller onedir bundle
    └── insight-engine                # Backend executable

insight/src-tauri/bin/
├── insight-engine-aarch64-apple-darwin/  # ARM64 backend bundle
└── insight-engine-x86_64-apple-darwin/   # x86_64 backend bundle

insight/src-tauri/target/release/bundle/
├── dmg/                                # DMG installer
│   └── Insight_<version>_<arch>.dmg
└── macos/                              # Unsigned .app bundle
    └── Insight.app
```

## Manual Build Steps

If you prefer to build components manually:

### 1. Build Backend

```bash
cd backend
pyinstaller --noconfirm --clean insight_engine_portable.spec

# Test the backend
./dist/insight-engine/insight-engine --smoketest
```

### 2. Copy to Tauri Resources

```bash
# Detect your architecture
ARCH=$(uname -m)  # arm64 or x86_64

# Set target name
if [[ "$ARCH" == "arm64" ]]; then
    TARGET="insight-engine-aarch64-apple-darwin"
else
    TARGET="insight-engine-x86_64-apple-darwin"
fi

# Copy binary
cp -R backend/dist/insight-engine "insight/src-tauri/bin/$TARGET"
chmod +x "insight/src-tauri/bin/$TARGET/insight-engine"
chmod +x "insight/src-tauri/bin/$TARGET"
```

### 3. Build Tauri App

```bash
cd insight

# Install dependencies (first time only)
pnpm install

# Build the app
pnpm tauri build
```

## Troubleshooting

### PyInstaller Build Fails

**Issue:** PyInstaller fails with import errors

**Solution:**
```bash
# Verify all dependencies are installed
cd backend
pip install -r requirements.txt  # if you have requirements.txt

# Test PyInstaller dry-run
pyinstaller --onefile --dry-run engine.py
```

### Smoke Test Failures

**Issue:** `--smoketest` returns non-zero exit code

**Solution:** Check logs for details:
```bash
# View boot trace
cat ~/.insight/logs/boot_trace_*.log | tail -100

# View app log
cat ~/.insight/logs/app.log | tail -100
```

Common issues:
- **SSL certificate errors:** Verify `certifi` is installed: `pip install certifi`
- **Workspace errors:** Clear workspace: `rm -rf ~/.insight`
- **Native library errors:** Verify architecture matches your system

### Tauri Build Fails

**Issue:** Tauri build fails with "binary not found"

**Solution:**
```bash
# Verify backend binary was copied
ls -la insight/src-tauri/bin/

# Check architecture matches
uname -m  # Should match binary name
file insight/src-tauri/bin/insight-engine-*
```

**Issue:** Tauri build fails with Rust errors

**Solution:**
```bash
# Update Rust
rustup update

# Clean Tauri build cache
cd insight
rm -rf src-tauri/target
pnpm tauri build
```

### Architecture Mismatch

**Issue:** "Wrong CPU type" error

**Solution:** Ensure backend binary architecture matches your system:
```bash
# Check your system architecture
uname -m

# Check binary architecture
file backend/dist/insight-engine/insight-engine

# Force correct architecture in build script
./scripts/build_app.sh --arch=arm64    # for Apple Silicon
./scripts/build_app.sh --arch=x86_64   # for Intel
```

### Cross-Architecture Builds

To build for a different architecture (e.g., Intel on Apple Silicon):

**Option 1: Use Rosetta 2 (Intel builds on Apple Silicon)**
```bash
# Install Intel Python in Rosetta environment
arch -x86_64 python3 -m pip install pyinstaller

# Build with explicit architecture
arch -x86_64 ./scripts/build_app.sh --arch=x86_64
```

**Option 2: Build on native hardware**
Build Intel version on an Intel Mac, build ARM64 version on Apple Silicon Mac.

### Signing and Notarization

The build produces an unsigned `.app` bundle. For distribution:

1. **Code Sign:**
   ```bash
   codesign --force --deep --sign "Developer ID Application: Your Name" \
       insight/src-tauri/target/release/bundle/macos/Insight.app
   ```

2. **Create DMG:**
   ```bash
   hdiutil create -volname "Insight" -srcfolder \
       insight/src-tauri/target/release/bundle/macos/Insight.app \
       Insight.dmg
   ```

3. **Notarize:**
   ```bash
   xcrun notarytool submit Insight.dmg \
       --apple-id "your@email.com" \
       --password "app-specific-password" \
       --team-id "TEAM_ID" \
       --wait
   ```

## CI/CD Integration

For GitHub Actions or similar CI:

```yaml
- name: Build Insight
  run: ./scripts/build_app.sh --arch=arm64

- name: Upload artifacts
  uses: actions/upload-artifact@v3
  with:
    name: Insight-macOS-ARM64
    path: |
      backend/dist/insight-engine/
      insight/src-tauri/target/release/bundle/dmg/*.dmg
```

## Development vs Production

### Development Build (Fast Iteration)

```bash
# Terminal 1: Frontend dev server
cd insight
pnpm dev

# Terminal 2: Backend (use directly, no packaging)
cd backend
python3 engine.py
```

### Production Build (Full Bundle)

```bash
# Complete build with all optimizations
./scripts/build_app.sh
```

## Support

For issues or questions:
1. Check logs: `~/.insight/logs/*.log`
2. Review build script output for error details
3. Verify prerequisites are installed correctly
4. Check [PACKAGING_DIAGNOSIS_AND_FIX.md](PACKAGING_DIAGNOSIS_AND_FIX.md) for known issues
