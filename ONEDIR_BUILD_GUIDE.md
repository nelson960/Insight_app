# Insight Onedir Production Build Guide

## Overview

This guide covers building Insight for macOS in **onedir production mode** - an offline, standalone application with all dependencies bundled locally.

---

#### Enhanced Excludes List
```python
excludes=[
    # Scientific computing (not needed)
    "matplotlib", "scipy", "pandas",
    "IPython", "jupyter", "jupyterlab", "notebook",
    # Testing
    "pytest", "test", "tests", "unittest", "doctest", "hypothesis",
    # Development tools
    "sphinx", "docs", "black", "isort", "flake8", "mypy", "pylint",
    "pydocstyle", "bandit",
    # GUI toolkits (not needed, we use Tauri)
    "tkinter", "tk",
    # ML frameworks not used directly
    "torch", "tensorflow",
    # Optional heavy dependencies
    "PIL", "Pillow", "cv2", "opencv",
],
```

#### Added Version Metadata (for Windows compatibility)
```python
if sys.platform == "win32":
    version_info = VSVersionInfo(
        ffi=FixedFileInfo(filevers=(0, 1, 0, 0), ...),
        ...
    )
```

---

### 2. Tauri Bundle Configuration (`insight/src-tauri/tauri.conf.json`)

#### Changed Targets to App Only
```json
{
  "bundle": {
    "active": true,
    "targets": "app",  // ✅ Only build .app bundle, no DMG
    "icon": [...],
    "macOS": {
      "frameworks": [],
      "minimumSystemVersion": "10.13"  // ✅ Set minimum macOS version
    }
  }
}
```

**Why "app" instead of "all"?**
- `"all"` would create both `.app` and `.dmg`
- For onedir mode, we only need the `.app` bundle
- The `.app` bundle IS the onedir package for macOS

---

### 3. New Build Script (`scripts/build_onedir_prod.sh`)

Created a comprehensive build script that:

✅ Builds PyInstaller backend with optimizations
✅ Runs smoke tests
✅ Builds Tauri app
✅ Creates distributable onedir package
✅ Optional code signing with ad-hoc signature
✅ Architecture detection (ARM64/x86_64)
✅ Skip flags for testing

---

## 🚀 How to Build

### Option A: Use the New Build Script (Recommended)

```bash
# Full build with code signing
./scripts/build_onedir_prod.sh

# Build without code signing (for testing)
./scripts/build_onedir_prod.sh --no-sign

# Skip PyInstaller (use existing backend binary)
./scripts/build_onedir_prod.sh --no-pyinstaller

# Force specific architecture
./scripts/build_onedir_prod.sh --arch=arm64

# Show help
./scripts/build_onedir_prod.sh --help
```

**Output:** `dist/onedir/Insight.app` (~2GB)

---

### Option B: Manual Build Steps

```bash
# 1. Build backend with PyInstaller
cd backend
pyinstaller --noconfirm --clean insight_engine_portable.spec

# 2. Test backend
INSIGHT_SMOKETEST=1 ./dist/insight-engine/insight-engine

# 3. Copy backend to Tauri resources
ARCH=$(uname -m)
BINARY_NAME="insight-engine-${ARCH}-apple-darwin"
mkdir -p ../insight/src-tauri/bin/${BINARY_NAME}
cp -R dist/insight-engine/* ../insight/src-tauri/bin/${BINARY_NAME}/

# 4. Build frontend
cd ../insight
pnpm install
pnpm build

# 5. Build Tauri app
cd src-tauri
cargo tauri build --bundles app

# 6. (Optional) Code sign with ad-hoc signature
codesign --force --deep --sign - \
  target/release/bundle/macos/Insight.app

# 7. Create distribution package
mkdir -p ../../dist/onedir
cp -R target/release/bundle/macos/Insight.app ../../dist/onedir/
```

---

## 📦 What You Get

### Directory Structure
```
dist/onedir/
├── Insight.app/          # macOS application bundle (onedir package)
│   ├── Contents/
│   │   ├── MacOS/
│   │   │   └── insight          # Rust executable
│   │   ├── Resources/
│   │   │   ├── bin/             # Backend engine
│   │   │   │   └── insight-engine-aarch64-apple-darwin/
│   │   │   │       ├── insight-engine     # Python executable
│   │   │   │       └── _internal/         # Dependencies
│   │   │   └── *_resources            # Frontend assets
│   │   └── Info.plist
│   └── icon
└── README.txt
```

---

## 🔐 Code Signing

### Ad-Hoc Signature (Default - Offline Distribution)

The build script uses **ad-hoc signing** by default:

```bash
codesign --force --deep --sign - Insight.app
```

**What this does:**
- ✅ Allows app to run on macOS without Gatekeeper blocking
- ✅ No Apple Developer account required
- ✅ Perfect for offline/personal distribution
- ⚠️ Users will see "unidentified developer" warning on first run
- ⚠️ Users must right-click → Open to run (one-time)

### To Remove Warning (Requires Apple Developer Account)

If you have an Apple Developer account:

1. Get a certificate from Apple
2. Update build script to use your identity:
```bash
codesign --force --deep --sign "Developer ID Application: Your Name" Insight.app
```

3. Notarize the app (required for macOS 10.15+):
```bash
xcrun notarytool submit Insight.app \
  --apple-id "your@email.com" \
  --password "app-specific-password" \
  --team-id "TEAM_ID"
```

---

## 📊 Expected Sizes

| Component | Size |
|-----------|------|
| `insight-engine` executable | ~130 MB |
| `_internal/` dependencies | ~1.7 GB |
| `.app` bundle total | **~2.0 GB** |

**Size optimizations applied:**
- ✅ `strip=True` reduces size by ~10-15%
- ✅ Comprehensive excludes removes unnecessary packages
- ✅ No debug symbols in production

---

## ✅ Verification Checklist

Before distributing:

- [ ] Smoke test passes: `INSIGHT_SMOKETEST=1 ./dist/insight-engine/insight-engine`
- [ ] App launches: `open dist/onedir/Insight.app`
- [ ] Backend responds: Check logs in `~/.insight/logs/`
- [ ] Code signature valid: `codesign -v dist/onedir/Insight.app`
- [ ] Size acceptable: `du -h dist/onedir`

---

## 🎯 Distribution

### For Personal Use / Offline

Simply zip the `.app` bundle:

```bash
cd dist/onedir
zip -r Insight-macos.zip Insight.app
```

Share the zip file. Users:
1. Unzip
2. Copy `Insight.app` to `/Applications/`
3. Right-click → Open (to bypass Gatekeeper warning)
4. Click "Open" in the dialog

---

## 🐛 Troubleshooting

### "Insight.app is damaged and can't be opened"

**Cause:** Gatekeeper quarantine attribute

**Fix:**
```bash
xattr -cr Insight.app
```

### "App won't open / crashes immediately"

**Check:**
1. Console.app for crash logs
2. `~/.insight/logs/backend_stderr_*.log` for backend errors
3. Run backend manually: `./Insight.app/Contents/MacOS/bin/*/insight-engine`

### Code signing fails

**Skip signing:**
```bash
./scripts/build_onedir_prod.sh --no-sign
```

---

## 📝 Changes Summary

| File | Changes | Impact |
|------|---------|--------|
| `backend/insight_engine_portable.spec` | Added icon, strip, excludes | Smaller, professional binary |
| `insight/src-tauri/tauri.conf.json` | Changed targets to "app" | Onedir mode enabled |
| `scripts/build_onedir_prod.sh` | New build script | Automated production builds |

---

## 🚀 Next Steps

1. **Build the app:**
   ```bash
   ./scripts/build_onedir_prod.sh
   ```

2. **Test thoroughly:**
   ```bash
   open dist/onedir/Insight.app
   ```

3. **Verify all features work:**
   - Chat functionality
   - File uploads
   - Settings
   - Model loading

4. **Distribute:**
   ```bash
   cd dist/onedir
   zip -r Insight-0.1.0-macos.zip Insight.app
   ```



For issues or questions:
- Check logs: `~/.insight/logs/`
- Run smoke test: `INSIGHT_SMOKETEST=1 backend/dist/insight-engine/insight-engine`
- Review build output for warnings

---

**Version:** 0.1.0
**Platform:** macOS 10.13+ (ARM64 and x86_64)
**Build Mode:** Onedir (Offline/Standalone)
