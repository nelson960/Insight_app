# Fresh Machine Verification Checklist

Complete checklist for verifying the packaged app works on a clean machine.

## Prerequisites for Test Machine

- Clean macOS user account (no conda, no previous Insight installs)
- Internet connection (for embedding model download test)
- Approximately 2GB free disk space

---

## Phase 1: Initial Installation

### Step 1: Install the App
```bash
# Mount and open the DMG
open Insight-_<version>_<arch>.dmg

# Drag to Applications folder
# Launch from Applications
open -a /Applications/Insight.app
```

**Expected Result:**
- App launches without error
- No crash on startup
- Window appears within 30 seconds

**If It Fails:**
- Check Console.app for crash logs
- Check `~/.insight/logs/boot_trace_*.log`
- Check `~/.insight/logs/backend_stderr_*.log`

---

### Step 2: Verify Codesign and Quarantine

```bash
# Check the app signature
codesign -dv --verbose=4 /Applications/Insight.app 2>&1 | head -20

# Check quarantine attribute
xattr -l /Applications/Insight.app

# Expected: "com.apple.quarantine" should be present
```

**Expected Result:**
- `codesign` should show "valid on disk"
- `com.apple.quarantine` attribute present

**If It Fails:**
```bash
# Re-sign if needed
codesign --force --deep --sign - /Applications/Insight.app
```

---

### Step 3: Verify Workspace Created

```bash
# Check workspace directory
ls -la ~/.insight/

# Expected subdirectories:
# uploads/  qdrant/  cache/  logs/  kv_sessions/  em_models/  db.sqlite
```

**Expected Result:**
- All subdirectories created
- `db.sqlite` file exists
- `logs/` directory contains `boot_trace_*.log`

---

## Phase 2: Boot Trace Verification

### Step 4: Check Boot Trace

```bash
# Find latest boot trace
LATEST_LOG=$(ls -t ~/.insight/logs/boot_trace_*.log 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    echo "ERROR: No boot trace log found!"
    exit 1
fi

echo "Latest boot trace: $LATEST_LOG"
cat "$LATEST_LOG"
```

**Expected Result:**
```
[timestamp] [INFO] engine_import: START
[timestamp] [INFO] before_app_creation
[timestamp] [INFO] configure_logging_start
[timestamp] [INFO] configure_logging_done
[timestamp] [INFO] create_app_start
[timestamp] [INFO] embedding_prefetch_start
[timestamp] [INFO] after_app_creation: OK
[timestamp] [INFO] main_async_start
[timestamp] [INFO] manager_started: OK
[timestamp] [INFO] startup_health_check_start
[timestamp] [INFO] startup_health_check: OK
```

**Critical Failures to Look For:**
- `[ERROR]` entries
- `TIMEOUT` entries
- `startup_health_check: ERROR`
- Missing `manager_started: OK`

---

### Step 5: Check for Critical Errors

```bash
# Search for errors in boot trace
LATEST_LOG=$(ls -t ~/.insight/logs/boot_trace_*.log 2>/dev/null | head -1)

echo "=== Checking for errors ==="
grep -i "error" "$LATEST_LOG" | head -20

echo ""
echo "=== Checking for timeouts ==="
grep -i "timeout" "$LATEST_LOG"

echo ""
echo "=== Checking session_manager errors ==="
grep -i "session_manager" "$LATEST_LOG"
```

**Expected Result:**
- No `[ERROR]` entries related to critical components
- No `session_manager_error: model_not_found` (unless no model configured)

---

## Phase 3: Diagnostics Verification

### Step 6: Run Diagnostics via IPC

From the app, open Developer Console and run:
```javascript
// Test packaging diagnostics
fetch('/diagnostics/packaging')
  .then(r => r.json())
  .then(d => console.log('Diagnostics:', JSON.stringify(d, null, 2)))
```

**Expected Result:**
```json
{
  "runtime": {
    "packaged": true,
    "meipass": "/var/folders/.../AppTranslocation/.../insight-engine",
    "python_version": "3.x.x"
  },
  "certificates": {
    "certifi_available": true,
    "certifi_exists": true,
    "certifi_bundle": "/.../cacert.pem"
  },
  "onnxruntime": {
    "available": true,
    "version": "1.x.x"
  },
  "llama_cpp": {
    "available": true,
    "native_lib_loaded": true
  },
  "tokenizers": {
    "available": true
  },
  "sqlite": {
    "database_exists": true,
    "database_writable": true
  }
}
```

**Critical Checks:**
- `certificates.certifi_exists: true`
- `onnxruntime.available: true`
- `llama_cpp.native_lib_loaded: true`

---

## Phase 4: Smoke Test

### Step 7: Run Smoke Test

```bash
# Find the bundled binary
BINARY=$(find /Applications/Insight.app -name "insight-engine-*" -type f 2>/dev/null | head -1)

if [ -z "$BINARY" ]; then
    echo "ERROR: insight-engine binary not found!"
    exit 1
fi

echo "Binary: $BINARY"

# Run smoke test
"$BINARY" --smoketest
```

**Expected Result:**
```
============================================================
  INSIGHT SMOKE TEST
============================================================
Time: 2025-01-14T...

============================================================
  Environment
============================================================
  ✓ Environment

============================================================
  Workspace
============================================================
  ✓ Workspace

============================================================
  SQLite
============================================================
  ✓ SQLite

============================================================
  SSL Certificates
============================================================
  ✓ SSL Certificates

============================================================
  ONNX Runtime
============================================================
  ✓ ONNX Runtime

============================================================
  Tokenizers
============================================================
  ✓ Tokenizers

============================================================
  llama.cpp
============================================================
  ✓ llama_cpp

============================================================
  Qdrant
============================================================
  ✓ Qdrant

============================================================
  SUMMARY
============================================================
  ✓ PASS: Environment
  ✓ PASS: Workspace
  ✓ PASS: SQLite
  ✓ PASS: SSL Certificates
  ✓ PASS: ONNX Runtime
  ✓ PASS: Tokenizers
  ✓ PASS: llama.cpp
  ✓ PASS: Qdrant

Total: 8/8 tests passed in X.Xs

✓ All tests passed!
```

**Exit Code:** 0

---

## Phase 5: Functional Testing

### Step 8: Configure Model

1. Open Settings → Model
2. Click "Choose Model File"
3. Select a valid GGUF model
4. App should load the model without crash

**Check:**
- Model path appears in Settings
- No crash on model load
- Status shows "Model ready"

---

### Step 9: Test Embedding Download

1. Open Settings → Embeddings
2. Click "Download Embeddings" (if not already downloaded)
3. Wait for download to complete (or timeout after 30s)

**Check:**
```bash
# Check embedding files
ls -lh ~/.insight/em_models/nomic-embed-text/

# Expected:
# tokenizer.json (several MB)
# onnx/
#   model.onnx (hundreds of MB)
```

**Download Status Check:**
- If complete: Status shows "Embeddings: Downloaded"
- If timeout: Status shows "Embeddings: Downloading..." but app still usable
- If error: Check `~/.insight/logs/backend_stderr_*.log`

---

### Step 10: Test Document Ingestion

1. Create a test text file: `echo "Hello world" > /tmp/test.txt`
2. In the app, upload the file
3. Wait for ingestion to complete

**Check:**
```bash
# Check if document was indexed
sqlite3 ~/.insight/db.sqlite "SELECT COUNT(*) FROM file_text;"

# Expected: > 0
```

---

### Step 11: Test Chat

1. Type a message in the chat
2. Wait for response
3. Verify streaming works

**Check:**
- Tokens appear incrementally
- No crash during generation
- Response completes

---

## Phase 6: Error Recovery Testing

### Step 12: Test Network Failure

```bash
# Block network temporarily (simulates offline/cert issues)
# Note: This requires sudo
sudo pfctl -e
sudo pfctl -f /dev/stdin <<EOF
block drop out on any from any to huggingface.co
EOF

# Then try to download embeddings (should timeout gracefully)
# Restore network:
sudo pfctl -f /etc/pf.conf
```

**Expected Result:**
- Download times out after 30 seconds
- App remains responsive
- Error message is shown (not silent hang)

---

### Step 13: Test Model Load Failure

1. Configure an invalid model path (non-existent file)
2. Try to send a chat message

**Expected Result:**
- Clear error message: "Model not found at..."
- No crash
- Can recover by configuring valid model

---

## Phase 7: Log File Verification

### Step 14: Check All Log Files

```bash
# List all log files
ls -lh ~/.insight/logs/

# Expected files:
# boot_trace_*.log - Main boot trace
# backend_stderr_*.log - Python stderr output

# Check for critical errors
echo "=== Boot trace errors ==="
grep -i "\[error\]" ~/.insight/logs/boot_trace_*.log | tail -20

echo ""
echo "=== Backend stderr errors ==="
grep -i "error\|exception\|traceback" ~/.insight/logs/backend_stderr_*.log | tail -20
```

**Expected Result:**
- No critical errors (exceptions that crash the app)
- SSL certificate errors should NOT appear
- Native library import errors should NOT appear

---

## Phase 8: Performance Verification

### Step 15: Check Startup Time

```bash
# Measure startup time
time open -a /Applications/Insight.app

# Expected: < 10 seconds to main window
```

### Step 16: Check Memory Usage

```bash
# Check memory after startup
ps aux | grep -i insight | grep -v grep

# Check memory during idle
# Expected: < 500MB RSS
```

---

## Phase 9: Uninstall Verification

### Step 17: Clean Uninstall

```bash
# Close the app
killall Insight 2>/dev/null

# Remove app
rm -rf /Applications/Insight.app

# Remove user data (optional)
rm -rf ~/.insight

# Verify cleanup
ls -la ~/ | grep -i insight  # Should return nothing
```

---

## Troubleshooting Guide

### Problem: App crashes on startup

**Diagnosis:**
```bash
# Check boot trace
cat ~/.insight/logs/boot_trace_*.log

# Check crash log
log show --predicate 'process == "Insight"' --last 1m
```

**Common Causes:**
1. Missing certificates → Rebuild with certifi bundled
2. Missing native libs → Check PyInstaller spec
3. Database lock → Delete `~/.insight/db.sqlite` and restart

---

### Problem: Infinite loading screen

**Diagnosis:**
```bash
# Check if backend process is running
ps aux | grep insight-engine

# Check backend logs
cat ~/.insight/logs/backend_stderr_*.log

# Look for: "Backend failed to start within 30 seconds"
```

**Common Causes:**
1. Backend crash → Check stderr logs
2. Import error → Check boot trace for import failures
3. Network timeout → Check if download is blocking

---

### Problem: Embedding download hangs

**Diagnosis:**
```bash
# Check packaging/dependency status
cat ~/.insight/logs/boot_summary_latest.json | jq '.dependencies'

# Check certificate
ls -la ~/.insight/em_models/nomic-embed-text/tokenizer.json
```

**Common Causes:**
1. No SSL certificates → Check certifi bundle
2. Network blocked → Check firewall/proxy
3. HuggingFace down → Check https://status.huggingface.co

---

### Problem: Model loading crashes

**Diagnosis:**
```bash
# Check model file
file /path/to/model.gguf

# Check if GGUF is valid
# (Use a GGUF validator or hexdump to check magic bytes)

# Check llama_cpp logs
grep -i llama ~/.insight/logs/backend_stderr_*.log
```

**Common Causes:**
1. Corrupted GGUF file → Re-download model
2. llama_cpp native lib missing → Rebuild PyInstaller
3. Insufficient memory → Use smaller model or reduce context

---

## Success Criteria

The packaged app is considered stable when ALL of the following pass:

- ✅ No crashes during normal operation
- ✅ Boot trace shows all steps completing with OK status
- ✅ Smoke test passes (8/8 tests)
- ✅ All diagnostic checks pass (SSL, native libs, DB)
- ✅ Model loading works consistently
- ✅ Embedding download completes or times out gracefully
- ✅ Document ingestion and retrieval work
- ✅ Chat functionality works end-to-end
- ✅ No silent hangs (all errors surface to user)
- ✅ Startup time < 10 seconds
- ✅ Memory usage < 500MB at idle
- ✅ Clean uninstall removes all traces

---

## Build Commands Reference

### macOS arm64 (Apple Silicon)

```bash
cd /Users/nelson/py/insight/insight_app

# Activate Python environment
conda activate ml

# Build Python backend
pyinstaller --noconfirm --clean backend/insight_engine_portable.spec

# Copy to Tauri bin directory (onedir bundle)
cp -R dist/insight-engine insight/src-tauri/bin/insight-engine-aarch64-apple-darwin
chmod +x insight/src-tauri/bin/insight-engine-aarch64-apple-darwin/insight-engine

# Build Tauri app
cd insight
npm run tauri build

# Output: insight/src-tauri/target/release/bundle/dmg/Insight_<version>_aarch64.dmg
```

### macOS x86_64 (Intel)

```bash
# Same as arm64, but the output bundle name differs:
cp -R dist/insight-engine insight/src-tauri/bin/insight-engine-x86_64-apple-darwin
chmod +x insight/src-tauri/bin/insight-engine-x86_64-apple-darwin/insight-engine

# Build Tauri app (on Intel Mac or with rosetta)
cd insight
npm run tauri build

# Output: insight/src-tauri/target/release/bundle/dmg/Insight_<version>_x86_64.dmg
```

### Universal Binary (both arch)

```bash
# Build both architectures separately, then use Tauri universal build
cd insight
npm run tauri build -- --target universal-apple-darwin
```

---

## Verification Script

Save this as `verify_packaged_app.sh` and run on fresh machine:

```bash
#!/bin/bash
set -e

APP_PATH="/Applications/Insight.app"

echo "=== Insight Packaged App Verification ==="
echo ""

# 1. Check app exists
if [ ! -d "$APP_PATH" ]; then
    echo "FAIL: App not installed at $APP_PATH"
    exit 1
fi
echo "✓ App installed"

# 2. Check codesign
if ! codesign -v "$APP_PATH" 2>&1 | grep -q "valid on disk"; then
    echo "FAIL: App signature invalid"
    exit 1
fi
echo "✓ App code signed"

# 3. Check workspace
if [ ! -d "$HOME/.insight" ]; then
    echo "FAIL: Workspace not created"
    exit 1
fi
echo "✓ Workspace created"

# 4. Find and run smoke test
BINARY=$(find "$APP_PATH" -name "insight-engine-*" -type f 2>/dev/null | head -1)
if [ -z "$BINARY" ]; then
    echo "FAIL: insight-engine binary not found"
    exit 1
fi
echo "✓ Binary found: $BINARY"

if ! "$BINARY" --smoketest; then
    echo "FAIL: Smoke test failed"
    exit 1
fi
echo "✓ Smoke test passed"

# 5. Check boot trace
LATEST_LOG=$(ls -t ~/.insight/logs/boot_trace_*.log 2>/dev/null | head -1)
if [ -z "$LATEST_LOG" ]; then
    echo "FAIL: No boot trace log"
    exit 1
fi
echo "✓ Boot trace exists"

if grep -q "\[ERROR\]" "$LATEST_LOG"; then
    echo "WARNING: Boot trace contains errors"
    grep "\[ERROR\]" "$LATEST_LOG"
fi

echo ""
echo "=== All Checks Passed ==="
```

Make executable:
```bash
chmod +x verify_packaged_app.sh
./verify_packaged_app.sh
```
