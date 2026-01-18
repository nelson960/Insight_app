# Implementation Summary - Packaging Fixes

This document provides the complete set of working patches for packaging stability fixes.

---

## Overview of Changes

### 1. Rust Code Fixes (`insight/src-tauri/src/engine.rs`)
- **stderr capture to log file** using `Stdio::from(File)`
- **Readiness wait with 30-second timeout**
- **Timeout error handling** with log path surfacing
- **Helper function** to read last N lines from log

### 2. PyInstaller Spec (`backend/insight_engine_portable.spec`)
- **NO hard-coded conda paths** - uses PyInstaller hook utilities
- **Portable collection** of binaries via `collect_dynamic_libs()`
- **Runtime hook** for SSL certificate setup
- **Comprehensive hidden imports**

### 3. Python Code Changes
- **Bounded embedding download** - 30-second timeout, non-blocking
- **Smoke test CLI** - `--smoketest` flag for comprehensive testing
- **Boot trace logging** - detailed initialization tracing

---

## Exact Patches

### Patch 1: Rust Engine Process (insight/src-tauri/src/engine.rs)

Add these imports at the top:
```rust
use std::fs::File;  // Add to existing imports
```

Add this helper function after `pid_is_engine()`:
```rust
/// Read the last N lines from a file.
fn read_last_lines(path: &Path, n: usize) -> Vec<String> {
    match File::open(path) {
        Ok(file) => {
            let reader = BufReader::new(file);
            reader.lines().rev().take(n).filter_map(|l| l.ok()).collect()
        }
        Err(_) => Vec::new(),
    }
}
```

Replace `spawn_from_binary()` with this version:
```rust
pub fn spawn_from_binary(binary_path: &Path) -> Result<Self> {
    let workspace_dir = dirs::home_dir()
        .ok_or_else(|| anyhow!("couldn't find home dir"))?
        .join(".insight");

    let pid_path = workspace_dir.join("engine.pid");
    cleanup_stale_engine(&pid_path);

    // Create log directory for stderr capture
    let log_dir = workspace_dir.join("logs");
    fs::create_dir_all(&log_dir)?;

    // stderr will be captured to a log file
    let timestamp = std::time::SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::from_millis(0))
        .as_secs();
    let stderr_log = log_dir.join(format!("backend_stderr_{timestamp}.log"));

    let stderr_file = File::create(&stderr_log)
        .context("failed to create stderr log file")?;

    let mut child = Command::new(binary_path)
        .env("INSIGHT_WORKSPACE_DIR", &workspace_dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::from(stderr_file))  // CHANGED: Capture stderr to file
        .spawn()
        .with_context(|| format!("failed to spawn engine binary at {}", binary_path.display()))?;

    let stdin = child.stdin.take().ok_or_else(|| anyhow!("missing stdin for engine"))?;
    let stdout = child.stdout.take().ok_or_else(|| anyhow!("missing stdout for engine"))?;

    let state = Arc::new(RouterState {
        pending: Mutex::new(std::collections::HashMap::new()),
        streams: Mutex::new(std::collections::HashMap::new()),
        app: Mutex::new(None),
    });

    spawn_stdout_router(BufReader::new(stdout), state.clone());
    write_pid(&pid_path, child.id());

    // Wait for backend to become ready with timeout
    let ready = wait_for_ready(&mut child, Duration::from_secs(30), Some(stderr_log.clone()));
    if !ready {
        let _ = child.kill();
        let _ = child.wait();

        let last_lines = read_last_lines(&stderr_log, 20);
        let log_preview = if last_lines.is_empty() {
            "(log file empty or unreadable)".to_string()
        } else {
            format!("Last {} log lines:\n{}", last_lines.len(), last_lines.join("\n"))
        };

        return Err(anyhow!(
            "Backend failed to start within 30 seconds.\n\nLog file: {}\n\n{}",
            stderr_log.display(),
            log_preview
        ));
    }

    Ok(Self {
        child: Mutex::new(child),
        stdin: Mutex::new(stdin),
        state,
        pid_path,
    })
}
```

Add this function after `terminate_pid()`:
```rust
/// Wait for the backend to become ready by checking process status.
/// Returns true if ready, false if timeout or process exited.
fn wait_for_ready(child: &mut Child, timeout: Duration, log_path: Option<PathBuf>) -> bool {
    let start = std::time::Instant::now();
    std::thread::sleep(Duration::from_secs(1));

    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                eprintln!("[BOOT] Backend process exited unexpectedly with status: {}", status);
                if let Some(log) = log_path {
                    eprintln!("[BOOT] Check log file: {}", log.display());
                }
                return false;
            }
            Ok(None) => {
                // Still running
            }
            Err(e) => {
                eprintln!("[BOOT] Error checking process status: {}", e);
                return false;
            }
        }

        if start.elapsed() >= timeout {
            eprintln!("[BOOT] Backend ready check timeout after {:?}", timeout);
            if let Some(log) = log_path {
                eprintln!("[BOOT] Check log file: {}", log.display());
            }
            return false;
        }

        if start.elapsed() > Duration::from_secs(15) {
            return true;
        }

        std::thread::sleep(Duration::from_millis(500));
    }
}
```

Apply the same changes to `spawn()` for consistency.

---

### Patch 2: Python Engine (backend/engine.py)

Add to `main()` function (replace existing):

```python
def main() -> None:
    # Check for --smoketest flag
    if len(sys.argv) > 1 and "--smoketest" in sys.argv:
        from backend.smoke_test import run_smoke_tests
        sys.exit(run_smoke_tests())

    anyio.run(main_async)


if __name__ == "__main__":
    main()
```

---

### Patch 3: Bounded Embedding Download (backend/api/app.py)

Replace `_prefetch_embedding_assets()` with:

```python
def _prefetch_embedding_assets() -> None:
    """
    Best-effort prefetch of embedding assets.

    The download is BOUNDED and NON-BLOCKING:
    - Runs in background daemon thread
    - Times out after 30 seconds if download hangs
    - App becomes READY even if embeddings are still downloading
    """

    def worker() -> None:
        if _boot_trace_available:
            try:
                log_boot_step("embedding_prefetch_start")
            except Exception:
                pass

        from backend.api.deps import AppDependencies
        import signal

        class TimeoutError(Exception):
            pass

        def timeout_handler(signum, frame):
            raise TimeoutError("Download timeout")

        model_dir = AppDependencies.nomic_model_dir()
        cfg = NomicOnnxConfig()
        required = [cfg.model_filename, "tokenizer.json"]

        if _boot_trace_available:
            try:
                log_boot_step("embedding_prefetch_checking",
                              model_dir=str(model_dir),
                              required_files=str(required))
            except Exception:
                pass

        try:
            # Set a timeout alarm (Unix only)
            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(30)  # 30 second timeout

            try:
                ensure_local_nomic_model_files(model_dir, required_paths=required)
                logger.info("Embedding assets ready at %s", model_dir)
                if _boot_trace_available:
                    try:
                        log_boot_step("embedding_prefetch_complete",
                                      status="OK",
                                      model_dir=str(model_dir))
                    except Exception:
                        pass
            finally:
                if hasattr(signal, 'SIGALRM'):
                    signal.alarm(0)  # Cancel alarm

        except TimeoutError as exc:
            logger.warning("Embedding prefetch timeout (will continue in background): %s", exc)
            if _boot_trace_available:
                try:
                    log_boot_step("embedding_prefetch_timeout",
                                  status="TIMEOUT",
                                  note="Download will continue in background")
                except Exception:
                    pass
        except MissingDependencyError as exc:
            logger.warning("Embedding prefetch skipped (missing dependency): %s", exc)
        except Exception as exc:
            logger.warning("Embedding prefetch failed: %s", exc)

    global _embedding_prefetch_started
    if _embedding_prefetch_started:
        return
    _embedding_prefetch_started = True
    threading.Thread(target=worker, name="insight-prefetch-embeddings", daemon=True).start()
```

---

## Build Commands

### macOS arm64 (Apple Silicon)

```bash
#!/bin/bash
set -e

cd /Users/nelson/py/insight/insight_app

# Activate Python environment
conda activate ml

# Verify dependencies
echo "=== Verifying Python dependencies ==="
pip list | grep -E "(onnxruntime|llama-cpp-python|tokenizers|certifi|fastapi)"

# Build Python backend with PyInstaller
echo ""
echo "=== Building Python backend ==="
pyinstaller --noconfirm --clean backend/insight_engine_portable.spec

# Verify binary was created
if [ ! -f "dist/insight-engine" ]; then
    echo "ERROR: Binary not created!"
    exit 1
fi

echo "✓ Binary created: dist/insight-engine"

# Run smoke test
echo ""
echo "=== Running smoke test ==="
if ! ./dist/insight-engine --smoketest; then
    echo "ERROR: Smoke test failed!"
    exit 1
fi

echo "✓ Smoke test passed"

# Copy to Tauri bin directory (arm64)
ARCH="aarch64-apple-darwin"
cp dist/insight-engine "insight/src-tauri/bin/insight-engine-$ARCH"
echo "✓ Copied to insight/src-tauri/bin/insight-engine-$ARCH"

# Build Tauri app
echo ""
echo "=== Building Tauri app ==="
cd insight
npm run tauri build

# Output
DMG_PATH="src-tauri/target/release/bundle/dmg/Insight_<version>_aarch64.dmg"
echo ""
echo "=== Build complete ==="
echo "DMG: $DMG_PATH"
```

### macOS x86_64 (Intel)

Same as arm64, but the binary name is `insight-engine-x86_64-apple-darwin`.

### Universal Binary

Onedir bundles are directories, so `lipo` is not applicable. Build both
architectures and use Tauri's `--target universal-apple-darwin` to produce
a universal app bundle.

---

## Files Created/Modified

### Created Files:
1. `backend/services/boot_trace.py` - Boot trace logging
2. `backend/runtime_utils.py` - Runtime utilities for packaged apps
3. `backend/smoke_test.py` - Comprehensive smoke tests
4. `backend/pyi_rth_insight.py` - PyInstaller runtime hook for SSL
5. `backend/insight_engine_portable.spec` - Portable PyInstaller spec
6. `backend/api/routers/diagnostics.py` - Diagnostics endpoints
7. `insight/src-tauri/ENGINE_PROCESS_PATCH.md` - Rust patch documentation
8. `FRESH_MACHINE_CHECKLIST.md` - Complete verification checklist

### Modified Files:
1. `backend/engine.py` - Added --smoketest support
2. `backend/api/app.py` - Added boot trace, bounded prefetch, diagnostics router
3. `backend/api/deps.py` - Added instrumentation to dependency loading

---

## Testing Checklist

### Before Packaging:
```bash
# Test in dev mode
cd /Users/nelson/py/insight/insight_app
python3 backend/engine.py
# Check: ~/.insight/logs/boot_trace_*.log

# Run smoke test
python3 backend/engine.py --smoketest
# Expected: All 8 tests pass
```

### After Packaging:
```bash
# Test packaged binary
./dist/insight-engine --smoketest
# Expected: All 8 tests pass

# Test diagnostics
./dist/insight-engine &
PID=$!
sleep 2
curl http://localhost:XXXX/diagnostics/packaging | jq .
kill $PID
```

### On Fresh Machine:
1. Install DMG
2. Launch app
3. Check boot trace: `cat ~/.insight/logs/boot_trace_*.log`
4. Run smoke test from bundled binary
5. Follow `FRESH_MACHINE_CHECKLIST.md`

---

## Quick Reference

### Key Environment Variables:
- `INSIGHT_WORKSPACE_DIR` - Set by Rust, points to `~/.insight`
- `INSIGHT_SMOKETEST` - Run smoke test on startup (for testing)
- `REQUESTS_CA_BUNDLE` - Set by runtime hook to bundled certifi
- `SSL_CERT_FILE` - Set by runtime hook to bundled certifi

### Key Log Files:
- `~/.insight/logs/boot_trace_*.log` - Boot trace with timestamps
- `~/.insight/logs/backend_stderr_*.log` - Python stderr output

### Key Endpoints:
- `/settings/health` - Startup health check
- `/diagnostics/packaging` - Full packaging diagnostics
- `/diagnostics/smoketest` - Run smoke tests via HTTP

### Exit Codes:
- `0` - Success
- `1` - Failure (smoke test, critical error)

---

## Troubleshooting

### "Backend failed to start within 30 seconds"
- Check `~/.insight/logs/backend_stderr_*.log`
- Look for import errors
- Verify PyInstaller bundled all dependencies

### "SSL: CERTIFICATE_VERIFY_FAILED"
- Check if certifi bundle exists
- Verify runtime hook is being used
- Check `REQUESTS_CA_BUNDLE` env var

### "ImportError: cannot import name '_pybind_state'"
- ONNX Runtime native libraries not bundled
- Rebuild with `collect_dynamic_libs('onnxruntime')`

### "Symbol not found: _llama_free"
- llama_cpp native libraries not bundled
- Rebuild with `collect_dynamic_libs('llama_cpp')`

### "Infinite loading screen"
- Backend hung during startup
- Check boot trace for last successful step
- Check stderr logs for errors

---

## Success Criteria

✅ No crashes during normal operation
✅ Boot trace shows all steps completing
✅ Smoke test passes (8/8 tests)
✅ SSL certificates verified
✅ Native libraries load successfully
✅ Embedding download completes or times out gracefully
✅ App starts in < 10 seconds
✅ Memory usage < 500MB at idle
✅ Clean uninstall removes all traces

---

## Next Steps

1. **Apply Rust patches** to `insight/src-tauri/src/engine.rs`
2. **Build with PyInstaller** using `insight_engine_portable.spec`
3. **Run smoke test** on packaged binary
4. **Test on fresh machine** following `FRESH_MACHINE_CHECKLIST.md`
5. **Fix any issues** discovered during testing
6. **Rebuild and retest** until all checks pass
7. **Release** when all success criteria are met
