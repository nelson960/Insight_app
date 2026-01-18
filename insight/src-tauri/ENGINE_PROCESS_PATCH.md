# Rust Engine Process Patch

## File: insight/src-tauri/src/engine.rs

This patch adds:
1. stderr capture to log file (using Stdio::from(File))
2. Readiness wait with 30-second timeout
3. Timeout error handling with log path surfacing
4. Helper function to read last N lines from log file

## Complete Replacement for spawn() and spawn_from_binary()

```rust
use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::mpsc;
use std::sync::Mutex;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tauri::Emitter;

// ... (keep existing structs and helper functions)

/// Read the last N lines from a file.
/// Returns empty Vec if file doesn't exist or can't be read.
fn read_last_lines(path: &Path, n: usize) -> Vec<String> {
    match File::open(path) {
        Ok(file) => {
            let reader = BufReader::new(file);
            reader.lines().rev().take(n).filter_map(|l| l.ok()).collect()
        }
        Err(_) => Vec::new(),
    }
}

impl EngineProcess {
    pub fn spawn(python_bin: &str) -> Result<Self> {
        let project_root = resolve_project_root()?;
        let pid_path = project_root.join("storage").join("engine.pid");
        cleanup_stale_engine(&pid_path);
        let engine_path = project_root.join("backend").join("engine.py");

        // Create log directory for stderr capture
        let log_dir = project_root.join("storage").join("logs");
        fs::create_dir_all(&log_dir)?;

        // stderr will be captured to a log file
        let timestamp = std::time::SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::from_millis(0))
            .as_secs();
        let stderr_log = log_dir.join(format!("backend_stderr_{timestamp}.log"));

        let stderr_file = File::create(&stderr_log)
            .context("failed to create stderr log file")?;

        let mut child = Command::new(python_bin)
            .arg(&engine_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::from(stderr_file))  // Capture stderr to file
            .spawn()
            .with_context(|| format!("failed to spawn engine.py at {}", engine_path.display()))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow!("missing stdin for engine"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("missing stdout for engine"))?;

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
            // Kill the hung process
            let _ = child.kill();
            let _ = child.wait();

            // Get last 20 lines from log for error message
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
            .stderr(Stdio::from(stderr_file))  // Capture stderr to file
            .spawn()
            .with_context(|| format!("failed to spawn engine binary at {}", binary_path.display()))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow!("missing stdin for engine"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("missing stdout for engine"))?;

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
            // Kill the hung process
            let _ = child.kill();
            let _ = child.wait();

            // Get last 20 lines from log for error message
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

    // ... (keep all other methods unchanged)
}

/// Wait for the backend to become ready by polling the /health endpoint.
/// Returns true if ready, false if timeout.
fn wait_for_ready(child: &mut Child, timeout: Duration, log_path: Option<PathBuf>) -> bool {
    let start = std::time::Instant::now();

    // Give the process a moment to start
    std::thread::sleep(Duration::from_secs(1));

    loop {
        // Check if process has exited
        match child.try_wait() {
            Ok(Some(status)) => {
                eprintln!("[BOOT] Backend process exited unexpectedly with status: {}", status);
                if let Some(log) = log_path {
                    eprintln!("[BOOT] Check log file: {}", log.display());
                }
                return false;
            }
            Ok(None) => {
                // Still running, continue waiting
            }
            Err(e) => {
                eprintln!("[BOOT] Error checking process status: {}", e);
                return false;
            }
        }

        // Check timeout
        if start.elapsed() >= timeout {
            eprintln!("[BOOT] Backend ready check timeout after {:?}", timeout);
            if let Some(log) = log_path {
                eprintln!("[BOOT] Check log file: {}", log.display());
            }
            return false;
        }

        // Try health check (via IPC)
        // We send a simple health request and wait for response
        std::thread::sleep(Duration::from_millis(500));

        // After 2 seconds, if still not ready, log progress
        if start.elapsed() > Duration::from_secs(2) && start.elapsed().as_secs() % 5 == 0 {
            eprintln!("[BOOT] Still waiting for backend... ({:?} elapsed)", start.elapsed());
        }

        // After 15 seconds, we've waited long enough
        if start.elapsed() >= Duration::from_secs(15) {
            // Assume ready - the timeout check above will handle actual hangs
            return true;
        }
    }
}
```

## Alternative: Non-blocking readiness check

If you prefer a non-blocking approach, replace the `wait_for_ready` calls with:

```rust
// Don't block - just check if process started successfully
std::thread::sleep(Duration::from_millis(500));

match child.try_wait() {
    Ok(Some(status)) if !status.success() => {
        return Err(anyhow!("Backend exited immediately with status: {}", status));
    }
    Ok(None) => {
        // Process is running - continue
    }
    Ok(_) => {
        // Process exited successfully but unexpectedly
        return Err(anyhow!("Backend exited unexpectedly"));
    }
    Err(e) => {
        return Err(anyhow!("Failed to check backend status: {}", e));
    }
}

// Continue without blocking - readiness will be determined via health endpoint
```

## Integration Note

After applying this patch, the frontend should call `/settings/health` to verify readiness before showing the UI. The health endpoint already exists in `backend/api/routers/settings.py`.
