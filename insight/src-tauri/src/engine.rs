use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::VecDeque;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::mpsc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::sync::{Arc, OnceLock};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tauri::Emitter;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EngineRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    pub endpoint: String,
    #[serde(default = "default_method")]
    pub method: String,
    #[serde(default)]
    pub payload: Value,
    #[serde(default = "default_stream")]
    pub stream: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EngineResponse {
    pub ok: bool,
    pub status: u16,
    pub data: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

fn default_method() -> String {
    "POST".to_string()
}

fn default_stream() -> bool {
    false
}

/// Get the path to the engine stderr log file.
///
/// In production, this is ~/.insight/logs/backend_stderr_<timestamp>.log
/// In development, this is storage/logs/backend_stderr_<timestamp>.log
fn stderr_log_path() -> PathBuf {
    if let Some(home) = dirs::home_dir() {
        let workspace = home.join(".insight");
        let logs_dir = workspace.join("logs");
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::from_millis(0))
            .as_secs();
        logs_dir.join(format!("backend_stderr_{}.log", timestamp))
    } else {
        // Fallback for dev
        let project_root = resolve_project_root().unwrap_or_else(|_| PathBuf::from("."));
        let logs_dir = project_root.join("storage").join("logs");
        let _ = fs::create_dir_all(&logs_dir);
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::from_millis(0))
            .as_secs();
        logs_dir.join(format!("backend_stderr_{}.log", timestamp))
    }
}

/// Read the last N lines from a file efficiently.
///
/// Returns an empty string if the file doesn't exist or cannot be read.
/// Uses a circular buffer to avoid storing all lines in memory.
fn read_last_n_lines(path: &Path, n: usize) -> String {
    let file = match File::open(path) {
        Ok(f) => f,
        Err(_) => return String::new(),
    };

    let reader = BufReader::new(file);
    let mut lines: VecDeque<String> = VecDeque::with_capacity(n);

    // Use circular buffer to only keep last N lines
    for line in reader.lines().filter_map(|l| l.ok()) {
        if lines.len() == n {
            lines.pop_front();
        }
        lines.push_back(line);
    }

    lines.into_iter().collect::<Vec<_>>().join("\n")
}

fn should_wait_for_engine_ready() -> bool {
    // Environment variable override (0/false/no = disable waiting, anything else = enable)
    if let Ok(value) = std::env::var("INSIGHT_ENGINE_WAIT_READY") {
        let normalized = value.trim().to_ascii_lowercase();
        return normalized != "0" && normalized != "false" && normalized != "no";
    }
    // Default to always waiting - both dev and production need engine ready
    true
}

fn get_engine_ready_timeout_secs() -> u64 {
    if let Ok(value) = std::env::var("INSIGHT_ENGINE_READY_TIMEOUT_SECS") {
        if let Ok(secs) = value.trim().parse::<u64>() {
            return secs.max(5); // Minimum 5 seconds
        }
    }
    30 // Default: 30 seconds (faster than old 60s)
}

fn get_watchdog_interval_secs() -> u64 {
    if let Ok(value) = std::env::var("INSIGHT_WATCHDOG_INTERVAL_SECS") {
        if let Ok(secs) = value.trim().parse::<u64>() {
            return secs.max(1);
        }
    }
    3
}

/// Wait for the engine to become ready by polling the stdout stream.
///
/// This checks for the "ASGI engine ready" message in stderr.
/// Uses a 30-second default timeout (configurable via INSIGHT_ENGINE_READY_TIMEOUT_SECS).
/// Optimized to only read new content from the log file each iteration.
fn wait_for_engine_ready(stderr_log_path: &Path, timeout_secs: u64) -> Result<()> {
    let start = SystemTime::now();
    let timeout = Duration::from_secs(timeout_secs);
    let mut last_size = 0u64;
    let mut file_ready_found = false;

    loop {
        let elapsed = start.elapsed().unwrap_or(Duration::from_millis(0));
        if elapsed > timeout {
            // Timeout: surface the last 50 lines of stderr for debugging
            let last_lines = read_last_n_lines(stderr_log_path, 50);
            return Err(anyhow!(
                "Engine failed to become ready within {}s\n\n\
                 Last 50 lines of stderr log ({}):\n\
                 ---\n{}\n---",
                timeout_secs,
                stderr_log_path.display(),
                if last_lines.is_empty() {
                    "(log file empty or missing)"
                } else {
                    &last_lines
                }
            ));
        }

        // Efficiently check only new content in the log file
        let metadata = match fs::metadata(stderr_log_path) {
            Ok(meta) => meta,
            Err(_) => {
                thread::sleep(Duration::from_millis(100));
                continue;
            }
        };

        let current_size = metadata.len();

        // Only read if file has grown (avoid re-reading entire file)
        if current_size > last_size || !file_ready_found {
            if let Ok(content) = fs::read_to_string(stderr_log_path) {
                if content.contains("ASGI engine ready") || content.contains("startup_health_check") {
                    eprintln!("[INFO] Engine ready in {:.1}s", elapsed.as_secs_f64());
                    return Ok(());
                }
                file_ready_found = true; // File exists, check it again
            }
            last_size = current_size;
        }

        thread::sleep(Duration::from_millis(100));
    }
}

/// Resolve the project root.
///
/// When running dev, cwd is usually insight/src-tauri. The project root is two levels up.
/// When running a bundled app, cwd is the exe dir; in that case we just walk up until we find
/// a marker (e.g. backend/ or storage/).
pub fn resolve_project_root() -> Result<PathBuf> {
    let mut root = std::env::current_dir()?;

    // Dev: /path/to/insight_app/insight/src-tauri -> /path/to/insight_app
    if root.ends_with("src-tauri") {
        if let Some(parent) = root.parent().and_then(|p| p.parent()) {
            return Ok(parent.to_path_buf());
        }
    }

    // Try to detect project root heuristically in other cases
    for _ in 0..5 {
        if root.join("backend").exists() && root.join("storage").exists() {
            return Ok(root);
        }
        if !root.pop() {
            break;
        }
    }

    Err(anyhow!("Could not resolve project root from cwd={}", std::env::current_dir()?.display()))
}

/// Helper function to retry an operation with exponential backoff
fn retry_with_backoff<T, E>(
    operation: impl Fn() -> Result<T, E>,
    max_retries: u32,
    initial_delay_ms: u64,
    context: &str,
) -> Result<T, E>
where
    E: std::fmt::Display,
{
    let mut delay = initial_delay_ms;

    for attempt in 0..=max_retries {
        match operation() {
            Ok(result) => {
                if attempt > 0 {
                    eprintln!("[INFO] {} succeeded on attempt {}", context, attempt + 1);
                }
                return Ok(result);
            }
            Err(e) => {
                if attempt < max_retries {
                    eprintln!(
                        "[WARNING] {} failed (attempt {}/{}), retrying in {}ms: {}",
                        context,
                        attempt + 1,
                        max_retries + 1,
                        delay,
                        e
                    );
                    thread::sleep(Duration::from_millis(delay));
                    delay = (delay * 2).min(5000); // Exponential backoff, max 5s
                } else {
                    eprintln!("[ERROR] {} failed after {} attempts: {}", context, max_retries + 1, e);
                    return Err(e);
                }
            }
        }
    }
    unreachable!()
}

pub struct EngineProcess {
    child: Arc<Mutex<Child>>,
    stdin: Arc<Mutex<ChildStdin>>,
    state: Arc<RouterState>,
    crashed: Arc<Mutex<bool>>,
    crash_reason: Arc<Mutex<Option<String>>>,
    watchdog_started: AtomicBool,
}

struct RouterState {
    pending: Mutex<std::collections::HashMap<String, mpsc::Sender<EngineResponse>>>,
    streams: Mutex<std::collections::HashMap<String, StreamSink>>,
    app: Mutex<Option<tauri::AppHandle>>,
}

#[derive(Clone)]
struct StreamSink {
    app: tauri::AppHandle,
    chat_id: Option<String>,
}

static REQUEST_SEQ: OnceLock<std::sync::atomic::AtomicU64> = OnceLock::new();

fn next_request_id() -> String {
    let seq = REQUEST_SEQ
        .get_or_init(|| std::sync::atomic::AtomicU64::new(1))
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::from_millis(0))
        .as_millis();
    format!("req-{now}-{seq}")
}

impl EngineProcess {
    pub fn spawn(python_bin: &str) -> Result<Self> {
        let project_root = resolve_project_root()?;
        // Remove legacy pid file if present (no longer used).
        let _ = fs::remove_file(project_root.join("storage").join("engine.pid"));

        // Create logs directory and open stderr log file
        let logs_dir = project_root.join("storage").join("logs");
        fs::create_dir_all(&logs_dir)?;
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::from_millis(0))
            .as_secs();
        let stderr_path = logs_dir.join(format!("backend_stderr_{}.log", timestamp));

        // IMPORTANT:
        // Run as a module from project_root so `import backend.*` works.
        // Also force unbuffered output for reliable IPC.
        // Use retry logic with exponential backoff for transient failures
        let python_bin = python_bin.to_string();
        let project_root_clone = project_root.clone();
        let stderr_path_clone = stderr_path.clone();

        let mut child = retry_with_backoff(
            || {
                let stderr_file = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&stderr_path_clone)
                    .with_context(|| format!("failed to open stderr log at {}", stderr_path_clone.display()))?;

                Command::new(&python_bin)
                    .current_dir(&project_root_clone)
                    .env("PYTHONPATH", project_root_clone.to_string_lossy().to_string())
                    .env("PYTHONUNBUFFERED", "1")
                    .arg("-u")
                    .arg("-m")
                    .arg("backend.engine")
                    .env("INSIGHT_WORKSPACE_DIR", project_root_clone.join("storage"))
                    .stdin(Stdio::piped())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::from(stderr_file))
                    .spawn()
                    .with_context(|| {
                        format!(
                            "failed to spawn python engine (python_bin={}, cwd={})",
                            python_bin,
                            project_root_clone.display()
                        )
                    })
            },
            2, // Max 2 retries (3 total attempts)
            500, // Start with 500ms delay
            "Python engine spawn"
        )?;

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

        // Wait for engine to become ready (default 30s, configurable via INSIGHT_ENGINE_READY_TIMEOUT_SECS)
        if should_wait_for_engine_ready() {
            wait_for_engine_ready(&stderr_path, get_engine_ready_timeout_secs())?;
        }

        Ok(Self {
            child: Arc::new(Mutex::new(child)),
            stdin: Arc::new(Mutex::new(stdin)),
            state,
            crashed: Arc::new(Mutex::new(false)),
            crash_reason: Arc::new(Mutex::new(None)),
            watchdog_started: AtomicBool::new(false),
        })
    }

    /// Spawn the bundled sidecar binary (for production builds).
    ///
    /// This is used when the app is bundled - the sidecar binary is already
    /// compiled and included in the .app bundle. We just need to execute it
    /// and set the INSIGHT_WORKSPACE_DIR environment variable.
    pub fn spawn_from_binary(binary_path: &Path) -> Result<Self> {
        let workspace_dir = dirs::home_dir()
            .ok_or_else(|| anyhow!("couldn't find home dir"))?
            .join(".insight");
        // Remove legacy pid file if present (no longer used).
        let _ = fs::remove_file(workspace_dir.join("engine.pid"));

        // Create logs directory and open stderr log file
        let logs_dir = workspace_dir.join("logs");
        fs::create_dir_all(&logs_dir)?;
        let stderr_path = stderr_log_path();
        let binary_path_clone = binary_path.to_path_buf();
        let workspace_dir_clone = workspace_dir.clone();
        let stderr_path_clone = stderr_path.clone();

        let mut child = retry_with_backoff(
            || {
                let stderr_file = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&stderr_path_clone)
                    .with_context(|| format!("failed to open stderr log at {}", stderr_path_clone.display()))?;

                Command::new(&binary_path_clone)
                    .env("INSIGHT_WORKSPACE_DIR", &workspace_dir_clone)
                    .stdin(Stdio::piped())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::from(stderr_file))
                    .spawn()
                    .with_context(|| format!("failed to spawn engine binary at {}", binary_path_clone.display()))
            },
            2, // Max 2 retries (3 total attempts)
            500, // Start with 500ms delay
            "Bundled engine spawn"
        )?;

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

        // Single stdout reader thread that demuxes all responses/tokens by request_id.
        // This is required to support /chat streaming + concurrent /files/* requests.
        spawn_stdout_router(BufReader::new(stdout), state.clone());

        // Wait for engine to become ready (default 30s, configurable via INSIGHT_ENGINE_READY_TIMEOUT_SECS)
        if should_wait_for_engine_ready() {
            wait_for_engine_ready(&stderr_path, get_engine_ready_timeout_secs())?;
        }

        Ok(Self {
            child: Arc::new(Mutex::new(child)),
            stdin: Arc::new(Mutex::new(stdin)),
            state,
            crashed: Arc::new(Mutex::new(false)),
            crash_reason: Arc::new(Mutex::new(None)),
            watchdog_started: AtomicBool::new(false),
        })
    }

    pub fn set_app_handle(&self, app: tauri::AppHandle) {
        if let Ok(mut slot) = self.state.app.lock() {
            *slot = Some(app);
        }
        if self
            .watchdog_started
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            if let Ok(slot) = self.state.app.lock() {
                if let Some(app) = slot.as_ref() {
                    spawn_process_watchdog(
                        Arc::clone(&self.child),
                        Arc::clone(&self.crashed),
                        Arc::clone(&self.crash_reason),
                        app.clone(),
                    );
                }
            }
        }
    }

    pub fn is_crashed(&self) -> bool {
        self.crashed.lock().map(|c| *c).unwrap_or(false)
    }

    pub fn get_crash_reason(&self) -> Option<String> {
        self.crash_reason.lock().ok()?.clone()
    }

    fn emit_crash_event(&self, message: &str, suggestion: Option<&str>) {
        let payload = serde_json::json!({
            "message": message,
            "suggestion": suggestion.unwrap_or("Please restart the application. If this persists, try reducing GPU layers in Settings."),
        });
        if let Ok(slot) = self.state.app.lock() {
            if let Some(app) = slot.as_ref() {
                let _ = app.emit("engine-crashed", payload);
            }
        }
    }

    pub fn mark_crashed(&self, reason: String) {
        let mut should_emit = false;
        if let Ok(mut crashed) = self.crashed.lock() {
            if !*crashed {
                *crashed = true;
                should_emit = true;
            }
        }
        if let Ok(mut reason_guard) = self.crash_reason.lock() {
            *reason_guard = Some(reason.clone());
        }
        if should_emit {
            self.emit_crash_event(&reason, None);
        }
    }

    /// Non-streaming request: send one JSON and wait for one JSON line response.
    pub fn send(&self, req: &EngineRequest) -> Result<EngineResponse> {
        let mut req = req.clone();
        if req.request_id.is_none() {
            req.request_id = Some(next_request_id());
        }
        let request_id = req
            .request_id
            .clone()
            .ok_or_else(|| anyhow!("missing request_id"))?;

        let (tx, rx) = mpsc::channel::<EngineResponse>();
        {
            let mut pending = self
                .state
                .pending
                .lock()
                .map_err(|_| anyhow!("pending mutex poisoned"))?;
            pending.insert(request_id.clone(), tx);
        }

        if self.is_crashed() {
            if let Ok(mut pending) = self.state.pending.lock() {
                pending.remove(&request_id);
            }
            let reason = self
                .get_crash_reason()
                .unwrap_or_else(|| "Engine has crashed".to_string());
            return Err(anyhow!(reason));
        }

        let line = serde_json::to_string(&req)?;
        {
            let mut stdin = self.stdin.lock().map_err(|_| anyhow!("stdin mutex poisoned"))?;
            match writeln!(stdin, "{line}") {
                Ok(_) => {}
                Err(e) if e.kind() == ErrorKind::BrokenPipe => {
                    self.mark_crashed("Broken pipe: engine process died unexpectedly".to_string());
                    if let Ok(mut pending) = self.state.pending.lock() {
                        pending.remove(&request_id);
                    }
                    return Err(anyhow!(
                        "Broken pipe: engine has crashed. Please restart the application."
                    ));
                }
                Err(e) => {
                    if let Ok(mut pending) = self.state.pending.lock() {
                        pending.remove(&request_id);
                    }
                    return Err(anyhow!("Failed to write to engine stdin: {}", e));
                }
            }
            if let Err(e) = stdin.flush() {
                if e.kind() == ErrorKind::BrokenPipe {
                    self.mark_crashed("Broken pipe: engine process died unexpectedly".to_string());
                    if let Ok(mut pending) = self.state.pending.lock() {
                        pending.remove(&request_id);
                    }
                    return Err(anyhow!(
                        "Broken pipe: engine has crashed. Please restart the application."
                    ));
                }
                if let Ok(mut pending) = self.state.pending.lock() {
                    pending.remove(&request_id);
                }
                return Err(anyhow!("Failed to flush engine stdin: {}", e));
            }
        }

        // Wait for the router thread to deliver the response for this request id.
        match rx.recv_timeout(Duration::from_secs(120)) {
            Ok(res) => Ok(res),
            Err(mpsc::RecvTimeoutError::Timeout) => {
                // Clean up pending entry so future responses don't leak.
                // The channel sender (tx) will be dropped when this function returns.
                eprintln!("[ERROR] Engine request timeout: request_id={request_id}");
                if let Ok(mut pending) = self.state.pending.lock() {
                    pending.remove(&request_id);
                }
                Err(anyhow!("timeout waiting for engine response request_id={request_id}"))
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                eprintln!("[ERROR] Engine channel disconnected: request_id={request_id}");
                if let Ok(mut pending) = self.state.pending.lock() {
                    pending.remove(&request_id);
                }
                Err(anyhow!("engine response channel disconnected request_id={request_id}"))
            }
        }
    }

    pub fn stream(
        &self,
        request_id: &str,
        endpoint: &str,
        payload: &Value,
        app: &tauri::AppHandle,
    ) -> Result<()> {
        let chat_id = payload
            .get("chat_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        {
            let mut streams = self
                .state
                .streams
                .lock()
                .map_err(|_| anyhow!("streams mutex poisoned"))?;
            streams.insert(
                request_id.to_string(),
                StreamSink {
                    app: app.clone(),
                    chat_id: chat_id.clone(),
                },
            );
        }

        if self.is_crashed() {
            if let Ok(mut streams) = self.state.streams.lock() {
                streams.remove(request_id);
            }
            let reason = self
                .get_crash_reason()
                .unwrap_or_else(|| "Engine has crashed".to_string());
            return Err(anyhow!(reason));
        }

        let mut payload = payload.clone();
        // Ensure backend sees the request id so it can cancel model generation.
        if let Some(obj) = payload.as_object_mut() {
            obj.entry("request_id".to_string())
                .or_insert(Value::String(request_id.to_string()));
        }

        let req = EngineRequest {
            request_id: Some(request_id.to_string()),
            endpoint: endpoint.to_string(),
            method: "POST".to_string(),
            payload,
            stream: true,
        };
        let line = serde_json::to_string(&req)?;
        {
            let mut stdin = self.stdin.lock().map_err(|_| anyhow!("stdin mutex poisoned"))?;
            match writeln!(stdin, "{line}") {
                Ok(_) => {}
                Err(e) if e.kind() == ErrorKind::BrokenPipe => {
                    self.mark_crashed("Broken pipe during stream start".to_string());
                    if let Ok(mut streams) = self.state.streams.lock() {
                        streams.remove(request_id);
                    }
                    return Err(anyhow!("Broken pipe: engine crashed"));
                }
                Err(e) => {
                    if let Ok(mut streams) = self.state.streams.lock() {
                        streams.remove(request_id);
                    }
                    return Err(anyhow!("Failed to write to engine stdin: {}", e));
                }
            }
            if let Err(e) = stdin.flush() {
                if e.kind() == ErrorKind::BrokenPipe {
                    self.mark_crashed("Broken pipe during stream start".to_string());
                    if let Ok(mut streams) = self.state.streams.lock() {
                        streams.remove(request_id);
                    }
                    return Err(anyhow!("Broken pipe: engine crashed"));
                }
                if let Ok(mut streams) = self.state.streams.lock() {
                    streams.remove(request_id);
                }
                return Err(anyhow!("Failed to flush engine stdin: {}", e));
            }
        }
        Ok(())
    }

    pub fn cancel(&self, request_id: &str) -> Result<()> {
        let line = serde_json::json!({
            "cmd": "cancel",
            "request_id": request_id,
        })
        .to_string();
        let mut stdin = self.stdin.lock().map_err(|_| anyhow!("stdin mutex poisoned"))?;
        writeln!(stdin, "{line}")?;
        stdin.flush()?;
        Ok(())
    }

    pub fn shutdown(&self) {
        if let Ok(mut stdin) = self.stdin.lock() {
            let _ = writeln!(stdin, r#"{{"cmd":"shutdown"}}"#);
            let _ = stdin.flush();
        }
        if let Ok(mut child) = self.child.lock() {
            // Give the engine a moment to shut down cleanly so it can
            // stop any child services (e.g., raw engine server).
            for _ in 0..20 {
                if let Ok(Some(_)) = child.try_wait() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl Drop for EngineProcess {
    fn drop(&mut self) {
        // Use try_lock to avoid potential deadlocks during shutdown
        // If another thread holds the lock, we'll gracefully skip cleanup
        if let Ok(mut stdin) = self.stdin.try_lock() {
            let _ = writeln!(stdin, r#"{{"cmd":"shutdown"}}"#);
            let _ = stdin.flush();
        } else {
            eprintln!("[WARNING] Failed to acquire stdin lock during drop, skipping shutdown command");
        }

        if let Ok(mut child) = self.child.try_lock() {
            for _ in 0..10 {
                if let Ok(Some(_)) = child.try_wait() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            let _ = child.kill();
            let _ = child.wait();
        } else {
            eprintln!("[WARNING] Failed to acquire child lock during drop, process may not be killed cleanly");
        }
    }
}


fn spawn_stdout_router(mut stdout: BufReader<ChildStdout>, state: Arc<RouterState>) {
    std::thread::spawn(move || {
        let debug = std::env::var("INSIGHT_IPC_DEBUG").is_ok();
        loop {
            let mut line = String::new();
            let read = match stdout.read_line(&mut line) {
                Ok(n) => n,
                Err(_) => break,
            };
            if read == 0 {
                break;
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            let parsed: serde_json::Value = match serde_json::from_str(trimmed) {
                Ok(v) => v,
                Err(_) => continue,
            };

            // Out-of-band events from the Python engine (not tied to a request_id).
            if parsed.get("type").and_then(|v| v.as_str()) == Some("event") {
                let app = {
                    match state.app.lock() {
                        Ok(guard) => guard.as_ref().cloned(),
                        Err(poisoned) => {
                            eprintln!("[WARNING] App mutex poisoned, recovering...");
                            // Recover from poisoned mutex - the data is still valid
                            poisoned.into_inner().as_ref().cloned()
                        }
                    }
                };
                if let Some(app) = app {
                    let name = parsed.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    if let Err(e) = app.emit("engine-event", parsed.clone()) {
                        eprintln!("[ERROR] Failed to emit engine-event: {}", e);
                    }
                    if name == "files_changed" {
                        if let Err(e) = app.emit("files-changed", parsed.clone()) {
                            eprintln!("[ERROR] Failed to emit files-changed: {}", e);
                        }
                    } else if name == "file_text_ready" {
                        if let Err(e) = app.emit("file-text-ready", parsed.clone()) {
                            eprintln!("[ERROR] Failed to emit file-text-ready: {}", e);
                        }
                    } else if name == "file_status" {
                        if let Err(e) = app.emit("file-status", parsed.clone()) {
                            eprintln!("[ERROR] Failed to emit file-status: {}", e);
                        }
                    } else if name == "llm_stream_end" {
                        if let Err(e) = app.emit("llm_stream_end", parsed.clone()) {
                            eprintln!("[ERROR] Failed to emit llm_stream_end: {}", e);
                        }
                    } else if name == "chat_sources" {
                        if let Err(e) = app.emit("chat-sources", parsed.clone()) {
                            eprintln!("[ERROR] Failed to emit chat-sources: {}", e);
                        }
                    }
                }
                continue;
            }
            let request_id = parsed
                .get("request_id")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());

            if debug {
                if let Some(rid) = &request_id {
                    eprintln!("[ipc] stdout <- request_id={} bytes={} line={}", rid, read, trimmed);
                } else {
                    eprintln!("[ipc] stdout <- bytes={} line={}", read, trimmed);
                }
            }

            // Streaming messages
            if parsed.get("stream_start").is_some()
                || parsed.get("stream_token").is_some()
                || parsed.get("stream_error").is_some()
                || parsed.get("stream_end").is_some()
            {
                let rid = match request_id {
                    Some(r) => r,
                    None => continue,
                };
                let sink = {
                    let streams = state.streams.lock().ok();
                    streams.and_then(|m| m.get(&rid).cloned())
                };
                let Some(sink) = sink else { continue };

                if let Some(err) = parsed.get("stream_error").and_then(|v| v.as_str()) {
                    if let Err(e) = sink.app.emit(
                        "llm-error",
                        serde_json::json!({
                            "request_id": rid,
                            "chat_id": sink.chat_id,
                            "error": err,
                        }),
                    ) {
                        eprintln!("[ERROR] Failed to emit llm-error for {}: {}", rid, e);
                    }
                }

                if let Some(token) = parsed.get("stream_token").and_then(|v| v.as_str()) {
                    if !token.contains("[DONE]") && !token.is_empty() {
                        if let Err(e) = sink.app.emit(
                            "llm-token",
                            serde_json::json!({
                                "token": token,
                                "chat_id": sink.chat_id,
                                "request_id": rid,
                            }),
                        ) {
                            eprintln!("[ERROR] Failed to emit llm-token for {}: {}", rid, e);
                        }
                    }
                }

                if parsed.get("stream_end").is_some() {
                    if let Err(e) = sink.app.emit(
                        "llm-done",
                        serde_json::json!({ "request_id": rid, "chat_id": sink.chat_id }),
                    ) {
                        eprintln!("[ERROR] Failed to emit llm-done for {}: {}", rid, e);
                    }
                    if let Ok(mut streams) = state.streams.lock() {
                        streams.remove(&rid);
                    }
                }
                continue;
            }

            // Non-stream response
            let rid = match request_id {
                Some(r) => r,
                None => continue,
            };
            let resp: EngineResponse = match serde_json::from_value(parsed) {
                Ok(r) => r,
                Err(_) => continue,
            };
            let tx = {
                let mut pending = match state.pending.lock() {
                    Ok(m) => m,
                    Err(_) => continue,
                };
                pending.remove(&rid)
            };
            if let Some(tx) = tx {
                let _ = tx.send(resp);
            }
        }
    });
}

fn spawn_process_watchdog(
    child: Arc<Mutex<Child>>,
    crashed: Arc<Mutex<bool>>,
    crash_reason: Arc<Mutex<Option<String>>>,
    app: tauri::AppHandle,
) {
    thread::spawn(move || {
        let check_interval = Duration::from_secs(get_watchdog_interval_secs());
        loop {
            thread::sleep(check_interval);

            if let Ok(c) = crashed.lock() {
                if *c {
                    break;
                }
            }

            let exit_status = match child.lock() {
                Ok(mut child_guard) => match child_guard.try_wait() {
                    Ok(Some(status)) => Some(status),
                    Ok(None) => None,
                    Err(_) => None,
                },
                Err(_) => None,
            };

            if let Some(status) = exit_status {
                let reason = if status.success() {
                    "Engine exited cleanly".to_string()
                } else {
                    format!("Engine crashed with code: {:?}", status)
                };

                if let Ok(mut c) = crashed.lock() {
                    *c = true;
                }
                if let Ok(mut r) = crash_reason.lock() {
                    *r = Some(reason.clone());
                }

                let _ = app.emit(
                    "engine-crashed",
                    serde_json::json!({
                        "message": reason,
                        "suggestion": "Please restart the application. If this persists, try reducing GPU layers in Settings.",
                    }),
                );
                break;
            }
        }
    });
}
