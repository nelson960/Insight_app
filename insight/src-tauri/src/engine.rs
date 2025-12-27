use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::mpsc;
use std::sync::Mutex;
use std::sync::{Arc, OnceLock};
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

pub struct EngineProcess {
    child: Child,
    stdin: Mutex<ChildStdin>,
    state: Arc<RouterState>,
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
        let engine_path = project_root.join("backend").join("engine.py");

        let mut child = Command::new(python_bin)
            .arg(&engine_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            // Python engine logs to stderr, so inherit it for debugging
            .stderr(Stdio::inherit())
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

        // Single stdout reader thread that demuxes all responses/tokens by request_id.
        // This is required to support /chat streaming + concurrent /files/* requests.
        spawn_stdout_router(BufReader::new(stdout), state.clone());

        Ok(Self {
            child,
            stdin: Mutex::new(stdin),
            state,
        })
    }

    pub fn set_app_handle(&self, app: tauri::AppHandle) {
        if let Ok(mut slot) = self.state.app.lock() {
            *slot = Some(app);
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

        let line = serde_json::to_string(&req)?;
        {
            let mut stdin = self.stdin.lock().map_err(|_| anyhow!("stdin mutex poisoned"))?;
            writeln!(stdin, "{line}")?;
            stdin.flush()?;
        }

        // Wait for the router thread to deliver the response for this request id.
        match rx.recv_timeout(Duration::from_secs(120)) {
            Ok(res) => Ok(res),
            Err(mpsc::RecvTimeoutError::Timeout) => {
                // Clean up so future responses don't leak.
                if let Ok(mut pending) = self.state.pending.lock() {
                    pending.remove(&request_id);
                }
                Err(anyhow!("timeout waiting for engine response request_id={request_id}"))
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
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
            writeln!(stdin, "{line}")?;
            stdin.flush()?;
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
}

impl Drop for EngineProcess {
    fn drop(&mut self) {
        if let Ok(mut stdin) = self.stdin.lock() {
            let _ = writeln!(stdin, r#"{{"cmd":"shutdown"}}"#);
        }
        let _ = self.child.kill();
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
                    let slot = state.app.lock().ok();
                    slot.and_then(|s| s.clone())
                };
                if let Some(app) = app {
                    let name = parsed.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    let _ = app.emit("engine-event", parsed.clone());
                    if name == "files_changed" {
                        let _ = app.emit("files-changed", parsed.clone());
                    } else if name == "file_text_ready" {
                        let _ = app.emit("file-text-ready", parsed.clone());
                    } else if name == "file_status" {
                        let _ = app.emit("file-status", parsed.clone());
                    } else if name == "llm_stream_end" {
           				let _ = app.emit("llm_stream_end", parsed.clone());
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
                    let _ = sink.app.emit(
                        "llm-error",
                        serde_json::json!({
                            "request_id": rid,
                            "chat_id": sink.chat_id,
                            "error": err,
                        }),
                    );
                }

                if let Some(token) = parsed.get("stream_token").and_then(|v| v.as_str()) {
                    if !token.contains("[DONE]") && !token.is_empty() {
                        let _ = sink.app.emit(
                            "llm-token",
                            serde_json::json!({
                                "token": token,
                                "chat_id": sink.chat_id,
                                "request_id": rid,
                            }),
                        );
                    }
                }

                if parsed.get("stream_end").is_some() {
                    let _ = sink.app.emit(
                        "llm-done",
                        serde_json::json!({ "request_id": rid, "chat_id": sink.chat_id }),
                    );
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
