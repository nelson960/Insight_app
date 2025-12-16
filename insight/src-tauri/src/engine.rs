use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::Mutex;
use tauri::Emitter;

#[derive(Debug, Deserialize, Serialize)]
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
    stdout: Mutex<BufReader<ChildStdout>>,
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

        Ok(Self {
            child,
            stdin: Mutex::new(stdin),
            stdout: Mutex::new(BufReader::new(stdout)),
        })
    }

    /// Non-streaming request: send one JSON and wait for one JSON line response.
    pub fn send(&self, req: &EngineRequest) -> Result<EngineResponse> {
        let line = serde_json::to_string(req)?;
        {
            let mut stdin = self.stdin.lock().map_err(|_| anyhow!("stdin mutex poisoned"))?;
            writeln!(stdin, "{line}")?;
            stdin.flush()?;
        }

        let mut response = String::new();
        {
            let mut stdout = self
                .stdout
                .lock()
                .map_err(|_| anyhow!("stdout mutex poisoned"))?;
            stdout
                .read_line(&mut response)
                .context("engine did not return a response")?;
        }
        if response.trim().is_empty() {
            return Err(anyhow!(
                "engine returned empty response; check Python errors in stderr"
            ));
        }
        let val: EngineResponse = serde_json::from_str(&response)
            .with_context(|| format!("invalid JSON from engine: {response}"))?;
        Ok(val)
    }

    /// Streaming request: send one JSON, then consume streaming JSON lines:
    ///   {"stream_token": "<chunk>"}
    ///   {"stream_end": true}
    pub fn stream(
        &self,
        request_id: &str,
        endpoint: &str,
        payload: &Value,
        app: &tauri::AppHandle,
    ) -> Result<()> {
        let debug = std::env::var("INSIGHT_IPC_DEBUG").is_ok();

        let chat_id = payload
            .get("chat_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        if debug {
            eprintln!(
                "[ipc] stream -> request_id={} endpoint={} chat_id={:?}",
                request_id, endpoint, chat_id
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

        let mut stdout = self
            .stdout
            .lock()
            .map_err(|_| anyhow!("stdout mutex poisoned"))?;

        loop {
            let mut response = String::new();
            let read = stdout.read_line(&mut response)?;
            if read == 0 {
                // EOF from engine
                break;
            }
            let trimmed = response.trim();
            if trimmed.is_empty() {
                continue;
            }
            if debug {
                eprintln!("[ipc] stream <- bytes={} line={}", read, trimmed);
            }

            let parsed: serde_json::Value = match serde_json::from_str(trimmed) {
                Ok(v) => v,
                Err(_) => {
                    // Ignore malformed JSON lines (shouldn't happen, but be defensive)
                    continue;
                }
            };

            // Filter out interleaved messages from other requests.
            if parsed
                .get("request_id")
                .and_then(|v| v.as_str())
                .is_some_and(|rid| rid != request_id)
            {
                continue;
            }

            // End-of-stream marker
            if parsed.get("stream_end").is_some() {
                if debug {
                    eprintln!("[ipc] stream_end");
                }
                let _ = app.emit(
                    "llm-done",
                    serde_json::json!({ "request_id": request_id, "chat_id": chat_id }),
                );
                break;
            }

            if let Some(err) = parsed.get("stream_error").and_then(|v| v.as_str()) {
                if debug {
                    eprintln!("[ipc] stream_error={}", err);
                }
                let _ = app.emit(
                    "llm-error",
                    serde_json::json!({
                        "request_id": request_id,
                        "chat_id": chat_id,
                        "error": err,
                    }),
                );
            }

            // Normal token line
            if let Some(token) = parsed.get("stream_token").and_then(|v| v.as_str()) {
                // Backend emits a human-readable marker; Rust already gets `stream_end`.
                if token.contains("[DONE]") {
                    continue;
                }
                let payload_json = serde_json::json!({
                    "token": token,
                    "chat_id": chat_id,
                    "request_id": request_id,
                });
                if let Err(err) = app.emit("llm-token", payload_json) {
                    if debug {
                        eprintln!("[ipc] emit llm-token failed: {err}");
                    }
                }
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
}

impl Drop for EngineProcess {
    fn drop(&mut self) {
        if let Ok(mut stdin) = self.stdin.lock() {
            let _ = writeln!(stdin, r#"{{"cmd":"shutdown"}}"#);
        }
        let _ = self.child.kill();
    }
}
