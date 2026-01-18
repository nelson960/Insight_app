mod engine;
use engine::{resolve_project_root, EngineProcess, EngineRequest, EngineResponse};
use notify::{RecommendedWatcher, RecursiveMode, Watcher};
use serde_json::Value;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use tauri::{Emitter, Manager, RunEvent, WindowEvent};
use rusqlite::Connection;

type SharedEngine = Arc<EngineProcess>;

fn ipc_debug() -> bool {
    std::env::var("INSIGHT_IPC_DEBUG").is_ok()
}

/// Get the workspace directory for the current build type.
///
/// - Dev/debug builds: returns `<project_root>/storage`
/// - Release builds: returns `~/.insight`
fn get_workspace_dir() -> Result<PathBuf, String> {
    if cfg!(debug_assertions) {
        // Dev mode: use project-local storage
        let root = resolve_project_root()
            .map_err(|e| format!("Failed to resolve project root: {}", e))?;
        Ok(root.join("storage"))
    } else {
        // Release mode: use ~/.insight
        Ok(dirs::home_dir()
            .ok_or_else(|| String::from("couldn't find home dir"))?
            .join(".insight"))
    }
}

fn shutdown_engine(app: &tauri::AppHandle) {
    // Cleanup the file watcher if it exists
    if let Some(watcher_mutex) = app.try_state::<Mutex<notify::RecommendedWatcher>>() {
        if let Ok(_watcher) = watcher_mutex.lock() {
            // The watcher will be properly dropped when the mutex is released
            eprintln!("[INFO] File watcher cleaned up");
        }
    }

    // Shutdown the engine process
    let engine = app.state::<SharedEngine>();
    engine.shutdown();
}

fn _extract_title_from_content_json(content_json: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(content_json).ok()?;
    let text = v
        .get("text")
        .and_then(|t| t.as_str())
        .or_else(|| v.as_str())
        .unwrap_or("")
        .trim();
    if text.is_empty() {
        return None;
    }
    let first_line = text.lines().next().unwrap_or(text).trim();
    if first_line.is_empty() {
        return None;
    }
    // Keep titles compact for the sidebar.
    let mut out = first_line.to_string();
    const MAX: usize = 48;
    if out.chars().count() > MAX {
        out = out.chars().take(MAX).collect::<String>() + "…";
    }
    Some(out)
}

#[tauri::command]
async fn engine_request(
    endpoint: String,
    method: Option<String>,
    payload: Option<Value>,
    state: tauri::State<'_, SharedEngine>,
) -> Result<EngineResponse, String> {
    // Avoid spamming logs for the Settings busy poll endpoint.
    if ipc_debug() && endpoint != "/settings/busy" {
        eprintln!("[cmd] engine_request endpoint={endpoint}");
    }
    let engine = state.inner().clone();
    let req = EngineRequest {
        request_id: None,
        endpoint,
        method: method.unwrap_or_else(|| "POST".into()),
        payload: payload.unwrap_or(Value::Null),
        stream: false,
    };
    let res = tauri::async_runtime::spawn_blocking(move || {
        engine
            .send(&req)
            .map_err(|e| format!("Python engine error: {}", e))
    })
    .await
    .map_err(|e| format!("engine_request join error: {e}"))??;
    Ok(res)
}

#[tauri::command]
fn list_sessions() -> Result<Value, String> {
    let workspace = get_workspace_dir()?;
    let mut sessions = Vec::new();
    let mut seen = std::collections::HashSet::new();

    // Prefer SQLite transcript store for session listing (clean UI chats).
    let db_path = workspace.join("db.sqlite");
    if db_path.exists() {
        if let Ok(conn) = Connection::open(db_path) {
            // Query for chats with messages, including last message timestamp and content
            if let Ok(mut stmt) = conn.prepare(
                "SELECT chat_id, MAX(created_at) AS last_ts FROM messages GROUP BY chat_id ORDER BY last_ts DESC",
            ) {
                let rows = stmt.query_map([], |row| {
                    let chat_id: String = row.get(0)?;
                    let last_ts: String = row.get(1)?;
                    Ok((chat_id, last_ts))
                });
                if let Ok(rows) = rows {
                    for row in rows.flatten() {
                        let (chat_id, last_ts) = row;
                        if seen.insert(chat_id.clone()) {
                            let mut title: Option<String> = None;
                            if let Ok(mut tstmt) = conn.prepare(
                                "SELECT content_json FROM messages WHERE chat_id=? AND role='user' ORDER BY created_at ASC LIMIT 1",
                            ) {
                                if let Ok(mut trows) = tstmt.query([chat_id.clone()]) {
                                    if let Ok(Some(r)) = trows.next() {
                                        let content_json: String = r.get(0).unwrap_or_default();
                                        title = _extract_title_from_content_json(&content_json);
                                    }
                                }
                            }

                            // Get last message content (any role)
                            let mut last_message_content: Option<String> = None;
                            if let Ok(mut lstmt) = conn.prepare(
                                "SELECT content_json FROM messages WHERE chat_id=? ORDER BY created_at DESC LIMIT 1",
                            ) {
                                if let Ok(mut lrows) = lstmt.query([chat_id.clone()]) {
                                    if let Ok(Some(r)) = lrows.next() {
                                        let content_json: String = r.get(0).unwrap_or_default();
                                        // Extract text from content_json
                                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&content_json) {
                                            if let Some(text) = v.get("text").and_then(|t| t.as_str()) {
                                                last_message_content = Some(truncate_string(text, 200));
                                            } else if let Some(text) = v.as_str() {
                                                last_message_content = Some(truncate_string(text, 200));
                                            }
                                        }
                                    }
                                }
                            }

                            // Get file count
                            let mut file_count: i64 = 0;
                            if let Ok(mut fstmt) = conn.prepare(
                                "SELECT COUNT(DISTINCT cf.file_id) FROM chat_files cf WHERE cf.chat_id=?",
                            ) {
                                if let Ok(mut frows) = fstmt.query([chat_id.clone()]) {
                                    if let Ok(Some(r)) = frows.next() {
                                        file_count = r.get(0).unwrap_or(0);
                                    }
                                }
                            }

                            let mut session_obj = serde_json::json!({
                                "chat_id": chat_id,
                                "last_message_at": last_ts,
                                "file_count": file_count
                            });
                            if let Some(t) = title {
                                session_obj["title"] = serde_json::json!(t);
                            }
                            if let Some(msg) = last_message_content {
                                session_obj["last_message_content"] = serde_json::json!(msg);
                            }
                            sessions.push(session_obj);
                        }
                    }
                }
            }

            // Also include chats that have files/documents attached even if they have no messages yet.
            // This ensures cards persist when the user only uploads/edits documents without chatting.
            //
            // IMPORTANT: use `chat_files` mapping (multi-chat sharing) rather than legacy `files.chat_id`.
            if let Ok(mut stmt) = conn.prepare(
                "SELECT chat_id, MAX(created_at) AS last_ts FROM chat_files GROUP BY chat_id ORDER BY last_ts DESC",
            ) {
                let rows = stmt.query_map([], |row| {
                    let chat_id: String = row.get(0)?;
                    let last_ts: String = row.get(1)?;
                    Ok((chat_id, last_ts))
                });
                if let Ok(rows) = rows {
                    for row in rows.flatten() {
                        let (chat_id, last_ts) = row;
                        if !seen.insert(chat_id.clone()) {
                            continue;
                        }
                        // Best-effort title: first user message if present, else first filename.
                        let mut title: Option<String> = None;
                        if let Ok(mut tstmt) = conn.prepare(
                            "SELECT content_json FROM messages WHERE chat_id=? AND role='user' ORDER BY created_at ASC LIMIT 1",
                        ) {
                            if let Ok(mut trows) = tstmt.query([chat_id.clone()]) {
                                if let Ok(Some(r)) = trows.next() {
                                    let content_json: String = r.get(0).unwrap_or_default();
                                    title = _extract_title_from_content_json(&content_json);
                                }
                            }
                        }
                        if title.is_none() {
                            if let Ok(mut fstmt) = conn.prepare(
                                "SELECT f.filename FROM chat_files cf JOIN files f ON f.id = cf.file_id WHERE cf.chat_id=? ORDER BY cf.created_at ASC LIMIT 1",
                            ) {
                                if let Ok(mut frows) = fstmt.query([chat_id.clone()]) {
                                    if let Ok(Some(r)) = frows.next() {
                                        let filename: String = r.get(0).unwrap_or_default();
                                        let f = filename.trim();
                                        if !f.is_empty() {
                                            title = Some(f.to_string());
                                        }
                                    }
                                }
                            }
                        }

                        // Get last message content if messages exist
                        let mut last_message_content: Option<String> = None;
                        if let Ok(mut lstmt) = conn.prepare(
                            "SELECT content_json FROM messages WHERE chat_id=? ORDER BY created_at DESC LIMIT 1",
                        ) {
                            if let Ok(mut lrows) = lstmt.query([chat_id.clone()]) {
                                if let Ok(Some(r)) = lrows.next() {
                                    let content_json: String = r.get(0).unwrap_or_default();
                                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&content_json) {
                                        if let Some(text) = v.get("text").and_then(|t| t.as_str()) {
                                            last_message_content = Some(truncate_string(text, 200));
                                        } else if let Some(text) = v.as_str() {
                                            last_message_content = Some(truncate_string(text, 200));
                                        }
                                    }
                                }
                            }
                        }

                        // Get file count
                        let mut file_count: i64 = 0;
                        if let Ok(mut fstmt) = conn.prepare(
                            "SELECT COUNT(DISTINCT cf.file_id) FROM chat_files cf WHERE cf.chat_id=?",
                        ) {
                            if let Ok(mut frows) = fstmt.query([chat_id.clone()]) {
                                if let Ok(Some(r)) = frows.next() {
                                    file_count = r.get(0).unwrap_or(0);
                                }
                            }
                        }

                        let mut session_obj = serde_json::json!({
                            "chat_id": chat_id,
                            "last_message_at": last_ts,
                            "file_count": file_count
                        });
                        if let Some(t) = title {
                            session_obj["title"] = serde_json::json!(t);
                        }
                        if let Some(msg) = last_message_content {
                            session_obj["last_message_content"] = serde_json::json!(msg);
                        }
                        sessions.push(session_obj);
                    }
                }
            }
        }
    }

    // Fallback: include any KV sessions that may not have a transcript yet.
    let kv_dir = workspace.join("kv_sessions");
    if kv_dir.exists() {
        if let Ok(entries) = std::fs::read_dir(kv_dir) {
            for entry in entries.flatten() {
                if let Some(name) = entry.file_name().to_str() {
                    if name.ends_with(".kv") || name.ends_with(".json") || name.ends_with(".bin") {
                        let chat_id = name.split('.').next().unwrap_or(name).to_string();
                        if seen.insert(chat_id.clone()) {
                            sessions.push(serde_json::json!({ "chat_id": chat_id }));
                        }
                    }
                }
            }
        }
    }
    Ok(serde_json::json!({ "sessions": sessions }))
}

// Helper function to truncate a string to a maximum length
fn truncate_string(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        // Truncate at a word boundary if possible
        let truncated = &s[..max_len];
        if let Some(last_space) = truncated.rfind(' ') {
            truncated[..last_space].to_string() + "..."
        } else {
            truncated.to_string() + "..."
        }
    }
}

#[tauri::command]
async fn engine_stream_request(
    request_id: String,
    payload: Value,
    app: tauri::AppHandle,
    state: tauri::State<'_, SharedEngine>,
) -> Result<(), String> {
    if ipc_debug() {
        eprintln!("[cmd] engine_stream_request request_id={request_id}");
    }
    // Start streaming in the background and return immediately, otherwise the
    // WebView may not process emitted events until the command resolves.
    let engine = state.inner().clone();
    let app = app.clone();
    tauri::async_runtime::spawn_blocking(move || {
        if ipc_debug() {
            eprintln!("[cmd] stream thread start request_id={request_id}");
        }
        let res = engine
            .stream(&request_id, "/chat", &payload, &app)
            .map_err(|e| format!("stream error: {}", e));
        if ipc_debug() {
            match &res {
                Ok(_) => eprintln!("[cmd] stream thread done request_id={request_id} ok"),
                Err(e) => eprintln!("[cmd] stream thread done request_id={request_id} err={e}"),
            }
        }
        res
    });
    Ok(())
}

#[tauri::command]
fn engine_cancel_request(
    request_id: String,
    state: tauri::State<'_, SharedEngine>,
) -> Result<(), String> {
    if ipc_debug() {
        eprintln!("[cmd] engine_cancel_request request_id={request_id}");
    }
    let engine = state.inner();
    engine.cancel(&request_id).map_err(|e| e.to_string())
}

#[tauri::command]
fn get_session_messages(chat_id: String) -> Result<Value, String> {
    let workspace = get_workspace_dir()?;
    let db_path = workspace.join("db.sqlite");
    if !db_path.exists() {
        return Ok(serde_json::json!({ "messages": [] }));
    }

    let conn = Connection::open(db_path).map_err(|e| format!("Failed to open db.sqlite: {e}"))?;
    let mut stmt = conn
        .prepare("SELECT role, content_json, citations_json, created_at FROM messages WHERE chat_id=? ORDER BY created_at ASC")
        .map_err(|e| format!("Failed to prepare query: {e}"))?;

    let mut out: Vec<Value> = Vec::new();
    let rows = stmt
        .query_map([chat_id], |row| {
            let role: String = row.get(0)?;
            let content_json: String = row.get(1)?;
            let citations_json: Option<String> = row.get(2)?;
            let created_at: String = row.get(3)?;
            Ok((role, content_json, citations_json, created_at))
        })
        .map_err(|e| format!("Query failed: {e}"))?;

    for row in rows.flatten() {
        let (role, content_json, citations_json, created_at) = row;
        let content: String;
        let mut attachments: Vec<String> = Vec::new();
        let mut selection: Option<Value> = None;
        let mut focus_document_id: Option<String> = None;
        match serde_json::from_str::<serde_json::Value>(&content_json) {
            Ok(v) => {
                if let Some(text) = v.get("text").and_then(|t| t.as_str()) {
                    content = text.to_string();
                } else if let Some(text) = v.as_str() {
                    content = text.to_string();
                } else {
                    content = content_json.clone();
                }
                if let Some(arr) = v.get("attachments").and_then(|a| a.as_array()) {
                    for item in arr {
                        if let Some(s) = item.as_str() {
                            attachments.push(s.to_string());
                        }
                    }
                }
                if let Some(sel) = v.get("selection") {
                    if sel.is_object() {
                        selection = Some(sel.clone());
                    }
                }
                if let Some(fid) = v.get("focus_document_id").and_then(|x| x.as_str()) {
                    focus_document_id = Some(fid.to_string());
                }
            }
            Err(_) => {
                // Fallback: treat as plain text
                content = content_json.clone();
            }
        }
        let mut msg = serde_json::json!({ "role": role, "content": content, "created_at": created_at });
        if !attachments.is_empty() {
            msg["attachments"] = serde_json::json!(attachments);
        }
        if let Some(sel) = selection {
            msg["selection"] = sel;
        }
        if let Some(fid) = focus_document_id {
            msg["focus_document_id"] = serde_json::json!(fid);
        }
        if let Some(raw) = citations_json {
            let trimmed = raw.trim();
            if !trimmed.is_empty() {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed) {
                    if v.is_array() {
                        msg["sources"] = v;
                    }
                }
            }
        }
        out.push(msg);
    }

    Ok(serde_json::json!({ "messages": out }))
}

#[tauri::command]
fn pick_files() -> Result<Value, String> {
    // UI picker guardrail:
    // - Backend decides Plan A vs Plan B at 5 MiB (INSIGHT_MAX_MULTI_FILE_BYTES).
    // - Backend hard-caps single-file size at 50 MiB (INSIGHT_MAX_SINGLE_LARGE_FILE_BYTES).
    // Keep the picker aligned with the backend hard cap so "raw_large" files can be selected.
    const MAX_ATTACHMENT_BYTES: u64 = 50 * 1024 * 1024; // 50 MiB
    let files = rfd::FileDialog::new().pick_files().unwrap_or_default();
    let out: Vec<Value> = files
        .into_iter()
        .filter_map(|p| {
            let path = p.to_str()?.to_string();
            let name = p.file_name()?.to_str().unwrap_or("").to_string();
            let size_bytes = std::fs::metadata(&p).ok().map(|m| m.len()).unwrap_or(0);
            Some(serde_json::json!({
                "path": path,
                "name": name,
                "size_bytes": size_bytes,
                "max_bytes": MAX_ATTACHMENT_BYTES,
            }))
        })
        .collect();
    Ok(serde_json::json!({ "files": out }))
}

#[tauri::command]
fn pick_model_file() -> Result<Value, String> {
    let file = rfd::FileDialog::new()
        .add_filter("GGUF", &["gguf"])
        .pick_file();
    if let Some(p) = file {
        let path = p.to_str().unwrap_or("").to_string();
        let name = p
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();
        let size_bytes = std::fs::metadata(&p).ok().map(|m| m.len()).unwrap_or(0);
        return Ok(serde_json::json!({
            "path": path,
            "name": name,
            "size_bytes": size_bytes
        }));
    }
    Ok(serde_json::json!({ "path": null }))
}

#[tauri::command]
fn save_export_file(default_name: String, content: String) -> Result<Value, String> {
    let name = default_name.trim();
    let file = rfd::FileDialog::new()
        .set_file_name(if name.is_empty() { "export.txt" } else { name })
        .save_file();
    if let Some(path) = file {
        std::fs::write(&path, content.as_bytes())
            .map_err(|e| format!("Failed to write {}: {}", path.display(), e))?;
        return Ok(serde_json::json!({ "path": path.to_string_lossy().to_string(), "cancelled": false }));
    }
    Ok(serde_json::json!({ "path": null, "cancelled": true }))
}

#[tauri::command]
fn print_current_webview(window: tauri::WebviewWindow) -> Result<(), String> {
    window.print().map_err(|e| format!("Print failed: {e}"))
}

#[tauri::command]
async fn export_pdf_file(
    default_name: String,
    html: String,
    window: tauri::WebviewWindow,
) -> Result<Value, String> {
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (default_name, html, window);
        return Err("PDF export is currently supported on macOS only.".to_string());
    }

    #[cfg(target_os = "macos")]
    {
        use std::ptr::NonNull;
        use std::sync::mpsc;
        use std::time::{Duration, SystemTime, UNIX_EPOCH};
        use tauri::{WebviewUrl, WebviewWindowBuilder};

        let name = default_name.trim();
        let file = rfd::FileDialog::new()
            .set_file_name(if name.is_empty() { "document.pdf" } else { name })
            .add_filter("PDF", &["pdf"])
            .save_file();
        let Some(path) = file else {
            return Ok(serde_json::json!({ "path": null, "cancelled": true }));
        };

        let app = window.app_handle();
        let label = {
            let ts = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis();
            format!("export-pdf-{ts}")
        };

        let about_blank = tauri::Url::parse("about:blank")
            .map_err(|e| format!("Failed to parse about:blank URL: {e}"))?;
        let export_window = WebviewWindowBuilder::new(app, label, WebviewUrl::External(about_blank))
            .title("Export PDF")
            .visible(false)
            .decorations(false)
            .resizable(false)
            .skip_taskbar(true)
            .inner_size(1024.0, 768.0)
            .build()
            .map_err(|e| format!("Failed to create export window: {e}"))?;

        let load_html = html.clone();
        export_window
            .with_webview(move |webview| unsafe {
                let view: &objc2_web_kit::WKWebView = &*webview.inner().cast();
                let ns_html = objc2_foundation::NSString::from_str(&load_html);
                view.loadHTMLString_baseURL(&ns_html, None);
            })
            .map_err(|e| format!("Failed to load export HTML into webview: {e}"))?;

        // Allow the WebView to finish laying out the document before exporting.
        // The export HTML is self-contained (no external resources), so a small
        // delay is sufficient and avoids impacting the visible app UI.
        tauri::async_runtime::spawn_blocking(|| {
            std::thread::sleep(Duration::from_millis(120));
        })
        .await
        .map_err(|e| format!("PDF export wait task failed: {e}"))?;

        let (tx, rx) = mpsc::channel::<Result<Vec<u8>, String>>();
        export_window
            .with_webview(move |webview| unsafe {
                let view: &objc2_web_kit::WKWebView = &*webview.inner().cast();
                let completion = block2::StackBlock::new(
                    move |data: *mut objc2_foundation::NSData, error: *mut objc2_foundation::NSError| {
                        let out = if !error.is_null() {
                            let err: &objc2_foundation::NSError = &*error;
                            let desc = err.localizedDescription();
                            Err(desc.to_string())
                        } else if data.is_null() {
                            Err("WKWebView returned no PDF data.".to_string())
                        } else {
                            let d: &objc2_foundation::NSData = &*data;
                            let len = d.length() as usize;
                            let mut buf = vec![0u8; len];
                            if len > 0 {
                                let ptr = NonNull::new(buf.as_mut_ptr().cast())
                                    .expect("Vec pointer should not be null");
                                d.getBytes_length(ptr, len as objc2_foundation::NSUInteger);
                            }
                            Ok(buf)
                        };
                        let _ = tx.send(out);
                    },
                );

                view.createPDFWithConfiguration_completionHandler(None, &completion);
            })
            .map_err(|e| format!("Failed to access export webview: {e}"))?;

        let path_str = path.to_string_lossy().to_string();
        let res = tauri::async_runtime::spawn_blocking(move || -> Result<(), String> {
            let bytes = rx
                .recv_timeout(Duration::from_secs(120))
                .map_err(|_| "Timed out generating PDF.".to_string())??;
            std::fs::write(&path, &bytes)
                .map_err(|e| format!("Failed to write {}: {}", path.display(), e))?;
            Ok(())
        })
        .await
        .map_err(|e| format!("PDF export task failed: {e}"))?;
        res?;

        let _ = export_window.close();
        Ok(serde_json::json!({ "path": path_str, "cancelled": false }))
    }
}

fn spawn_kv_watcher(app: tauri::AppHandle, kv_dir: PathBuf) -> notify::Result<RecommendedWatcher> {
    let mut watcher = notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
        if let Ok(event) = res {
            if event.kind.is_modify() || event.kind.is_create() || event.kind.is_remove() {
                let _ = app.emit("sessions_updated", {});
            }
        }
    })?;
    watcher.watch(&kv_dir, RecursiveMode::NonRecursive)?;
    Ok(watcher)
}

pub fn run() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            // Try to detect and use the bundled sidecar binary first.
            // This checks if we're running from a bundled .app build.
            let engine = if cfg!(debug_assertions) {
                // Dev build: use Python with backend/engine.py
                let python_bin = std::env::var("PYTHON_BIN").unwrap_or_else(|_| "python3".into());
                EngineProcess::spawn(&python_bin)
                    .map_err(|e| format!("Failed to spawn Python engine: {}", e))?
            } else {
                // Release build: try to use the bundled sidecar binary
                use tauri::path::BaseDirectory;
                let candidates = [
                    // Standard layout: bin/<arch>/insight-engine
                    "bin/insight-engine-aarch64-apple-darwin/insight-engine",
                    "bin/insight-engine-x86_64-apple-darwin/insight-engine",
                    // Fallback layout (older versions): bin/insight-engine-<arch>
                    "bin/insight-engine-aarch64-apple-darwin",
                    "bin/insight-engine-x86_64-apple-darwin",
                ];
                let mut resolved: Option<PathBuf> = None;
                for rel in candidates {
                    if let Ok(path) = app.path().resolve(rel, BaseDirectory::Resource) {
                        if path.is_file() {
                            resolved = Some(path);
                            break;
                        }
                    }
                }
                if let Some(sidecar_path) = resolved {
                    eprintln!("[INFO] Using bundled engine: {}", sidecar_path.display());
                    EngineProcess::spawn_from_binary(&sidecar_path)
                        .map_err(|e| format!("Failed to spawn bundled engine ({}): {}", sidecar_path.display(), e))?
                } else {
                    // Fallback: try Python (useful for testing release builds locally)
                    eprintln!("[WARNING] No bundled sidecar found, falling back to Python");
                    let python_bin = std::env::var("PYTHON_BIN").unwrap_or_else(|_| "python3".into());
                    EngineProcess::spawn(&python_bin)
                        .map_err(|e| format!("Failed to spawn Python engine ({}): {}", python_bin, e))?
                }
            };

            // Allow the stdout router thread to emit out-of-band backend events
            // (e.g. files_changed / file_text_ready) to the frontend.
            engine.set_app_handle(app.app_handle().clone());
            app.manage(Arc::new(engine));

            // Determine workspace directory based on build type
            let workspace_dir = if cfg!(debug_assertions) {
                // Dev mode: use project-local storage
                resolve_project_root()
                    .map_err(|e| format!("Failed to resolve project root: {}", e))?
                    .join("storage")
            } else {
                // Release mode: use ~/.insight
                dirs::home_dir()
                    .ok_or_else(|| String::from("couldn't find home dir"))?
                    .join(".insight")
            };

            std::fs::create_dir_all(&workspace_dir)
                .map_err(|e| format!("Failed to create workspace directory: {}", e))?;

            // Watch KV sessions for real-time updates.
            let kv_dir = workspace_dir.join("kv_sessions");
            std::fs::create_dir_all(&kv_dir)
                .map_err(|e| format!("Failed to create kv_sessions directory: {}", e))?;
            if let Ok(watcher) = spawn_kv_watcher(app.app_handle().clone(), kv_dir.clone()) {
                // Keep watcher alive in state
                app.manage(Mutex::new(watcher));
                eprintln!("[INFO] Watching KV sessions directory: {}", kv_dir.display());
            } else {
                eprintln!("[WARNING] Failed to spawn KV sessions watcher");
            }
            Ok(())
        })
        .on_window_event(|window, event| {
            if let WindowEvent::CloseRequested { .. } = event {
                shutdown_engine(&window.app_handle());
                window.app_handle().exit(0);
            }
        })
        .invoke_handler(tauri::generate_handler![
            engine_request,
            list_sessions,
            get_session_messages,
            pick_files,
            pick_model_file,
            save_export_file,
            print_current_webview,
            export_pdf_file,
            engine_stream_request,
            engine_cancel_request
        ])
        .build(tauri::generate_context!())
        .expect("error running Tauri application");

    app.run(|app_handle, event| {
        if let RunEvent::ExitRequested { .. } = event {
            shutdown_engine(app_handle);
        }
    });
}
