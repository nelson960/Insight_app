mod engine;
use engine::{resolve_project_root, EngineProcess, EngineRequest, EngineResponse};
use notify::{RecommendedWatcher, RecursiveMode, Watcher};
use serde_json::Value;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use tauri::{Emitter, Manager};
use rusqlite::Connection;

type SharedEngine = Arc<EngineProcess>;

fn ipc_debug() -> bool {
    std::env::var("INSIGHT_IPC_DEBUG").is_ok()
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
fn engine_request(
    endpoint: String,
    method: Option<String>,
    payload: Option<Value>,
    state: tauri::State<'_, SharedEngine>,
) -> Result<EngineResponse, String> {
    if ipc_debug() {
        eprintln!("[cmd] engine_request endpoint={endpoint}");
    }
    let engine = state.inner();
    let req = EngineRequest {
        request_id: None,
        endpoint,
        method: method.unwrap_or_else(|| "POST".into()),
        payload: payload.unwrap_or(Value::Null),
        stream: false,
    };
    engine
        .send(&req)
        .map_err(|e| format!("Python engine error: {}", e))
}

#[tauri::command]
fn list_sessions() -> Result<Value, String> {
    let root = resolve_project_root().map_err(|e| e.to_string())?;
    let mut sessions = Vec::new();
    let mut seen = std::collections::HashSet::new();

    // Prefer SQLite transcript store for session listing (clean UI chats).
    let db_path = root.join("storage").join("db.sqlite");
    if db_path.exists() {
        if let Ok(conn) = Connection::open(db_path) {
            if let Ok(mut stmt) = conn.prepare(
                "SELECT chat_id, MAX(created_at) AS last_ts FROM messages GROUP BY chat_id ORDER BY last_ts DESC",
            ) {
                let rows = stmt.query_map([], |row| {
                    let chat_id: String = row.get(0)?;
                    Ok(chat_id)
                });
                if let Ok(rows) = rows {
                    for row in rows.flatten() {
                        if seen.insert(row.clone()) {
                            let mut title: Option<String> = None;
                            if let Ok(mut tstmt) = conn.prepare(
                                "SELECT content_json FROM messages WHERE chat_id=? AND role='user' ORDER BY created_at ASC LIMIT 1",
                            ) {
                                if let Ok(mut trows) = tstmt.query([row.clone()]) {
                                    if let Ok(Some(r)) = trows.next() {
                                        let content_json: String = r.get(0).unwrap_or_default();
                                        title = _extract_title_from_content_json(&content_json);
                                    }
                                }
                            }
                            if let Some(t) = title {
                                sessions.push(serde_json::json!({ "chat_id": row, "title": t }));
                            } else {
                                sessions.push(serde_json::json!({ "chat_id": row }));
                            }
                        }
                    }
                }
            }
        }
    }

    // Fallback: include any KV sessions that may not have a transcript yet.
    let kv_dir = root.join("storage").join("kv_sessions");
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
    let root = resolve_project_root().map_err(|e| e.to_string())?;
    let db_path = root.join("storage").join("db.sqlite");
    if !db_path.exists() {
        return Ok(serde_json::json!({ "messages": [] }));
    }

    let conn = Connection::open(db_path).map_err(|e| format!("Failed to open db.sqlite: {e}"))?;
    let mut stmt = conn
        .prepare("SELECT role, content_json FROM messages WHERE chat_id=? ORDER BY created_at ASC")
        .map_err(|e| format!("Failed to prepare query: {e}"))?;

    let mut out: Vec<Value> = Vec::new();
    let rows = stmt
        .query_map([chat_id], |row| {
            let role: String = row.get(0)?;
            let content_json: String = row.get(1)?;
            Ok((role, content_json))
        })
        .map_err(|e| format!("Query failed: {e}"))?;

    for row in rows.flatten() {
        let (role, content_json) = row;
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
        let mut msg = serde_json::json!({ "role": role, "content": content });
        if !attachments.is_empty() {
            msg["attachments"] = serde_json::json!(attachments);
        }
        if let Some(sel) = selection {
            msg["selection"] = sel;
        }
        if let Some(fid) = focus_document_id {
            msg["focus_document_id"] = serde_json::json!(fid);
        }
        out.push(msg);
    }

    Ok(serde_json::json!({ "messages": out }))
}

#[tauri::command]
fn pick_files() -> Result<Value, String> {
    const MAX_ATTACHMENT_BYTES: u64 = 5 * 1024 * 1024; // 5 MiB (frontend pre-check; backend also enforces)
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
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            let python_bin = std::env::var("PYTHON_BIN").unwrap_or_else(|_| "python3".into());
            let engine = EngineProcess::spawn(&python_bin)
                .map_err(|e| format!("Failed to spawn Python engine: {}", e))?;
            app.manage(Arc::new(engine));

            // Watch KV sessions for real-time updates.
            let root = resolve_project_root().map_err(|e| format!("{e}"))?;
            let kv_dir = root.join("storage").join("kv_sessions");
            std::fs::create_dir_all(&kv_dir).ok();
            if let Ok(watcher) = spawn_kv_watcher(app.app_handle().clone(), kv_dir) {
                // Keep watcher alive in state
                app.manage(Mutex::new(watcher));
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            engine_request,
            list_sessions,
            get_session_messages,
            pick_files,
            engine_stream_request,
            engine_cancel_request
        ])
        .run(tauri::generate_context!())
        .expect("error running Tauri application");
}
