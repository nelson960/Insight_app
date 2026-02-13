mod engine;
use engine::{resolve_project_root, EngineProcess, EngineRequest, EngineResponse};
use notify::{RecommendedWatcher, RecursiveMode, Watcher};
use serde_json::Value;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use tauri::{Emitter, Manager, RunEvent, WindowEvent};

type SharedEngine = Arc<EngineProcess>;

fn shutdown_engine(app: &tauri::AppHandle) {
    // Cleanup the file watcher if it exists
    if let Some(watcher_mutex) = app.try_state::<Mutex<notify::RecommendedWatcher>>() {
        drop(watcher_mutex.lock());
    }

    // Shutdown the engine process
    let engine = app.state::<SharedEngine>();
    engine.shutdown();
}

#[tauri::command]
async fn engine_request(
    endpoint: String,
    method: Option<String>,
    payload: Option<Value>,
    state: tauri::State<'_, SharedEngine>,
) -> Result<EngineResponse, String> {
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
async fn list_sessions(state: tauri::State<'_, SharedEngine>) -> Result<Value, String> {
    let engine = state.inner().clone();
    let req = EngineRequest {
        request_id: None,
        endpoint: "/chat/session_summaries".into(),
        method: "GET".into(),
        payload: Value::Null,
        stream: false,
    };
    let resp = tauri::async_runtime::spawn_blocking(move || engine.send(&req))
        .await
        .map_err(|e| format!("list_sessions join error: {e}"));
    match resp {
        Ok(Ok(r)) if r.ok => Ok(r.data),
        Ok(Ok(r)) => {
            eprintln!(
                "[WARN] list_sessions backend error status={} err={:?}",
                r.status, r.error
            );
            Ok(serde_json::json!({ "sessions": [] }))
        }
        Ok(Err(e)) => {
            eprintln!("[WARN] list_sessions backend send failed: {}", e);
            Ok(serde_json::json!({ "sessions": [] }))
        }
        Err(e) => {
            eprintln!("[WARN] list_sessions join failed: {}", e);
            Ok(serde_json::json!({ "sessions": [] }))
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
    // Start streaming in the background and return immediately, otherwise the
    // WebView may not process emitted events until the command resolves.
    let engine = state.inner().clone();
    let app = app.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let res = engine
            .stream(&request_id, "/chat", &payload, &app)
            .map_err(|e| format!("stream error: {}", e));
        res
    });
    Ok(())
}

#[tauri::command]
fn engine_cancel_request(
    request_id: String,
    state: tauri::State<'_, SharedEngine>,
) -> Result<(), String> {
    let engine = state.inner();
    engine.cancel(&request_id).map_err(|e| e.to_string())
}

#[tauri::command]
fn engine_emit_event(event_name: String, payload: Value, app: tauri::AppHandle) -> Result<(), String> {
    app.emit(&event_name, payload).map_err(|e| e.to_string())
}

#[tauri::command]
async fn get_session_messages(
    chat_id: String,
    state: tauri::State<'_, SharedEngine>,
) -> Result<Value, String> {
    let engine = state.inner().clone();
    let req = EngineRequest {
        request_id: None,
        endpoint: "/chat/session_messages".into(),
        method: "POST".into(),
        payload: serde_json::json!({ "chat_id": chat_id }),
        stream: false,
    };
    let resp = tauri::async_runtime::spawn_blocking(move || engine.send(&req))
        .await
        .map_err(|e| format!("get_session_messages join error: {e}"));
    match resp {
        Ok(Ok(r)) if r.ok => Ok(r.data),
        Ok(Ok(r)) => {
            eprintln!(
                "[WARN] get_session_messages backend error status={} err={:?}",
                r.status, r.error
            );
            Ok(serde_json::json!({ "messages": [] }))
        }
        Ok(Err(e)) => {
            eprintln!("[WARN] get_session_messages backend send failed: {}", e);
            Ok(serde_json::json!({ "messages": [] }))
        }
        Err(e) => {
            eprintln!("[WARN] get_session_messages join failed: {}", e);
            Ok(serde_json::json!({ "messages": [] }))
        }
    }
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
            engine_cancel_request,
            engine_emit_event
        ])
        .build(tauri::generate_context!())
        .expect("error running Tauri application");

    app.run(|app_handle, event| {
        if let RunEvent::ExitRequested { .. } = event {
            shutdown_engine(app_handle);
        }
    });
}
