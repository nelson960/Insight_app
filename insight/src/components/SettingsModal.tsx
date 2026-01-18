import { useEffect, useMemo, useRef, useState } from "react";
import { engine } from "../api/engine";
import { invoke } from "@tauri-apps/api/core";

export type ThemeMode = "system" | "dark" | "light";

type SettingsState = {
  theme_mode: ThemeMode;
  llm_model_path: string;
  llm_gpu_layers: number;
  llm_ctx_size: number;
  raw_engine_host: string;
  raw_engine_port: number;
  raw_engine_model_path: string;
  raw_engine_ctx: number | null;
  raw_engine_threads: number | null;
  raw_engine_gpu_layers: number | null;
  raw_engine_max_tokens: number;
  raw_engine_embedding_model: string;
  raw_engine_embedding_auto_download: boolean;
  raw_engine_log_preview_chars: number;
  raw_engine_log_prompts: boolean;
  raw_engine_log_completions: boolean;
};

type LlmModelInfo = {
  path?: string;
  architecture?: string;
  name?: string;
  size_label?: string;
  file_type?: number | null;
  quantization_version?: number | null;
  ctx_train?: number | null;
  ctx_runtime?: number | null;
  chat_template_kind?: string;
  chat_format?: string;
  prompt_renderer?: string;
  tokenizer_model?: string;
  add_bos_token?: string;
  bos_token_id?: number | null;
  eos_token_id?: number | null;
};

type SettingsResponse = {
  settings?: SettingsState;
  llm?: { loaded?: boolean; model_info?: LlmModelInfo | null };
  raw_engine?: { defaults?: Partial<SettingsState> };
};

type StorageUsage = {
  base: string;
  total_bytes: number;
  total_files: number;
  breakdown: Record<string, { bytes: number; files: number; path: string }>;
};

type BusyState = {
  busy: boolean;
  reasons?: string[];
  active_jobs?: number;
  llm?: any;
};

type HealthIssue = {
  code: string;
  severity: "error" | "warning";
  message: string;
  fix: string;
  action?: string;
};

type HealthReport = {
  ok: boolean;
  issues: HealthIssue[];
  checks?: Record<string, any>;
};

type RawEngineStatus = {
  ok?: boolean;
  running: boolean;
  starting?: boolean;
  pid?: number | null;
  host?: string | null;
  port?: number | null;
  requested_host?: string | null;
  requested_port?: number | null;
  log_dir?: string | null;
  command?: string | null;
  started_at?: number | null;
  start_requested_at?: number | null;
  exit_code?: number | null;
  error?: string | null;
};

type RawEngineLogLine = {
  ts: number;
  line: string;
};

function formatBytes(bytes: number) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let v = bytes;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(i === 0 ? 0 : 2)} ${units[i]}`;
}

export function SettingsModal(props: {
  open: boolean;
  onClose: () => void;
  themeMode: ThemeMode;
  onThemeModeChange: (mode: ThemeMode) => void;
}) {
  const { open, onClose, themeMode, onThemeModeChange } = props;
  const RAW_ENGINE_DEFAULTS: SettingsState = {
    theme_mode: themeMode,
    llm_model_path: "",
    llm_gpu_layers: 99,
    llm_ctx_size: 32768,
    raw_engine_host: "127.0.0.1",
    raw_engine_port: 11435,
    raw_engine_model_path: "",
    raw_engine_ctx: null,
    raw_engine_threads: null,
    raw_engine_gpu_layers: null,
    raw_engine_max_tokens: 1024,
    raw_engine_embedding_model: "nomic-embed-text-v1.5",
    raw_engine_embedding_auto_download: true,
    raw_engine_log_preview_chars: 400,
    raw_engine_log_prompts: false,
    raw_engine_log_completions: false,
  };
  const [settings, setSettings] = useState<SettingsState>({
    theme_mode: themeMode,
    llm_model_path: "",
    llm_gpu_layers: 99,
    llm_ctx_size: 32768,
    raw_engine_host: "127.0.0.1",
    raw_engine_port: 11435,
    raw_engine_model_path: "",
    raw_engine_ctx: null,
    raw_engine_threads: null,
    raw_engine_gpu_layers: null,
    raw_engine_max_tokens: 1024,
    raw_engine_embedding_model: "nomic-embed-text-v1.5",
    raw_engine_embedding_auto_download: true,
    raw_engine_log_preview_chars: 400,
    raw_engine_log_prompts: false,
    raw_engine_log_completions: false,
  });
  const [storage, setStorage] = useState<StorageUsage | null>(null);
  const [restartRequired, setRestartRequired] = useState(false);
  const [modelValidation, setModelValidation] = useState<{ ok: boolean; msg: string } | null>(null);
  const [_cleanResult, setCleanResult] = useState<string | null>(null);
  const [resetOpen, setResetOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [engineBusy, setEngineBusy] = useState<BusyState | null>(null);
  const [llmLoaded, setLlmLoaded] = useState(false);
  const [llmInfo, setLlmInfo] = useState<LlmModelInfo | null>(null);
  const [healthReport, setHealthReport] = useState<HealthReport | null>(null);
  const [embeddingDownloading, setEmbeddingDownloading] = useState(false);
  const [embeddingDownloadError, setEmbeddingDownloadError] = useState<string | null>(null);
  const [rawEngineStatus, setRawEngineStatus] = useState<RawEngineStatus | null>(null);
  const [rawEngineLogs, setRawEngineLogs] = useState<RawEngineLogLine[]>([]);
  const [rawEngineError, setRawEngineError] = useState<string | null>(null);
  const [rawEngineBusy, setRawEngineBusy] = useState(false);
  const [rawLogsOpen, setRawLogsOpen] = useState(false);
  const [rawAdvancedOpen, setRawAdvancedOpen] = useState(false);
  const [rawEngineDefaults, setRawEngineDefaults] = useState<Partial<SettingsState>>(RAW_ENGINE_DEFAULTS);
  const resetWrapRef = useRef<HTMLDivElement | null>(null);
  const rawLogRef = useRef<HTMLDivElement | null>(null);
  const modalRef = useRef<HTMLDivElement | null>(null);
  const scrollTimersRef = useRef<Map<HTMLElement, number>>(new Map());

  function getErrorText(res: { error?: string; data?: any }, fallback: string) {
    return (
      res.error ||
      res.data?.error ||
      res.data?.detail?.error ||
      res.data?.detail ||
      fallback
    );
  }

  function formatLogTime(ts: number) {
    if (!Number.isFinite(ts)) return "";
    try {
      return new Date(ts * 1000).toLocaleTimeString();
    } catch {
      return "";
    }
  }

  useEffect(() => {
    if (!open) return;
    setResetOpen(false);
    setCleanResult(null);
    setModelValidation(null);
    setRestartRequired(false);
    setEmbeddingDownloading(false);
    setEmbeddingDownloadError(null);
    setRawEngineError(null);
    setBusy(true);
    (async () => {
      const res = await engine<SettingsResponse>(
        "/settings",
        undefined,
        "GET"
      );
      if (res.ok && (res.data as any)?.settings) {
        const next = (res.data as any).settings as SettingsState;
        setSettings((prev) => ({ ...prev, ...next }));
        const mode = next.theme_mode;
        if (mode) onThemeModeChange(mode);
      }
      if (res.ok) {
        const defaults = (res.data as any)?.raw_engine?.defaults as Partial<SettingsState> | undefined;
        if (defaults) setRawEngineDefaults(defaults);
      }
      if (res.ok) {
        const loaded = Boolean((res.data as any)?.llm?.loaded);
        setLlmLoaded(loaded);
        setLlmInfo(((res.data as any)?.llm?.model_info as any) || null);
      } else {
        setLlmLoaded(false);
        setLlmInfo(null);
      }
      const st = await engine<StorageUsage>("/settings/storage", undefined, "GET");
      if (st.ok) setStorage(st.data as any);
      setBusy(false);
    })().catch(() => setBusy(false));
  }, [open]);

  useEffect(() => {
    if (!resetOpen) return;
    function handleOutsideClick(event: MouseEvent) {
      if (!resetWrapRef.current) return;
      if (!resetWrapRef.current.contains(event.target as Node)) {
        setResetOpen(false);
      }
    }
    document.addEventListener("mousedown", handleOutsideClick);
    return () => document.removeEventListener("mousedown", handleOutsideClick);
  }, [resetOpen]);


  useEffect(() => {
    if (!open) return;
    let alive = true;
    const tick = async () => {
      const bs = await engine<BusyState>("/settings/busy", undefined, "GET");
      if (!alive) return;
      if (bs.ok) setEngineBusy(bs.data as any);
    };
    tick();
    const id = window.setInterval(tick, 15000);
    return () => {
      alive = false;
      window.clearInterval(id);
    };
  }, [open]);

  useEffect(() => {
    if (!open) return;
    let alive = true;
    const tick = async () => {
      if (!alive) return;
      await refreshHealth();
    };
    tick();
    const id = window.setInterval(tick, 10000);
    return () => {
      alive = false;
      window.clearInterval(id);
    };
  }, [open]);

  useEffect(() => {
    if (!open) return;
    let alive = true;
    const tick = async () => {
      if (!alive) return;
      await refreshRawEngineStatus();
      if (rawLogsOpen) {
        await refreshRawEngineLogs();
      }
    };
    tick();
    const id = window.setInterval(tick, rawLogsOpen ? 2500 : 12000);
    return () => {
      alive = false;
      window.clearInterval(id);
    };
  }, [open, rawLogsOpen]);

  useEffect(() => {
    const checks = healthReport?.checks || {};
    const status = checks.embedding_download_status;
    const err = checks.embedding_download_error;
    if (status === "downloading") {
      setEmbeddingDownloading(true);
    } else if (status === "ready" || status === "idle") {
      setEmbeddingDownloading(false);
    }
    if (err) {
      setEmbeddingDownloadError(err);
    }
  }, [healthReport]);

  useEffect(() => {
    if (!rawLogsOpen) return;
    if (!rawLogRef.current) return;
    rawLogRef.current.scrollTop = rawLogRef.current.scrollHeight;
  }, [rawEngineLogs, rawLogsOpen]);

  useEffect(() => {
    if (!open) return;
    const root = modalRef.current;
    if (!root) return;
    const nodes = Array.from(root.querySelectorAll<HTMLElement>(".settings-scrollable"));
    if (nodes.length === 0) return;
    const timers = scrollTimersRef.current;
    const markScrolling = (el: HTMLElement) => {
      el.classList.add("is-scrolling");
      const prior = timers.get(el);
      if (prior) window.clearTimeout(prior);
      const timeout = window.setTimeout(() => {
        el.classList.remove("is-scrolling");
        timers.delete(el);
      }, 1000);
      timers.set(el, timeout);
    };
    const onScroll = (event: Event) => {
      markScrolling(event.currentTarget as HTMLElement);
    };
    const onWheel = (event: Event) => {
      markScrolling(event.currentTarget as HTMLElement);
    };
    nodes.forEach((el) => {
      el.addEventListener("scroll", onScroll, { passive: true });
      el.addEventListener("wheel", onWheel, { passive: true });
      el.addEventListener("touchmove", onWheel, { passive: true });
    });
    return () => {
      nodes.forEach((el) => {
        el.removeEventListener("scroll", onScroll);
        el.removeEventListener("wheel", onWheel);
        el.removeEventListener("touchmove", onWheel);
        el.classList.remove("is-scrolling");
      });
      timers.forEach((id) => window.clearTimeout(id));
      timers.clear();
    };
  }, [open, rawLogsOpen]);

  const sortedBreakdown = useMemo(() => {
    if (!storage?.breakdown) return [];
    return Object.entries(storage.breakdown).sort((a, b) => (b[1].bytes || 0) - (a[1].bytes || 0));
  }, [storage]);

  const rawRunning = Boolean(rawEngineStatus?.running);
  const rawStarting = Boolean(rawEngineStatus?.starting);
  const rawBaseUrl =
    rawRunning && rawEngineStatus?.host && rawEngineStatus?.port
      ? `http://${rawEngineStatus.host}:${rawEngineStatus.port}`
      : "";
  const rawConfigLocked = rawRunning || rawStarting;
  const rawDefaults = rawEngineDefaults || RAW_ENGINE_DEFAULTS;
  const rawDefaultLabel = (val: any) => (val === null || val === undefined || val === "" ? "Auto" : String(val));
  const rawMaxTokensDefault =
    typeof rawDefaults.raw_engine_max_tokens === "number"
      ? rawDefaults.raw_engine_max_tokens
      : RAW_ENGINE_DEFAULTS.raw_engine_max_tokens;

  async function refreshStorage() {
    const st = await engine<StorageUsage>("/settings/storage", undefined, "GET");
    if (st.ok) setStorage(st.data as any);
  }

  async function refreshHealth() {
    const res = await engine<HealthReport>("/settings/health?full=1", undefined, "GET");
    if (res.ok) setHealthReport(res.data as any);
  }

  async function refreshRawEngineStatus() {
    const res = await engine<RawEngineStatus>("/settings/raw_engine/status", undefined, "GET");
    if (res.ok) setRawEngineStatus(res.data as any);
  }

  async function refreshRawEngineLogs() {
    const res = await engine<{ lines?: RawEngineLogLine[] }>("/settings/raw_engine/logs?limit=250", undefined, "GET");
    if (res.ok) setRawEngineLogs((res.data as any)?.lines || []);
  }

  async function saveRawEngineSettings() {
    if (rawConfigLocked || rawEngineBusy) {
      setRawEngineError("Stop the raw server before changing its settings.");
      return;
    }
    setRawEngineError(null);
    const patch: Partial<SettingsState> = {
      raw_engine_host: settings.raw_engine_host,
      raw_engine_port: settings.raw_engine_port,
      raw_engine_ctx: settings.raw_engine_ctx,
      raw_engine_threads: settings.raw_engine_threads,
      raw_engine_gpu_layers: settings.raw_engine_gpu_layers,
      raw_engine_max_tokens: settings.raw_engine_max_tokens,
    };
    const res = await engine<any>("/settings", { settings: patch }, "POST");
    if (!res.ok) {
      setRawEngineError(getErrorText(res, "Failed to save raw server settings"));
      return;
    }
    await refreshRawEngineStatus();
  }

  async function toggleRawEngine() {
    if (rawEngineBusy) return;
    setRawEngineBusy(true);
    setRawEngineError(null);
    const running = Boolean(rawEngineStatus?.running);
    const endpoint = running ? "/settings/raw_engine/stop" : "/settings/raw_engine/start";
    const res = await engine<any>(endpoint, {}, "POST");
    if (!res.ok || (res.data as any)?.ok === false) {
      setRawEngineError(getErrorText(res, "Failed to toggle raw API server"));
    }
    await refreshRawEngineStatus();
    await refreshRawEngineLogs();
    setRawEngineBusy(false);
  }

  async function saveSettingsPatch(patch: Partial<SettingsState>) {
    const prev = settings;
    const next = { ...prev, ...patch };
    setSettings(next);
    const res = await engine<{ ok: boolean; restart_required?: boolean }>(
      "/settings",
      { settings: patch },
      "POST"
    );
    if (res.ok) {
      if ((res.data as any)?.restart_required) setRestartRequired(true);
      setCleanResult(null);
    } else {
      setSettings(prev);
      setCleanResult(getErrorText(res, "Failed to save settings"));
    }
  }

  async function validateModel(path: string) {
    setModelValidation(null);
    const trimmed = path.trim();
    if (!trimmed) {
      setModelValidation({ ok: false, msg: "Model path is required" });
      return false;
    }
    if (!trimmed.toLowerCase().endsWith(".gguf")) {
      setModelValidation({ ok: false, msg: "Model must be a .gguf file" });
      return false;
    }
    const res = await engine<any>("/settings/model/validate", { model_path: trimmed }, "POST");
    if (!res.ok || !(res.data as any)?.ok) {
      setModelValidation({ ok: false, msg: (res.data as any)?.error || res.error || "Invalid model" });
      return false;
    }
    setModelValidation({ ok: true, msg: (res.data as any)?.note || "Looks valid" });
    return true;
  }

  async function browseModel() {
    try {
      const res = await invoke<{ path?: string | null }>("pick_model_file");
      const path = res?.path || "";
      if (!path) return;
      const ok = await validateModel(path);
      if (!ok) return;
      setSettings((p) => ({ ...p, llm_model_path: path }));
    } catch (e: any) {
      setModelValidation({ ok: false, msg: e?.message || String(e) });
    }
  }

  async function applyLlmSettings() {
    if (engineBusy?.busy) {
      setCleanResult("Background work is running. Wait for it to finish before applying model settings.");
      return;
    }

    const ok = await validateModel(settings.llm_model_path);
    if (!ok) {
      return;
    }

    setBusy(true);
    const res = await engine<any>(
      "/settings/llm/apply",
      {
        confirm: true,
        settings: {
          llm_model_path: settings.llm_model_path,
          llm_ctx_size: settings.llm_ctx_size,
        },
      },
      "POST"
    );
    setBusy(false);
    if (!res.ok) {
      setCleanResult(getErrorText(res, "Failed to apply model settings"));
      return;
    }

    setRestartRequired(true);
    setLlmLoaded(false);
    setLlmInfo(null);
    await refreshStorage();
    await refreshHealth();
    setCleanResult("Model settings applied and workspace cleared. Restart the app/engine to reload the model.");
  }

  async function resetAll() {
    if (engineBusy?.busy) {
      setCleanResult("Background work is running. Wait for it to finish before resetting.");
      return;
    }
    setBusy(true);
    const res = await engine<any>("/settings/storage/reset", { confirm: true }, "POST");
    setBusy(false);
    if (!res.ok) {
      setCleanResult(getErrorText(res, "Reset failed"));
      return;
    }
    setResetOpen(false);
    setRestartRequired(true);
    const st = await engine<StorageUsage>("/settings/storage", undefined, "GET");
    if (st.ok) setStorage(st.data as any);
    const settingsRes = await engine<SettingsResponse>("/settings", undefined, "GET");
    if (settingsRes.ok && (settingsRes.data as any)?.settings) {
      const next = (settingsRes.data as any).settings as SettingsState;
      setSettings((prev) => ({ ...prev, ...next }));
      const mode = next.theme_mode;
      if (mode) onThemeModeChange(mode);
    }
    if (settingsRes.ok) {
      const loaded = Boolean((settingsRes.data as any)?.llm?.loaded);
      setLlmLoaded(loaded);
      setLlmInfo(((settingsRes.data as any)?.llm?.model_info as any) || null);
    } else {
      setLlmLoaded(false);
      setLlmInfo(null);
    }
    setModelValidation(null);
    const total = st.ok ? formatBytes((st.data as any)?.total_bytes ?? 0) : "0 B";
    setCleanResult(
      `Delete complete. Workspace cleared (KV/Qdrant/DB/uploads/cache/logs). Current storage: ${total}. Please restart the app.`
    );
    onClose();
    window.setTimeout(() => {
      window.location.reload();
    }, 60);
  }

  async function downloadEmbeddings() {
    if (embeddingDownloading) return;
    setEmbeddingDownloadError(null);
    if (engineBusy?.busy) {
      setEmbeddingDownloadError(
        "Background work is running. Wait for it to finish before downloading embeddings."
      );
      return;
    }
    setEmbeddingDownloading(true);
    const res = await engine<any>("/settings/embedding/download", {}, "POST");
    if (!res.ok || !(res.data as any)?.ok) {
      setEmbeddingDownloadError(getErrorText(res, "Failed to download embeddings"));
      setEmbeddingDownloading(false);
    }
    await refreshHealth();
  }

  if (!open) return null;

  return (
    <div
      className="settings-backdrop"
      role="dialog"
      aria-modal="true"
      aria-label="Settings"
      onPointerDown={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        className="settings-modal"
        onPointerDown={(e) => e.stopPropagation()}
        ref={modalRef}
      >
        <div className="settings-header">
          <div className="settings-title">Settings</div>
          <button className="settings-icon-btn settings-close" type="button" onClick={onClose} aria-label="Close">
            <svg className="settings-icon" viewBox="0 0 24 24" aria-hidden="true">
              <path d="M6 6l12 12M18 6l-12 12" />
            </svg>
          </button>
        </div>

        {restartRequired ? (
          <div className="settings-banner">
            Changes require restarting the app/engine to fully apply.
          </div>
        ) : null}
        {engineBusy?.busy ? (
          <div className="settings-banner warn">
            Background work running. Wait before cleaning/resetting storage or changing context.
          </div>
        ) : null}

        <div className="settings-body settings-scrollable">
          <section className="settings-section">
            <div className="settings-section-title">Health</div>
            {healthReport ? (
              (() => {
                const checks = (healthReport as any)?.checks || {};
                const embeddingPresent = checks.embedding_present === true;
                const embeddingStatus = checks.embedding_download_status;
                const showEmbeddingPanel =
                  !embeddingPresent ||
                  embeddingStatus === "downloading" ||
                  embeddingStatus === "error";
                const embeddingBusy = embeddingDownloading || embeddingStatus === "downloading";
                const issues = (healthReport.issues || []).filter(
                  (issue) =>
                    ![
                      "embedding_missing",
                      "embedding_downloading",
                      "embedding_download_failed",
                    ].includes(issue.code)
                );
                if (issues.length === 0 && !showEmbeddingPanel) {
                  return (
                    <div className="settings-health-ok">
                      <span className="settings-health-dot ok" />
                      All systems operational
                    </div>
                  );
                }
                return (
                  <div className="settings-health-list">
                    {showEmbeddingPanel ? (
                      <div className="settings-health-item warning">
                        <div className="settings-health-title">
                          {embeddingBusy
                            ? "Embedding model downloading"
                            : embeddingStatus === "error"
                            ? "Embedding download failed"
                            : "Embedding model missing"}
                        </div>
                        <div className="settings-health-fix">
                          {embeddingBusy
                            ? "Keep Insight open until the download completes."
                            : "Downloads the local ONNX embedding model required for ingestion/search."}
                        </div>
                        <div className="settings-row">
                          <button
                            className="settings-icon-btn"
                            type="button"
                            onClick={downloadEmbeddings}
                            disabled={embeddingBusy || !!engineBusy?.busy}
                            aria-label={
                              embeddingBusy ? "Embedding download in progress" : "Download embeddings"
                            }
                            title={embeddingBusy ? "Downloading…" : "Download embeddings"}
                          >
                            <svg className="settings-icon" viewBox="0 0 24 24" aria-hidden="true">
                              <path d="M12 4v10" />
                              <path d="M8.5 10.5L12 14l3.5-3.5" />
                              <path d="M5 18h14" />
                            </svg>
                          </button>
                          {embeddingDownloadError ? (
                            <div className="settings-hint err">{embeddingDownloadError}</div>
                          ) : null}
                        </div>
                        {embeddingBusy ? (
                          <div className="settings-health-bar">
                            <span />
                          </div>
                        ) : null}
                      </div>
                    ) : null}
                    {issues.map((issue) => (
                      <div
                        key={issue.code}
                        className={`settings-health-item ${issue.severity}`}
                      >
                        <div className="settings-health-title">
                          {issue.message}
                        </div>
                        <div className="settings-health-fix">{issue.fix}</div>
                      </div>
                    ))}
                  </div>
                );
              })()
            ) : (
              <div className="settings-muted">Checking system health…</div>
            )}
          </section>

          <section className="settings-section">
            <div className="settings-section-title">Model</div>
            <div className="settings-row settings-col">
              <label className="settings-label">GGUF model path</label>
              <div className="settings-inline">
                <input
                  className="settings-input"
                  value={settings.llm_model_path}
                  placeholder="Path to .gguf"
                  onChange={(e) => {
                    setSettings((p) => ({ ...p, llm_model_path: e.target.value }));
                  }}
                  onBlur={async () => {
                    if (!settings.llm_model_path) return;
                    await validateModel(settings.llm_model_path);
                  }}
                />
                <button
                  className="settings-icon-btn"
                  type="button"
                  onClick={browseModel}
                  aria-label="Browse for a GGUF model"
                  title="Browse model file"
                >
                  <svg className="settings-icon" viewBox="0 0 24 24" aria-hidden="true">
                    <path d="M4 8h6l2 2h8v8H4z" />
                    <path d="M4 8V6h6l2 2" />
                  </svg>
                </button>
              </div>
              {modelValidation ? (
                <div className={`settings-hint ${modelValidation.ok ? "ok" : "err"}`}>
                  {modelValidation.msg}
                </div>
              ) : null}
              <div className="settings-hint">
                {llmLoaded && llmInfo ? (
                  <div>
                    Loaded: <code>{llmInfo.name || "Unknown model"}</code>{" "}
                    {llmInfo.architecture ? <span>({llmInfo.architecture})</span> : null}
                    <div className="settings-muted">
                      ctx runtime{" "}
                      <code>{String(llmInfo.ctx_runtime ?? "")}</code>{" "}
                      {llmInfo.ctx_train ? (
                        <>
                          • ctx train <code>{String(llmInfo.ctx_train)}</code>
                        </>
                      ) : null}
                      {typeof llmInfo.file_type === "number" ? (
                        <>
                          {" "}
                          • file_type <code>{String(llmInfo.file_type)}</code>
                        </>
                      ) : null}
                      {llmInfo.chat_template_kind ? (
                        <>
                          {" "}
                          • template <code>{llmInfo.chat_template_kind}</code>
                        </>
                      ) : null}
                    </div>
                  </div>
                ) : (
                  <div className="settings-muted">
                    Model info appears after the engine loads the model (start a chat or restart after applying settings).
                  </div>
                )}
              </div>
            </div>

            <div className="settings-row settings-col">
              <label className="settings-label">Context length</label>
              <div className="settings-inline">
                <select
                  className="settings-select settings-select-compact"
                  value={String(settings.llm_ctx_size || 32768)}
                  onChange={(e) => {
                    const v = Number(e.target.value);
                    setSettings((p) => ({ ...p, llm_ctx_size: v }));
                  }}
                >
                  <option value="8192">8k</option>
                  <option value="32768">32k</option>
                </select>
                <div className="settings-apply-wrap">
                  <button
                    className="settings-reset-confirm settings-apply-confirm"
                    type="button"
                    disabled={busy || !!engineBusy?.busy}
                    onClick={applyLlmSettings}
                    aria-label="Apply model settings (resets all data)"
                    title="Apply (resets all data)"
                  >
                    Apply model
                  </button>
                </div>
              </div>
              <div className="settings-hint">
                Changing model/context clears chats, files, indexes, and KV sessions.
              </div>
            </div>
          </section>

          <section className="settings-section">
            <div className="settings-section-title">Raw API Server</div>
            <div className="settings-row settings-col">
              <label className="settings-label">Toggle</label>
              <div className="settings-inline settings-inline-tight">
                <button
                  className={`settings-icon-btn settings-toggle-btn ${rawRunning ? "active" : ""}`}
                  type="button"
                  onClick={toggleRawEngine}
                  disabled={rawEngineBusy || rawStarting}
                  aria-label={rawRunning ? "Stop raw API server" : "Start raw API server"}
                  title={rawRunning ? "Stop raw API server" : "Start raw API server"}
                >
                  <svg className="settings-icon" viewBox="0 0 24 24" aria-hidden="true">
                    <path d="M12 3.5v8.2" />
                    <path d="M7.3 5.8a7 7 0 1 0 9.4 0" />
                  </svg>
                </button>
                <div className={`settings-raw-status ${rawRunning ? "on" : "off"}`}>
                  {rawRunning ? "Running" : rawStarting ? "Starting…" : "Stopped"}
                </div>
                {rawEngineBusy || rawStarting ? <div className="settings-muted">Working…</div> : null}
              </div>
              <div className="settings-hint settings-raw-hint">
                {rawBaseUrl ? (
                  <>
                    Base URL <code>{rawBaseUrl}</code>
                    {rawEngineStatus?.requested_port &&
                    rawEngineStatus.port &&
                    rawEngineStatus.requested_port !== rawEngineStatus.port ? (
                      <>
                        {" "}
                        • requested <code>{rawEngineStatus.requested_port}</code>
                      </>
                    ) : null}
                  </>
                ) : (
                  rawStarting
                    ? "Starting raw server. This may take a moment while the model loads."
                    : "Raw server is off. Starting it will load the model a second time."
                )}
              </div>
              {rawEngineError || rawEngineStatus?.error ? (
                <div className="settings-hint err settings-raw-hint">
                  {rawEngineError || rawEngineStatus?.error}
                </div>
              ) : null}

              <details
                className="settings-dropdown settings-raw-dropdown"
                open={rawLogsOpen}
                onToggle={(e) => setRawLogsOpen((e.target as HTMLDetailsElement).open)}
              >
                <summary>Raw server logs & options</summary>
                <div className="settings-dropdown-panel settings-raw-panel settings-scrollable">
                  <div className="settings-raw-config">
                    <div className="settings-row settings-col">
                      <label className="settings-label">Host</label>
                      <div className="settings-inline">
                        <input
                          className="settings-input settings-input-compact"
                          type="text"
                          value={settings.raw_engine_host}
                          inputMode="decimal"
                          pattern="[0-9.]*"
                          onChange={(e) => {
                            const next = e.target.value.replace(/[^0-9.]/g, "");
                            setSettings((p) => ({ ...p, raw_engine_host: next }));
                          }}
                          disabled={rawConfigLocked || rawEngineBusy}
                        />
                        <div className="settings-muted">Default: {rawDefaultLabel(rawDefaults.raw_engine_host)}</div>
                      </div>
                    </div>

                    <div className="settings-row settings-col">
                      <label className="settings-label">Port</label>
                      <div className="settings-inline">
                        <input
                          className="settings-input settings-input-compact"
                          type="text"
                          inputMode="numeric"
                          pattern="[0-9]*"
                          value={settings.raw_engine_port ? String(settings.raw_engine_port) : ""}
                          onChange={(e) => {
                            const next = e.target.value.replace(/[^0-9]/g, "");
                            setSettings((p) => ({
                              ...p,
                              raw_engine_port: next ? Number(next) : 0,
                            }));
                          }}
                          disabled={rawConfigLocked || rawEngineBusy}
                        />
                        <div className="settings-muted">Default: {rawDefaultLabel(rawDefaults.raw_engine_port)}</div>
                      </div>
                    </div>

                    <div className="settings-row settings-col">
                      <label className="settings-label">Context size</label>
                      <div className="settings-inline">
                        <select
                          className="settings-select settings-input-compact"
                          value={settings.raw_engine_ctx ?? ""}
                          onChange={(e) => {
                            const v = e.target.value;
                            setSettings((p) => ({
                              ...p,
                              raw_engine_ctx: v ? Number(v) : null,
                            }));
                          }}
                          disabled={rawConfigLocked || rawEngineBusy}
                        >
                          <option value="">Auto</option>
                          <option value="8192">8k</option>
                          <option value="32768">32k</option>
                        </select>
                        <div className="settings-muted">Default: {rawDefaultLabel(rawDefaults.raw_engine_ctx)}</div>
                      </div>
                    </div>

                    <div className="settings-row">
                      <button
                        className={`settings-btn ${rawAdvancedOpen ? "active" : ""}`}
                        type="button"
                        onClick={() => setRawAdvancedOpen((prev) => !prev)}
                      >
                        Advanced options
                      </button>
                    </div>

                    {rawAdvancedOpen ? (
                      <div className="settings-raw-advanced">
                        <div className="settings-row settings-col">
                          <label className="settings-label">Threads</label>
                          <div className="settings-inline">
                            <input
                              className="settings-input settings-input-compact"
                              type="text"
                              inputMode="numeric"
                              pattern="[0-9]*"
                              value={
                                settings.raw_engine_threads === null
                                  ? ""
                                  : String(settings.raw_engine_threads)
                              }
                              onChange={(e) => {
                                const next = e.target.value.replace(/[^0-9]/g, "");
                                setSettings((p) => ({
                                  ...p,
                                  raw_engine_threads: next ? Number(next) : null,
                                }));
                              }}
                              disabled={rawConfigLocked || rawEngineBusy}
                            />
                            <div className="settings-muted">
                              Default: {rawDefaultLabel(rawDefaults.raw_engine_threads)}
                            </div>
                          </div>
                        </div>

                        <div className="settings-row settings-col">
                          <label className="settings-label">GPU layers</label>
                          <div className="settings-inline">
                            <input
                              className="settings-input settings-input-compact"
                              type="text"
                              inputMode="numeric"
                              pattern="[0-9]*"
                              value={
                                settings.raw_engine_gpu_layers === null
                                  ? ""
                                  : String(settings.raw_engine_gpu_layers)
                              }
                              onChange={(e) => {
                                const next = e.target.value.replace(/[^0-9]/g, "");
                                setSettings((p) => ({
                                  ...p,
                                  raw_engine_gpu_layers: next ? Number(next) : null,
                                }));
                              }}
                              disabled={rawConfigLocked || rawEngineBusy}
                            />
                            <div className="settings-muted">
                              Default: {rawDefaultLabel(rawDefaults.raw_engine_gpu_layers)}
                            </div>
                          </div>
                          <div className="settings-muted">Use 0 for CPU-only.</div>
                        </div>

                        <div className="settings-row settings-col">
                          <label className="settings-label">Default max tokens</label>
                          <div className="settings-inline">
                            <input
                              className="settings-input settings-input-compact"
                              type="text"
                              inputMode="numeric"
                              pattern="[0-9]*"
                              value={String(settings.raw_engine_max_tokens)}
                              onChange={(e) => {
                                const next = e.target.value.replace(/[^0-9]/g, "");
                                setSettings((p) => ({
                                  ...p,
                                  raw_engine_max_tokens: next ? Number(next) : rawMaxTokensDefault,
                                }));
                              }}
                              disabled={rawConfigLocked || rawEngineBusy}
                            />
                            <div className="settings-muted">
                              Default: {rawDefaultLabel(rawDefaults.raw_engine_max_tokens)}
                            </div>
                          </div>
                        </div>
                      </div>
                    ) : null}

                    <div className="settings-row">
                      <button
                        className="settings-btn"
                        type="button"
                        onClick={saveRawEngineSettings}
                        disabled={rawConfigLocked || rawEngineBusy}
                      >
                        Save raw server settings
                      </button>
                      {rawConfigLocked ? (
                        <div className="settings-muted">Stop the server to edit settings.</div>
                      ) : null}
                    </div>
                  </div>
                  <div className="settings-raw-console settings-scrollable" ref={rawLogRef}>
                    {rawEngineLogs.length ? (
                      rawEngineLogs.map((entry, idx) => (
                        <div key={`${entry.ts}-${idx}`} className="settings-raw-line">
                          <span className="settings-raw-time">{formatLogTime(entry.ts)}</span>
                          <span className="settings-raw-text">{entry.line}</span>
                        </div>
                      ))
                    ) : (
                      <div className="settings-muted">No raw server logs yet.</div>
                    )}
                  </div>
                </div>
              </details>
            </div>
          </section>

          <section className="settings-section">
            <div className="settings-section-title">Appearance</div>
            <div className="settings-row">
              <button
                className={`settings-icon-btn ${themeMode === "system" ? "active" : ""}`}
                type="button"
                onClick={() => {
                  onThemeModeChange("system");
                  saveSettingsPatch({ theme_mode: "system" });
                }}
                aria-label="Use system theme"
                title="System"
              >
                <svg className="settings-icon" viewBox="0 0 24 24" aria-hidden="true">
                  <rect x="3.5" y="4.5" width="17" height="12" rx="2" />
                  <path d="M8 19.5h8" />
                  <path d="M12 16.5v3" />
                </svg>
              </button>
              <button
                className={`settings-icon-btn ${themeMode === "dark" ? "active" : ""}`}
                type="button"
                onClick={() => {
                  onThemeModeChange("dark");
                  saveSettingsPatch({ theme_mode: "dark" });
                }}
                aria-label="Use dark theme"
                title="Dark"
              >
                <svg className="settings-icon" viewBox="0 0 24 24" aria-hidden="true">
                  <path d="M14.5 4.5a7 7 0 1 0 5 12.5a7.5 7.5 0 0 1 -5 -12.5z" />
                </svg>
              </button>
              <button
                className={`settings-icon-btn ${themeMode === "light" ? "active" : ""}`}
                type="button"
                onClick={() => {
                  onThemeModeChange("light");
                  saveSettingsPatch({ theme_mode: "light" });
                }}
                aria-label="Use light theme"
                title="Light"
              >
                <svg className="settings-icon" viewBox="0 0 24 24" aria-hidden="true">
                  <circle cx="12" cy="12" r="4.2" />
                  <path d="M12 2.5v3.2" />
                  <path d="M12 18.3v3.2" />
                  <path d="M2.5 12h3.2" />
                  <path d="M18.3 12h3.2" />
                  <path d="M4.6 4.6l2.3 2.3" />
                  <path d="M17.1 17.1l2.3 2.3" />
                  <path d="M19.4 4.6l-2.3 2.3" />
                  <path d="M6.9 17.1l-2.3 2.3" />
                </svg>
              </button>
            </div>
          </section>

          <section className="settings-section danger">
            <div className="settings-section-title">Storage</div>
            <div className="settings-row">
              <div className="settings-kv">
                <div className="k">Workspace</div>
                <div className="v">{storage?.base || "…"}</div>
              </div>
              <div className="settings-kv settings-kv-total">
                <div className="k">Total</div>
                <div className="v settings-total-row">
                  <span>{storage ? formatBytes(storage.total_bytes) : "…"}</span>
                  <button
                    className="settings-icon-btn settings-refresh-btn"
                    type="button"
                    onClick={refreshStorage}
                    aria-label="Refresh storage usage"
                    title="Refresh storage"
                  >
                    <svg className="settings-icon" viewBox="0 0 24 24" aria-hidden="true">
                      <path d="M20 12a8 8 0 1 1-2.3-5.6" />
                      <path d="M20 5v5h-5" />
                    </svg>
                  </button>
                </div>
              </div>
            </div>
            <details className="settings-dropdown">
              <summary>Storage files</summary>
              <div className="settings-dropdown-panel">
                <div className="settings-breakdown settings-scrollable">
                  {sortedBreakdown.map(([name, info]) => (
                    <div key={name} className="settings-breakdown-row">
                      <div className="name">{name}</div>
                      <div className="bytes">{formatBytes(info.bytes)}</div>
                    </div>
                  ))}
                </div>
              </div>
            </details>
            <div className="settings-divider" />
            <div className="settings-section-title">Reset</div>
            <div className="settings-row">
              <div className="settings-reset-wrap" ref={resetWrapRef}>
                <button
                  className="settings-icon-btn danger reset-trigger"
                  type="button"
                  disabled={busy || !!engineBusy?.busy}
                  onClick={() => setResetOpen((prev) => !prev)}
                  aria-label="Reset all data"
                  title="Reset all data"
                >
                  <svg className="settings-icon" viewBox="0 0 24 24" aria-hidden="true">
                    <path d="M4 7h16" />
                    <path d="M9 7V5h6v2" />
                    <path d="M7 7l1 12h8l1-12" />
                  </svg>
                </button>
                {resetOpen && (
                  <div className="settings-reset-popover" role="dialog" aria-label="Confirm reset">
                    <button
                      className="settings-reset-confirm"
                      type="button"
                      disabled={busy || !!engineBusy?.busy}
                      onClick={resetAll}
                    >
                      Reset
                    </button>
                  </div>
                )}
              </div>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
