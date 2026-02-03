import { useEffect, useMemo, useRef, useState } from "react";
import { EngineResponse, engine } from "../api/engine";
import { invoke } from "@tauri-apps/api/core";

export type ThemeMode = "system" | "dark" | "light";

type SettingsState = {
  theme_mode: ThemeMode;
  llm_model_path: string;
  llm_gpu_layers: number;
  llm_ctx_size: number;
  rag_default_mode: "small_doc" | "rag";
  rag_default_detail: number;
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
  n_layer?: number | null;
  n_head?: number | null;
  n_head_kv?: number | null;
  n_embd?: number | null;
  rope_type?: string | null;
  rope_freq_base?: number | null;
  vocab_size?: number | null;
  kv_cache_gib?: number | null;
  ctx_train?: number | null;
  ctx_runtime?: number | null;
  chat_template_kind?: string;
  chat_template_name?: string;
  chat_format?: string;
  prompt_renderer?: string;
  tokenizer_model?: string;
  add_bos_token?: string;
  bos_token_id?: number | null;
  eos_token_id?: number | null;
};

type SettingsResponse = {
  settings?: SettingsState;
  llm?: {
    loaded?: boolean;
    model_info?: LlmModelInfo | null;
    ctx_sizes?: number[];
    ctx_max?: number | null;
  };
  raw_engine?: {
    defaults?: Partial<SettingsState>;
    ctx_sizes?: number[];
    ctx_max?: number | null;
  };
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

function formatCtxLabel(value: number) {
  if (!Number.isFinite(value) || value <= 0) return String(value);
  if (value % 1024 === 0) return `${value / 1024}k`;
  return String(value);
}

export function SettingsModal(props: {
  open: boolean;
  onClose: () => void;
  themeMode: ThemeMode;
  onThemeModeChange: (mode: ThemeMode) => void;
  initialTab?: "general" | "model" | "retrieval" | "raw" | "storage";
}) {
  const { open, onClose, themeMode, onThemeModeChange, initialTab = "general" } = props;
  const normalizeCtxValue = (
    value: number | null | undefined,
    options: number[],
    fallback?: number | null
  ) => {
    if (value !== null && value !== undefined && options.includes(value)) return value;
    if (fallback !== null && fallback !== undefined && options.includes(fallback)) return fallback;
    return options[options.length - 1] ?? (fallback ?? 32768);
  };
  const RAW_ENGINE_DEFAULTS: SettingsState = {
    theme_mode: themeMode,
    llm_model_path: "",
    llm_gpu_layers: 99,
    llm_ctx_size: 32768,
    rag_default_mode: "small_doc",
    rag_default_detail: 3,
    raw_engine_host: "127.0.0.1",
    raw_engine_port: 11435,
    raw_engine_model_path: "",
    raw_engine_ctx: 32768,
    raw_engine_threads: null,
    raw_engine_gpu_layers: null,
    raw_engine_max_tokens: 1024,
    raw_engine_embedding_model: "nomic-embed-text-v1.5",
    raw_engine_embedding_auto_download: false,
    raw_engine_log_preview_chars: 400,
    raw_engine_log_prompts: false,
    raw_engine_log_completions: false,
  };
  const [settings, setSettings] = useState<SettingsState>({
    theme_mode: themeMode,
    llm_model_path: "",
    llm_gpu_layers: 99,
    llm_ctx_size: 32768,
    rag_default_mode: "small_doc",
    rag_default_detail: 3,
    raw_engine_host: "127.0.0.1",
    raw_engine_port: 11435,
    raw_engine_model_path: "",
    raw_engine_ctx: 32768,
    raw_engine_threads: null,
    raw_engine_gpu_layers: null,
    raw_engine_max_tokens: 1024,
    raw_engine_embedding_model: "nomic-embed-text-v1.5",
    raw_engine_embedding_auto_download: false,
    raw_engine_log_preview_chars: 400,
    raw_engine_log_prompts: false,
    raw_engine_log_completions: false,
  });
  const [llmCtxOptions, setLlmCtxOptions] = useState<number[]>([8192, 32768]);
  const [rawCtxOptions, setRawCtxOptions] = useState<number[]>([8192, 32768]);
  const [llmCtxMax, setLlmCtxMax] = useState<number | null>(null);
  const [rawCtxMax, setRawCtxMax] = useState<number | null>(null);
  const [storage, setStorage] = useState<StorageUsage | null>(null);
  const [modelValidation, setModelValidation] = useState<{
    ok: boolean;
    msg: string;
    templateName?: string;
    templateKind?: string;
  } | null>(null);
  const [modelApplyPhase, setModelApplyPhase] = useState<"idle" | "validating" | "applying" | "loading">("idle");
  const [_cleanResult, setCleanResult] = useState<string | null>(null);
  const [resetOpen, setResetOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [engineBusy, setEngineBusy] = useState<BusyState | null>(null);
  const [llmLoaded, setLlmLoaded] = useState(false);
  const [llmInfo, setLlmInfo] = useState<LlmModelInfo | null>(null);
  const [_, setPendingModelPath] = useState<string | null>(null);
  const [healthReport, setHealthReport] = useState<HealthReport | null>(null);
  const [embeddingPresent, setEmbeddingPresent] = useState<boolean | null>(null);
  const [embeddingDownloading, setEmbeddingDownloading] = useState(false);
  const [embeddingDownloadActive, setEmbeddingDownloadActive] = useState(false);
  const [embeddingDownloadError, setEmbeddingDownloadError] = useState<string | null>(null);
  const [embeddingForceStopped, setEmbeddingForceStopped] = useState(false);
  const [rawEngineStatus, setRawEngineStatus] = useState<RawEngineStatus | null>(null);
  const [rawEngineLogs, setRawEngineLogs] = useState<RawEngineLogLine[]>([]);
  const [rawEngineError, setRawEngineError] = useState<string | null>(null);
  const lastCtxTrainRef = useRef<number | null>(null);
  const [rawEngineBusy, setRawEngineBusy] = useState(false);
  const [rawAdvancedOpen, setRawAdvancedOpen] = useState(false);
  const [rawEngineDefaults, setRawEngineDefaults] = useState<Partial<SettingsState>>(RAW_ENGINE_DEFAULTS);
  const resetWrapRef = useRef<HTMLDivElement | null>(null);
  const rawLogRef = useRef<HTMLDivElement | null>(null);
  const modalRef = useRef<HTMLDivElement | null>(null);
  const scrollTimersRef = useRef<Map<HTMLElement, number>>(new Map());
  const modelLoadPollRef = useRef<number | null>(null);
  const modelLoadStartRef = useRef<number | null>(null);
  const modelApplyPhaseRef = useRef(modelApplyPhase);
  const modelValidationCacheRef = useRef<{path: string, time: number} | null>(null);
  const [activeTab, setActiveTab] = useState<"general" | "model" | "retrieval" | "raw" | "storage">(initialTab);
  const rawLogsOpen = activeTab === "raw";
  const ragDefaultsSaveRef = useRef<number | null>(null);

  function clearModelLoadPoll() {
    if (modelLoadPollRef.current != null) {
      window.clearTimeout(modelLoadPollRef.current);
      modelLoadPollRef.current = null;
    }
    modelLoadStartRef.current = null;
  }

  function scheduleRagDefaultsSave(patch: Partial<SettingsState>) {
    if (ragDefaultsSaveRef.current != null) {
      window.clearTimeout(ragDefaultsSaveRef.current);
    }
    ragDefaultsSaveRef.current = window.setTimeout(() => {
      ragDefaultsSaveRef.current = null;
      saveSettingsPatch(patch);
    }, 250);
  }


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
    modelApplyPhaseRef.current = modelApplyPhase;
  }, [modelApplyPhase]);

  useEffect(() => {
    if (!open) return;
    setActiveTab(initialTab);
    setResetOpen(false);
    setCleanResult(null);
    setModelValidation(null);
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
      let llmCtxList = llmCtxOptions;
      let rawCtxList = rawCtxOptions;
      if (res.ok) {
        const llm = (res.data as any)?.llm as SettingsResponse["llm"];
        const raw = (res.data as any)?.raw_engine as SettingsResponse["raw_engine"];
        const nextLlmCtx = Array.isArray(llm?.ctx_sizes)
          ? llm!.ctx_sizes!.filter((v) => Number.isFinite(v))
          : [];
        const nextRawCtx = Array.isArray(raw?.ctx_sizes)
          ? raw!.ctx_sizes!.filter((v) => Number.isFinite(v))
          : [];
        llmCtxList = nextLlmCtx.length ? nextLlmCtx : [8192, 32768];
        rawCtxList = nextRawCtx.length ? nextRawCtx : llmCtxList;
        setLlmCtxOptions(llmCtxList);
        setRawCtxOptions(rawCtxList);
        setLlmCtxMax(typeof llm?.ctx_max === "number" ? llm?.ctx_max : null);
        setRawCtxMax(typeof raw?.ctx_max === "number" ? raw?.ctx_max : null);
      }

      if (res.ok && (res.data as any)?.settings) {
        const next = (res.data as any).settings as SettingsState;
        const llmCtx = normalizeCtxValue(next.llm_ctx_size, llmCtxList, next.llm_ctx_size);
        const rawCtx = normalizeCtxValue(next.raw_engine_ctx, rawCtxList, llmCtx);
        setSettings((prev) => ({ ...prev, ...next, llm_ctx_size: llmCtx, raw_engine_ctx: rawCtx }));
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
        const embedPresent = (res.data as any)?.embedding?.present;
        if (typeof embedPresent === "boolean") {
          setEmbeddingPresent(embedPresent);
        }
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
    if (!open) return;
    const ctxTrain = llmInfo?.ctx_train ?? null;
    if (ctxTrain === lastCtxTrainRef.current) return;
    lastCtxTrainRef.current = ctxTrain;
    refreshCtxOptionsFromSettings().catch(() => {});
  }, [open, llmInfo?.ctx_train]);

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
    const id = window.setInterval(tick, 30000);  // Reduced from 15s to 30s
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
      if (modelApplyPhaseRef.current !== "idle") return;
      await refreshHealth();
    };
    tick();
    const id = window.setInterval(tick, 30000);  // Reduced from 10s to 30s
    return () => {
      alive = false;
      window.clearInterval(id);
    };
  }, [open]);

  useEffect(() => {
    return () => {
      clearModelLoadPoll();
    };
  }, []);

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
    const id = window.setInterval(tick, rawLogsOpen ? 2500 : 5000);  // 5s base, 2.5s when logs open
    return () => {
      alive = false;
      window.clearInterval(id);
    };
  }, [open, rawLogsOpen]);

  useEffect(() => {
    const checks = healthReport?.checks || {};
    const status = checks.embedding_download_status;
    const err = checks.embedding_download_error;
    const present = checks.embedding_present === true;
    if (typeof checks.embedding_present === "boolean") {
      setEmbeddingPresent(checks.embedding_present);
    }
    if (present) {
      setEmbeddingDownloading(false);
      setEmbeddingDownloadActive(false);
    } else if (status === "downloading") {
      if (!embeddingForceStopped) {
        setEmbeddingDownloading(true);
      }
      setEmbeddingDownloadActive(true);
    } else if (status === "ready" || status === "error") {
      setEmbeddingDownloading(false);
      setEmbeddingDownloadActive(false);
    } else if (status === "idle") {
      // If we already kicked off a download, keep showing activity until we see ready/error/present.
      setEmbeddingDownloading(embeddingDownloadActive && !embeddingForceStopped);
    } else if (status === undefined || status === null) {
      // Fallback health reports may omit status; keep download active unless we know it's done.
      setEmbeddingDownloading(embeddingDownloadActive && !embeddingForceStopped);
    }
    if (status === "ready" || status === "idle") {
      setEmbeddingForceStopped(false);
    }
    if (err && err !== "Download cancelled.") {
      setEmbeddingDownloadError(err);
    } else if (!err || err === "Download cancelled.") {
      setEmbeddingDownloadError(null);
    }
  }, [healthReport, embeddingForceStopped, embeddingDownloadActive]);

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

  const quantizationTag = useMemo(() => {
    const path = settings.llm_model_path || "";
    const base = path.split(/[\\/]/).pop() || "";
    const match = base.match(/(Q\d+(?:_[A-Z0-9]+)+|F16|BF16|FP16|FP32)/i);
    return match ? match[1].toUpperCase() : "";
  }, [settings.llm_model_path]);

  const displayModelInfo = llmLoaded && llmInfo ? llmInfo : null;
  const displayModelInfoLabel = llmLoaded && llmInfo ? "Model info" : "";

  const healthModelLabel = useMemo(() => {
    if (llmInfo?.name) {
      return llmInfo.size_label ? `${llmInfo.name} · ${llmInfo.size_label}` : llmInfo.name;
    }
    const path = settings.llm_model_path || "";
    const base = path.split(/[\\/]/).pop() || "";
    return base || "Not configured";
  }, [llmInfo?.name, llmInfo?.size_label, settings.llm_model_path]);

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

  function withTimeout<T>(promise: Promise<T>, timeoutMs: number, label: string) {
    return new Promise<T>((resolve, reject) => {
      const timer = window.setTimeout(() => {
        reject(new Error(`timeout: ${label}`));
      }, timeoutMs);
      promise.then(
        (value) => {
          window.clearTimeout(timer);
          resolve(value);
        },
        (err) => {
          window.clearTimeout(timer);
          reject(err);
        }
      );
    });
  }

  async function refreshStorage() {
    const st = await engine<StorageUsage>("/settings/storage", undefined, "GET");
    if (st.ok) setStorage(st.data as any);
  }

  async function refreshHealth(opts?: { full?: boolean; includeLlmInfo?: boolean }) {
    const full = opts?.full ?? false;
    const includeLlmInfo = opts?.includeLlmInfo ?? true;
    const fallbackFromSettings = async (message: string, code: "health_timeout" | "health_unavailable") => {
      const issues: HealthIssue[] = [];
      const checks: Record<string, any> = { health_fallback: true };
      try {
        const settingsRes = await engine<SettingsResponse>("/settings", undefined, "GET");
        if (settingsRes.ok && settingsRes.data) {
          const modelPath = (settingsRes.data as any)?.settings?.llm_model_path;
          if (!modelPath) {
            issues.push({
              code: "model_not_configured",
              severity: "error",
              message: "No model is configured yet.",
              fix: "Open Settings → Model and choose a GGUF model file.",
              action: "open_settings",
            });
          }
          const embedPresent = (settingsRes.data as any)?.embedding?.present;
          if (embedPresent === false) {
            issues.push({
              code: "embedding_missing",
              severity: "warning",
              message: "Embedding model files are missing.",
              fix: "Download embeddings in Settings → Model.",
              action: "open_settings",
            });
            checks.embedding_present = false;
            checks.embedding_download_status = "idle";
            setEmbeddingPresent(false);
          } else if (embedPresent === true) {
            checks.embedding_present = true;
            checks.embedding_download_status = "ready";
            setEmbeddingPresent(true);
          }
        }
      } catch {
        // Ignore fallback errors; keep the health timeout issue only.
      }
      if (issues.length === 0) {
        issues.push({
          code,
          severity: "error",
          message: code === "health_timeout" ? "Health check timed out" : "Health check failed",
          fix: message || "Unable to fetch system health. Try again or restart the app.",
        });
      }
      setHealthReport({ ok: false, issues, checks });
    };
    try {
      const res = await withTimeout(
        engine<HealthReport>(`/settings/health${full ? "?full=1" : ""}`, undefined, "GET"),
        8000,
        "health"
      );
      if (res.ok && res.data) {
        setHealthReport(res.data as any);
      } else {
        await fallbackFromSettings(getErrorText(res, "Unable to fetch system health. Try again or restart the app."), "health_unavailable");
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      await fallbackFromSettings(message || "Unable to fetch system health. Try again or restart the app.", "health_timeout");
    }

    const canFetchLlmInfo =
      includeLlmInfo &&
      modelApplyPhaseRef.current === "idle" &&
      modelLoadPollRef.current == null;
    if (canFetchLlmInfo) {
      try {
        // Also fetch LLM info separately (health endpoint doesn't include it)
        const llmRes = await withTimeout(
          engine<{ loaded: boolean; model_info: LlmModelInfo | null }>("/settings/llm/info?load=0", undefined, "GET"),
          12000,
          "llm info"
        );
        if (llmRes.ok) {
          setLlmLoaded(Boolean(llmRes.data?.loaded));
          setLlmInfo(llmRes.data?.model_info || null);
        }
      } catch {
        // If the model is still loading, keep the previous info and let polling handle it.
      }
    }
  }

  async function pollLlmInfoUntilReady() {
    if (modelLoadPollRef.current != null) return;
    const start = Date.now();
    modelLoadStartRef.current = start;
    let firstLoad = true;

    const tick = async () => {
      try {
        const llmRes = await withTimeout(
          engine<{ loaded: boolean; model_info: LlmModelInfo | null }>(
            `/settings/llm/info?load=${firstLoad ? "1" : "0"}`,
            undefined,
            "GET"
          ),
          firstLoad ? 120000 : 45000,
          "llm info"
        );
        firstLoad = false;
        if (llmRes.ok && llmRes.data?.loaded) {
          setLlmLoaded(true);
          setLlmInfo(llmRes.data?.model_info || null);
          setPendingModelPath(null);
          setModelApplyPhase("idle");
          clearModelLoadPoll();
          return;
        }
      } catch {
        // keep polling; model may still be loading or errored
      }

      const elapsed = Date.now() - start;
      if (elapsed > 90_000) {
        setCleanResult("Model settings applied. Still waiting for the model to load; restart the app/engine if needed.");
      }
      modelLoadPollRef.current = window.setTimeout(tick, 3000);
    };

    modelLoadPollRef.current = window.setTimeout(tick, 1000);
  }

  async function refreshRawEngineStatus() {
    const res = await engine<RawEngineStatus>("/settings/raw_engine/status", undefined, "GET");
    if (res.ok) setRawEngineStatus(res.data as any);
  }

  async function refreshRawEngineLogs() {
    const res = await engine<{ lines?: RawEngineLogLine[] }>("/settings/raw_engine/logs?limit=250", undefined, "GET");
    if (res.ok) setRawEngineLogs((res.data as any)?.lines || []);
  }

  async function refreshCtxOptionsFromSettings() {
    const res = await engine<SettingsResponse>("/settings", undefined, "GET");
    if (!res.ok) return;
    const llm = (res.data as any)?.llm as SettingsResponse["llm"];
    const raw = (res.data as any)?.raw_engine as SettingsResponse["raw_engine"];
    const nextLlmCtx = Array.isArray(llm?.ctx_sizes)
      ? llm!.ctx_sizes!.filter((v) => Number.isFinite(v))
      : [];
    const nextRawCtx = Array.isArray(raw?.ctx_sizes)
      ? raw!.ctx_sizes!.filter((v) => Number.isFinite(v))
      : [];
    const llmCtxList = nextLlmCtx.length ? nextLlmCtx : [8192, 32768];
    const rawCtxList = nextRawCtx.length ? nextRawCtx : llmCtxList;
    setLlmCtxOptions(llmCtxList);
    setRawCtxOptions(rawCtxList);
    setLlmCtxMax(typeof llm?.ctx_max === "number" ? llm?.ctx_max : null);
    setRawCtxMax(typeof raw?.ctx_max === "number" ? raw?.ctx_max : null);
    setSettings((prev) => {
      const llmCtx = normalizeCtxValue(prev.llm_ctx_size, llmCtxList, prev.llm_ctx_size);
      const rawCtx = normalizeCtxValue(prev.raw_engine_ctx, rawCtxList, llmCtx);
      if (llmCtx === prev.llm_ctx_size && rawCtx === prev.raw_engine_ctx) return prev;
      return { ...prev, llm_ctx_size: llmCtx, raw_engine_ctx: rawCtx };
    });
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
      setCleanResult(null);
    } else {
      setSettings(prev);
      setCleanResult(getErrorText(res, "Failed to save settings"));
    }
  }

  async function validateModel(path: string, options?: { skipCache?: boolean }) {
    // Debounce: skip API call if validated same path within 5 seconds
    const last = modelValidationCacheRef.current;
    if (!options?.skipCache && last && last.path === path && Date.now() - last.time < 5000) {
      // Still show the cached validation state in UI
      return true;
    }

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
    setModelValidation({
      ok: true,
      msg: (res.data as any)?.note || "Looks valid",
      templateName: (res.data as any)?.chat_template_name || undefined,
      templateKind: (res.data as any)?.chat_template_kind || undefined,
    });
    const ctxSizesRaw = (res.data as any)?.ctx_sizes;
    const ctxMaxRaw = (res.data as any)?.ctx_max;
    if (Array.isArray(ctxSizesRaw)) {
      const list = ctxSizesRaw
        .filter((v: any) => Number.isFinite(v))
        .sort((a: number, b: number) => a - b);
      if (list.length) {
        setLlmCtxOptions(list);
        setLlmCtxMax(typeof ctxMaxRaw === "number" ? ctxMaxRaw : null);
        setSettings((prev) => {
          const nextCtx = normalizeCtxValue(prev.llm_ctx_size, list, prev.llm_ctx_size);
          if (nextCtx === prev.llm_ctx_size) return prev;
          return { ...prev, llm_ctx_size: nextCtx };
        });
      }
    }
    // Cache the successful validation
    modelValidationCacheRef.current = {path, time: Date.now()};
    return true;
  }

  async function browseModel() {
    try {
      const res = await invoke<{ path?: string | null }>("pick_model_file");
      const path = res?.path || "";
      if (!path) return;
      setSettings((p) => ({ ...p, llm_model_path: path }));
      await validateModel(path);
    } catch (e: any) {
      setModelValidation({ ok: false, msg: e?.message || String(e) });
    }
  }

  async function applyLlmSettings() {
    if (engineBusy?.busy) {
      setCleanResult("Background work is running. Wait for it to finish before applying model settings.");
      return;
    }

    clearModelLoadPoll();
    setModelApplyPhase("validating");
    const ok = await validateModel(settings.llm_model_path, { skipCache: true });
    if (!ok) {
      setModelApplyPhase("idle");
      return;
    }

    setModelApplyPhase("applying");
    setBusy(true);
    let res: EngineResponse<any>;
    try {
      res = await withTimeout(
        engine<any>(
          "/settings/llm/apply",
          {
            confirm: true,
            settings: {
              llm_model_path: settings.llm_model_path,
              llm_ctx_size: settings.llm_ctx_size,
            },
          },
          "POST"
        ),
        20_000,
        "apply model"
      );
    } catch (err: any) {
      setBusy(false);
      setModelApplyPhase("idle");
      setCleanResult(`Apply timed out. The engine may still be resetting. Restart the app/engine if this persists. (${err?.message || "timeout"})`);
      return;
    }
    setBusy(false);
    if (!res.ok) {
      setModelApplyPhase("idle");
      setCleanResult(getErrorText(res, "Failed to apply model settings"));
      return;
    }

    // Keep UI in loading state until the engine reports the model is loaded.
    setModelApplyPhase("loading");
    setPendingModelPath(settings.llm_model_path);
    setLlmLoaded(false);
    setLlmInfo(null);
    try {
      await withTimeout(refreshStorage(), 8_000, "refresh storage");
      await withTimeout(refreshHealth({ full: false, includeLlmInfo: false }), 8_000, "refresh health");
      setCleanResult("Model settings applied and workspace cleared. Model info will appear once the engine loads the model.");
      await pollLlmInfoUntilReady();
    } catch (err: any) {
      setCleanResult(
        `Model settings applied. Health refresh timed out; model info will appear once the engine loads the model. (${err?.message || "timeout"})`
      );
      await pollLlmInfoUntilReady();
    }
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
    try {
      localStorage.removeItem("insight_setup_seen");
    } catch {
      // ignore storage errors
    }
    window.setTimeout(() => {
      window.location.reload();
    }, 60);
  }

  async function downloadEmbeddings() {
    if (embeddingDownloading) return;
    setEmbeddingDownloadError(null);
    setEmbeddingForceStopped(false);
    setEmbeddingDownloadActive(true);
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

  async function cancelEmbeddingDownload() {
    setEmbeddingDownloadError(null);
    setEmbeddingDownloading(false);
    setEmbeddingForceStopped(true);
    setEmbeddingDownloadActive(false);
    setHealthReport((prev) => {
      if (!prev) return prev;
      const nextChecks = { ...(prev as any).checks };
      nextChecks.embedding_download_status = "idle";
      nextChecks.embedding_download_error = null;
      return { ...(prev as any), ok: false, checks: nextChecks } as HealthReport;
    });
    try {
      const res = await withTimeout(
        engine<any>("/settings/embedding/cancel", {}, "POST"),
        8000,
        "cancel embedding download"
      );
      if (!res.ok || !(res.data as any)?.ok) {
        setEmbeddingDownloadError(getErrorText(res, "Failed to cancel embedding download"));
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setEmbeddingDownloadError(message || "Failed to cancel embedding download");
    }
    await refreshHealth();
  }

  const derivedHealthIssues = useMemo<HealthIssue[]>(() => {
    const issues: HealthIssue[] = [];
    if (!settings.llm_model_path) {
      issues.push({
        code: "model_not_configured",
        severity: "error",
        message: "No model is configured yet.",
        fix: "Open Settings → Model and choose a GGUF model file.",
        action: "open_settings",
      });
    }
    if (embeddingPresent === false) {
      issues.push({
        code: "embedding_missing",
        severity: "warning",
        message: "Embedding model files are missing.",
        fix: "Download embeddings in Settings → Model.",
        action: "open_settings",
      });
    }
    return issues;
  }, [settings.llm_model_path, embeddingPresent]);

  const healthIssues = useMemo(() => {
    const reportIssues =
      Array.isArray(healthReport?.issues) && healthReport!.issues.length
        ? healthReport!.issues
        : [];
    if (!reportIssues.length) return derivedHealthIssues;
    const reportCodes = reportIssues
      .map((issue) => (typeof issue.code === "string" ? issue.code : ""))
      .filter(Boolean);
    const reportOnlyTimeout =
      reportCodes.length > 0 &&
      reportCodes.every((code) => code === "health_timeout" || code === "health_unavailable");
    if (reportOnlyTimeout && derivedHealthIssues.length) return derivedHealthIssues;
    return reportIssues;
  }, [healthReport, derivedHealthIssues]);
  const uniqueHealthIssues = useMemo(() => {
    const seen = new Set<string>();
    return healthIssues.filter((issue) => {
      const code = typeof issue.code === "string" && issue.code ? issue.code : "";
      if (!code) return true;
      if (seen.has(code)) return false;
      seen.add(code);
      return true;
    });
  }, [healthIssues]);
  const hideHealthDetails = uniqueHealthIssues.some((issue) => issue.code === "model_not_configured");
  const healthHasError = uniqueHealthIssues.some((issue) => issue.severity === "error");
  const healthCodes = uniqueHealthIssues
    .map((issue) => (typeof issue.code === "string" ? issue.code : ""))
    .filter(Boolean);
  const healthCodesPreview = healthCodes.slice(0, 3).join(", ");
  const healthCodesSuffix = healthCodes.length > 3 ? ` +${healthCodes.length - 3}` : "";
  const hasHealthReport = Boolean(healthReport);
  const healthBadgeClass = !hasHealthReport && uniqueHealthIssues.length === 0
    ? "muted"
    : healthHasError
      ? "error"
      : uniqueHealthIssues.length
        ? "warn"
        : "ok";
  const healthBadgeText = !hasHealthReport && uniqueHealthIssues.length === 0
    ? "Checking"
    : healthHasError
      ? `Errors: ${healthCodesPreview}${healthCodesSuffix}`
      : uniqueHealthIssues.length
        ? `Warnings: ${healthCodesPreview}${healthCodesSuffix}`
        : "All systems operational";
  const shouldShowEmbeddings =
    embeddingPresent !== true || embeddingDownloading || embeddingDownloadActive || !!embeddingDownloadError;

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
        <aside className="settings-nav">
          <div className="nav-title-row">
            <div className="nav-title">Insight App</div>
            <button className="settings-icon-btn settings-close" type="button" onClick={onClose} aria-label="Close">
              <svg className="settings-icon" viewBox="0 0 24 24" aria-hidden="true">
                <path d="M6 6l12 12M18 6l-12 12" />
              </svg>
            </button>
          </div>
          <nav>
            <button
              className={`nav-item ${activeTab === "general" ? "active" : ""}`}
              type="button"
              onClick={() => setActiveTab("general")}
            >
              General
            </button>
            <button
              className={`nav-item ${activeTab === "model" ? "active" : ""}`}
              type="button"
              onClick={() => setActiveTab("model")}
            >
              Model
            </button>
            <button
              className={`nav-item ${activeTab === "retrieval" ? "active" : ""}`}
              type="button"
              onClick={() => setActiveTab("retrieval")}
            >
              Retrieval
            </button>
            <button
              className={`nav-item ${activeTab === "raw" ? "active" : ""}`}
              type="button"
              onClick={() => setActiveTab("raw")}
            >
              Raw Server
            </button>
            <button
              className={`nav-item ${activeTab === "storage" ? "active" : ""}`}
              type="button"
              onClick={() => setActiveTab("storage")}
            >
              Storage
            </button>
          </nav>
        </aside>

        <main className="settings-content content-area">
          {engineBusy?.busy ? (
            <div className="settings-banner warn">
              Background work running. Wait before cleaning/resetting storage or changing context.
            </div>
          ) : null}

          {activeTab === "general" ? (
            <section className="settings-section">
              <h2 className="section-header">General</h2>

              <div className="setting-item stack">
                <div className="setting-info">
                  <div className="label">System Health</div>
                </div>
                <div className="setting-control">
                  <div className={`status-badge ${healthBadgeClass}`}>
                    <span className="status-dot" />
                    <span className="status-text">{healthBadgeText}</span>
                  </div>
                </div>
                <div className="setting-control full-width">
                  {healthReport || uniqueHealthIssues.length ? null : (
                    <div className="settings-muted">Checking system health…</div>
                  )}
                  {!hideHealthDetails ? (
                    <div className="settings-health-meta">
                      <div>Model: {healthModelLabel}</div>
                      {quantizationTag ? <div>Quant: {quantizationTag}</div> : null}
                    </div>
                  ) : null}
                </div>
              </div>

              <div className="setting-item">
                <div className="setting-info">
                  <div className="label">Appearance</div>
                  <div className="description">Customize the interface theme</div>
                </div>
                <div className="setting-control">
                  <div className="segmented-control">
                    <button
                      className={`segment ${themeMode === "dark" ? "active" : ""}`}
                      type="button"
                      onClick={() => {
                        onThemeModeChange("dark");
                        saveSettingsPatch({ theme_mode: "dark" });
                      }}
                    >
                      Dark
                    </button>
                    <button
                      className={`segment ${themeMode === "light" ? "active" : ""}`}
                      type="button"
                      onClick={() => {
                        onThemeModeChange("light");
                        saveSettingsPatch({ theme_mode: "light" });
                      }}
                    >
                      Light
                    </button>
                    <button
                      className={`segment ${themeMode === "system" ? "active" : ""}`}
                      type="button"
                      onClick={() => {
                        onThemeModeChange("system");
                        saveSettingsPatch({ theme_mode: "system" });
                      }}
                    >
                      System
                    </button>
                  </div>
                </div>
              </div>
            </section>
          ) : null}

          {activeTab === "model" ? (
            <div className="settings-tab-body settings-scrollable">
              <section className="settings-section">
                <h2 className="section-header">Model</h2>
              <div className="setting-item stack">
                <div className="setting-info">
                  <div className="label">Model Path</div>
                  <div className="description">gguf model file</div>
                </div>
                <div className="setting-control full-width">
                  <div className="input-wrapper">
                    <input
                      className="input-field"
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
                      className="settings-icon-btn icon-btn"
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
                </div>
                {modelValidation ? (
                  <div className={`settings-hint ${modelValidation.ok ? "ok" : "err"}`}>
                    {modelValidation.msg}
                  </div>
                ) : null}
              </div>
              {shouldShowEmbeddings ? (() => {
                const checks = (healthReport as any)?.checks || {};
                const embeddingPresentCheck = checks.embedding_present === true || embeddingPresent === true;
                const embeddingStatus = checks.embedding_download_status;
                const effectiveEmbeddingStatus =
                  embeddingForceStopped && embeddingStatus === "downloading" ? "error" : embeddingStatus;
                const embeddingBusy =
                  embeddingDownloading ||
                  embeddingDownloadActive ||
                  effectiveEmbeddingStatus === "downloading";
                return (
                  <div className="setting-item stack">
                    <div className="setting-info">
                      <div className="label">Embeddings</div>
                      <div className="description">Local ONNX model required for ingestion/search.</div>
                    </div>
                    <div
                      className={`settings-health-item ${
                        effectiveEmbeddingStatus === "error" ? "error" : "warning"
                      }`}
                    >
                      <div className="settings-health-title">
                        {embeddingBusy
                          ? "Embedding model downloading"
                          : effectiveEmbeddingStatus === "error"
                          ? "Embedding download failed"
                          : embeddingPresentCheck
                          ? "Embedding model installed"
                          : "Embedding model missing"}
                      </div>
                      <div className="settings-health-fix">
                        {embeddingBusy
                          ? "Keep Insight open until the download completes."
                          : "Downloads the local ONNX embedding model required for ingestion/search."}
                      </div>
                      <div className="settings-row">
                        {embeddingBusy ? (
                          <button
                            className="settings-icon-btn"
                            type="button"
                            onClick={cancelEmbeddingDownload}
                            aria-label="Stop embedding download"
                            title="Stop download"
                          >
                            <svg className="settings-icon" viewBox="0 0 24 24" aria-hidden="true">
                              <path d="M6 6h12v12H6z" />
                            </svg>
                          </button>
                        ) : (
                          <button
                            className="settings-icon-btn"
                            type="button"
                            onClick={downloadEmbeddings}
                            disabled={!!engineBusy?.busy}
                            aria-label="Download embeddings"
                            title="Download embeddings"
                          >
                            <svg className="settings-icon" viewBox="0 0 24 24" aria-hidden="true">
                              <path d="M12 4v10" />
                              <path d="M8.5 10.5L12 14l3.5-3.5" />
                              <path d="M5 18h14" />
                            </svg>
                          </button>
                        )}
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
                  </div>
                );
              })() : null}
              <div className="setting-item stack">
                  <div className="setting-info">
                    <div className="label">Context Window</div>
                  </div>
                  <div className="setting-control">
                    <select
                      className="settings-select settings-select-compact"
                      value={String(settings.llm_ctx_size || llmCtxOptions[llmCtxOptions.length - 1] || 32768)}
                      onChange={(e) => {
                        const v = Number(e.target.value);
                        setSettings((p) => ({ ...p, llm_ctx_size: v }));
                      }}
                    >
                      {llmCtxOptions.map((opt) => (
                        <option key={opt} value={String(opt)}>
                          {formatCtxLabel(opt)}
                        </option>
                      ))}
                    </select>
                  </div>
                  {settings.llm_ctx_size > 32768 ? (
                    <div className="settings-hint warn">
                      Larger context windows require much more VRAM and may fail on some systems.
                      {llmCtxMax && settings.llm_ctx_size >= llmCtxMax
                        ? " This matches the model’s max context."
                        : null}
                    </div>
                  ) : null}
                  <div className="setting-control">
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
                </div>
                <div className="settings-hint">
                  {modelApplyPhase !== "idle" ? (
                    <div className="settings-muted">
                      {modelApplyPhase === "validating"
                        ? "Validating model…"
                        : modelApplyPhase === "applying"
                          ? "Applying model settings…"
                          : "Applying model…"}
                    </div>
                  ) : displayModelInfo ? (
                    <div className="model-info-section">
                      <div className="model-info-title">{displayModelInfoLabel || "Model info"}</div>
                      <div className="model-info-box">
                      <div className="model-info-row">
                        <div className="model-info-label">Name</div>
                        <div className="model-info-value"><code>{displayModelInfo.name || "Unknown model"}</code></div>
                      </div>
                      {displayModelInfo.size_label || displayModelInfo.architecture ? (
                        <div className="model-info-row">
                          <div className="model-info-label">Size / Arch</div>
                          <div className="model-info-value">
                            <code>{[displayModelInfo.size_label, displayModelInfo.architecture].filter(Boolean).join(" · ")}</code>
                          </div>
                        </div>
                      ) : null}
                      {displayModelInfo.ctx_runtime ? (
                        <div className="model-info-row">
                          <div className="model-info-label">Ctx runtime</div>
                          <div className="model-info-value"><code>{String(displayModelInfo.ctx_runtime)}</code></div>
                        </div>
                      ) : null}
                      {displayModelInfo.ctx_train || displayModelInfo.ctx_runtime ? (
                        <div className="model-info-row">
                          <div className="model-info-label">Ctx max</div>
                          <div className="model-info-value"><code>{String(displayModelInfo.ctx_train ?? displayModelInfo.ctx_runtime)}</code></div>
                        </div>
                      ) : null}
                      {typeof displayModelInfo.n_layer === "number" ? (
                        <div className="model-info-row">
                          <div className="model-info-label">Layers</div>
                          <div className="model-info-value"><code>{String(displayModelInfo.n_layer)}</code></div>
                        </div>
                      ) : null}
                      {typeof displayModelInfo.n_head === "number" ? (
                        <div className="model-info-row">
                          <div className="model-info-label">Heads</div>
                          <div className="model-info-value">
                            <code>
                              {String(displayModelInfo.n_head)}
                              {typeof displayModelInfo.n_head_kv === "number" ? `/${String(displayModelInfo.n_head_kv)}` : ""}
                            </code>
                          </div>
                        </div>
                      ) : null}
                      {typeof displayModelInfo.n_embd === "number" ? (
                        <div className="model-info-row">
                          <div className="model-info-label">Hidden</div>
                          <div className="model-info-value"><code>{String(displayModelInfo.n_embd)}</code></div>
                        </div>
                      ) : null}
                      {displayModelInfo.rope_type || displayModelInfo.rope_freq_base ? (
                        <div className="model-info-row">
                          <div className="model-info-label">Rope</div>
                          <div className="model-info-value">
                            <code>
                              {displayModelInfo.rope_type ? displayModelInfo.rope_type : "?"}
                              {displayModelInfo.rope_freq_base ? `@${Number(displayModelInfo.rope_freq_base).toLocaleString()}` : ""}
                            </code>
                          </div>
                        </div>
                      ) : null}
                    {typeof displayModelInfo.file_type === "number" ? (
                      <div className="model-info-row">
                        <div className="model-info-label">File type</div>
                        <div className="model-info-value"><code>{String(displayModelInfo.file_type)}</code></div>
                      </div>
                    ) : null}
                    {quantizationTag ? (
                      <div className="model-info-row">
                        <div className="model-info-label">Quantization</div>
                        <div className="model-info-value"><code>{quantizationTag}</code></div>
                      </div>
                    ) : null}
                    {typeof displayModelInfo.quantization_version === "number" ? (
                      <div className="model-info-row">
                        <div className="model-info-label">Quant</div>
                          <div className="model-info-value"><code>v{String(displayModelInfo.quantization_version)}</code></div>
                        </div>
                      ) : null}
                      {displayModelInfo.prompt_renderer ? (
                        <div className="model-info-row">
                          <div className="model-info-label">Renderer</div>
                          <div className="model-info-value"><code>{displayModelInfo.prompt_renderer}</code></div>
                        </div>
                      ) : null}
                      {displayModelInfo.tokenizer_model ? (
                        <div className="model-info-row">
                          <div className="model-info-label">Tokenizer</div>
                          <div className="model-info-value">
                            <code>
                              {displayModelInfo.tokenizer_model}
                              {typeof displayModelInfo.vocab_size === "number" ? ` (vocab ${displayModelInfo.vocab_size})` : ""}
                            </code>
                          </div>
                        </div>
                      ) : null}
                      {typeof displayModelInfo.kv_cache_gib === "number" ? (
                        <div className="model-info-row">
                          <div className="model-info-label">KV@ctx</div>
                          <div className="model-info-value"><code>≈ {displayModelInfo.kv_cache_gib.toFixed(2)} GiB</code></div>
                        </div>
                      ) : null}
                      {displayModelInfo.bos_token_id != null || displayModelInfo.eos_token_id != null ? (
                        <div className="model-info-row">
                          <div className="model-info-label">BOS/EOS</div>
                          <div className="model-info-value">
                            <code>
                              {displayModelInfo.bos_token_id != null ? displayModelInfo.bos_token_id : "?"}/
                              {displayModelInfo.eos_token_id != null ? displayModelInfo.eos_token_id : "?"}
                            </code>
                          </div>
                        </div>
                      ) : null}
                      </div>
                    </div>
                  ) : (
                    <div className="settings-muted">
                      Model info appears after the engine loads the model (start a chat or restart after applying settings).
                    </div>
                  )}
                </div>
              </section>
            </div>
          ) : null}

          {activeTab === "retrieval" ? (
            <section className="settings-section">
              <h2 className="section-header">Retrieval</h2>
            <div className="setting-item">
              <div className="setting-info">
                <div className="label">Document Mode</div>
                <div className="description">Small‑Doc uses full text if fits else retrieval.</div>
              </div>
              <div className="setting-control">
                <div className="segmented-control">
                  <button
                    className={`segment ${settings.rag_default_mode === "small_doc" ? "active" : ""}`}
                    type="button"
                    onClick={() => {
                      setSettings((p) => ({ ...p, rag_default_mode: "small_doc" }));
                      saveSettingsPatch({ rag_default_mode: "small_doc" });
                    }}
                  >
                    Small‑Doc
                  </button>
                  <button
                    className={`segment ${settings.rag_default_mode === "rag" ? "active" : ""}`}
                    type="button"
                    onClick={() => {
                      setSettings((p) => ({ ...p, rag_default_mode: "rag" }));
                      saveSettingsPatch({ rag_default_mode: "rag" });
                    }}
                  >
                    RAG
                  </button>
                </div>
              </div>
            </div>
            <div className="setting-item stack">
              <div className="setting-info">
                <div className="label">Detail ↔ Precision</div>
                <div className="description">Lower = tighter matches. Higher = more coverage + neighbor expansion.</div>
              </div>
              <div className="setting-control">
                <select
                  className="settings-select settings-select-compact"
                  value={String(settings.rag_default_detail || 3)}
                  onChange={(e) => {
                    const v = Math.min(5, Math.max(1, Number(e.target.value || 3)));
                    setSettings((p) => ({ ...p, rag_default_detail: v }));
                    scheduleRagDefaultsSave({ rag_default_detail: v });
                  }}
                >
                  <option value="1">1</option>
                  <option value="2">2</option>
                  <option value="3">3</option>
                  <option value="4">4</option>
                  <option value="5">5</option>
                </select>
              </div>
            </div>
            </section>
          ) : null}

          {activeTab === "raw" ? (
            <div className="settings-tab-body settings-tab-body-fixed">
              <section className="settings-section settings-raw-section">
                <h2 className="section-header">Raw Server</h2>
              <div className="settings-row settings-col settings-raw-top">
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
              </div>

              <div className="settings-raw-layout">
                <div className="settings-raw-left settings-scrollable">
                  <div className="settings-raw-meta">
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
                  </div>

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
                      <label className="settings-label">Context window</label>
                      <div className="settings-inline">
                        <select
                          className="settings-select settings-input-compact"
                          value={String(settings.raw_engine_ctx ?? rawCtxOptions[rawCtxOptions.length - 1] ?? 32768)}
                          onChange={(e) => {
                            const v = e.target.value;
                            setSettings((p) => ({
                              ...p,
                              raw_engine_ctx: Number(v),
                            }));
                          }}
                          disabled={rawConfigLocked || rawEngineBusy}
                        >
                          {rawCtxOptions.map((opt) => (
                            <option key={opt} value={String(opt)}>
                              {formatCtxLabel(opt)}
                            </option>
                          ))}
                        </select>
                        <div className="settings-muted">
                          Default: {formatCtxLabel(Number(rawDefaults.raw_engine_ctx || 32768))}
                        </div>
                      </div>
                      {settings.raw_engine_ctx && settings.raw_engine_ctx > 32768 ? (
                        <div className="settings-hint warn">
                          Large context windows use a lot of VRAM and may fail to load on some GPUs.
                          {rawCtxMax && settings.raw_engine_ctx >= rawCtxMax
                            ? " This matches the model’s max context."
                            : null}
                        </div>
                      ) : null}
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
                </div>

                <div className="settings-raw-right">
                  <div className="settings-raw-console settings-scrollable" ref={rawLogRef}>
                    {rawEngineError || rawEngineStatus?.error ? (
                      <div className="settings-raw-line settings-raw-line-error">
                        <span className="settings-raw-time">error</span>
                        <span className="settings-raw-text">{rawEngineError || rawEngineStatus?.error}</span>
                      </div>
                    ) : null}
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
              </div>
              </section>
            </div>
          ) : null}

          {activeTab === "storage" ? (
            <section className="settings-section danger">
              <h2 className="section-header">Storage</h2>
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
            <div className="section-header section-subheader">Reset</div>
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
          ) : null}
        </main>
      </div>
    </div>
  );
}
