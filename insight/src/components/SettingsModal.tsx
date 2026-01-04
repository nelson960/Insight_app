import { useEffect, useMemo, useState } from "react";
import { engine } from "../api/engine";
import { invoke } from "@tauri-apps/api/core";

export type ThemeMode = "system" | "dark" | "light";

type SettingsState = {
  theme_mode: ThemeMode;
  llm_model_path: string;
  llm_gpu_layers: number;
  llm_ctx_size: number;
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
  const [settings, setSettings] = useState<SettingsState>({
    theme_mode: themeMode,
    llm_model_path: "",
    llm_gpu_layers: 99,
    llm_ctx_size: 32768,
  });
  const [storage, setStorage] = useState<StorageUsage | null>(null);
  const [restartRequired, setRestartRequired] = useState(false);
  const [modelValidation, setModelValidation] = useState<{ ok: boolean; msg: string } | null>(null);
  const [cleanResult, setCleanResult] = useState<string | null>(null);
  const [resetArmed, setResetArmed] = useState(false);
  const [llmApplyArmed, setLlmApplyArmed] = useState(false);
  const [busy, setBusy] = useState(false);
  const [engineBusy, setEngineBusy] = useState<BusyState | null>(null);
  const [llmLoaded, setLlmLoaded] = useState(false);
  const [llmInfo, setLlmInfo] = useState<LlmModelInfo | null>(null);

  function getErrorText(res: { error?: string; data?: any }, fallback: string) {
    return (
      res.error ||
      res.data?.error ||
      res.data?.detail?.error ||
      res.data?.detail ||
      fallback
    );
  }

  useEffect(() => {
    if (!open) return;
    setResetArmed(false);
    setLlmApplyArmed(false);
    setCleanResult(null);
    setModelValidation(null);
    setRestartRequired(false);
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
    if (!open) return;
    let alive = true;
    const tick = async () => {
      const bs = await engine<BusyState>("/settings/busy", undefined, "GET");
      if (!alive) return;
      if (bs.ok) setEngineBusy(bs.data as any);
    };
    tick();
    const id = window.setInterval(tick, 3000);
    return () => {
      alive = false;
      window.clearInterval(id);
    };
  }, [open]);

  const sortedBreakdown = useMemo(() => {
    if (!storage?.breakdown) return [];
    return Object.entries(storage.breakdown).sort((a, b) => (b[1].bytes || 0) - (a[1].bytes || 0));
  }, [storage]);

  async function refreshStorage() {
    const st = await engine<StorageUsage>("/settings/storage", undefined, "GET");
    if (st.ok) setStorage(st.data as any);
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
    const res = await engine<any>("/settings/model/validate", { model_path: path }, "POST");
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
      setLlmApplyArmed(false);
    } catch (e: any) {
      setModelValidation({ ok: false, msg: e?.message || String(e) });
    }
  }

  async function applyLlmSettings() {
    if (engineBusy?.busy) {
      setCleanResult("Background work is running. Wait for it to finish before applying model settings.");
      return;
    }
    if (!llmApplyArmed) {
      setLlmApplyArmed(true);
      setCleanResult("Click again to confirm. This will delete all chats, files, indexes, and KV sessions.");
      return;
    }

    const ok = await validateModel(settings.llm_model_path);
    if (!ok) {
      setLlmApplyArmed(false);
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
    setLlmApplyArmed(false);

    if (!res.ok) {
      setCleanResult(getErrorText(res, "Failed to apply model settings"));
      return;
    }

    setRestartRequired(true);
    await refreshStorage();
    setCleanResult("Model settings applied and workspace cleared. Restart the app/engine to reload the model.");
  }

  async function cleanCache(trimLogs: boolean) {
    if (engineBusy?.busy) {
      setCleanResult("Background work is running. Wait for it to finish before cleaning.");
      return;
    }
    setCleanResult(null);
    const res = await engine<any>("/settings/storage/clean_cache", { trim_logs: trimLogs }, "POST");
    if (!res.ok) {
      setCleanResult(getErrorText(res, "Failed to clean cache"));
      return;
    }
    const freed = (res.data as any)?.freed_bytes ?? 0;
    setCleanResult(`Freed ${formatBytes(freed)}`);
    await refreshStorage();
  }

  async function resetAll() {
    if (engineBusy?.busy) {
      setCleanResult("Background work is running. Wait for it to finish before resetting.");
      return;
    }
    if (!resetArmed) {
      setResetArmed(true);
      return;
    }
    setBusy(true);
    const res = await engine<any>("/settings/storage/reset", { confirm: true }, "POST");
    setBusy(false);
    if (!res.ok) {
      setCleanResult(getErrorText(res, "Reset failed"));
      setResetArmed(false);
      return;
    }
    setResetArmed(false);
    setRestartRequired(true);
    const st = await engine<StorageUsage>("/settings/storage", undefined, "GET");
    if (st.ok) setStorage(st.data as any);
    const total = st.ok ? formatBytes((st.data as any)?.total_bytes ?? 0) : "0 B";
    setCleanResult(
      `Delete complete. Workspace cleared (KV/Qdrant/DB/uploads/cache/logs/keys/config). Current storage: ${total}. Please restart the app.`
    );
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
      >
        <div className="settings-header">
          <div className="settings-title">Settings</div>
          <button className="settings-close" type="button" onClick={onClose} aria-label="Close">
            ×
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

        <div className="settings-body">
          <section className="settings-section">
            <div className="settings-section-title">Storage</div>
            <div className="settings-row">
              <div className="settings-kv">
                <div className="k">Workspace</div>
                <div className="v">{storage?.base || "…"}</div>
              </div>
              <div className="settings-kv">
                <div className="k">Total</div>
                <div className="v">{storage ? formatBytes(storage.total_bytes) : "…"}</div>
              </div>
              <button className="settings-btn" type="button" onClick={refreshStorage}>
                Refresh
              </button>
            </div>
            <div className="settings-breakdown">
              {sortedBreakdown.map(([name, info]) => (
                <div key={name} className="settings-breakdown-row">
                  <div className="name">{name}</div>
                  <div className="bytes">{formatBytes(info.bytes)}</div>
                </div>
              ))}
            </div>
            <div className="settings-row">
              <button className="settings-btn" type="button" disabled={!!engineBusy?.busy} onClick={() => cleanCache(false)}>
                Clean cache
              </button>
              <button className="settings-btn" type="button" disabled={!!engineBusy?.busy} onClick={() => cleanCache(true)}>
                Clean cache + logs
              </button>
              <div className="settings-muted">{cleanResult || ""}</div>
            </div>
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
                  onChange={(e) => setSettings((p) => ({ ...p, llm_model_path: e.target.value }))}
                  onBlur={async () => {
                    if (!settings.llm_model_path) return;
                    const ok = await validateModel(settings.llm_model_path);
                    if (ok) setLlmApplyArmed(false);
                  }}
                />
                <button className="settings-btn" type="button" onClick={browseModel}>
                  Browse…
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
                  className="settings-select"
                  value={String(settings.llm_ctx_size || 32768)}
                  onChange={(e) => {
                    const v = Number(e.target.value);
                    setSettings((p) => ({ ...p, llm_ctx_size: v }));
                    setLlmApplyArmed(false);
                  }}
                >
                  <option value="8192">8k</option>
                  <option value="32768">32k</option>
                </select>
                <button
                  className={`settings-btn danger ${llmApplyArmed ? "confirm" : ""}`}
                  type="button"
                  disabled={busy || !!engineBusy?.busy}
                  onClick={applyLlmSettings}
                >
                  {llmApplyArmed ? "Click again to confirm" : "Apply (resets all data)"}
                </button>
              </div>
              <div className="settings-hint">
                Changing model/context clears chats, files, indexes, and KV sessions.
              </div>
            </div>
          </section>

          <section className="settings-section">
            <div className="settings-section-title">Appearance</div>
            <div className="settings-row">
              <button
                className={`settings-btn ${themeMode === "system" ? "active" : ""}`}
                type="button"
                onClick={() => {
                  onThemeModeChange("system");
                  saveSettingsPatch({ theme_mode: "system" });
                }}
              >
                System
              </button>
              <button
                className={`settings-btn ${themeMode === "dark" ? "active" : ""}`}
                type="button"
                onClick={() => {
                  onThemeModeChange("dark");
                  saveSettingsPatch({ theme_mode: "dark" });
                }}
              >
                Dark
              </button>
              <button
                className={`settings-btn ${themeMode === "light" ? "active" : ""}`}
                type="button"
                onClick={() => {
                  onThemeModeChange("light");
                  saveSettingsPatch({ theme_mode: "light" });
                }}
              >
                Light
              </button>
            </div>
          </section>

          <section className="settings-section danger">
            <div className="settings-section-title">Reset</div>
            <div className="settings-hint">
              Deletes all data under <code>storage/</code> (DB, Qdrant, KV sessions, uploads, cache, logs, keys, config). Does not touch <code>models/</code>.
            </div>
            <div className="settings-row">
              <button
                className={`settings-btn danger ${resetArmed ? "confirm" : ""}`}
                type="button"
                disabled={busy || !!engineBusy?.busy}
                onClick={resetAll}
              >
                {resetArmed ? "Click again to confirm delete" : "Delete all data"}
              </button>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
