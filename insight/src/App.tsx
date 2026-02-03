import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import "./App.css";
import { Canvas, CanvasLink, CanvasNote } from "./components/Canvas";
import type { CardLayout } from "./components/Canvas";
import { CardOverlay } from "./components/CardOverlay";
import { ChatWindow } from "./components/ChatWindow";
import { SettingsModal, ThemeMode } from "./components/SettingsModal";
import { FirstRunSetupModal } from "./components/FirstRunSetupModal";
import { useSessions } from "./state/useSessions";
import { engine } from "./api/engine";
import { setTheme as setAppTheme } from "@tauri-apps/api/app";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { listen } from "@tauri-apps/api/event";
import { HealthReport } from "./components/StartupHealthModal";

const NOTES_STORAGE_KEY = "insight.canvas.notes.v1";
const LINKS_STORAGE_KEY = "insight.canvas.links.v1";
const CARD_LAYOUT_STORAGE_KEY = "insight.card.layout.default.v1";
const DEFAULT_CARD_LAYOUT: CardLayout = {
  showChat: true,
  showDocs: true,
  chatOnRight: true,
  splitRatio: 0.6,
};
const DEFAULT_DOCK_DEAD_ZONE = { x: 0, y: 0, w: 240, h: 200 };
const STARTUP_BLOCKING_CODES = new Set([
  "model_not_configured",
  "model_missing",
  "model_not_file",
  "model_wrong_extension",
  "model_invalid",
  "chat_router_failed",
]);

function pickBlockingIssues(
  report: HealthReport | null | undefined,
  blockingCodes: Set<string>
) {
  if (!report) return null;
  const issues = Array.isArray(report.issues)
    ? report.issues.filter((issue) => blockingCodes.has(issue.code))
    : [];
  if (!issues.length) return null;
  return { ...report, ok: false, issues };
}

function coerceCardLayout(raw: unknown): CardLayout | null {
  if (!raw || typeof raw !== "object") return null;
  const obj = raw as Record<string, unknown>;

  const showChat = typeof obj.showChat === "boolean" ? obj.showChat : DEFAULT_CARD_LAYOUT.showChat;
  const showDocs = typeof obj.showDocs === "boolean" ? obj.showDocs : DEFAULT_CARD_LAYOUT.showDocs;
  const chatOnRight =
    typeof obj.chatOnRight === "boolean" ? obj.chatOnRight : DEFAULT_CARD_LAYOUT.chatOnRight;
  const splitRatioRaw = Number(obj.splitRatio);
  const splitRatio = Number.isFinite(splitRatioRaw)
    ? Math.max(0.05, Math.min(0.95, splitRatioRaw))
    : DEFAULT_CARD_LAYOUT.splitRatio;
  const activeFileId =
    typeof obj.activeFileId === "string" && obj.activeFileId
      ? obj.activeFileId
      : obj.activeFileId === null
        ? null
        : undefined;

  // Ensure at least one pane is visible.
  if (!showChat && !showDocs) return { ...DEFAULT_CARD_LAYOUT };

  return { showChat, showDocs, chatOnRight, splitRatio, activeFileId };
}

type SpawnHint = {
  anchor: { x: number; y: number };
  bounds?: { w: number; h: number };
  avoidZones?: Array<{ x: number; y: number; w: number; h: number }>;
};

function rectOverlapArea(
  ax: number,
  ay: number,
  aw: number,
  ah: number,
  bx: number,
  by: number,
  bw: number,
  bh: number
) {
  const x1 = Math.max(ax, bx);
  const y1 = Math.max(ay, by);
  const x2 = Math.min(ax + aw, bx + bw);
  const y2 = Math.min(ay + ah, by + bh);
  if (x2 <= x1 || y2 <= y1) return 0;
  return (x2 - x1) * (y2 - y1);
}

function equalLayout(a?: CardLayout, b?: CardLayout) {
  if (!a && !b) return true;
  if (!a || !b) return false;
  return (
    a.showChat === b.showChat &&
    a.showDocs === b.showDocs &&
    a.chatOnRight === b.chatOnRight &&
    a.splitRatio === b.splitRatio &&
    (a.activeFileId ?? null) === (b.activeFileId ?? null)
  );
}

function findOpenPosition(
  start: { x: number; y: number },
  size: { w: number; h: number },
  notes: CanvasNote[],
  bounds?: { w: number; h: number },
  avoidZones?: Array<{ x: number; y: number; w: number; h: number }>
) {
  const clampToBounds = (pos: { x: number; y: number }) => {
    if (!bounds) return pos;
    const maxX = Math.max(0, bounds.w - size.w);
    const maxY = Math.max(0, bounds.h - size.h);
    return {
      x: Math.max(0, Math.min(maxX, pos.x)),
      y: Math.max(0, Math.min(maxY, pos.y)),
    };
  };

  const inBounds = (pos: { x: number; y: number }) => {
    if (!bounds) return true;
    return pos.x >= 0 && pos.y >= 0 && pos.x + size.w <= bounds.w && pos.y + size.h <= bounds.h;
  };

  const startPos = clampToBounds(start);
  if (!notes.length) return startPos;
  const margin = 16;
  const step = 40;
  const maxRing = 18;
  const candidates: Array<{ x: number; y: number }> = [];
  candidates.push({ x: startPos.x, y: startPos.y });
  for (let r = 1; r <= maxRing; r += 1) {
    for (let dx = -r; dx <= r; dx += 1) {
      for (let dy = -r; dy <= r; dy += 1) {
        if (Math.abs(dx) !== r && Math.abs(dy) !== r) continue;
        candidates.push({ x: startPos.x + dx * step, y: startPos.y + dy * step });
      }
    }
  }
  let best = candidates[0] ?? startPos;
  let bestOverlap = Number.POSITIVE_INFINITY;
  const avoidPenalty = avoidZones && avoidZones.length ? size.w * size.h * 4 : 0;
  for (const c of candidates) {
    if (!inBounds(c)) continue;
    let overlap = 0;
    let hit = false;
    for (const n of notes) {
      const area = rectOverlapArea(
        c.x,
        c.y,
        size.w,
        size.h,
        n.x - margin,
        n.y - margin,
        n.w + margin * 2,
        n.h + margin * 2
      );
      if (area > 0) {
        hit = true;
        overlap += area;
      }
    }
    if (avoidZones && avoidZones.length) {
      for (const zone of avoidZones) {
        const area = rectOverlapArea(c.x, c.y, size.w, size.h, zone.x, zone.y, zone.w, zone.h);
        if (area > 0) {
          hit = true;
          overlap += avoidPenalty;
        }
      }
    }
    if (!hit) return c;
    if (overlap < bestOverlap) {
      bestOverlap = overlap;
      best = c;
    }
  }
  return clampToBounds(best);
}

function loadPersistedCardLayout(): CardLayout | null {
  try {
    const raw = localStorage.getItem(CARD_LAYOUT_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    const layout = coerceCardLayout(parsed);
    if (!layout) return null;
    if (Math.abs(layout.splitRatio - 0.5) < 1e-6) {
      layout.splitRatio = DEFAULT_CARD_LAYOUT.splitRatio;
    }
    return { ...layout, activeFileId: undefined };
  } catch {
    return null;
  }
}

function persistCardLayout(layout: CardLayout) {
  try {
    const cleaned = { ...layout, activeFileId: undefined };
    localStorage.setItem(CARD_LAYOUT_STORAGE_KEY, JSON.stringify(cleaned));
  } catch {
    // ignore
  }
}

function loadPersistedNotes(): CanvasNote[] | null {
  try {
    const raw = localStorage.getItem(NOTES_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return null;
    const out: CanvasNote[] = [];
    for (const item of parsed) {
      if (!item || typeof item !== "object") continue;
      const obj = item as Record<string, unknown>;
      const chatId = obj.chatId;
      const x = Number(obj.x);
      const y = Number(obj.y);
      const w = Number(obj.w);
      const h = Number(obj.h);
      const z = Number(obj.z);
      if (typeof chatId !== "string" || !chatId) continue;
      if (![x, y, w, h, z].every(Number.isFinite)) continue;
      let layout = coerceCardLayout(obj.layout);
      if (layout && Math.abs(layout.splitRatio - 0.5) < 1e-6) {
        layout = { ...layout, splitRatio: DEFAULT_CARD_LAYOUT.splitRatio };
      }
      const locked = typeof obj.locked === "boolean" ? obj.locked : undefined;
      const collapsed =
        typeof obj.collapsed === "boolean" ? obj.collapsed : undefined;
      const groupColorRaw = obj.groupColor;
      const groupColor =
        typeof groupColorRaw === "string" && groupColorRaw.trim()
          ? groupColorRaw.trim()
          : undefined;
      out.push({
        chatId,
        x,
        y,
        w,
        h,
        z,
        title: typeof obj.title === "string" ? obj.title : undefined,
        locked,
        collapsed,
        groupColor,
        layout: layout ?? undefined,
      });
    }
    return out;
  } catch (err) {
    console.error("[Storage] Failed to load notes:", err);
    return null;
  }
}

function persistNotes(notes: CanvasNote[]) {
  try {
    const serialized = JSON.stringify(notes);
    // Check if we're about to exceed quota (rough estimate: 2 bytes per character + overhead)
    if (serialized.length > 4_000_000) { // 4MB safety limit (localStorage is typically 5-10MB)
      throw new Error("Notes data too large to save");
    }
    localStorage.setItem(NOTES_STORAGE_KEY, serialized);
  } catch (err) {
    // Log to console for debugging
    console.error("[Storage] Failed to save notes:", err);
    // Return error message for display
    throw err;
  }
}

function loadPersistedLinks(): CanvasLink[] | null {
  try {
    const raw = localStorage.getItem(LINKS_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return null;
    const out: CanvasLink[] = [];
    for (const item of parsed) {
      if (!item || typeof item !== "object") continue;
      const obj = item as Record<string, unknown>;
      const id = obj.id;
      const fromChatId = obj.fromChatId;
      const toChatId = obj.toChatId;
      const kind = obj.kind;
      if (typeof id !== "string" || !id) continue;
      if (typeof fromChatId !== "string" || !fromChatId) continue;
      if (typeof toChatId !== "string" || !toChatId) continue;
      if (kind !== "branch") continue;
      out.push({ id, fromChatId, toChatId, kind });
    }
    return out;
  } catch (err) {
    console.error("[Storage] Failed to load links:", err);
    return null;
  }
}

function persistLinks(links: CanvasLink[]) {
  try {
    const serialized = JSON.stringify(links);
    // Check if we're about to exceed quota
    if (serialized.length > 4_000_000) {
      throw new Error("Links data too large to save");
    }
    localStorage.setItem(LINKS_STORAGE_KEY, serialized);
  } catch (err) {
    console.error("[Storage] Failed to save links:", err);
    throw err;
  }
}

function App() {
  const { sessions, loading, reload, removeSession, addLocalChat } = useSessions();
  const [activeChat, setActiveChat] = useState<string | null>(null);
  const [notes, setNotes] = useState<CanvasNote[]>([]);
  const [links, setLinks] = useState<CanvasLink[]>([]);
  const [defaultCardLayout, setDefaultCardLayout] = useState<CardLayout>(DEFAULT_CARD_LAYOUT);
  const defaultCardLayoutRef = useRef<CardLayout>(DEFAULT_CARD_LAYOUT);
  const notesRef = useRef<CanvasNote[]>([]);
  const spawnHintRef = useRef<SpawnHint | null>(null);
  const layoutPersistTimerRef = useRef<number | null>(null);
  const [overlayChatId, setOverlayChatId] = useState<string | null>(null);
  const [overlayCardId, setOverlayCardId] = useState<string | null>(null);
  const [overlayChatClosing, setOverlayChatClosing] = useState(false);
  const [overlayCardClosing, setOverlayCardClosing] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settingsInitialTab, setSettingsInitialTab] = useState<"general" | "model" | "retrieval" | "raw" | "storage">(
    "general"
  );
  const [themeMode, setThemeMode] = useState<ThemeMode>("system");
  const [engineCrashed, setEngineCrashed] = useState(false);
  const [crashDetails, setCrashDetails] = useState<string | null>(null);
  const [confirmDeleteChatId, setConfirmDeleteChatId] = useState<string | null>(
    null
  );
  const [deletedChatIds, setDeletedChatIds] = useState<string[]>([]);
  const [setupOpen, setSetupOpen] = useState(false);
  const [storageError, setStorageError] = useState<string | null>(null);

  const pickStartupIssues = useCallback(
    (report?: HealthReport | null) => pickBlockingIssues(report, STARTUP_BLOCKING_CODES),
    []
  );

  const maybeOpenSetup = useCallback((report: HealthReport | null) => {
    if (!report) {
      setSetupOpen(false);
      return false;
    }
    const issues = Array.isArray(report.issues) ? report.issues : [];
    const needsSetup = issues.some((issue) =>
      [
        "model_not_configured",
        "model_missing",
        "model_not_file",
        "model_wrong_extension",
        "model_validation_required",
      ].includes(issue.code)
    );
    if (needsSetup) {
      setSetupOpen(true);
      return true;
    }
    setSetupOpen(false);
    return false;
  }, []);

  const buildSetupFallbackReport = useCallback(async (): Promise<HealthReport | null> => {
    const res = await engine<any>("/settings", undefined, "GET");
    if (!res.ok) return null;
    const issues: HealthReport["issues"] = [];
    const settings = (res.data as any)?.settings || {};
    const embedding = (res.data as any)?.embedding || {};
    const modelPath = settings?.llm_model_path;
    if (!modelPath) {
      issues.push({
        code: "model_not_configured",
        severity: "error",
        message: "No model is configured yet.",
        fix: "Open Settings → Model and choose a GGUF model file.",
        action: "open_settings",
      });
    }
    if (embedding?.present === false) {
      issues.push({
        code: "embedding_missing",
        severity: "warning",
        message: "Embedding model files are missing.",
        fix: "Download embeddings in Settings → Model.",
        action: "open_settings",
      });
    }
    if (!issues.length) return null;
    return { ok: false, issues, checks: { health_fallback: true } };
  }, []);

  const ensureModelReady = useCallback(async () => {
    const res = await engine<HealthReport>("/settings/health", undefined, "GET");
    if (!res.ok) return true;
    const trimmed = pickStartupIssues(res.data as any);
    if (trimmed) return !maybeOpenSetup(trimmed);
    const checks = (res.data as any)?.checks || {};
    const chatReady = checks.chat_router_ready;
    const chatLoading = checks.chat_router_loading;
    const chatError = checks.chat_router_error;
    if (chatError) {
      return false;
    }
    if (chatLoading) {
      return true;
    }
    if (chatReady === false) {
      return true;
    }
    return true;
  }, [pickStartupIssues, maybeOpenSetup]);

  // Load persisted app settings (theme) from backend.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      const res = await engine<{ settings?: { theme_mode?: ThemeMode } }>(
        "/settings",
        undefined,
        "GET"
      );
      if (!res.ok || cancelled) return;
      const mode = (res.data as any)?.settings?.theme_mode;
      if (mode === "system" || mode === "dark" || mode === "light") {
        setThemeMode(mode);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Startup health checks (model/config validation).
  useEffect(() => {
    let cancelled = false;
    const handleReport = (report?: HealthReport | null) => {
      if (cancelled) return;
      maybeOpenSetup(report || null);
    };

    (async () => {
      const res = await engine<HealthReport>("/settings/health", undefined, "GET");
      if (cancelled) return;
      if (res.ok) {
        handleReport(res.data as any);
        return;
      }
      const fallback = await buildSetupFallbackReport();
      if (cancelled) return;
      if (fallback) handleReport(fallback);
    })();

    const unlistenPromise = listen<{ report?: HealthReport }>("engine-event", (event) => {
      const payload = event.payload as any;
      if (payload?.name !== "startup_health") return;
      handleReport(payload?.report || payload);
    });

    return () => {
      cancelled = true;
      unlistenPromise
        .then((unsub) => unsub())
        .catch((err) => console.warn("[App] Failed to unsubscribe from engine events:", err));
    };
  }, []);

  // Apply theme to the document root (CSS can key off data-theme).
  useEffect(() => {
    const root = document.documentElement;
    const apply = (mode: ThemeMode) => {
      const prefersDark = window.matchMedia?.("(prefers-color-scheme: dark)").matches;
      const effectiveTheme: "dark" | "light" =
        mode === "system" ? (prefersDark ? "dark" : "light") : mode;

      root.dataset.theme = effectiveTheme;

      // Keep the native title bar in sync with the in-app theme toggle.
      // In system mode we explicitly apply the current system theme and update on OS changes.
      const isTauriRuntime = typeof (window as any).__TAURI_INTERNALS__ !== "undefined";
      if (isTauriRuntime) {
        const titleBarBg = effectiveTheme === "dark" ? "#0f172a" : "#ffffff";
        const win = getCurrentWindow();
        // Ensure the title bar uses our window background color (macOS).
        void win.setTitleBarStyle("transparent").catch((err) => {
          console.warn("[theme] setTitleBarStyle failed (may not be supported on this platform):", err);
        });
        // On macOS, theme is app-wide; use `null` to follow system.
        const tauriTheme = mode === "system" ? null : effectiveTheme;
        void win.setTheme(tauriTheme).catch((e) => {
          console.warn("[theme] window.setTheme failed", e);
        });
        void win.setBackgroundColor(titleBarBg).catch((err) => {
          console.warn("[theme] setBackgroundColor failed (may not be supported on this platform):", err);
        });
        void setAppTheme(tauriTheme).catch((e) => {
          console.warn("[theme] app.setTheme failed", e);
        });
      }
    };
    apply(themeMode);

    if (themeMode !== "system" || !window.matchMedia) return;
    const mql = window.matchMedia("(prefers-color-scheme: dark)");
    const handler = () => apply("system");
    try {
      mql.addEventListener("change", handler);
      return () => mql.removeEventListener("change", handler);
    } catch {
      // Safari fallback
      mql.addListener(handler);
      return () => mql.removeListener(handler);
    }
  }, [themeMode]);

  useEffect(() => {
    const unlisten = listen<{
      message?: string;
      suggestion?: string;
    }>("engine-crashed", (event) => {
      console.error("[App] Engine crash detected:", event.payload);
      setEngineCrashed(true);
      setCrashDetails(
        event.payload?.suggestion ||
          event.payload?.message ||
          "The Python engine has crashed. Please restart the application."
      );
    });

    return () => {
      unlisten.then((fn) => fn()).catch(console.error);
    };
  }, []);

  // Rehydrate canvas notes on startup (persistent card positions).
  useEffect(() => {
    const restored = loadPersistedNotes();
    if (restored && restored.length) {
      setNotes(restored);
    }
  }, []);

  useEffect(() => {
    const restored = loadPersistedLinks();
    if (restored && restored.length) {
      setLinks(restored);
    }
  }, []);

  useEffect(() => {
    const restored = loadPersistedCardLayout();
    if (restored) {
      setDefaultCardLayout(restored);
      defaultCardLayoutRef.current = restored;
    }
  }, []);

  // Persist notes as they change (debounced).
  useEffect(() => {
    const t = window.setTimeout(() => {
      try {
        persistNotes(notes);
        setStorageError(null); // Clear error on successful save
      } catch (err) {
        setStorageError("Failed to save canvas notes. Your data may not persist.");
      }
    }, 250);
    return () => window.clearTimeout(t);
  }, [notes]);

  // Persist links as they change (debounced).
  useEffect(() => {
    const t = window.setTimeout(() => {
      try {
        persistLinks(links);
        setStorageError(null); // Clear error on successful save
      } catch (err) {
        setStorageError("Failed to save canvas links. Your data may not persist.");
      }
    }, 250);
    return () => window.clearTimeout(t);
  }, [links]);

  const deletedChatIdSet = useMemo(() => new Set(deletedChatIds), [deletedChatIds]);
  const sortedSessions = useMemo(
    () => [...sessions].sort((a, b) => a.chat_id.localeCompare(b.chat_id)),
    [sessions]
  );
  const visibleSessions = useMemo(
    () => sortedSessions.filter((s) => !deletedChatIdSet.has(s.chat_id)),
    [sortedSessions, deletedChatIdSet]
  );

  // Pick the first session automatically when loaded.
  useEffect(() => {
    if (!loading && visibleSessions.length > 0 && !activeChat) {
      setActiveChat(visibleSessions[0].chat_id);
    }
  }, [visibleSessions, loading, activeChat]);

  useEffect(() => {
    notesRef.current = notes;
  }, [notes]);

  const handleSpawnHint = useCallback((hint: SpawnHint) => {
    spawnHintRef.current = hint;
  }, []);

  useEffect(() => {
    if (!deletedChatIds.length) return;
    const sessionIds = new Set(sessions.map((s) => s.chat_id));
    const next = deletedChatIds.filter((id) => sessionIds.has(id));
    if (next.length !== deletedChatIds.length) {
      setDeletedChatIds(next);
    }
  }, [sessions, deletedChatIds]);

  function focusChat(chatId: string) {
    if (!chatId) return;
    setActiveChat(chatId);
    setNotes((prev) => {
      const nextZ = (prev.reduce((m, n) => Math.max(m, n.z), 0) || 0) + 1;
      return prev.map((n) => (n.chatId === chatId ? { ...n, z: nextZ } : n));
    });
  }

  function closeOverlay() {
    if (!overlayChatId) return;
    setOverlayChatClosing(true);
    window.setTimeout(() => {
      setOverlayChatId(null);
      setOverlayChatClosing(false);
    }, 240);
  }

  function closeOverlayCard() {
    if (!overlayCardId) return;
    setOverlayCardClosing(true);
    window.setTimeout(() => {
      setOverlayCardId(null);
      setOverlayCardClosing(false);
    }, 240);
  }

  useEffect(() => {
    if (overlayChatId) setOverlayChatClosing(false);
  }, [overlayChatId]);

  useEffect(() => {
    if (overlayCardId) setOverlayCardClosing(false);
  }, [overlayCardId]);

  useEffect(() => {
    if (!overlayChatId && !overlayCardId) return;
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") {
        closeOverlay();
        closeOverlayCard();
        setSettingsOpen(false);
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [overlayChatId, overlayCardId]);

  const updateNote = useCallback((chatId: string, patch: Partial<CanvasNote>) => {
    if (!chatId) return;
    if (!patch || typeof patch !== "object") return;

    setNotes((prev) => {
      let changed = false;
      const next = prev.map((n) => {
        if (n.chatId !== chatId) return n;
        const merged: CanvasNote = { ...n, ...patch };

        // Avoid infinite update loops: only update when something actually changed.
        for (const [key, value] of Object.entries(patch)) {
          if (key === "layout") {
            if (!equalLayout(n.layout, (value as any) ?? undefined)) {
              changed = true;
            }
            continue;
          }
          if (!Object.is((n as any)[key], value)) {
            changed = true;
          }
        }

        return changed ? merged : n;
      });
      return changed ? next : prev;
    });
  }, []);

  // Keep this callback stable while an overlay is open. If it changes every render,
  // CardOverlay will re-run its "layout sync" effect continuously and can trigger
  // React's "Maximum update depth exceeded".
  const handleOverlayLayoutChange = useCallback(
    (nextLayout: CardLayout) => {
      if (!overlayCardId) return;
      const noteLayout = notesRef.current.find((n) => n.chatId === overlayCardId)?.layout;
      const isSync = noteLayout ? equalLayout(noteLayout, nextLayout) : false;
      updateNote(overlayCardId, { layout: nextLayout });
      if (nextLayout.showChat && nextLayout.showDocs && !isSync) {
        const updated: CardLayout = {
          ...DEFAULT_CARD_LAYOUT,
          chatOnRight: nextLayout.chatOnRight,
          splitRatio: nextLayout.splitRatio,
        };
        defaultCardLayoutRef.current = updated;
        setDefaultCardLayout(updated);
        if (layoutPersistTimerRef.current != null) {
          window.clearTimeout(layoutPersistTimerRef.current);
        }
        layoutPersistTimerRef.current = window.setTimeout(() => {
          layoutPersistTimerRef.current = null;
          persistCardLayout(updated);
        }, 200);
      }
    },
    [overlayCardId, updateNote]
  );

  const handleOverlayTitleChange = useCallback(
    (newTitle: string) => {
      if (!overlayCardId) return;
      updateNote(overlayCardId, { title: newTitle || undefined });
    },
    [overlayCardId, updateNote]
  );

  // Keep one global canvas note per chat (auto-place missing ones).
  useEffect(() => {
    if (loading) return;
    setNotes((prev) => {
      const existingIds = new Set(prev.map((n) => n.chatId));
      const sessionIds = new Set(visibleSessions.map((s) => s.chat_id));
      // Drop notes for chats that no longer exist.
      const filtered = prev.filter((n) => sessionIds.has(n.chatId));
      if (filtered.length !== prev.length) {
        prev = filtered;
      }
      const next: CanvasNote[] = [...prev];
      let z = (prev.reduce((m, n) => Math.max(m, n.z), 0) || 0) + 1;

      const toAdd = visibleSessions.filter((s) => !existingIds.has(s.chat_id));
      if (!toAdd.length) return prev;

      const spawnHint = spawnHintRef.current;
      const anchor = spawnHint?.anchor ?? { x: 80, y: 80 };
      const bounds = spawnHint?.bounds;
      const avoidZones = spawnHint?.avoidZones ?? [DEFAULT_DOCK_DEAD_ZONE];
      const colW = 260;
      const rowH = 170;
      const startIndex = prev.length;
      for (let i = 0; i < toAdd.length; i++) {
        const idx = startIndex + i;
        let desired = anchor;
        if (idx > 0) {
          const localIdx = idx - 1;
          const col = localIdx % 3;
          const row = Math.floor(localIdx / 3);
          desired = { x: anchor.x + col * colW, y: anchor.y + row * rowH };
        }
        const pos = findOpenPosition(desired, { w: 240, h: 140 }, next, bounds, avoidZones);
        next.push({
          chatId: toAdd[i].chat_id,
          title: toAdd[i].title,
          x: pos.x,
          y: pos.y,
          w: 240,
          h: 140,
          z: z++,
          layout: { ...defaultCardLayoutRef.current },
        });
      }
      return next;
    });
  }, [visibleSessions, loading]);

  // Drop links for chats that no longer exist.
  useEffect(() => {
    if (loading) return;
    const sessionIds = new Set(visibleSessions.map((s) => s.chat_id));
    setLinks((prev) => {
      const next = prev.filter((l) => sessionIds.has(l.fromChatId) && sessionIds.has(l.toChatId));
      return next.length === prev.length ? prev : next;
    });
  }, [visibleSessions, loading]);

  async function deleteChat(chatId: string) {
    // Note: window.confirm/alert can be blocked in the Tauri WebView.
    // Use an inline two-click confirmation instead.
    if (confirmDeleteChatId !== chatId) {
      setConfirmDeleteChatId(chatId);
      return;
    }
    setConfirmDeleteChatId(null);

    // Optimistic UI update.
    setDeletedChatIds((prev) => (prev.includes(chatId) ? prev : [...prev, chatId]));
    removeSession(chatId);
    if (activeChat === chatId) {
      const remaining = visibleSessions.filter((s) => s.chat_id !== chatId);
      setActiveChat(remaining.length ? remaining[0].chat_id : null);
    }
    setNotes((prev) => prev.filter((n) => n.chatId !== chatId));
    setLinks((prev) => prev.filter((l) => l.fromChatId !== chatId && l.toChatId !== chatId));
    setOverlayChatId((prev) => (prev === chatId ? null : prev));

    console.log("Deleting chat", chatId);
    const res = await engine(`/chat/sessions/${encodeURIComponent(chatId)}`, undefined, "DELETE");
    if (!res.ok) {
      console.error("Delete failed", res);
      setDeletedChatIds((prev) => prev.filter((id) => id !== chatId));
      // Re-sync from backend on failure.
      // Re-sync from backend on failure.
      await reload();
      return;
    }

    console.log("Deleted chat", chatId, res.data);
    // Ensure we converge to backend truth (also refreshes ordering).
    await reload();
  }

  function collectChatTreeIds(rootChatId: string): string[] {
    const childrenByParent = new Map<string, string[]>();
    for (const l of links) {
      if (l.kind !== "branch") continue;
      const arr = childrenByParent.get(l.fromChatId) || [];
      arr.push(l.toChatId);
      childrenByParent.set(l.fromChatId, arr);
    }
    const out: string[] = [];
    const stack = [rootChatId];
    const seen = new Set<string>();
    while (stack.length) {
      const id = stack.pop();
      if (!id) continue;
      if (seen.has(id)) continue;
      seen.add(id);
      out.push(id);
      const kids = childrenByParent.get(id);
      if (kids && kids.length) stack.push(...kids);
    }
    // Delete children first, parent last.
    return out.reverse();
  }

  async function deleteChatTree(rootChatId: string) {
    if (!rootChatId) return;
    const ids = collectChatTreeIds(rootChatId);
    if (!ids.length) return;

    setConfirmDeleteChatId(null);

    setDeletedChatIds((prev) => {
      const next = new Set(prev);
      ids.forEach((id) => next.add(id));
      return [...next];
    });
    // Optimistic UI update.
    for (const id of ids) removeSession(id);

    if (activeChat && ids.includes(activeChat)) {
      const remaining = visibleSessions.filter((s) => !ids.includes(s.chat_id));
      setActiveChat(remaining.length ? remaining[0].chat_id : null);
    }
    setNotes((prev) => prev.filter((n) => !ids.includes(n.chatId)));
    setLinks((prev) => prev.filter((l) => !ids.includes(l.fromChatId) && !ids.includes(l.toChatId)));
    setOverlayChatId((prev) => (prev && ids.includes(prev) ? null : prev));
    setOverlayCardId((prev) => (prev && ids.includes(prev) ? null : prev));

    console.log("Deleting chat group", rootChatId, ids);
    for (const id of ids) {
      const local = sessions.find((s) => s.chat_id === id)?.local;
      if (local) continue;
      const res = await engine(`/chat/sessions/${encodeURIComponent(id)}`, undefined, "DELETE");
      if (!res.ok) {
        console.error("Delete group failed", id, res);
        setDeletedChatIds((prev) => prev.filter((cid) => !ids.includes(cid)));
        await reload();
        return;
      }
    }

    console.log("Deleted chat group", rootChatId);
    await reload();
  }

  function createChatCardAt(
    pos: { x: number; y: number },
    opts?: { bounds?: { w: number; h: number }; avoidZones?: Array<{ x: number; y: number; w: number; h: number }> }
  ) {
    const id = `chat-${Date.now()}`;
    addLocalChat(id);
    setActiveChat(id);
    setNotes((prev) => {
      const nextZ = (prev.reduce((m, n) => Math.max(m, n.z), 0) || 0) + 1;
      const size = { w: 320, h: 200 };
      const placed = findOpenPosition(pos, size, prev, opts?.bounds, opts?.avoidZones);
      return [
        ...prev,
        {
          chatId: id,
          x: placed.x,
          y: placed.y,
          w: size.w,
          h: size.h,
          z: nextZ,
          layout: { ...defaultCardLayoutRef.current },
        },
      ];
    });
    return id;
  }

  // Branch/open a new child card (triggered from ChatWindow "Branch" action).
  useEffect(() => {
    function onOpenCard(e: Event) {
      const ce = e as CustomEvent;
      const childId = ce?.detail?.chatId;
      const parentId = ce?.detail?.parentChatId;
      const requestedTitle =
        typeof ce?.detail?.title === "string" ? (ce.detail.title as string).trim() : "";
      if (typeof childId !== "string" || !childId) return;

      addLocalChat(childId);
      setActiveChat(childId);

      if (typeof parentId === "string" && parentId) {
        const linkId = `branch:${parentId}:${childId}`;
        setLinks((prev) => {
          if (prev.some((l) => l.id === linkId)) return prev;
          return [...prev, { id: linkId, fromChatId: parentId, toChatId: childId, kind: "branch" }];
        });
      }

      setNotes((prev) => {
        const existing = prev.find((n) => n.chatId === childId);
        const nextZ = (prev.reduce((m, n) => Math.max(m, n.z), 0) || 0) + 1;
        if (existing) {
          return prev.map((n) => {
            if (n.chatId !== childId) return n;
            const hasTitle = typeof n.title === "string" && n.title.trim().length > 0;
            const nextTitle = !hasTitle && requestedTitle ? requestedTitle : n.title;
            return { ...n, z: nextZ, title: nextTitle };
          });
        }

        const parent = typeof parentId === "string" ? prev.find((n) => n.chatId === parentId) : null;
        const desired = {
          x: parent ? parent.x + Math.max(260, parent.w) + 40 : 80,
          y: parent ? parent.y + 20 : 80,
        };
        const size = { w: 360, h: 220 };
        const spawnHint = spawnHintRef.current;
        const bounds = spawnHint?.bounds;
        const avoidZones = spawnHint?.avoidZones ?? [DEFAULT_DOCK_DEAD_ZONE];
        const placed = findOpenPosition(desired, size, prev, bounds, avoidZones);

        return [
          ...prev,
          {
            chatId: childId,
            x: placed.x,
            y: placed.y,
            w: size.w,
            h: size.h,
            z: nextZ,
            title: requestedTitle || undefined,
            layout: { ...defaultCardLayoutRef.current },
          },
        ];
      });

      // Auto-open the child card immediately.
      setOverlayCardId(childId);
      setOverlayChatId(null);
    }
    window.addEventListener("insight:open-card", onOpenCard as any);
    return () => window.removeEventListener("insight:open-card", onOpenCard as any);
  }, [addLocalChat]);

  return (
    <div className="app-root">
      <div className="shell">
        <main className="main">
          <Canvas
            sessions={visibleSessions}
            activeChatId={activeChat}
            notes={notes}
            links={links}
            onFocusChat={focusChat}
            onUpdateNote={updateNote}
            onOpenChat={(chatId) => {
              setActiveChat(chatId);
              setOverlayChatId(chatId);
              setOverlayChatClosing(false);
              setOverlayCardId(null);
            }}
            onCreateChatAt={createChatCardAt}
            onOpenCard={(chatId) => {
              setActiveChat(chatId);
              setOverlayCardId(chatId);
              setOverlayCardClosing(false);
              setOverlayChatId(null);
            }}
            onOpenSettings={() => {
              setSettingsInitialTab("general");
              setSettingsOpen(true);
            }}
            onDeleteChat={deleteChat}
            onDeleteChatTree={deleteChatTree}
            confirmDeleteChatId={confirmDeleteChatId}
            onResetConfirmDelete={() => setConfirmDeleteChatId(null)}
            loadingSessions={loading}
            onSpawnHint={handleSpawnHint}
            dockVisible={!overlayCardId && !overlayChatId}
          />
          <FirstRunSetupModal
            open={setupOpen}
            onOpenSettings={() => {
              setSetupOpen(false);
              setSettingsInitialTab("model");
              setSettingsOpen(true);
            }}
          />
          <SettingsModal
            open={settingsOpen}
            onClose={() => setSettingsOpen(false)}
            themeMode={themeMode}
            onThemeModeChange={(mode) => setThemeMode(mode)}
            initialTab={settingsInitialTab}
          />
          {engineCrashed ? (
            <div className="crash-backdrop" role="dialog" aria-modal="true">
              <div className="crash-modal">
                <div className="crash-header">
                  <h2 className="crash-title">Engine Crash Detected</h2>
                </div>
                <div className="crash-body">
                  <div className="crash-message">
                    {crashDetails || "The Python engine has crashed unexpectedly."}
                  </div>
                  <div className="crash-suggestions">
                    <h3>Recovery Options:</h3>
                    <ul>
                      <li>Restart the application using the button below</li>
                      <li>If crashes persist, open Settings → Model and reduce GPU layers</li>
                      <li>Current setting may be too high for your system</li>
                      <li>Try setting GPU layers to 35 or lower</li>
                    </ul>
                  </div>
                </div>
                <div className="crash-actions">
                  <button
                    className="crash-btn primary"
                    onClick={() => window.location.reload()}
                    type="button"
                  >
                    Restart Application
                  </button>
                </div>
              </div>
            </div>
          ) : null}
          {storageError && (
            <div
              style={{
                position: "fixed",
                top: 10,
                left: "50%",
                transform: "translateX(-50%)",
                zIndex: 9999,
                backgroundColor: "#f59e0b",
                color: "#000",
                padding: "12px 20px",
                borderRadius: "8px",
                boxShadow: "0 4px 12px rgba(0,0,0,0.3)",
                display: "flex",
                alignItems: "center",
                gap: "12px",
                maxWidth: "90vw",
              }}
            >
              <span style={{ fontSize: "14px", fontWeight: 500 }}>
                ⚠️ {storageError}
              </span>
              <button
                onClick={() => setStorageError(null)}
                style={{
                  background: "rgba(0,0,0,0.1)",
                  border: "none",
                  borderRadius: "4px",
                  padding: "4px 8px",
                  cursor: "pointer",
                  fontSize: "12px",
                }}
              >
                Dismiss
              </button>
            </div>
          )}
          {overlayCardId ? (
            <OverlayBoundary title="Card UI crashed" onClose={closeOverlayCard}>
              {(() => {
                const note = notes.find((n) => n.chatId === overlayCardId);
                const layout = note?.layout ?? defaultCardLayout;
                return (
                  <CardOverlay
                    key={overlayCardId}
                    title={
                      note?.title ||
                      visibleSessions.find((s) => s.chat_id === overlayCardId)?.title ||
                      overlayCardId
                    }
                    chatId={overlayCardId}
                    onClose={closeOverlayCard}
                    closing={overlayCardClosing}
                    initialLayout={layout}
                    onLayoutChange={handleOverlayLayoutChange}
                    onTitleChange={handleOverlayTitleChange}
                    sessions={visibleSessions}
                    loadingSessions={loading}
                    onOpenCard={(chatId) => {
                      setActiveChat(chatId);
                      setOverlayCardId(chatId);
                      setOverlayCardClosing(false);
                      setOverlayChatId(null);
                    }}
                    onCreateCard={() => {
        const spawnHint = spawnHintRef.current;
        const anchor = spawnHint?.anchor ?? { x: 80, y: 80 };
                      const avoidZones = spawnHint?.avoidZones ?? [DEFAULT_DOCK_DEAD_ZONE];
                      return createChatCardAt(anchor, {
                        bounds: spawnHint?.bounds,
                        avoidZones,
                      });
                    }}
                    onOpenSettings={() => {
                      setSettingsInitialTab("general");
                      setSettingsOpen(true);
                    }}
                    onDeleteChat={deleteChat}
                    confirmDeleteChatId={confirmDeleteChatId}
                    onRequireModel={ensureModelReady}
                  />
                );
              })()}
            </OverlayBoundary>
          ) : null}
          {overlayChatId ? (
            <div
              className="chat-overlay"
              data-state={overlayChatClosing ? "closing" : "open"}
              role="dialog"
              aria-modal="true"
              aria-label="Chat"
              onPointerDown={(e) => {
                if (e.target === e.currentTarget) closeOverlay();
              }}
            >
              <div
                className="chat-overlay-window"
                onPointerDown={(e) => {
                  // Prevent backdrop-close from firing when clicking inside the window.
                  e.stopPropagation();
                }}
              >
                <div className="chat-overlay-header">
                  <div className="chat-overlay-title">
                    {notes.find((n) => n.chatId === overlayChatId)?.title ||
                      visibleSessions.find((s) => s.chat_id === overlayChatId)?.title ||
                      overlayChatId}
                  </div>
                  <button
                    className="chat-overlay-close"
                    type="button"
                    onClick={closeOverlay}
                    aria-label="Close"
                    title="Close"
                  >
                    ×
                  </button>
                </div>
                <div className="chat-overlay-body">
                  <OverlayBoundary title="Chat UI crashed" onClose={closeOverlay}>
                    <ChatWindow
                      chatId={overlayChatId}
                      active={true}
                      embedded={false}
                      showTopbar={false}
                      docsVisible={false}
                      onRequireModel={ensureModelReady}
                    />
                  </OverlayBoundary>
                </div>
              </div>
            </div>
          ) : null}
        </main>
      </div>
    </div>
  );
}

export default App;

class OverlayBoundary extends React.Component<
  { title: string; onClose: () => void; children: React.ReactNode },
  { error: Error | null }
> {
  constructor(props: { title: string; onClose: () => void; children: React.ReactNode }) {
    super(props);
    this.state = { error: null };
  }
  static getDerivedStateFromError(error: Error) {
    return { error };
  }
  componentDidCatch(error: Error) {
    console.error(this.props.title, error);
  }
  render() {
    if (this.state.error) {
      return (
        <div className="overlay-error">
          <div className="overlay-error-title">{this.props.title}</div>
          <div className="overlay-error-msg">
            {String(this.state.error?.message || this.state.error)}
          </div>
          <button
            className="overlay-error-btn"
            onClick={this.props.onClose}
            type="button"
          >
            Close
          </button>
        </div>
      );
    }
    return this.props.children as any;
  }
}
