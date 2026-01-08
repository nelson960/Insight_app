import React, { useCallback, useEffect, useMemo, useState } from "react";
import "./App.css";
import { Canvas, CanvasLink, CanvasNote } from "./components/Canvas";
import type { CardLayout } from "./components/Canvas";
import { CardOverlay } from "./components/CardOverlay";
import { ChatWindow } from "./components/ChatWindow";
import { SettingsModal, ThemeMode } from "./components/SettingsModal";
import { useSessions } from "./state/useSessions";
import { engine } from "./api/engine";
import { setTheme as setAppTheme } from "@tauri-apps/api/app";
import { getCurrentWindow } from "@tauri-apps/api/window";

const NOTES_STORAGE_KEY = "insight.canvas.notes.v1";
const LINKS_STORAGE_KEY = "insight.canvas.links.v1";
const DEFAULT_CARD_LAYOUT: CardLayout = {
  showChat: true,
  showDocs: true,
  chatOnRight: true,
  splitRatio: 0.5,
};

function coerceCardLayout(raw: unknown): CardLayout | null {
  if (!raw || typeof raw !== "object") return null;
  const obj = raw as any;

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

function loadPersistedNotes(): CanvasNote[] | null {
  try {
    const raw = localStorage.getItem(NOTES_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return null;
    const out: CanvasNote[] = [];
    for (const item of parsed) {
      if (!item || typeof item !== "object") continue;
      const chatId = (item as any).chatId;
      const x = Number((item as any).x);
      const y = Number((item as any).y);
      const w = Number((item as any).w);
      const h = Number((item as any).h);
      const z = Number((item as any).z);
      if (typeof chatId !== "string" || !chatId) continue;
      if (![x, y, w, h, z].every(Number.isFinite)) continue;
      const layout = coerceCardLayout((item as any).layout);
      const locked = typeof (item as any).locked === "boolean" ? (item as any).locked : undefined;
      const collapsed =
        typeof (item as any).collapsed === "boolean" ? (item as any).collapsed : undefined;
      const groupColorRaw = (item as any).groupColor;
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
        title: (item as any).title,
        locked,
        collapsed,
        groupColor,
        layout: layout ?? undefined,
      });
    }
    return out;
  } catch {
    return null;
  }
}

function persistNotes(notes: CanvasNote[]) {
  try {
    localStorage.setItem(NOTES_STORAGE_KEY, JSON.stringify(notes));
  } catch {
    // ignore
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
      const id = (item as any).id;
      const fromChatId = (item as any).fromChatId;
      const toChatId = (item as any).toChatId;
      const kind = (item as any).kind;
      if (typeof id !== "string" || !id) continue;
      if (typeof fromChatId !== "string" || !fromChatId) continue;
      if (typeof toChatId !== "string" || !toChatId) continue;
      if (kind !== "branch") continue;
      out.push({ id, fromChatId, toChatId, kind });
    }
    return out;
  } catch {
    return null;
  }
}

function persistLinks(links: CanvasLink[]) {
  try {
    localStorage.setItem(LINKS_STORAGE_KEY, JSON.stringify(links));
  } catch {
    // ignore
  }
}

function App() {
  const { sessions, loading, reload, removeSession, addLocalChat } = useSessions();
  const [activeChat, setActiveChat] = useState<string | null>(null);
  const [notes, setNotes] = useState<CanvasNote[]>([]);
  const [links, setLinks] = useState<CanvasLink[]>([]);
  const [overlayChatId, setOverlayChatId] = useState<string | null>(null);
  const [overlayCardId, setOverlayCardId] = useState<string | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [themeMode, setThemeMode] = useState<ThemeMode>("system");
  const [confirmDeleteChatId, setConfirmDeleteChatId] = useState<string | null>(
    null
  );

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
        void win.setTitleBarStyle("transparent").catch(() => {});
        // On macOS, theme is app-wide; use `null` to follow system.
        const tauriTheme = mode === "system" ? null : effectiveTheme;
        void win.setTheme(tauriTheme).catch((e) => {
          console.warn("[theme] window.setTheme failed", e);
        });
        void win.setBackgroundColor(titleBarBg).catch(() => {});
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

  // Persist notes as they change (debounced).
  useEffect(() => {
    const t = window.setTimeout(() => persistNotes(notes), 250);
    return () => window.clearTimeout(t);
  }, [notes]);

  // Persist links as they change (debounced).
  useEffect(() => {
    const t = window.setTimeout(() => persistLinks(links), 250);
    return () => window.clearTimeout(t);
  }, [links]);

  // Pick the first session automatically when loaded.
  useEffect(() => {
    if (!loading && sessions.length > 0 && !activeChat) {
      setActiveChat(sessions[0].chat_id);
    }
  }, [sessions, loading, activeChat]);

  const sortedSessions = useMemo(
    () => [...sessions].sort((a, b) => a.chat_id.localeCompare(b.chat_id)),
    [sessions]
  );

  function focusChat(chatId: string) {
    if (!chatId) return;
    setActiveChat(chatId);
    setNotes((prev) => {
      const nextZ = (prev.reduce((m, n) => Math.max(m, n.z), 0) || 0) + 1;
      return prev.map((n) => (n.chatId === chatId ? { ...n, z: nextZ } : n));
    });
  }

  function closeOverlay() {
    setOverlayChatId(null);
  }

  useEffect(() => {
    if (!overlayChatId && !overlayCardId) return;
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") {
        setOverlayChatId(null);
        setOverlayCardId(null);
        setSettingsOpen(false);
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [overlayChatId, overlayCardId]);

  const updateNote = useCallback((chatId: string, patch: Partial<CanvasNote>) => {
    if (!chatId) return;
    if (!patch || typeof patch !== "object") return;

    function equalLayout(a?: CardLayout, b?: CardLayout) {
      if (a === b) return true;
      if (!a || !b) return false;
      return (
        a.showChat === b.showChat &&
        a.showDocs === b.showDocs &&
        a.chatOnRight === b.chatOnRight &&
        a.splitRatio === b.splitRatio &&
        (a.activeFileId ?? null) === (b.activeFileId ?? null)
      );
    }

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
      updateNote(overlayCardId, { layout: nextLayout });
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
      const sessionIds = new Set(sessions.map((s) => s.chat_id));
      // Drop notes for chats that no longer exist.
      const filtered = prev.filter((n) => sessionIds.has(n.chatId));
      if (filtered.length !== prev.length) {
        prev = filtered;
      }
      const next: CanvasNote[] = [...prev];
      let z = (prev.reduce((m, n) => Math.max(m, n.z), 0) || 0) + 1;

      const toAdd = sessions.filter((s) => !existingIds.has(s.chat_id));
      if (!toAdd.length) return prev;

      // Simple grid-ish placement.
      const baseX = 60;
      const baseY = 60;
      const colW = 260;
      const rowH = 170;
      const startIndex = prev.length;
      for (let i = 0; i < toAdd.length; i++) {
        const idx = startIndex + i;
        const col = idx % 4;
        const row = Math.floor(idx / 4);
        next.push({
          chatId: toAdd[i].chat_id,
          title: toAdd[i].title,
          x: baseX + col * colW,
          y: baseY + row * rowH,
          w: 240,
          h: 140,
          z: z++,
          layout: { ...DEFAULT_CARD_LAYOUT },
        });
      }
      return next;
    });
  }, [sessions, loading]);

  // Drop links for chats that no longer exist.
  useEffect(() => {
    if (loading) return;
    const sessionIds = new Set(sessions.map((s) => s.chat_id));
    setLinks((prev) => {
      const next = prev.filter((l) => sessionIds.has(l.fromChatId) && sessionIds.has(l.toChatId));
      return next.length === prev.length ? prev : next;
    });
  }, [sessions, loading]);

  async function deleteChat(chatId: string) {
    // Note: window.confirm/alert can be blocked in the Tauri WebView.
    // Use an inline two-click confirmation instead.
    if (confirmDeleteChatId !== chatId) {
      setConfirmDeleteChatId(chatId);
      return;
    }
    setConfirmDeleteChatId(null);

    // Optimistic UI update.
    removeSession(chatId);
    if (activeChat === chatId) {
      const remaining = sessions.filter((s) => s.chat_id !== chatId);
      setActiveChat(remaining.length ? remaining[0].chat_id : null);
    }
    setNotes((prev) => prev.filter((n) => n.chatId !== chatId));
    setLinks((prev) => prev.filter((l) => l.fromChatId !== chatId && l.toChatId !== chatId));
    setOverlayChatId((prev) => (prev === chatId ? null : prev));

    console.log("Deleting chat", chatId);
    const res = await engine(`/chat/sessions/${encodeURIComponent(chatId)}`, undefined, "DELETE");
    if (!res.ok) {
      console.error("Delete failed", res);
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

    // Optimistic UI update.
    for (const id of ids) removeSession(id);

    if (activeChat && ids.includes(activeChat)) {
      const remaining = sessions.filter((s) => !ids.includes(s.chat_id));
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
        await reload();
        return;
      }
    }

    console.log("Deleted chat group", rootChatId);
    await reload();
  }

  function createChatCardAt(pos: { x: number; y: number }) {
    const id = `chat-${Date.now()}`;
    addLocalChat(id);
    setActiveChat(id);
    setNotes((prev) => {
      const nextZ = (prev.reduce((m, n) => Math.max(m, n.z), 0) || 0) + 1;
      return [
        ...prev,
        {
          chatId: id,
          x: pos.x,
          y: pos.y,
          w: 320,
          h: 200,
          z: nextZ,
          layout: { ...DEFAULT_CARD_LAYOUT },
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
        const x = parent ? parent.x + Math.max(260, parent.w) + 40 : 80;
        const y = parent ? parent.y + 20 : 80;

        return [
          ...prev,
          {
            chatId: childId,
            x,
            y,
            w: 360,
            h: 220,
            z: nextZ,
            title: requestedTitle || undefined,
            layout: { ...DEFAULT_CARD_LAYOUT },
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
            sessions={sortedSessions}
            activeChatId={activeChat}
            notes={notes}
            links={links}
            onFocusChat={focusChat}
            onUpdateNote={updateNote}
            onOpenChat={(chatId) => {
              setActiveChat(chatId);
              setOverlayChatId(chatId);
              setOverlayCardId(null);
            }}
            onCreateChatAt={createChatCardAt}
            onOpenCard={(chatId) => {
              setActiveChat(chatId);
              setOverlayCardId(chatId);
              setOverlayChatId(null);
            }}
            onOpenSettings={() => setSettingsOpen(true)}
            onDeleteChat={deleteChat}
            onDeleteChatTree={deleteChatTree}
            confirmDeleteChatId={confirmDeleteChatId}
            onResetConfirmDelete={() => setConfirmDeleteChatId(null)}
            loadingSessions={loading}
            dockVisible={!overlayCardId && !overlayChatId}
          />
          <SettingsModal
            open={settingsOpen}
            onClose={() => setSettingsOpen(false)}
            themeMode={themeMode}
            onThemeModeChange={(mode) => setThemeMode(mode)}
          />
          {overlayCardId ? (
            <OverlayBoundary title="Card UI crashed" onClose={() => setOverlayCardId(null)}>
              {(() => {
                const note = notes.find((n) => n.chatId === overlayCardId);
                const layout = note?.layout ?? DEFAULT_CARD_LAYOUT;
                return (
                  <CardOverlay
                    key={overlayCardId}
                    title={
                      note?.title ||
                      sortedSessions.find((s) => s.chat_id === overlayCardId)?.title ||
                      overlayCardId
                    }
                    chatId={overlayCardId}
                    onClose={() => setOverlayCardId(null)}
                    initialLayout={layout}
                    onLayoutChange={handleOverlayLayoutChange}
                    onTitleChange={handleOverlayTitleChange}
                    sessions={sortedSessions}
                    loadingSessions={loading}
                    onOpenCard={(chatId) => {
                      setActiveChat(chatId);
                      setOverlayCardId(chatId);
                      setOverlayChatId(null);
                    }}
                    onCreateCard={() => {
                      const idx = notes.length;
                      const col = idx % 4;
                      const row = Math.floor(idx / 4);
                      return createChatCardAt({
                        x: 60 + col * 260,
                        y: 60 + row * 170,
                      });
                    }}
                    onOpenSettings={() => setSettingsOpen(true)}
                    onDeleteChat={deleteChat}
                    confirmDeleteChatId={confirmDeleteChatId}
                  />
                );
              })()}
            </OverlayBoundary>
          ) : null}
          {overlayChatId ? (
            <div
              className="chat-overlay"
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
                      sortedSessions.find((s) => s.chat_id === overlayChatId)?.title ||
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
