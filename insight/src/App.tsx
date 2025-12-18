import React, { useEffect, useMemo, useState } from "react";
import "./App.css";
import { Canvas, CanvasNote } from "./components/Canvas";
import { CardOverlay } from "./components/CardOverlay";
import { ChatWindow } from "./components/ChatWindow";
import { useSessions } from "./state/useSessions";
import { engine } from "./api/engine";

const NOTES_STORAGE_KEY = "insight.canvas.notes.v1";

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
      out.push({ chatId, x, y, w, h, z, title: (item as any).title });
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

function App() {
  const { sessions, loading, reload, removeSession, addLocalChat } = useSessions();
  const [activeChat, setActiveChat] = useState<string | null>(null);
  const [notes, setNotes] = useState<CanvasNote[]>([]);
  const [overlayChatId, setOverlayChatId] = useState<string | null>(null);
  const [overlayCardId, setOverlayCardId] = useState<string | null>(null);
  const [overlayCardShowChat, setOverlayCardShowChat] = useState(false);
  const [confirmDeleteChatId, setConfirmDeleteChatId] = useState<string | null>(
    null
  );

  // Rehydrate canvas notes on startup (persistent card positions).
  useEffect(() => {
    const restored = loadPersistedNotes();
    if (restored && restored.length) {
      setNotes(restored);
    }
  }, []);

  // Persist notes as they change (debounced).
  useEffect(() => {
    const t = window.setTimeout(() => persistNotes(notes), 250);
    return () => window.clearTimeout(t);
  }, [notes]);

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
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [overlayChatId, overlayCardId]);

  function updateNote(chatId: string, patch: Partial<CanvasNote>) {
    setNotes((prev) => prev.map((n) => (n.chatId === chatId ? { ...n, ...patch } : n)));
  }

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
        });
      }
      return next;
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
        },
      ];
    });
  }

  return (
    <div className="app-root">
      <div className="shell">
        <main className="main">
          <Canvas
            sessions={sortedSessions}
            activeChatId={activeChat}
            notes={notes}
            onFocusChat={focusChat}
            onUpdateNote={updateNote}
            onOpenChat={(chatId) => {
              setActiveChat(chatId);
              setOverlayChatId(chatId);
              setOverlayCardId(null);
            }}
            onCreateChatAt={createChatCardAt}
            onOpenCard={(chatId, opts) => {
              setActiveChat(chatId);
              setOverlayCardId(chatId);
              setOverlayChatId(null);
              setOverlayCardShowChat(opts?.showChat ?? true);
            }}
            onDeleteChat={deleteChat}
            confirmDeleteChatId={confirmDeleteChatId}
            loadingSessions={loading}
          />
          {overlayCardId ? (
            <CardOverlay
              title={sortedSessions.find((s) => s.chat_id === overlayCardId)?.title || overlayCardId}
              chatId={overlayCardId}
              onClose={() => setOverlayCardId(null)}
              initialShowChat={overlayCardShowChat}
            />
          ) : null}
          {overlayChatId ? (
            <div
              className="chat-overlay"
              style={{ background: "rgba(0, 0, 0, 0.6)" }}
              role="dialog"
              aria-modal="true"
              aria-label="Chat"
              onPointerDown={(e) => {
                if (e.target === e.currentTarget) closeOverlay();
              }}
            >
              <div
                className="chat-overlay-window"
                style={{ background: "#0f172a" }}
                onPointerDown={(e) => {
                  // Prevent backdrop-close from firing when clicking inside the window.
                  e.stopPropagation();
                }}
              >
                <div className="chat-overlay-header">
                  <div className="chat-overlay-title">
                    {sortedSessions.find((s) => s.chat_id === overlayChatId)?.title || overlayChatId}
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
                  <ChatOverlayBoundary onClose={closeOverlay}>
                    <ChatWindow chatId={overlayChatId} active={true} embedded={false} showTopbar={false} />
                  </ChatOverlayBoundary>
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

class ChatOverlayBoundary extends React.Component<
  { onClose: () => void; children: React.ReactNode },
  { error: Error | null }
> {
  constructor(props: { onClose: () => void; children: React.ReactNode }) {
    super(props);
    this.state = { error: null };
  }
  static getDerivedStateFromError(error: Error) {
    return { error };
  }
  componentDidCatch(error: Error) {
    console.error("Chat overlay crashed", error);
  }
  render() {
    if (this.state.error) {
      return (
        <div style={{ padding: 12, color: "#e5e7eb" }}>
          <div style={{ fontWeight: 700, marginBottom: 8 }}>Chat UI crashed</div>
          <div style={{ fontSize: 12, opacity: 0.85, whiteSpace: "pre-wrap" }}>
            {String(this.state.error?.message || this.state.error)}
          </div>
          <button
            style={{
              marginTop: 12,
              height: 30,
              padding: "0 10px",
              borderRadius: 10,
              border: "1px solid rgba(75, 85, 99, 0.65)",
              background: "rgba(17, 24, 39, 0.6)",
              color: "#e5e7eb",
              cursor: "pointer",
            }}
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
