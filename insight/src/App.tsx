import { useEffect, useMemo, useState } from "react";
import "./App.css";
import { ChatWindow } from "./components/ChatWindow";
import { useSessions } from "./state/useSessions";
import { engine } from "./api/engine";

function App() {
  const { sessions, loading, reload, removeSession, addLocalChat } = useSessions();
  const [activeChat, setActiveChat] = useState<string | null>(null);
  const [confirmDeleteChatId, setConfirmDeleteChatId] = useState<string | null>(
    null
  );

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

  function newChat() {
    const id = `chat-${Date.now()}`;
    addLocalChat(id);
    setActiveChat(id);
  }

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

  return (
    <div className="app-root">
      <div className="shell">
        <aside className="sidebar">
          <div className="sidebar-header">
            <div className="sidebar-title">Chats</div>
            <button className="sidebar-new-btn" onClick={newChat}>
              + New chat
            </button>
          </div>
          <div className="sidebar-list">
            {sortedSessions.length === 0 ? (
              loading ? <div className="sidebar-item muted">Loading…</div> : null
            ) : null}
            {loading && sortedSessions.length > 0 ? (
              <div className="sidebar-item muted">Syncing…</div>
            ) : null}
            {sortedSessions.map((s) => (
              <div key={s.chat_id} className="sidebar-row">
                <button
                  className={`sidebar-item sidebar-item-btn ${
                    s.chat_id === activeChat ? "active" : ""
                  }`}
                  onClick={() => setActiveChat(s.chat_id)}
                  title={s.chat_id}
                >
                  {s.title || s.chat_id}
                </button>
                <button
                  className={`sidebar-delete-btn ${
                    confirmDeleteChatId === s.chat_id ? "confirm" : ""
                  }`}
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    deleteChat(s.chat_id);
                  }}
                  title={
                    confirmDeleteChatId === s.chat_id
                      ? "Click again to confirm delete"
                      : "Delete chat"
                  }
                  aria-label={`Delete chat ${s.chat_id}`}
                >
                  {confirmDeleteChatId === s.chat_id ? "Del" : "×"}
                </button>
              </div>
            ))}
          </div>
        </aside>
        <main className="main">
          <ChatWindow chatId={activeChat} />
        </main>
      </div>
    </div>
  );
}

export default App;
