import { useEffect, useRef, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";

export type ChatSummary = {
  chat_id: string;
  title?: string;
  local?: boolean;
  last_message_at?: string;
  last_message_content?: string;
  file_count?: number;
};

export function useSessions() {
  const [sessions, setSessions] = useState<ChatSummary[]>([]);
  const [loading, setLoading] = useState(true);

  // Coalesce multiple reload triggers (watcher + manual reload).
  // If reload is already in-flight, we schedule one more pass and return the in-flight promise.
  const inFlightRef = useRef(false);
  const queuedRef = useRef(false);
  const promiseRef = useRef<Promise<void> | null>(null);

  async function reload(): Promise<void> {
    if (inFlightRef.current) {
      queuedRef.current = true;
      return promiseRef.current ?? Promise.resolve();
    }
    inFlightRef.current = true;
    promiseRef.current = (async () => {
      do {
        queuedRef.current = false;
        setLoading(true);
        try {
          const res = await invoke<{ sessions?: ChatSummary[] }>("list_sessions");
          if (res && (res as any).sessions) {
            const next = ((res as any).sessions as ChatSummary[]).map((c) => ({
              ...c,
              local: false,
            }));
            // Preserve any local (unsent) chats not yet materialized in SQLite/KV.
            setSessions((prev) => {
              const localOnly = prev.filter(
                (c) => c.local && !next.some((n) => n.chat_id === c.chat_id)
              );
              return [...localOnly, ...next];
            });
          }
        } finally {
          setLoading(false);
        }
      } while (queuedRef.current);
      inFlightRef.current = false;
      promiseRef.current = null;
    })();
    return promiseRef.current;
  }

  function removeSession(chatId: string) {
    setSessions((prev) => prev.filter((s) => s.chat_id !== chatId));
  }

  function addLocalChat(chatId: string) {
    setSessions((prev) => {
      if (prev.some((s) => s.chat_id === chatId)) return prev;
      return [{ chat_id: chatId, title: "New chat", local: true }, ...prev];
    });
  }

  useEffect(() => {
    reload();
    const unlistenPromise = listen("sessions_updated", () => reload());
    return () => {
      unlistenPromise.then((unsub) => unsub()).catch(() => {});
    };
  }, []);

  return { sessions, loading, reload, removeSession, addLocalChat };
}
