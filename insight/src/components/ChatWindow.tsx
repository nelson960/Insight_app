import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen, UnlistenFn } from "@tauri-apps/api/event";
import {
  cancelActiveStreamAndWait,
  engineCancel,
  engineStreamChat,
  getActiveStream,
  waitForStreamToFinish,
} from "../api/engine";
import { engine } from "../api/engine";

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  attachments?: string[];
};

type Props = {
  chatId: string | null;
};

type ContextStatus = {
  chat_id: string;
  used_tokens: number;
  capacity_tokens: number;
  percent: number;
  compacted?: boolean;
};

export function ChatWindow({ chatId }: Props) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [attachedPaths, setAttachedPaths] = useState<string[]>([]);
  const [contextStatus, setContextStatus] = useState<ContextStatus | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const pendingTextRef = useRef("");
  const flushRafRef = useRef<number | null>(null);
  const activeRequestIdRef = useRef<string | null>(null);
  const assistantIdRef = useRef<string | null>(null);
  const unlistenTokenRef = useRef<UnlistenFn | null>(null);
  const unlistenDoneRef = useRef<UnlistenFn | null>(null);
  const unlistenErrorRef = useRef<UnlistenFn | null>(null);
  const activeChatIdRef = useRef<string | null>(null);
  const contextReqSeqRef = useRef(0);
  const inputElRef = useRef<HTMLTextAreaElement | null>(null);
  const INPUT_MAX_HEIGHT_PX = 120;

  function autosizeInput() {
    const el = inputElRef.current;
    if (!el) return;
    // Reset first so scrollHeight reflects current content.
    el.style.height = "auto";
    // A small extra pixel prevents the caret from being visually clipped on some platforms.
    const next = Math.min(el.scrollHeight + 2, INPUT_MAX_HEIGHT_PX);
    el.style.height = `${next}px`;
    el.style.overflowY = el.scrollHeight > INPUT_MAX_HEIGHT_PX ? "auto" : "hidden";
  }

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    activeChatIdRef.current = chatId;
  }, [chatId]);

  useLayoutEffect(() => {
    autosizeInput();
  }, [input]);

  function formatInt(n: number) {
    try {
      return new Intl.NumberFormat().format(n);
    } catch {
      return String(n);
    }
  }

  function cleanupListeners() {
    for (const ref of [unlistenTokenRef, unlistenDoneRef, unlistenErrorRef]) {
      if (!ref.current) continue;
      try {
        ref.current();
      } catch {
        // ignore
      }
      ref.current = null;
    }
  }

  function flushPendingToMessage() {
    if (flushRafRef.current != null) {
      cancelAnimationFrame(flushRafRef.current);
      flushRafRef.current = null;
    }
    const pending = pendingTextRef.current;
    pendingTextRef.current = "";
    const assistantId = assistantIdRef.current;
    if (!pending || !assistantId) return;
    setMessages((prev) =>
      prev.map((m) =>
        m.id === assistantId ? { ...m, content: (m.content || "") + pending } : m
      )
    );
  }

  // Load persisted messages when switching chats
  useEffect(() => {
    // Switching chats should not keep streaming UI/listeners from the previous chat.
    // A stream may still be finalizing in the background (KV save), but this view should remain responsive.
    cleanupListeners();
    pendingTextRef.current = "";
    assistantIdRef.current = null;
    activeRequestIdRef.current = null;
    setIsStreaming(false);
    setError(null);

    if (!chatId) {
      setAttachedPaths([]);
      setMessages([]);
      setContextStatus(null);
      return;
    }
    let cancelled = false;
    async function loadHistory() {
      try {
        const res = await invoke<{ messages?: { role: string; content: string; attachments?: string[] }[] }>(
          "get_session_messages",
          { chatId }
        );
        if (!cancelled && res && (res as any).messages) {
          const msgs = (res as any).messages.map((m: any, idx: number) => ({
            id: `${chatId}-${idx}-${m.role}`,
            role: m.role === "assistant" ? "assistant" : "user",
            content: m.content ?? "",
            attachments: Array.isArray(m.attachments) ? m.attachments : undefined,
          })) as ChatMessage[];
          setMessages(msgs);
        }
      } catch {
        // ignore; keep empty
      }
    }
    loadHistory();
    return () => {
      cancelled = true;
    };
  }, [chatId]);

  async function refreshContextStatus(activeChatId: string) {
    const mySeq = ++contextReqSeqRef.current;
    const res = await engine<ContextStatus>(
      `/chat/context/${encodeURIComponent(activeChatId)}`,
      undefined,
      "GET"
    );
    // Ignore late responses for a chat that is no longer active.
    if (activeChatIdRef.current !== activeChatId) return;
    // Ignore stale results if a newer refresh started.
    if (contextReqSeqRef.current !== mySeq) return;
    if (res.ok && res.data && (res.data as any).chat_id === activeChatId) {
      setContextStatus(res.data);
    } else {
      setContextStatus(null);
    }
  }

  useEffect(() => {
    if (!chatId) return;
    let cancelled = false;
    // Show a neutral state immediately so we don't display the previous chat's status.
    setContextStatus(null);

    // Avoid calling engine_request while any stream is still finalizing; the stream reader
    // can consume non-stream responses and make engine_request hang.
    const active = getActiveStream();
    if (!active) {
      refreshContextStatus(chatId).catch(() => {
        if (!cancelled) setContextStatus(null);
      });
      return () => {
        cancelled = true;
      };
    }

    // If another chat is still streaming/finalizing, wait for it to end then refresh.
    waitForStreamToFinish(active.requestId)
      .then(() => {
        if (!cancelled && chatId) {
          refreshContextStatus(chatId).catch(() => {
            if (!cancelled) setContextStatus(null);
          });
        }
      })
      .catch(() => {
        if (!cancelled) setContextStatus(null);
      });
    return () => {
      cancelled = true;
    };
  }, [chatId]);

  function filenameFromPath(p: string) {
    const parts = p.split(/[\\/]/g);
    return parts[parts.length - 1] || p;
  }

  function formatBytes(n: number) {
    if (!Number.isFinite(n) || n <= 0) return "0 B";
    const units = ["B", "KB", "MB", "GB"];
    let i = 0;
    let v = n;
    while (v >= 1024 && i < units.length - 1) {
      v /= 1024;
      i += 1;
    }
    const digits = i === 0 ? 0 : 1;
    return `${v.toFixed(digits)} ${units[i]}`;
  }

  async function pickAttachments() {
    if (!chatId) {
      setError("Select or create a chat first.");
      return;
    }
    if (isStreaming) return;

    setError(null);
    try {
      const res = await invoke<{
        files?: { path: string; name?: string; size_bytes?: number; max_bytes?: number }[];
      }>("pick_files");
      const files = res?.files || [];
      if (!files.length) return;

      const accepted: string[] = [];
      const rejected: { name: string; size: number; max: number }[] = [];

      for (const f of files) {
        const p = (f as any)?.path as string;
        if (!p) continue;
        const name = (f as any)?.name || filenameFromPath(p);
        const size = Number((f as any)?.size_bytes ?? 0);
        const max = Number((f as any)?.max_bytes ?? 0);
        if (max > 0 && size > max) {
          rejected.push({ name, size, max });
          continue;
        }
        accepted.push(p);
      }

      if (rejected.length) {
        const first = rejected[0];
        const extra = rejected.length > 1 ? ` (+${rejected.length - 1} more)` : "";
        setError(
          `File too large: ${first.name} (${formatBytes(first.size)}). Max ${formatBytes(first.max)}.${extra}`
        );
      }

      if (!accepted.length) return;

      // Only attach; actual ingestion happens when user clicks Send (with query).
      setAttachedPaths((prev) => {
        const set = new Set(prev);
        for (const p of accepted) set.add(p);
        return Array.from(set);
      });
    } catch (err: any) {
      setError(err?.message ?? String(err));
    }
  }

  function removeAttachment(path: string) {
    setAttachedPaths((prev) => prev.filter((p) => p !== path));
  }

  async function sendMessage() {
    const trimmed = input.trim();
    if (!trimmed || isStreaming) return;
    if (!chatId) {
      setError("Select or create a chat first.");
      return;
    }

    setError(null);
    cleanupListeners();

    // If any stream is still active/finalizing, we must wait before starting a new one.
    // Otherwise the old Rust stream reader can still be holding stdout and swallow
    // the new stream's tokens (appears as "no response").
    const active = getActiveStream();
    if (active) {
      try {
        setIsStreaming(true);
        if (active.chatId === chatId) {
          // Same chat: likely user pressed Stop. Don't spam cancel; just wait for stream_end.
          await waitForStreamToFinish(active.requestId);
        } else {
          // Different chat: cancel and wait for clean stream_end before starting.
          await cancelActiveStreamAndWait();
        }
      } catch {
        // ignore; we'll still attempt to start this chat
      } finally {
        setIsStreaming(false);
      }
    }

    const attached = attachedPaths.slice();
    const attachedNames = attached.map(filenameFromPath);

    const userMsg: ChatMessage = {
      id: `${Date.now()}-user`,
      role: "user",
      content: trimmed,
      attachments: attachedNames.length ? attachedNames : undefined,
    };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setIsStreaming(true);
    setAttachedPaths([]);

    const assistantId = `${Date.now()}-assistant`;
    assistantIdRef.current = assistantId;
    setMessages((prev) => [
      ...prev,
      { id: assistantId, role: "assistant", content: "" },
    ]);
    pendingTextRef.current = "";

    try {
      const requestId =
        (globalThis.crypto && "randomUUID" in globalThis.crypto
          ? (globalThis.crypto as any).randomUUID()
          : `${Date.now()}-${Math.random()}`) as string;
      activeRequestIdRef.current = requestId;
      const paths = attached;

      unlistenTokenRef.current = await listen<{ token: string; chat_id?: string; request_id?: string }>(
        "llm-token",
        (event) => {
          const payload = event.payload || {};
          if (payload.chat_id && payload.chat_id !== chatId) return;
          if (payload.request_id && payload.request_id !== requestId) return;
          const token = payload.token || "";
          if (!token) return;
          pendingTextRef.current += token;
          if (flushRafRef.current != null) return;
          flushRafRef.current = requestAnimationFrame(() => {
            flushRafRef.current = null;
            const pending = pendingTextRef.current;
            if (!pending) return;
            pendingTextRef.current = "";
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? { ...m, content: (m.content || "") + pending }
                  : m
              )
            );
          });
        }
      );

      unlistenDoneRef.current = await listen<{ request_id?: string; chat_id?: string }>(
        "llm-done",
        (event) => {
          const payload = event.payload || {};
          if (payload.chat_id && payload.chat_id !== chatId) return;
          if (payload.request_id && payload.request_id !== requestId) return;
          flushPendingToMessage();
          cleanupListeners();
          activeRequestIdRef.current = null;
          setIsStreaming(false);
          if (chatId) {
            refreshContextStatus(chatId).catch(() => {});
          }
        }
      );

      unlistenErrorRef.current = await listen<{ request_id?: string; chat_id?: string; error?: string }>(
        "llm-error",
        (event) => {
          const payload = event.payload || {};
          if (payload.chat_id && payload.chat_id !== chatId) return;
          if (payload.request_id && payload.request_id !== requestId) return;
          flushPendingToMessage();
          cleanupListeners();
          activeRequestIdRef.current = null;
          if (payload.error && payload.error !== "cancelled") {
            setError(payload.error);
          }
          setIsStreaming(false);
          if (chatId) {
            refreshContextStatus(chatId).catch(() => {});
          }
        }
      );

      // Starts stream in the background and returns immediately.
      await engineStreamChat({ chatId, query: trimmed, requestId, paths });
    } catch (err: any) {
      console.error("Chat error", err);
      setError(err?.message ?? String(err));
      flushPendingToMessage();
      cleanupListeners();
      activeRequestIdRef.current = null;
      setIsStreaming(false);
    }
  }

  async function stopStreaming() {
    const rid = activeRequestIdRef.current;
    if (!rid) {
      // Fallback: if another chat started a stream, still allow stop to cancel it.
      const active = getActiveStream();
      if (active) {
        engineCancel(active.requestId).catch(() => {});
      }
      return;
    }

    // Immediate UI stop: ignore further tokens and switch back to "Send".
    flushPendingToMessage();
    cleanupListeners();
    activeRequestIdRef.current = null;
    setIsStreaming(false);

    // Best-effort backend cancel (stops ASGI + llama.cpp compute).
    engineCancel(rid).catch(() => {});
  }

  function handleKeyDown(
    e: React.KeyboardEvent<HTMLInputElement | HTMLTextAreaElement>
  ) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  return (
    <div className="chat-root">
      <div className="chat-topbar" aria-hidden="true" />

      <div className="chat-messages">
        {messages.map((m) => (
          <div key={m.id} className={`chat-message chat-message-${m.role}`}>
            <div className="chat-message-role">
              {m.role === "user" ? "You" : "Insight"}
            </div>
            <div className="chat-message-content">{m.content}</div>
            {!!m.attachments?.length && (
              <div className="chat-message-attachments">
                {m.attachments.map((name) => (
                  <span key={name} className="chat-attachment-chip" title={name}>
                    {name}
                  </span>
                ))}
              </div>
            )}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      {error && <div className="chat-error">{error}</div>}

      {!!attachedPaths.length && (
        <div className="chat-attachments">
          {attachedPaths.map((p) => {
            const name = filenameFromPath(p);
            return (
              <div key={p} className="chat-attachment-pill" title={p}>
                <span className="chat-attachment-name">{name}</span>
                <button
                  className="chat-attachment-remove"
                  onClick={() => removeAttachment(p)}
                  disabled={isStreaming}
                  aria-label={`Remove attachment ${name}`}
                  title="Remove"
                >
                  ×
                </button>
              </div>
            );
          })}
        </div>
      )}

      <div className="chat-input-row">
        <div className="chat-input-shell">
          <button
            className="chat-input-attach"
            onClick={pickAttachments}
            disabled={!chatId || isStreaming}
            aria-label="Attach files"
            title={attachedPaths.length ? `Attached: ${attachedPaths.length}` : "Attach files"}
            type="button"
          >
            +
          </button>
          <textarea
            className="chat-input"
            placeholder="Ask Insight anything about your data…"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={1}
            ref={inputElRef}
          />
        </div>
        <div className="chat-context-wrap">
          {contextStatus ? (
            (() => {
              const used = Number(contextStatus.used_tokens || 0);
              const cap = Number(contextStatus.capacity_tokens || 0);
              const pctUsed = Math.max(0, Math.min(100, Number(contextStatus.percent || 0)));
              const left = Math.max(0, cap - used);
              const tooltip = `Used ${pctUsed}%\nRemaining ${formatInt(left)} / ${formatInt(cap)} tokens`;
              const r = 16;
              const cx = 18;
              const cy = 18;
              const circumference = 2 * Math.PI * r;
              const dashOffset = circumference * (1 - pctUsed / 100);
              return (
                <>
                  <div className="chat-context-hover" aria-label={tooltip}>
                    <svg
                      className="chat-context-ring"
                      viewBox="0 0 36 36"
                      role="img"
                      aria-hidden="true"
                    >
                      <circle
                        className="chat-context-ring-track"
                        cx={cx}
                        cy={cy}
                        r={r}
                      />
                      <circle
                        className="chat-context-ring-progress"
                        cx={cx}
                        cy={cy}
                        r={r}
                        strokeDasharray={circumference}
                        strokeDashoffset={dashOffset}
                        transform={`rotate(-90 ${cx} ${cy})`}
                      />
                      <circle className="chat-context-ring-center" cx={cx} cy={cy} r={11} />
                    </svg>
                    <div className="chat-context-tooltip" role="tooltip">
                      <div className="chat-context-tooltip-title">Context</div>
                      <div className="chat-context-tooltip-body">
                        Used {pctUsed}% · Remaining {formatInt(left)} / {formatInt(cap)} tokens
                      </div>
                    </div>
                  </div>
                </>
              );
            })()
          ) : (
            <div className="chat-context-hover" aria-label="Context status unavailable">
              <svg
                className="chat-context-ring disabled"
                viewBox="0 0 36 36"
                role="img"
                aria-hidden="true"
              >
                <circle className="chat-context-ring-track" cx={18} cy={18} r={16} />
                <circle className="chat-context-ring-progress" cx={18} cy={18} r={16} />
                <circle className="chat-context-ring-center" cx={18} cy={18} r={11} />
              </svg>
              <div className="chat-context-tooltip" role="tooltip">
                <div className="chat-context-tooltip-title">Context</div>
                <div className="chat-context-tooltip-body">Unavailable</div>
              </div>
            </div>
          )}
        </div>
        <button
          className="chat-send-btn"
          onClick={isStreaming ? stopStreaming : sendMessage}
          disabled={
            (isStreaming && !activeRequestIdRef.current) ||
            (!isStreaming && !input.trim())
          }
        >
          {isStreaming ? "Stop" : "Send"}
        </button>
      </div>
    </div>
  );
}
