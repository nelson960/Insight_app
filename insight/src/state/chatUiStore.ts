import { listen } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  request_id?: string;
  attachments?: string[];
  selection?: { text: string; file_id?: string; page?: number };
  focus_document_id?: string;
};

type PersistedMessage = {
  role: string;
  content: string;
  attachments?: string[];
  selection?: { text: string; file_id?: string; page?: number };
  focus_document_id?: string;
};

export type ChatUiSnapshot = {
  chatId: string;
  loaded: boolean;
  loading: boolean;
  messages: ChatMessage[];
  isStreaming: boolean;
  activeRequestId: string | null;
};

type ChatUiState = ChatUiSnapshot & {
  loadPromise?: Promise<void>;
};

type StreamIndexEntry = {
  chatId: string;
  userMessageId?: string;
  messageId: string;
  cancelled?: boolean;
};

const states = new Map<string, ChatUiState>();
const listenersByChat = new Map<string, Set<() => void>>();
const streamIndex = new Map<string, StreamIndexEntry>(); // request_id -> message info

let bridgeInit: Promise<void> | null = null;

function ensureState(chatId: string): ChatUiState {
  const existing = states.get(chatId);
  if (existing) return existing;
  const next: ChatUiState = {
    chatId,
    loaded: false,
    loading: false,
    messages: [],
    isStreaming: false,
    activeRequestId: null,
  };
  states.set(chatId, next);
  return next;
}

function emit(chatId: string) {
  const subs = listenersByChat.get(chatId);
  if (!subs) return;
  for (const fn of subs) {
    try {
      fn();
    } catch {
      // ignore
    }
  }
}

export function subscribeChatUi(chatId: string, fn: () => void) {
  if (!chatId) return () => {};
  const set = listenersByChat.get(chatId) ?? new Set<() => void>();
  set.add(fn);
  listenersByChat.set(chatId, set);
  return () => {
    const s = listenersByChat.get(chatId);
    if (!s) return;
    s.delete(fn);
    if (s.size === 0) listenersByChat.delete(chatId);
  };
}

export function getChatUiSnapshot(chatId: string | null): ChatUiSnapshot {
  if (!chatId) {
    return {
      chatId: "",
      loaded: true,
      loading: false,
      messages: [],
      isStreaming: false,
      activeRequestId: null,
    };
  }
  const s = ensureState(chatId);
  return {
    chatId: s.chatId,
    loaded: s.loaded,
    loading: s.loading,
    messages: s.messages,
    isStreaming: s.isStreaming,
    activeRequestId: s.activeRequestId,
  };
}

export async function ensureChatUiLoaded(chatId: string) {
  if (!chatId) return;
  await ensureBridge();
  const s = ensureState(chatId);
  if (s.loaded) return;
  if (s.loadPromise) return s.loadPromise;

  s.loading = true;
  emit(chatId);
  s.loadPromise = (async () => {
    try {
      const res = await invoke<{ messages?: PersistedMessage[] }>("get_session_messages", {
        chatId,
      });
      const rows = Array.isArray(res?.messages) ? res.messages : [];
      // Only set history if there is no newer local stream-in-progress message list.
      // If there is, keep it (it may contain a partial assistant response that the
      // backend hasn't persisted yet).
      if (s.messages.length === 0) {
        s.messages = rows.map((m, idx) => ({
          id: `${chatId}-${idx}-${m.role}`,
          role: m.role === "assistant" ? "assistant" : "user",
          content: m.content ?? "",
          attachments: Array.isArray(m.attachments) ? m.attachments : undefined,
          selection:
            m.selection && typeof m.selection === "object" ? (m.selection as any) : undefined,
          focus_document_id:
            typeof m.focus_document_id === "string" ? m.focus_document_id : undefined,
        }));
      }
    } catch {
      // ignore
    } finally {
      s.loaded = true;
      s.loading = false;
      s.loadPromise = undefined;
      emit(chatId);
    }
  })();

  return s.loadPromise;
}

export function beginStreamTurn(opts: {
  chatId: string;
  requestId: string;
  userText: string;
  attachments?: string[];
  selection?: { text: string; file_id?: string; page?: number };
}) {
  const { chatId, requestId, userText, attachments, selection } = opts;
  if (!chatId || !requestId) return;
  const s = ensureState(chatId);

  const now = Date.now();
  const userMsg: ChatMessage = {
    id: `${now}-user`,
    role: "user",
    content: userText,
    attachments: attachments?.length ? attachments : undefined,
    selection: selection?.text ? selection : undefined,
  };
  const assistantMsgId = `${now}-assistant`;
  const assistantMsg: ChatMessage = {
    id: assistantMsgId,
    role: "assistant",
    content: "",
    request_id: requestId,
  };
  s.messages = [...s.messages, userMsg, assistantMsg];
  s.isStreaming = true;
  s.activeRequestId = requestId;
  streamIndex.set(requestId, { chatId, userMessageId: userMsg.id, messageId: assistantMsgId });
  emit(chatId);
}

export function cancelStreamTurn(chatId: string) {
  if (!chatId) return;
  const s = ensureState(chatId);
  const rid = s.activeRequestId;
  if (!rid) return;
  const entry = streamIndex.get(rid);
  if (entry) entry.cancelled = true;
  s.isStreaming = false;
  s.activeRequestId = null;
  emit(chatId);
}

export function rollbackStreamTurn(chatId: string, requestId: string) {
  if (!chatId || !requestId) return;
  const s = ensureState(chatId);
  const entry = streamIndex.get(requestId);
  if (!entry || entry.chatId !== chatId) return;
  const ids = new Set<string>();
  if (entry.messageId) ids.add(entry.messageId);
  if (entry.userMessageId) ids.add(entry.userMessageId);
  if (ids.size) {
    s.messages = s.messages.filter((m) => !ids.has(m.id));
  }
  if (s.activeRequestId === requestId) s.activeRequestId = null;
  s.isStreaming = false;
  streamIndex.delete(requestId);
  emit(chatId);
}

function appendTokenInternal(chatId: string, requestId: string, token: string) {
  const s = ensureState(chatId);
  const entry = streamIndex.get(requestId);
  if (!entry || entry.chatId !== chatId) return;
  if (entry.cancelled) return;
  const idx = s.messages.findIndex((m) => m.request_id === requestId && m.role === "assistant");
  if (idx < 0) return;
  const prev = s.messages[idx];
  const next = { ...prev, content: (prev.content || "") + token };
  s.messages = [...s.messages.slice(0, idx), next, ...s.messages.slice(idx + 1)];
  emit(chatId);
}

function endStreamInternal(chatId: string, requestId: string) {
  const s = ensureState(chatId);
  if (s.activeRequestId === requestId) {
    s.activeRequestId = null;
  }
  s.isStreaming = false;
  streamIndex.delete(requestId);
  emit(chatId);
}

async function ensureBridge() {
  if (bridgeInit) return bridgeInit;
  bridgeInit = (async () => {
    await listen<{ token?: string; chat_id?: string; request_id?: string }>(
      "llm-token",
      (event) => {
        const p = event.payload || {};
        const token = p.token || "";
        const chatId = p.chat_id || "";
        const requestId = p.request_id || "";
        if (!chatId || !requestId || !token) return;
        appendTokenInternal(chatId, requestId, token);
      }
    );

    await listen<{ request_id?: string; chat_id?: string }>("llm-done", (event) => {
      const p = event.payload || {};
      const chatId = p.chat_id || "";
      const requestId = p.request_id || "";
      if (!chatId || !requestId) return;
      endStreamInternal(chatId, requestId);
    });

    await listen<{ request_id?: string; chat_id?: string; error?: string }>(
      "llm-error",
      (event) => {
        const p = event.payload || {};
        const chatId = p.chat_id || "";
        const requestId = p.request_id || "";
        if (!chatId || !requestId) return;
        // Don't end the stream here; Rust emits llm-error as soon as it sees stream_error,
        // but the stdout router will continue until stream_end and then emit llm-done.
        // We *do* stop the UI spinner quickly, like ChatWindow used to do, so switching panes
        // doesn't keep the UI in a "stuck streaming" state.
        const s = ensureState(chatId);
        if (s.activeRequestId === requestId) {
          s.isStreaming = false;
          emit(chatId);
        }
      }
    );

    await listen<any>("llm_stream_end", (event) => {
      const p = (event.payload as any) || {};
      const chatId = p.chat_id || "";
      const requestId = p.request_id || "";
      if (!chatId || !requestId) return;
      // Fast UI end signal (tokens are done, KV may still be saving).
      const s = ensureState(chatId);
      if (s.activeRequestId === requestId) {
        s.isStreaming = false;
        emit(chatId);
      }
    });
  })();
  return bridgeInit;
}
