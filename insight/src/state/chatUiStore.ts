import { listen } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";

export type Source = {
  filename: string;
  page_ranges?: Array<[number, number]>;
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  request_id?: string;
  attachments?: string[];
  selection?: { text: string; file_id?: string; page?: number };
  focus_document_id?: string;
  sources?: Source[];
  created_at?: string;
  // Version support for regeneration and editing
  versions?: string[]; // Array of alternative content strings (index 0 is original)
  activeVersionIndex?: number; // Currently displayed version (default 0)
  originalContent?: string; // Original content before any edits
};

type PersistedMessage = {
  id: string;
  role: string;
  content: string;
  attachments?: string[];
  selection?: { text: string; file_id?: string; page?: number };
  focus_document_id?: string;
  sources?: Source[];
  versions?: string[];
  activeVersionIndex?: number;
  created_at?: string;
};

export type ChatDraft = {
  input: string;
  attachments: string[];
};

export type ChatUiSnapshot = {
  chatId: string;
  loaded: boolean;
  loading: boolean;
  messages: ChatMessage[];
  isStreaming: boolean;
  activeRequestId: string | null;
  isCompacting: boolean;
  lastError?: string | null;
};

type ChatUiState = ChatUiSnapshot & {
  loadPromise?: Promise<void>;
  draft?: ChatDraft;
};

type StreamIndexEntry = {
  chatId: string;
  userMessageId?: string;
  messageId: string;
  cancelled?: boolean;
  isRegeneration?: boolean;
  regenSnapshot?: ChatMessage;
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
    isCompacting: false,
    lastError: null,
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
    } catch (err) {
      // Log but don't throw - one bad subscriber shouldn't break others
      console.error(`[chatUiStore/emit] Error in subscriber for chat ${chatId}:`, err);
    }
  }
}

export function subscribeChatUi(chatId: string, fn: () => void) {
  if (!chatId) return () => { };
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
      isCompacting: false,
      lastError: null,
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
    isCompacting: s.isCompacting,
    lastError: s.lastError ?? null,
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
          id: m.id || `${chatId}-${idx}-${m.role}`,
          role: m.role === "assistant" ? "assistant" : "user",
          content: m.content ?? "",
          attachments: Array.isArray(m.attachments) ? m.attachments : undefined,
          selection:
            m.selection && typeof m.selection === "object" ? (m.selection as any) : undefined,
          focus_document_id:
            typeof m.focus_document_id === "string" ? m.focus_document_id : undefined,
          sources: Array.isArray(m.sources) ? m.sources : undefined,
          created_at: typeof (m as any).created_at === "string" ? (m as any).created_at : undefined,
          versions: Array.isArray(m.versions) ? m.versions : undefined,
          activeVersionIndex: typeof m.activeVersionIndex === "number" ? m.activeVersionIndex : undefined,
        }));
      }
    } catch (err) {
      // Log error but continue - corrupted messages shouldn't crash the app
      console.error(`[chatUiStore/ensureChatUiLoaded] Failed to load messages for chat ${chatId}:`, err);
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
  messageId?: string;
  userMessageId?: string;
}) {
  const { chatId, requestId, userText, attachments, selection, messageId, userMessageId } = opts;
  if (!chatId || !requestId) return;
  const s = ensureState(chatId);

  const now = Date.now();
  const resolvedUserId = userMessageId || `${now}-user`;
  const userMsg: ChatMessage = {
    id: resolvedUserId,
    role: "user",
    content: userText,
    attachments: attachments?.length ? attachments : undefined,
    selection: selection?.text ? selection : undefined,
  };
  const assistantMsgId = messageId || `${now}-assistant`;
  const assistantMsg: ChatMessage = {
    id: assistantMsgId,
    role: "assistant",
    content: "",
    request_id: requestId,
  };
  s.messages = [...s.messages, userMsg, assistantMsg];
  s.isStreaming = true;
  s.lastError = null;
  s.activeRequestId = requestId;
  // No changes to beginStreamTurn, it behaves as before
  streamIndex.set(requestId, { chatId, userMessageId: userMsg.id, messageId: assistantMsgId });
  emit(chatId);
}

/**
 * Begin an assistant generation turn WITHOUT adding a user message.
 * Used for editing a user message where the user message is already updated in place.
 */
export function beginAssistantGeneration(opts: {
  chatId: string;
  requestId: string;
  messageId?: string;
}) {
  const { chatId, requestId, messageId } = opts;
  if (!chatId || !requestId) return;
  const s = ensureState(chatId);

  const now = Date.now();
  const assistantMsgId = messageId || `${now}-assistant`;
  const assistantMsg: ChatMessage = {
    id: assistantMsgId,
    role: "assistant",
    content: "",
    request_id: requestId,
  };

  // Append only the assistant message
  s.messages = [...s.messages, assistantMsg];
  s.isStreaming = true;
  s.lastError = null;
  s.activeRequestId = requestId;
  streamIndex.set(requestId, { chatId, messageId: assistantMsgId });
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
  const newContent = (prev.content || "") + token;

  // Update content and also the active version if it exists
  let versions = prev.versions;
  if (versions && prev.activeVersionIndex !== undefined && versions[prev.activeVersionIndex] !== undefined) {
    versions = [...versions];
    versions[prev.activeVersionIndex] = newContent;
  }

  const next = { ...prev, content: newContent, versions };
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

function attachSourcesInternal(chatId: string, requestId: string, sources: any[]) {
  if (!chatId || !requestId || !Array.isArray(sources) || sources.length === 0) return;
  const s = ensureState(chatId);
  const idx = s.messages.findIndex((m) => m.role === "assistant" && m.request_id === requestId);
  if (idx < 0) return;
  const prev = s.messages[idx];
  const next = { ...prev, sources };
  s.messages = [...s.messages.slice(0, idx), next, ...s.messages.slice(idx + 1)];
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

    await listen<any>("chat-sources", (event) => {
      const p = (event.payload as any) || {};
      const chatId = p.chat_id || "";
      const requestId = p.request_id || "";
      const sources = Array.isArray(p.sources) ? (p.sources as any[]) : [];
      if (!chatId || !requestId || sources.length === 0) return;
      attachSourcesInternal(chatId, requestId, sources);
    });

    await listen<any>("engine-event", (event) => {
      const payload = (event.payload as any) || {};
      const name = payload?.name || "";
      if (name === "chat_message_ids") {
        const chatId = payload?.chat_id || "";
        const requestId = payload?.request_id || "";
        if (!chatId || !requestId) return;
        const s = ensureState(chatId);
        const entry = streamIndex.get(requestId);
        if (!entry || entry.chatId !== chatId) return;

        const reconcileMessageId = (oldId?: string, newId?: string) => {
          if (!oldId || !newId || oldId === newId) return;
          const oldIdx = s.messages.findIndex((m) => m.id === oldId);
          if (oldIdx < 0) return;
          const newIdx = s.messages.findIndex((m) => m.id === newId);
          if (newIdx >= 0 && newIdx !== oldIdx) {
            const oldMsg = s.messages[oldIdx];
            const newMsg = s.messages[newIdx];
            const merged: ChatMessage = {
              ...newMsg,
              id: newId,
              content: newMsg.content || oldMsg.content,
              sources: newMsg.sources ?? oldMsg.sources,
              versions: newMsg.versions ?? oldMsg.versions,
              activeVersionIndex:
                newMsg.activeVersionIndex ?? oldMsg.activeVersionIndex,
              request_id: newMsg.request_id ?? oldMsg.request_id,
              selection: newMsg.selection ?? oldMsg.selection,
              attachments: newMsg.attachments ?? oldMsg.attachments,
            };
            let next = s.messages.slice();
            if (oldIdx < newIdx) {
              next.splice(oldIdx, 1);
              const adjustedIdx = newIdx - 1;
              next[adjustedIdx] = merged;
            } else {
              next[newIdx] = merged;
              next.splice(oldIdx, 1);
            }
            s.messages = next;
            return;
          }
          const updated = { ...s.messages[oldIdx], id: newId };
          s.messages = [
            ...s.messages.slice(0, oldIdx),
            updated,
            ...s.messages.slice(oldIdx + 1),
          ];
        };

        const persistedUserId =
          typeof payload?.user_message_id === "string"
            ? payload.user_message_id
            : undefined;
        const persistedAssistantId =
          typeof payload?.assistant_message_id === "string"
            ? payload.assistant_message_id
            : undefined;

        if (persistedUserId && entry.userMessageId) {
          reconcileMessageId(entry.userMessageId, persistedUserId);
          entry.userMessageId = persistedUserId;
        }
        if (persistedAssistantId && entry.messageId) {
          reconcileMessageId(entry.messageId, persistedAssistantId);
          entry.messageId = persistedAssistantId;
          if (entry.regenSnapshot) {
            entry.regenSnapshot = { ...entry.regenSnapshot, id: persistedAssistantId };
          }
        }

        emit(chatId);
        return;
      }

      if (name !== "compaction_start" && name !== "compaction_end") return;
      const chatId = payload?.chat_id || "";
      if (!chatId) return;
      const s = ensureState(chatId);
      s.isCompacting = name === "compaction_start";
      emit(chatId);
    });

    await listen<{ request_id?: string; chat_id?: string; error?: string }>(
      "llm-error",
      (event) => {
        const p = event.payload || {};
        const chatId = p.chat_id || "";
        const requestId = p.request_id || "";
        if (!chatId || !requestId) return;
        rollbackRegenerationTurn(chatId, requestId);
        // Don't end the stream here; Rust emits llm-error as soon as it sees stream_error,
        // but the stdout router will continue until stream_end and then emit llm-done.
        // We *do* stop the UI spinner quickly, like ChatWindow used to do, so switching panes
        // doesn't keep the UI in a "stuck streaming" state.
        const s = ensureState(chatId);
        const errText = p.error ? String(p.error) : "Model error";
        s.lastError = errText;
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

// Draft management functions
export function setChatDraft(chatId: string, draft: ChatDraft | null) {
  if (!chatId) return;
  const s = ensureState(chatId);
  s.draft = draft ?? undefined;
  emit(chatId);
}

export function getChatDraft(chatId: string): ChatDraft | null {
  if (!chatId) return null;
  const state = states.get(chatId);
  return state?.draft ?? null;
}

export function clearChatDraft(chatId: string) {
  if (!chatId) return;
  const s = states.get(chatId);
  if (s) {
    s.draft = undefined;
    emit(chatId);
  }
}

/**
 * Prepare for regenerating an assistant message.
 * Stores the current content as a version and sets up for streaming.
 * Returns the user message that should be resent, or null if not found.
 */
export function prepareRegeneration(
  chatId: string,
  assistantMessageId: string,
  requestId: string
): { userMessage: ChatMessage; assistantMessageId: string } | null {
  if (!chatId || !assistantMessageId || !requestId) return null;
  const s = ensureState(chatId);

  // Find the assistant message
  const assistantIdx = s.messages.findIndex((m) => m.id === assistantMessageId);
  if (assistantIdx < 0) return null;

  const assistantMsg = s.messages[assistantIdx];
  if (assistantMsg.role !== "assistant") return null;

  // Find the preceding user message
  let userIdx = assistantIdx - 1;
  while (userIdx >= 0 && s.messages[userIdx].role !== "user") {
    userIdx--;
  }
  if (userIdx < 0) return null;

  const userMsg = s.messages[userIdx];

  // Snapshot current assistant message for rollback on regen failure/cancel
  const regenSnapshot: ChatMessage = {
    ...assistantMsg,
    versions: assistantMsg.versions ? [...assistantMsg.versions] : undefined,
    sources: assistantMsg.sources ? [...assistantMsg.sources] : undefined,
  };

  // Store current content as a version (if not already versioned)
  const currentContent = assistantMsg.content;
  const versions = assistantMsg.versions ? [...assistantMsg.versions] : [];
  if (versions.length === 0 && currentContent) {
    // First regeneration: store original as version 0
    versions.push(currentContent);
  } else if (versions.length > 0) {
    // Update current version with any edits
    const activeIdx = assistantMsg.activeVersionIndex ?? 0;
    if (versions[activeIdx] !== currentContent && currentContent) {
      versions[activeIdx] = currentContent;
    }
  }

  // Prepare for new version: push placeholder
  versions.push("");

  // Update the assistant message: clear content, set new request_id for streaming
  const updatedAssistant: ChatMessage = {
    ...assistantMsg,
    content: "",
    versions,
    activeVersionIndex: versions.length - 1, // Point to the new empty version
    request_id: requestId, // Set the new request_id for token routing
    sources: undefined, // Clear prior sources; new citations (if any) will be attached per request_id
  };

  s.messages = [
    ...s.messages.slice(0, assistantIdx),
    updatedAssistant,
    ...s.messages.slice(assistantIdx + 1),
  ];

  // Register with streamIndex so tokens route to this message
  streamIndex.set(requestId, {
    chatId,
    messageId: assistantMessageId,
    // No userMessageId since we're not adding a new user message
    isRegeneration: true,
    regenSnapshot,
  });

  // Set streaming state
  s.isStreaming = true;
  s.lastError = null;
  s.activeRequestId = requestId;

  emit(chatId);

  return { userMessage: userMsg, assistantMessageId };
}

/**
 * Roll back a regeneration attempt, restoring the pre-regen assistant message.
 * Safe to call for non-regeneration streams (no-op).
 */
export function rollbackRegenerationTurn(chatId: string, requestId: string) {
  if (!chatId || !requestId) return;
  const s = ensureState(chatId);
  const entry = streamIndex.get(requestId);
  if (!entry || entry.chatId !== chatId || !entry.isRegeneration || !entry.regenSnapshot) return;

  const idx = s.messages.findIndex((m) => m.id === entry.messageId);
  if (idx >= 0) {
    const snap = entry.regenSnapshot;
    const restored: ChatMessage = {
      ...snap,
      versions: snap.versions ? [...snap.versions] : undefined,
      sources: snap.sources ? [...snap.sources] : undefined,
    };
    s.messages = [...s.messages.slice(0, idx), restored, ...s.messages.slice(idx + 1)];
  }
  if (s.activeRequestId === requestId) s.activeRequestId = null;
  s.isStreaming = false;
  streamIndex.delete(requestId);
  emit(chatId);
}

/**
 * Finalize a regeneration by adding the new content as a version.
 * Called when streaming completes for a regeneration.
 */
export function finalizeRegeneration(
  chatId: string,
  assistantMessageId: string,
  newContent: string
) {
  if (!chatId || !assistantMessageId) return;
  const s = ensureState(chatId);

  const idx = s.messages.findIndex((m) => m.id === assistantMessageId);
  if (idx < 0) return;

  const msg = s.messages[idx];
  const versions = msg.versions ? [...msg.versions] : [msg.originalContent || msg.content];
  versions.push(newContent);

  const updated: ChatMessage = {
    ...msg,
    content: newContent,
    versions,
    activeVersionIndex: versions.length - 1,
  };

  s.messages = [...s.messages.slice(0, idx), updated, ...s.messages.slice(idx + 1)];
  emit(chatId);
}

/**
 * Edit a user message and truncate all subsequent messages.
 * Returns the edited message info for resending, or null if not found.
 */
export function updateEditedUserMessage(
  chatId: string,
  messageId: string,
  newContent: string
): { editedMessage: ChatMessage; truncatedCount: number } | null {
  if (!chatId || !messageId) return null;
  const s = ensureState(chatId);

  const idx = s.messages.findIndex((m) => m.id === messageId);
  if (idx < 0) return null;

  const msg = s.messages[idx];
  if (msg.role !== "user") return null;

  // Store original content if not already stored
  const originalContent = msg.originalContent ?? msg.content;

  const editedMsg: ChatMessage = {
    ...msg,
    content: newContent,
    originalContent,
  };

  // Truncate all messages after this one
  const truncatedCount = s.messages.length - idx - 1;
  s.messages = [...s.messages.slice(0, idx), editedMsg];
  emit(chatId);

  return { editedMessage: editedMsg, truncatedCount };
}

/**
 * Switch to a different version of a message (for both user and assistant messages).
 * direction: -1 for previous, +1 for next
 */
export function switchMessageVersion(
  chatId: string,
  messageId: string,
  direction: -1 | 1
): boolean {
  if (!chatId || !messageId) return false;
  const s = ensureState(chatId);

  const idx = s.messages.findIndex((m) => m.id === messageId);
  if (idx < 0) return false;

  const msg = s.messages[idx];
  const versions = msg.versions;
  if (!versions || versions.length <= 1) return false;

  const currentIdx = msg.activeVersionIndex ?? 0;
  let newIdx = currentIdx + direction;

  // Wrap around
  if (newIdx < 0) newIdx = versions.length - 1;
  if (newIdx >= versions.length) newIdx = 0;

  if (newIdx === currentIdx) return false;

  const updated: ChatMessage = {
    ...msg,
    content: versions[newIdx],
    activeVersionIndex: newIdx,
  };

  s.messages = [...s.messages.slice(0, idx), updated, ...s.messages.slice(idx + 1)];
  emit(chatId);

  return true;
}

/**
 * Get version info for a message (for UI display).
 */
export function getMessageVersionInfo(
  chatId: string,
  messageId: string
): { current: number; total: number } | null {
  if (!chatId || !messageId) return null;
  const s = states.get(chatId);
  if (!s) return null;

  const msg = s.messages.find((m) => m.id === messageId);
  if (!msg || !msg.versions || msg.versions.length <= 1) return null;

  return {
    current: (msg.activeVersionIndex ?? 0) + 1, // 1-indexed for display
    total: msg.versions.length,
  };
}
