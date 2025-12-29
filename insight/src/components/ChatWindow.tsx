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
import { ChatMarkdown, ChatMarkdownStream } from "./ChatMarkdown";

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  request_id?: string;
  attachments?: string[];
  selection?: { text: string; file_id?: string; page?: number };
  focus_document_id?: string;
};

type Props = {
  chatId: string | null;
  active?: boolean;
  embedded?: boolean;
  showTopbar?: boolean;
  activeDocumentId?: string | null;
  docsVisible?: boolean;
  selection?: { text: string; file_id?: string } | null;
  onSetSelection?: (sel: { text: string; file_id?: string } | null) => void;
  onClearSelection?: () => void;
  onRequestDocsRefresh?: () => void;
};

type ContextStatus = {
  chat_id: string;
  used_tokens: number;
  capacity_tokens: number;
  percent: number;
  compacted?: boolean;
  last_gen_tps?: number;
  last_ttft_ms?: number;
  last_gen_tokens?: number;
};

type MarkdownStreamState = {
  blocks: string[];
  tail: string;
  inFence: boolean;
  fenceToken?: "```" | "~~~" | null;
};

const MAX_MD_TAIL_CHARS = 1800;

function _scanMarkdownTail(md: string): {
  commitIdx: number;
  inFence: boolean;
  fenceToken: "```" | "~~~" | null;
  fenceStartIdx: number;
} {
  let inFence = false;
  let fenceToken: "```" | "~~~" | null = null;
  let fenceStartIdx = -1;
  let lastParaIdx = -1;
  let lastLineIdx = -1;

  // Scan the tail for:
  // - fenced code blocks (``` / ~~~) to avoid splitting inside them
  // - safe commit boundaries (blank lines), and a fallback (single newline)
  for (let i = 0; i < md.length; i++) {
    const atLineStart = i === 0 || md[i - 1] === "\n";
    if (atLineStart && (md.startsWith("```", i) || md.startsWith("~~~", i))) {
      const tok = md.startsWith("```", i) ? "```" : "~~~";
      // Toggle fence. When opening, remember where it started so we can safely
      // commit everything *before* the fence, even if there wasn't a blank line.
      if (!inFence) {
        inFence = true;
        fenceToken = tok;
        fenceStartIdx = i;
      } else {
        inFence = false;
        fenceToken = null;
        fenceStartIdx = -1;
      }
    }

    if (inFence) continue;
    if (md[i] !== "\n") continue;

    lastLineIdx = i + 1;
    if (i + 1 < md.length && md[i + 1] === "\n") {
      lastParaIdx = i + 2;
    }
  }

  let commitIdx = lastParaIdx;
  // If we're currently inside a fence, commit everything *before* the fence so only
  // the code block tail re-renders during streaming (prevents "markdown flips").
  if (inFence && fenceStartIdx > 0) {
    commitIdx = fenceStartIdx;
  }
  if (commitIdx < 0 && !inFence && md.length > MAX_MD_TAIL_CHARS && lastLineIdx > 0) {
    commitIdx = lastLineIdx;
  }
  return {
    commitIdx: commitIdx > 0 ? commitIdx : 0,
    inFence,
    fenceToken,
    fenceStartIdx,
  };
}

function _ingestMarkdownChunk(state: MarkdownStreamState, chunk: string) {
  if (!chunk) return;
  state.tail += chunk;

  const scan = _scanMarkdownTail(state.tail);
  state.inFence = scan.inFence;
  state.fenceToken = scan.fenceToken;
  if (!scan.commitIdx) return;

  const committed = state.tail.slice(0, scan.commitIdx);
  state.blocks.push(committed);
  state.tail = state.tail.slice(scan.commitIdx);
}

export function ChatWindow({
  chatId,
  active = true,
  embedded = false,
  showTopbar = true,
  activeDocumentId = null,
  docsVisible = true,
  selection,
  onSetSelection,
  onClearSelection,
  onRequestDocsRefresh,
}: Props) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [attachedPaths, setAttachedPaths] = useState<string[]>([]);
  const [contextStatus, setContextStatus] = useState<ContextStatus | null>(null);
  const [localSelection, setLocalSelection] = useState<{ text: string; file_id?: string } | null>(
    null
  );
  const [chatSelectionText, setChatSelectionText] = useState<string>("");
  const [chatSelectionPos, setChatSelectionPos] = useState<{ x: number; y: number } | null>(null);
  const [pendingChatSelection, setPendingChatSelection] = useState<{ text: string } | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const messagesRef = useRef<HTMLDivElement | null>(null);
  const pendingTextRef = useRef("");
  const flushRafRef = useRef<number | null>(null);
  const activeRequestIdRef = useRef<string | null>(null);
  const assistantIdRef = useRef<string | null>(null);
  const mdStreamRef = useRef<Map<string, MarkdownStreamState>>(new Map());
  const unlistenTokenRef = useRef<UnlistenFn | null>(null);
  const unlistenDoneRef = useRef<UnlistenFn | null>(null);
  const unlistenErrorRef = useRef<UnlistenFn | null>(null);
  const activeChatIdRef = useRef<string | null>(null);
  const contextReqSeqRef = useRef(0);
  const inputElRef = useRef<HTMLTextAreaElement | null>(null);
  const INPUT_MAX_HEIGHT_PX = 120;
  const chatPopoverTimerRef = useRef<number | null>(null);
  const unlistenStreamEndRef = useRef<UnlistenFn | null>(null);


  const effectiveSelection = selection === undefined ? localSelection : selection;

  function setSelectionValue(next: { text: string; file_id?: string } | null) {
    if (selection === undefined) setLocalSelection(next);
    onSetSelection?.(next);
  }

  function clearSelectionValue() {
    if (selection === undefined) setLocalSelection(null);
    onClearSelection?.();
  }

  useEffect(() => {
    function onFocusChat(e: Event) {
      const ce = e as CustomEvent;
      const targetChatId = ce?.detail?.chatId;
      if (typeof targetChatId === "string" && chatId && targetChatId !== chatId) return;
      inputElRef.current?.focus?.();
    }
    window.addEventListener("insight:focus-chat", onFocusChat as any);
    return () => window.removeEventListener("insight:focus-chat", onFocusChat as any);
  }, [chatId]);

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
    if (chatPopoverTimerRef.current != null) {
      window.clearTimeout(chatPopoverTimerRef.current);
      chatPopoverTimerRef.current = null;
    }
    setChatSelectionText("");
    setChatSelectionPos(null);
    setPendingChatSelection(null);
  }, [chatId]);

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
    for (const ref of [unlistenTokenRef, unlistenDoneRef, unlistenErrorRef, unlistenStreamEndRef]) {
      if (!ref.current) continue;
      try {
        ref.current();
      } catch {
        // ignore
      }
      ref.current = null;
    }
  }

  function _closestChatMessage(node: Node | null): HTMLElement | null {
    if (!node) return null;
    const el =
      node.nodeType === Node.ELEMENT_NODE ? (node as Element) : node.parentElement;
    if (!el) return null;
    return el.closest(".chat-message") as HTMLElement | null;
  }

  function readChatSelectionFromWindow(opts?: { showPopover?: boolean }) {
    const sel = window.getSelection?.();
    const txt = (sel && typeof sel.toString === "function" ? sel.toString() : "") || "";
    const cleaned = txt.replace(/\s+/g, " ").trim();
    if (!cleaned) {
      setChatSelectionText("");
      setChatSelectionPos(null);
      setPendingChatSelection(null);
      return;
    }

    const body = messagesRef.current;
    if (!body) return;

    try {
      const anchorNode = sel?.anchorNode ?? null;
      const focusNode = sel?.focusNode ?? null;
      const anchorMsg = _closestChatMessage(anchorNode);
      const focusMsg = _closestChatMessage(focusNode);
      if (!anchorMsg || !focusMsg || anchorMsg !== focusMsg) {
        setChatSelectionText("");
        setChatSelectionPos(null);
        setPendingChatSelection(null);
        return;
      }
      if (!anchorMsg.classList.contains("chat-message-assistant")) {
        setChatSelectionText("");
        setChatSelectionPos(null);
        setPendingChatSelection(null);
        return;
      }
      const anchorOk = anchorNode ? body.contains(anchorNode) : false;
      const focusOk = focusNode ? body.contains(focusNode) : false;
      if (!anchorOk && !focusOk) {
        setChatSelectionText("");
        setChatSelectionPos(null);
        setPendingChatSelection(null);
        return;
      }
    } catch {
      // ignore
    }

    const capped = cleaned.length > 2000 ? cleaned.slice(0, 2000) + "…" : cleaned;
    setChatSelectionText(capped);
    setPendingChatSelection({ text: capped });

    try {
      if (!opts?.showPopover) return;
      if (!sel || sel.rangeCount === 0) return;
      const range = sel.getRangeAt(0);
      const rect = range.getBoundingClientRect();
      const bodyRect = body.getBoundingClientRect();
      const xRaw = rect.left - bodyRect.left + body.scrollLeft + rect.width / 2;
      const minX = body.scrollLeft + 28;
      const maxX = body.scrollLeft + body.clientWidth - 28;
      const x = Math.max(minX, Math.min(maxX, xRaw));

      const yRaw = rect.top - bodyRect.top + body.scrollTop - 48;
      const minY = body.scrollTop + 8;
      const y = Math.max(minY, yRaw);
      if (chatPopoverTimerRef.current != null) window.clearTimeout(chatPopoverTimerRef.current);
      chatPopoverTimerRef.current = window.setTimeout(() => {
        chatPopoverTimerRef.current = null;
        setChatSelectionPos({ x: Math.max(8, x), y: Math.max(8, y) });
      }, 0);
    } catch {
      // ignore
    }
  }

  useEffect(() => {
    function onSelectionChangeEvent() {
      readChatSelectionFromWindow({ showPopover: false });
    }
    document.addEventListener("selectionchange", onSelectionChangeEvent);
    return () => document.removeEventListener("selectionchange", onSelectionChangeEvent);
  }, []);

  function commitChatSelectionToInput() {
    if (!pendingChatSelection) return;
    setSelectionValue({ text: pendingChatSelection.text });
    inputElRef.current?.focus?.();
    setChatSelectionText("");
    setPendingChatSelection(null);
    setChatSelectionPos(null);
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
    const st =
      mdStreamRef.current.get(assistantId) ??
      ({ blocks: [], tail: "", inFence: false } satisfies MarkdownStreamState);
    _ingestMarkdownChunk(st, pending);
    mdStreamRef.current.set(assistantId, st);
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
    mdStreamRef.current.clear();
    setIsStreaming(false);
    setError(null);
    setChatSelectionText("");
    setChatSelectionPos(null);
    setPendingChatSelection(null);

    if (!chatId) {
      setAttachedPaths([]);
      setMessages([]);
      setContextStatus(null);
      return;
    }
    let cancelled = false;
    async function loadHistory() {
      try {
        const res = await invoke<{
          messages?: {
            role: string;
            content: string;
            attachments?: string[];
            selection?: { text: string; file_id?: string; page?: number };
            focus_document_id?: string;
          }[];
        }>(
          "get_session_messages",
          { chatId }
        );
        if (!cancelled && res && (res as any).messages) {
          const msgs = (res as any).messages.map((m: any, idx: number) => ({
            id: `${chatId}-${idx}-${m.role}`,
            role: m.role === "assistant" ? "assistant" : "user",
            content: m.content ?? "",
            attachments: Array.isArray(m.attachments) ? m.attachments : undefined,
            selection: m.selection && typeof m.selection === "object" ? m.selection : undefined,
            focus_document_id:
              typeof m.focus_document_id === "string" ? m.focus_document_id : undefined,
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
    if (!active) return;
    if (!chatId) return;
    let cancelled = false;
    // Show a neutral state immediately so we don't display the previous chat's status.
    setContextStatus(null);

    // Avoid calling engine_request while any stream is still finalizing; the stream reader
    // can consume non-stream responses and make engine_request hang.
    const activeStream = getActiveStream();
    if (!activeStream) {
      refreshContextStatus(chatId).catch(() => {
        if (!cancelled) setContextStatus(null);
      });
      return () => {
        cancelled = true;
      };
    }

    // If another chat is still streaming/finalizing, wait for it to end then refresh.
    waitForStreamToFinish(activeStream.requestId)
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
    if (isStreaming || !active) return;

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
    if (!trimmed || isStreaming || !active) return;
    if (!chatId) {
      setError("Select or create a chat first.");
      return;
    }

    setError(null);
    cleanupListeners();

    // If any stream is still active/finalizing, we must wait before starting a new one.
    // Otherwise the old Rust stream reader can still be holding stdout and swallow
    // the new stream's tokens (appears as "no response").
    const activeStream = getActiveStream();
    if (activeStream) {
      try {
        setIsStreaming(true);
        if (activeStream.chatId === chatId) {
          // Same chat: likely user pressed Stop. Don't spam cancel; just wait for stream_end.
          await waitForStreamToFinish(activeStream.requestId);
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
    const selectionPayload = effectiveSelection?.text
      ? effectiveSelection.file_id
        ? { text: effectiveSelection.text, file_id: effectiveSelection.file_id }
        : { text: effectiveSelection.text }
      : undefined;

    if (attachedNames.length) {
      try {
        window.dispatchEvent(
          new CustomEvent("insight:docs-pending", {
            detail: { chatId, names: attachedNames },
          })
        );
      } catch {
        // ignore
      }
      // Nudge the Documents pane to refresh its file list immediately.
      onRequestDocsRefresh?.();
    }

    const userMsg: ChatMessage = {
      id: `${Date.now()}-user`,
      role: "user",
      content: trimmed,
      attachments: attachedNames.length ? attachedNames : undefined,
      selection: selectionPayload,
    };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setIsStreaming(true);
    setAttachedPaths([]);
    // Selection is a one-shot context for this turn; clear the input-bar snippet after send.
    if (selectionPayload) clearSelectionValue();

    const assistantId = `${Date.now()}-assistant`;
    assistantIdRef.current = assistantId;
    setMessages((prev) => [
      ...prev,
      { id: assistantId, role: "assistant", content: "" },
    ]);
    pendingTextRef.current = "";
    mdStreamRef.current.set(assistantId, { blocks: [], tail: "", inFence: false });

    try {
      const requestId =
        (globalThis.crypto && "randomUUID" in globalThis.crypto
          ? (globalThis.crypto as any).randomUUID()
          : `${Date.now()}-${Math.random()}`) as string;
      activeRequestIdRef.current = requestId;
      // Attach the request_id to the placeholder assistant message so out-of-band
      // events (e.g. citations) can update the correct message even after streaming ends.
      setMessages((prev) =>
        prev.map((m) => (m.id === assistantId ? { ...m, request_id: requestId } : m))
      );
      const paths = attached;
      // Option A (backend contract): `focus_document_id` biases retrieval but does not hard-filter.
      // Do NOT send `documents` from the UI (it can cause the backend to inline-stitch that doc
      // and skip broader RAG). The backend already scopes retrieval to the chat's files.
      //
      // If the user attached new files in this same send, omit focus_document_id so the backend
      // can default focus to the first ingested doc for this turn.
      const focusDocForTurn =
        paths.length > 0
          ? undefined
          : (docsVisible ? activeDocumentId : null) || undefined;
      const docScopeMode = docsVisible ? "focused" : "all";

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
            const st =
              mdStreamRef.current.get(assistantId) ??
              ({ blocks: [], tail: "", inFence: false } satisfies MarkdownStreamState);
            _ingestMarkdownChunk(st, pending);
            mdStreamRef.current.set(assistantId, st);
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
          mdStreamRef.current.delete(assistantId);
          if (chatId) {
            refreshContextStatus(chatId).catch(() => {});
          }
        }
      );
	unlistenStreamEndRef.current = await listen<{
	request_id?: string;
	chat_id?: string;
	}>(
	"llm_stream_end",
	(event) => {
		const payload = event.payload || {};
		if (payload.chat_id && payload.chat_id !== chatId) return;
		if (payload.request_id && payload.request_id !== requestId) return;

		// Fast UI response when tokens stop
		setIsStreaming(false);
		try {
      unlistenStreamEndRef.current?.();
    } catch {}
    unlistenStreamEndRef.current = null;
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
          mdStreamRef.current.delete(assistantId);
          if (chatId) {
            refreshContextStatus(chatId).catch(() => {});
          }
        }
      );

      // Starts stream in the background and returns immediately.
      await engineStreamChat({
        chatId,
        query: trimmed,
        requestId,
        paths,
        focusDocumentId: focusDocForTurn,
        docScopeMode,
        selection: selectionPayload,
      });
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
    if (!active) return;
    const rid = activeRequestIdRef.current;
    if (!rid) {
      // Fallback: if another chat started a stream, still allow stop to cancel it.
      const activeStream = getActiveStream();
      if (activeStream) {
        engineCancel(activeStream.requestId).catch(() => {});
      }
      return;
    }

    // Immediate UI stop: ignore further tokens and switch back to "Send".
    flushPendingToMessage();
    cleanupListeners();
    if (assistantIdRef.current) {
      mdStreamRef.current.delete(assistantIdRef.current);
    }
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
      <div className={`chat-root ${embedded ? "chat-root-embedded" : ""}`}>
      {showTopbar ? <div className="chat-topbar" aria-hidden="true" /> : null}

      <div
        className="chat-messages"
        ref={messagesRef}
        onMouseUp={() => readChatSelectionFromWindow({ showPopover: true })}
        onKeyUp={() => readChatSelectionFromWindow({ showPopover: true })}
        onScroll={() => {
          if (chatSelectionPos) setChatSelectionPos(null);
        }}
      >
        {messages.map((m) => (
          <div key={m.id} className={`chat-message chat-message-${m.role}`}>
            <div className="chat-message-role">
              {m.role === "user" ? "You" : "Insight"}
            </div>
            <div className="chat-message-content">
              {m.role === "assistant" ? (
                isStreaming && m.id === assistantIdRef.current ? (
                  (() => {
                    const st = mdStreamRef.current.get(m.id);
                    if (!st) return <ChatMarkdown markdown={m.content} />;
                    return (
                      <ChatMarkdownStream
                        blocks={st.blocks}
                        tail={st.tail}
                        inFence={st.inFence}
                        fenceToken={st.fenceToken ?? null}
                      />
                    );
                  })()
                ) : (
                  <ChatMarkdown markdown={m.content} />
                )
              ) : (
                m.content
              )}
            </div>
            {!!m.selection?.text && (
              <div className="chat-message-selection">
                <span className="chat-selection-chip" title={m.selection.text}>
                  Selection
                </span>
              </div>
            )}
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
        {chatSelectionText && chatSelectionPos ? (
          <button
            type="button"
            className="chat-selection-ask chat-selection-popover"
            style={{ left: chatSelectionPos.x, top: chatSelectionPos.y }}
            onClick={commitChatSelectionToInput}
            title="Ask about this selection"
            onMouseDown={(e) => e.stopPropagation()}
            onPointerDown={(e) => e.stopPropagation()}
          >
            Ask
          </button>
        ) : null}
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
          {effectiveSelection?.text ? (
            <div
              className="chat-input-selection"
              title={effectiveSelection.text}
              onClick={() => inputElRef.current?.focus?.()}
              role="note"
              aria-label="Selected excerpt"
            >
              <div className="chat-input-selection-text">
                {(() => {
                  const cleaned = effectiveSelection.text.replace(/\s+/g, " ").trim();
                  const parts = cleaned.split(" ").filter(Boolean);
                  const preview = parts.slice(0, 5).join(" ");
                  return parts.length > 5 ? `${preview}…` : preview;
                })()}
              </div>
              <button
                className="chat-input-selection-clear"
                type="button"
                aria-label="Clear selection"
                title="Clear selection"
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  clearSelectionValue();
                }}
              >
                ×
              </button>
            </div>
          ) : null}

          <div className="chat-input-main">
            <button
              className="chat-input-attach"
              onClick={pickAttachments}
              disabled={!chatId || isStreaming || !active}
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
              readOnly={!active}
            />
          </div>
        </div>
        <div className="chat-context-wrap">
          {contextStatus ? (
            (() => {
              const used = Number(contextStatus.used_tokens || 0);
              const cap = Number(contextStatus.capacity_tokens || 0);
              const pctUsed = Math.max(0, Math.min(100, Number(contextStatus.percent || 0)));
              const left = Math.max(0, cap - used);
              const tps =
                typeof contextStatus.last_gen_tps === "number" && Number.isFinite(contextStatus.last_gen_tps)
                  ? contextStatus.last_gen_tps
                  : null;
              const speedLine = tps ? `\nSpeed ~${tps.toFixed(1)} tok/s` : "";
              const tooltip = `Used ${pctUsed}%\nRemaining ${formatInt(left)} / ${formatInt(cap)} tokens${speedLine}`;
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
                        <div>
                          Used {pctUsed}% · Remaining {formatInt(left)} / {formatInt(cap)} tokens
                        </div>
                        {tps ? <div>Speed ~{tps.toFixed(1)} tok/s</div> : null}
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
            !active ||
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
