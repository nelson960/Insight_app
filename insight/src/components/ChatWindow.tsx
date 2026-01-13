import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { Check, Copy, GitBranch, Square, X } from "lucide-react";
import SendHorizontalIcon from "./icons/SendHorizontalIcon";
import {
  cancelActiveStreamAndWait,
  engineCancel,
  engineStreamChat,
  getActiveStream,
  waitForStreamToFinish,
} from "../api/engine";
import { engine } from "../api/engine";
import { ChatMarkdown, ChatMarkdownStream } from "./ChatMarkdown";
import {
  beginStreamTurn,
  cancelStreamTurn,
  clearChatDraft,
  ensureChatUiLoaded,
  getChatDraft,
  getChatUiSnapshot,
  rollbackStreamTurn,
  setChatDraft,
  subscribeChatUi,
} from "../state/chatUiStore";

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
  onRequireModel?: () => Promise<boolean>;
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
  input_budget_tokens?: number;
  input_budget_reserved?: number;
};

type MarkdownStreamState = {
  blocks: string[];
  tail: string;
  inFence: boolean;
  fenceToken?: "```" | "~~~" | null;
};

type IngestProgressState = {
  fileIds: string[];
  total: number;
  done: number;
  failed: number;
};

const MAX_MD_TAIL_CHARS = 1800;

function formatPageRanges(pageRanges: unknown): string | null {
  if (!Array.isArray(pageRanges) || pageRanges.length === 0) return null;
  const parts: string[] = [];
  for (const r of pageRanges) {
    if (!Array.isArray(r) || r.length < 2) continue;
    const a = Number(r[0]);
    const b = Number(r[1]);
    if (!Number.isFinite(a) || !Number.isFinite(b)) continue;
    const start = Math.min(a, b);
    const end = Math.max(a, b);
    if (start === end) parts.push(String(start));
    else parts.push(`${start}–${end}`);
  }
  if (parts.length === 0) return null;
  const joined = parts.join(", ");
  return parts.length === 1 && !joined.includes("–") ? `page ${joined}` : `pages ${joined}`;
}

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
  onRequireModel,
}: Props) {
  const [chatUi, setChatUi] = useState(() => getChatUiSnapshot(chatId));
  const messages = chatUi.messages;
  const isStreaming = chatUi.isStreaming;
  const activeRequestId = chatUi.activeRequestId;

  // Track active streaming message content length for scroll updates
  const activeStreamContentLen = (() => {
    if (!activeRequestId) return 0;
    const m = messages.find(
      (x) => x.role === "assistant" && x.request_id === activeRequestId
    );
    return (m?.content ?? "").length;
  })();

  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);
  const [branchTarget, setBranchTarget] = useState<{
    id: string;
    role: "user" | "assistant";
    content: string;
  } | null>(null);
  const [branchShareDocs, setBranchShareDocs] = useState(true);
  const [branchIsWorking, setBranchIsWorking] = useState(false);
  const [branchError, setBranchError] = useState<string | null>(null);
  const branchPopoverRef = useRef<HTMLDivElement | null>(null);
  const [attachedPaths, setAttachedPaths] = useState<string[]>([]);
  const [_ingestProgress, setIngestProgress] = useState<IngestProgressState | null>(null);
  const [showPrestream, setShowPrestream] = useState(false);
  const [contextStatus, setContextStatus] = useState<ContextStatus | null>(null);
  const [localSelection, setLocalSelection] = useState<{ text: string; file_id?: string } | null>(
    null
  );
  const [chatSelectionText, setChatSelectionText] = useState<string>("");
  const [chatSelectionPos, setChatSelectionPos] = useState<{ x: number; y: number } | null>(null);
  const [pendingChatSelection, setPendingChatSelection] = useState<{ text: string } | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const messagesRef = useRef<HTMLDivElement | null>(null);
  const mdStreamRef = useRef<Map<string, { st: MarkdownStreamState; seen: number }>>(new Map());
  const activeChatIdRef = useRef<string | null>(null);
  const prevRequestIdRef = useRef<string | null>(null);
  const contextReqSeqRef = useRef(0);
  const inputElRef = useRef<HTMLTextAreaElement | null>(null);
  const INPUT_MAX_HEIGHT_PX = 120;
  const chatPopoverTimerRef = useRef<number | null>(null);
  const toastTimerRef = useRef<number | null>(null);
  const prestreamTimerRef = useRef<number | null>(null);
  const prestreamRequestIdRef = useRef<string | null>(null);
  const prestreamStartedRef = useRef(false);

  // Scroll state: stick-to-bottom pattern with refs
  const stickToBottomRef = useRef<boolean>(true);
  const programmaticScrollRef = useRef<boolean>(false);
  const savedPositionsRef = useRef<Map<string, number>>(new Map());
  const prevChatIdRef = useRef<string | null>(null);
  const scrollSaveTimeoutRef = useRef<number | null>(null);

  const effectiveSelection = selection === undefined ? localSelection : selection;

  function setSelectionValue(next: { text: string; file_id?: string } | null) {
    if (selection === undefined) setLocalSelection(next);
    onSetSelection?.(next);
  }

  function clearSelectionValue() {
    if (selection === undefined) setLocalSelection(null);
    onClearSelection?.();
  }

  function handleInputChange(value: string) {
    setInput(value);
    if (chatId) {
      setChatDraft(chatId, {
        input: value,
        attachments: attachedPaths,
      });
    }
  }

  function handleSetAttachments(paths: string[] | ((prev: string[]) => string[])) {
    const newPaths = typeof paths === "function" ? paths(attachedPaths) : paths;
    setAttachedPaths(paths);
    if (chatId) {
      setChatDraft(chatId, {
        input: input,
        attachments: newPaths,
      });
    }
  }

  function showToast(message: string) {
    if (!message) return;
    setToast(message);
    if (toastTimerRef.current != null) window.clearTimeout(toastTimerRef.current);
    toastTimerRef.current = window.setTimeout(() => {
      toastTimerRef.current = null;
      setToast(null);
    }, 4200);
  }

  function clearPrestream() {
    if (prestreamTimerRef.current != null) {
      window.clearTimeout(prestreamTimerRef.current);
      prestreamTimerRef.current = null;
    }
    prestreamRequestIdRef.current = null;
    prestreamStartedRef.current = false;
    setShowPrestream(false);
  }

  function startPrestream(requestId: string) {
    clearPrestream();
    prestreamRequestIdRef.current = requestId;
    prestreamStartedRef.current = false;
    prestreamTimerRef.current = window.setTimeout(() => {
      prestreamTimerRef.current = null;
      if (prestreamRequestIdRef.current !== requestId) return;
      if (prestreamStartedRef.current) return;
      if (chatId && getChatUiSnapshot(chatId).activeRequestId !== requestId) return;
      setShowPrestream(true);
    }, 500);
  }

  useEffect(() => {
    return () => {
      if (toastTimerRef.current != null) window.clearTimeout(toastTimerRef.current);
      if (scrollSaveTimeoutRef.current != null) window.clearTimeout(scrollSaveTimeoutRef.current);
    };
  }, []);

  // Scroll helper functions
  function isNearBottom(el: HTMLElement): boolean {
    const distance = el.scrollHeight - el.scrollTop - el.clientHeight;
    return distance <= 48;
  }

  function scrollToBottom() {
    const el = messagesRef.current;
    if (!el) return;

    programmaticScrollRef.current = true;
    el.scrollTop = el.scrollHeight; // browser clamps to max

    requestAnimationFrame(() => {
      programmaticScrollRef.current = false;
    });
  }

  function saveScrollPosition(chatId: string) {
    const messagesEl = messagesRef.current;
    if (!messagesEl) return;
    savedPositionsRef.current.set(chatId, messagesEl.scrollTop);

    // Debounced save to sessionStorage
    if (scrollSaveTimeoutRef.current != null) {
      clearTimeout(scrollSaveTimeoutRef.current);
    }
    scrollSaveTimeoutRef.current = window.setTimeout(() => {
      const data = Array.from(savedPositionsRef.current.entries());
      sessionStorage.setItem('chat-scroll-positions', JSON.stringify(data));
    }, 500);
  }

  // Fresh session detection and load saved positions on mount
  useEffect(() => {
    const hasSessionMarker = sessionStorage.getItem('insight-session-active');
    if (!hasSessionMarker) {
      // Fresh session: clear saved positions and set marker
      savedPositionsRef.current.clear();
      sessionStorage.setItem('insight-session-active', 'true');
    } else {
      // Existing session: load saved positions from sessionStorage
      const stored = sessionStorage.getItem('chat-scroll-positions');
      if (stored) {
        try {
          const data = JSON.parse(stored) as [string, number][];
          savedPositionsRef.current = new Map<string, number>(data);
        } catch {
          // Invalid data, start fresh
          console.warn('Failed to parse saved scroll positions');
        }
      }
    }
  }, []);

  // Update scroll mode when streaming starts/ends

  // User scroll handler - updates stickToBottomRef and saves position
  useEffect(() => {
    const messagesEl = messagesRef.current;
    if (!messagesEl || !chatId) return;

    const handleScroll = () => {
      // Ignore scroll events triggered by our own scrollToBottom calls
      if (programmaticScrollRef.current) return;

      // Update stickToBottom based on whether user is near bottom
      stickToBottomRef.current = isNearBottom(messagesEl);

      // Always save current position (debounced happens inside saveScrollPosition)
      saveScrollPosition(chatId);
    };

    messagesEl.addEventListener('scroll', handleScroll, { passive: true });
    return () => {
      messagesEl.removeEventListener('scroll', handleScroll);
    };
  }, [chatId]);

  // Streaming auto-scroll: pin to bottom during streaming AND after it ends (final render/layout)
  useLayoutEffect(() => {
    if (!chatId) return;
    if (!stickToBottomRef.current) return;

    // Pin to bottom during streaming AND after it ends
    scrollToBottom();
  }, [chatId, isStreaming, activeStreamContentLen, messages.length]);

  // ChatId switch: save old position, restore new position
  useEffect(() => {
    if (!chatId) return;

    const prevChatId = prevChatIdRef.current;
    if (prevChatId === chatId) return; // No change
    prevChatIdRef.current = chatId;

    const messagesEl = messagesRef.current;
    if (!messagesEl) return;

    // Save previous chat's position
    if (prevChatId) {
      savedPositionsRef.current.set(prevChatId, messagesEl.scrollTop);
    }

    // Restore new chat's position or scroll to bottom
    const savedPos = savedPositionsRef.current.get(chatId);
    if (savedPos !== undefined) {
      // Restore saved position
      requestAnimationFrame(() => {
        const el = messagesRef.current;
        if (el) {
          el.scrollTop = savedPos;
          stickToBottomRef.current = isNearBottom(el);
        }
      });
    } else {
      // No saved position: scroll to bottom and set stickToBottom
      scrollToBottom();
      stickToBottomRef.current = true;
    }
  }, [chatId]);

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
    if (!branchTarget) return;
    requestAnimationFrame(() => {
      branchPopoverRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    });
  }, [branchTarget?.id]);

  useEffect(() => {
    if (chatPopoverTimerRef.current != null) {
      window.clearTimeout(chatPopoverTimerRef.current);
      chatPopoverTimerRef.current = null;
    }
    setChatSelectionText("");
    setChatSelectionPos(null);
    setPendingChatSelection(null);
    setCopiedMessageId(null);
    setBranchTarget(null);
    setBranchError(null);
    setBranchIsWorking(false);
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

  // Load persisted messages when switching chats
  useEffect(() => {
    mdStreamRef.current.clear();
    setError(null);
    setChatSelectionText("");
    setChatSelectionPos(null);
    setPendingChatSelection(null);
    setIngestProgress(null);
    setIsSending(false);
    clearPrestream();

    setChatUi(getChatUiSnapshot(chatId));

    // Restore draft when chatId changes
    const draft = chatId ? getChatDraft(chatId) : null;
    if (draft) {
      setInput(draft.input);
      setAttachedPaths(draft.attachments);
    } else {
      setInput("");
      setAttachedPaths([]);
    }

    if (!chatId) {
      setContextStatus(null);
      return;
    }

    void ensureChatUiLoaded(chatId);
    return subscribeChatUi(chatId, () => {
      const snap = getChatUiSnapshot(chatId);

      // Update markdown streaming state for the active request, if any.
      const rid = snap.activeRequestId;
      if (rid) {
        const msg = snap.messages.find((m) => m.role === "assistant" && m.request_id === rid);
        if (msg) {
          const content = msg.content || "";
          const entry =
            mdStreamRef.current.get(rid) ??
            ({
              st: { blocks: [], tail: "", inFence: false },
              seen: 0,
            } satisfies { st: MarkdownStreamState; seen: number });
          const delta = content.slice(entry.seen);
          if (delta) _ingestMarkdownChunk(entry.st, delta);
          entry.seen = content.length;
          mdStreamRef.current.set(rid, entry);
          if (rid === prestreamRequestIdRef.current && delta) {
            prestreamStartedRef.current = true;
            setShowPrestream(false);
          }
        }
      }

      setChatUi(snap);
    });
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

  // After a stream fully finishes (llm-done), refresh the context meter.
  // Important: wait for llm-done to avoid stdout contention with engine_request.
  useEffect(() => {
    if (!chatId) return;
    const prev = prevRequestIdRef.current;
    prevRequestIdRef.current = activeRequestId;
    if (prev && !activeRequestId) {
      clearPrestream();
      waitForStreamToFinish(prev)
        .then(() => {
          if (chatId) refreshContextStatus(chatId).catch(() => {});
        })
        .catch(() => {});
    }
  }, [chatId, activeRequestId]);

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

  function formatIngestFailure(res: { status: number; data: any; error?: string }) {
    const status = Number(res?.status || 0);
    const payload = res?.data;
    const detail =
      payload && typeof payload === "object" && "detail" in payload ? (payload as any).detail : payload;
    if (detail && typeof detail === "object") {
      const code = String((detail as any).error || "");
      const filename = String((detail as any).filename || "");
      const sizeBytes = Number((detail as any).size_bytes || 0);
      const maxBytes = Number((detail as any).max_bytes || 0);
      const msg = String((detail as any).message || "");

      if (code === "file_too_large") {
        const filePart = filename ? `: ${filename}` : "";
        const sizePart = sizeBytes ? ` (${formatBytes(sizeBytes)})` : "";
        const maxPart = maxBytes ? ` Max ${formatBytes(maxBytes)}.` : "";
        return `File too large${filePart}${sizePart}.${maxPart}`;
      }
      if (msg) return msg;
      if (code) return `Upload rejected (${code})`;
    }
    if (typeof detail === "string" && detail.trim()) return detail.trim();
    if (res?.error) return res.error;
    return `Failed to ingest files (${status || "error"})`;
  }

  async function pickAttachments() {
    if (!chatId) {
      setError("Select or create a chat first.");
      return;
    }
    if (isStreaming || !active) return;
    if (onRequireModel) {
      const ok = await onRequireModel();
      if (!ok) return;
    }

    const LARGE_MODE_THRESHOLD_BYTES = 5 * 1024 * 1024; // keep aligned with backend default (INSIGHT_MAX_MULTI_FILE_BYTES)
    const LARGE_MODE_ALLOWED_SUFFIXES = new Set([".txt", ".log", ".json"]);

    setError(null);
    try {
      const res = await invoke<{
        files?: { path: string; name?: string; size_bytes?: number; max_bytes?: number }[];
      }>("pick_files");
      const files = res?.files || [];
      if (!files.length) return;

      const accepted: string[] = [];
      const rejected: { name: string; size: number; max: number }[] = [];
      const largeSelected: { path: string; name: string; size: number; suffix: string }[] = [];
      const largeTypeRejected: { name: string; suffix: string }[] = [];

      for (const f of files) {
        const p = (f as any)?.path as string;
        if (!p) continue;
        const name = (f as any)?.name || filenameFromPath(p);
        const size = Number((f as any)?.size_bytes ?? 0);
        const max = Number((f as any)?.max_bytes ?? 0);
        const suffix = (() => {
          const n = String(name || "");
          const dot = n.lastIndexOf(".");
          return dot >= 0 ? n.slice(dot).toLowerCase() : "";
        })();
        if (max > 0 && size > max) {
          rejected.push({ name, size, max });
          continue;
        }
        if (size > LARGE_MODE_THRESHOLD_BYTES) {
          if (!LARGE_MODE_ALLOWED_SUFFIXES.has(suffix)) {
            largeTypeRejected.push({ name, suffix });
            continue;
          }
          largeSelected.push({ path: p, name, size, suffix });
        }
        accepted.push(p);
      }

      if (largeTypeRejected.length) {
        const first = largeTypeRejected[0];
        const extra = largeTypeRejected.length > 1 ? ` (+${largeTypeRejected.length - 1} more)` : "";
        setError(
          `Large file mode supports only .txt/.log/.json. Unsupported: ${first.name}${first.suffix ? ` (${first.suffix})` : ""
          }.${extra}`
        );
      }

      if (rejected.length) {
        const first = rejected[0];
        const extra = rejected.length > 1 ? ` (+${rejected.length - 1} more)` : "";
        setError(
          `File too large: ${first.name} (${formatBytes(first.size)}). Max ${formatBytes(first.max)}.${extra}`
        );
      }

      if (!accepted.length) return;

      // Large-file ("raw_large") policy: one file only, and only into a new chat with no existing files.
      // Backend enforces this; we precheck to avoid a confusing 409 after the user types a message.
      if (largeSelected.length) {
        if (largeSelected.length > 1 || accepted.length > 1 || attachedPaths.length) {
          setError("Large file mode supports uploading exactly one file into a new card (no other attachments).");
          return;
        }
        try {
          const listRes = await engine<{ files?: { file_id: string }[] }>(
            `/files/chat/${encodeURIComponent(chatId)}`,
            undefined,
            "GET"
          );
          const existing = Array.isArray((listRes as any)?.data?.files) ? (listRes as any).data.files : [];
          if (existing.length) {
            setError("Large files can only be uploaded into a new card with no existing files.");
            return;
          }
        } catch {
          // If the precheck fails, fall back to backend enforcement.
        }
      }

      // Only attach; actual ingestion happens when user clicks Send (with query).
      handleSetAttachments((prev) => {
        const set = new Set(prev);
        for (const p of accepted) set.add(p);
        return Array.from(set);
      });
    } catch (err: any) {
      setError(err?.message ?? String(err));
    }
  }

  function removeAttachment(path: string) {
    handleSetAttachments((prev) => prev.filter((p) => p !== path));
  }

  async function sendMessage() {
    const trimmed = input.trim();
    if (!trimmed || isStreaming || isSending || !active) return;
    if (!chatId) {
      setError("Select or create a chat first.");
      return;
    }
    if (onRequireModel) {
      const ok = await onRequireModel();
      if (!ok) return;
    }

    const inputBeforeSend = input;
    const attachedBeforeSend = attachedPaths.slice();
    const selectionBeforeSend = effectiveSelection
      ? { text: effectiveSelection.text, file_id: effectiveSelection.file_id }
      : null;

    setError(null);
    setIsSending(true);

    // If any stream is still active/finalizing, we must wait before starting a new one.
    // Otherwise the old Rust stream reader can still be holding stdout and swallow
    // the new stream's tokens (appears as "no response").
    const activeStream = getActiveStream();
    if (activeStream) {
      try {
        if (activeStream.chatId === chatId) {
          // Same chat: likely user pressed Stop. Don't spam cancel; just wait for stream_end.
          await waitForStreamToFinish(activeStream.requestId);
        } else {
          // Different chat: cancel and wait for clean stream_end before starting.
          await cancelActiveStreamAndWait();
        }
      } catch {
        // ignore; we'll still attempt to start this chat
      }
    }

    const attached = attachedPaths.slice();
    const attachedNames = attached.map(filenameFromPath);
    let selectionPayload = effectiveSelection?.text
      ? effectiveSelection.file_id
        ? { text: effectiveSelection.text, file_id: effectiveSelection.file_id }
        : { text: effectiveSelection.text }
      : undefined;

    const requestId =
      (globalThis.crypto && "randomUUID" in globalThis.crypto
        ? (globalThis.crypto as any).randomUUID()
        : `${Date.now()}-${Math.random()}`) as string;

    beginStreamTurn({
      chatId,
      requestId,
      userText: trimmed,
      attachments: attachedNames.length ? attachedNames : undefined,
      selection: selectionPayload,
    });
    startPrestream(requestId);

    setInput("");
    setAttachedPaths([]);
    clearChatDraft(chatId);
    setIngestProgress(null);
    // Selection is a one-shot context for this turn; clear the input-bar snippet after send.
    if (selectionPayload) clearSelectionValue();

    try {
      // Option A (backend contract): `focus_document_id` biases retrieval but does not hard-filter.
      // Do NOT send `documents` from the UI (it can cause the backend to inline-stitch that doc
      // and skip broader RAG). The backend already scopes retrieval to the chat's files.
      const docPaneOpen = !!docsVisible;
      let focusDocForTurn: string | undefined = docPaneOpen ? activeDocumentId || undefined : undefined;

      // Strict RAG flow: if the user attached files (or if we're in all-doc scope),
      // we must wait for ingestion/indexing to complete before starting /chat.
      let docIdsForTurn: string[] = [];
      let attachmentNamesForTurn: string[] = attachedNames.slice();

      if (attached.length) {
        const ingestRes = await engine<{ files: { file_id: string; filename: string; job_id: string }[] }>(
          "/files/ingest_path",
          { chat_id: chatId, paths: attached },
          "POST"
        );
        if (!ingestRes.ok) {
          const msg = formatIngestFailure(ingestRes as any);
          showToast(msg);
          setError(msg);
        rollbackStreamTurn(chatId, requestId);
        setInput(inputBeforeSend);
        setAttachedPaths(attachedBeforeSend);
        if (selectionBeforeSend) setSelectionValue(selectionBeforeSend);
        clearPrestream();
        return;
      }
        const files = Array.isArray(ingestRes.data?.files) ? ingestRes.data.files : [];
        docIdsForTurn = files.map((f) => f.file_id).filter((s) => typeof s === "string" && s);
        const names = files.map((f) => f.filename).filter((s) => typeof s === "string" && s);
        if (names.length) attachmentNamesForTurn = names;

        if (attachmentNamesForTurn.length) {
          try {
            window.dispatchEvent(
              new CustomEvent("insight:docs-pending", {
                detail: { chatId, names: attachmentNamesForTurn },
              })
            );
          } catch {
            // ignore
          }
          // Nudge the Documents pane to refresh its file list immediately.
          onRequestDocsRefresh?.();
        }

        // If the docs pane is open, bias this turn to the newly attached file(s).
        // Important: prefer the new upload over any stale `activeDocumentId`.
        if (docPaneOpen && docIdsForTurn.length) {
          const newest = docIdsForTurn[docIdsForTurn.length - 1];
          focusDocForTurn = newest || activeDocumentId || undefined;
          if (newest) {
            try {
              window.dispatchEvent(
                new CustomEvent("insight:docs-select", {
                  detail: { chatId, fileId: newest },
                })
              );
            } catch {
              // ignore
            }
          }
        }

        // If a stale selection from another file is still present, drop it for this turn.
        if (selectionPayload && (selectionPayload as any).file_id) {
          const selFile = String((selectionPayload as any).file_id || "");
          if (selFile && !docIdsForTurn.includes(selFile)) {
            selectionPayload = undefined;
          }
        }
      }

      async function waitForIngestionIfNeeded() {
        // Determine which file_ids we must wait for:
        // - always wait for newly attached docs
        // - if a selection includes a file_id, wait for that file_id
        // - else if docs pane is open, wait for the focused doc (if any)
        // - else (chat pane only), wait for the latest uploaded file in this chat
        //   (multi-upload turns already wait for the newly attached docs)
        const start = Date.now();
        const maxWaitMs = 180_000;
        const pollMs = 450;

        type FileRow = { file_id: string; status?: string; created_at?: any };

        // Helper to fetch file statuses (IPC; cheap).
        async function fetchFiles(): Promise<FileRow[]> {
          const res = await engine<{ files: FileRow[] }>(
            `/files/chat/${encodeURIComponent(chatId as string)}`,
            undefined,
            "GET"
          );
          if (!res.ok) return [];
          const rows = Array.isArray(res.data?.files) ? res.data.files : [];
          return rows as any;
        }

        function buildStatusMap(rows: FileRow[]) {
          const m = new Map<string, string>();
          for (const r of rows) {
            const fid = (r as any)?.file_id;
            if (typeof fid !== "string" || !fid) continue;
            const st = String((r as any)?.status || "");
            m.set(fid, st);
          }
          return m;
        }

        function pickLatestFileId(rows: FileRow[]): string | undefined {
          let last: string | undefined;
          let best: { fid: string; ts: number } | null = null;
          for (const r of rows) {
            const fid = (r as any)?.file_id;
            if (typeof fid !== "string" || !fid) continue;
            last = fid;
            const createdAt = (r as any)?.created_at;
            const ts = typeof createdAt === "string" ? Date.parse(createdAt) : NaN;
            if (!Number.isFinite(ts)) continue;
            if (!best || ts >= best.ts) best = { fid, ts };
          }
          return best?.fid || last;
        }

        const rows0 = await fetchFiles();
        const statusById = buildStatusMap(rows0);
        const latestChatId = pickLatestFileId(rows0);

        let waitIds: string[] = [];
        if (selectionPayload && (selectionPayload as any).file_id) {
          waitIds = [String((selectionPayload as any).file_id)];
        } else if (docPaneOpen && focusDocForTurn) {
          waitIds = [focusDocForTurn];
        } else if (!docPaneOpen) {
          if (docIdsForTurn.length > 1) {
            waitIds = docIdsForTurn.slice();
          } else if (latestChatId) {
            waitIds = [latestChatId];
          }
        }
        if (!docPaneOpen) {
          for (const fid of docIdsForTurn) {
            if (!waitIds.includes(fid)) waitIds.push(fid);
          }
        }
        // Only wait for files that are not completed.
        waitIds = waitIds.filter((fid) => {
          const st = statusById.get(fid) || "";
          return st !== "completed";
        });

        if (!waitIds.length) return;

        const progressIds = Array.from(
          new Set((docIdsForTurn.length ? docIdsForTurn : waitIds).filter(Boolean))
        );
        setIngestProgress({
          fileIds: progressIds,
          total: progressIds.length,
          done: 0,
          failed: 0,
        });

        while (true) {
          // Cancelled (Stop clicked) or superseded by another send.
          if (getChatUiSnapshot(chatId).activeRequestId !== requestId) return;

          const stRows = await fetchFiles();
          const stMap = buildStatusMap(stRows);
          let done = 0;
          let failed = 0;
          let pending = 0;
          for (const fid of progressIds) {
            const st = stMap.get(fid) || "";
            if (st === "completed") done += 1;
            else if (st === "failed") {
              done += 1;
              failed += 1;
            } else pending += 1;
          }
          setIngestProgress({ fileIds: progressIds, total: progressIds.length, done, failed });

          let pendingWait = 0;
          for (const fid of waitIds) {
            const st = stMap.get(fid) || "";
            if (st !== "completed" && st !== "failed") pendingWait += 1;
          }
          if (pendingWait === 0) break;

          if (Date.now() - start > maxWaitMs) {
            throw new Error("Timed out waiting for document indexing to complete");
          }
          await new Promise((r) => window.setTimeout(r, pollMs));
        }
      }

      await waitForIngestionIfNeeded();
      // If we were cancelled while waiting, do not start the stream.
      if (getChatUiSnapshot(chatId).activeRequestId !== requestId) return;
      setIngestProgress(null);

      // Starts stream in the background and returns immediately.
      await engineStreamChat({
        chatId,
        query: trimmed,
        requestId,
        documents: docIdsForTurn,
        attachments: attachmentNamesForTurn,
        focusDocumentId: focusDocForTurn,
        docPaneOpen,
        selection: selectionPayload,
      });
    } catch (err: any) {
      console.error("Chat error", err);
      const msg = err?.message ?? String(err);
      showToast(msg);
      setError(msg);
      setIngestProgress(null);
      clearPrestream();
      // Ensure UI returns to Send state even if the stream failed to start.
      if (chatId && getChatUiSnapshot(chatId).activeRequestId === requestId) {
        cancelStreamTurn(chatId);
      }
    } finally {
      setIsSending(false);
    }
  }

  async function stopStreaming() {
    if (!active) return;
    if (!chatId) return;
    const rid = getChatUiSnapshot(chatId).activeRequestId;
    if (!rid) return;

    // Immediate UI stop: ignore further tokens and switch back to "Send".
    cancelStreamTurn(chatId);
    setIngestProgress(null);
    setIsSending(false);
    clearPrestream();

    // Scroll to bottom and set stickToBottom to true
    stickToBottomRef.current = true;
    scrollToBottom();

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

  async function copyMessage(m: { id: string; content: string }) {
    const text = (m.content || "").trim();
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      setCopiedMessageId(m.id);
      window.setTimeout(() => {
        setCopiedMessageId((prev) => (prev === m.id ? null : prev));
      }, 900);
    } catch (err) {
      console.warn("copy failed", err);
    }
  }

  async function branchFromMessage() {
    if (!chatId) return;
    if (!branchTarget) return;
    if (branchIsWorking) return;

    function titleFromBranchContent(text: string): string | null {
      const trimmed = (text || "").trim();
      if (!trimmed) return null;
      const firstLine = trimmed.split("\n")[0]?.trim() ?? "";
      if (!firstLine) return null;
      const compact = firstLine.replace(/\s+/g, " ");
      const max = 48;
      if (compact.length <= max) return compact;
      return compact.slice(0, max) + "…";
    }

    setBranchIsWorking(true);
    setBranchError(null);
    try {
      const res = await engine<{
        ok?: boolean;
        child_chat_id?: string;
        parent_chat_id?: string;
        shared_docs?: boolean;
        shared_file_ids?: string[];
      }>("/chat/branch", {
        parent_chat_id: chatId,
        message: {
          role: branchTarget.role,
          content: branchTarget.content,
        },
        share_docs: branchShareDocs,
      });
      if (!res.ok) throw new Error((res as any).error || "Branch failed");

      const childId = (res.data as any)?.child_chat_id;
      if (typeof childId !== "string" || !childId) {
        throw new Error("Branch failed: missing child_chat_id");
      }

      const title = titleFromBranchContent(branchTarget.content);
      window.dispatchEvent(
        new CustomEvent("insight:open-card", {
          detail: { chatId: childId, parentChatId: chatId, title },
        })
      );

      setBranchTarget(null);
    } catch (err: any) {
      console.error("branch failed", err);
      setBranchError(err?.message ?? String(err));
    } finally {
      setBranchIsWorking(false);
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
              {showPrestream &&
              isStreaming &&
              activeRequestId &&
              m.role === "assistant" &&
              m.request_id === activeRequestId ? (
                <div className="chat-prestream chat-prestream-inline" role="status" aria-label="Preparing response">
                  <span />
                  <span />
                  <span />
                </div>
              ) : null}
            </div>
            <div className="chat-message-content">
              {m.role === "assistant" ? (
                isStreaming && activeRequestId && m.request_id === activeRequestId ? (
                  (() => {
                    const st = mdStreamRef.current.get(activeRequestId);
                    if (!st) return <ChatMarkdown markdown={m.content} />;
                    return (
                      <ChatMarkdownStream
                        blocks={st.st.blocks}
                        tail={st.st.tail}
                        inFence={st.st.inFence}
                        fenceToken={st.st.fenceToken ?? null}
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

            {m.role === "assistant" && Array.isArray((m as any).sources) && (m as any).sources.length >= 2 ? (
              <div className="chat-message-sources" aria-label="Sources">
                {(m as any).sources.map((s: any, i: number) => {
                  const filename = typeof s?.filename === "string" ? s.filename : "Document";
                  const pages = formatPageRanges(s?.page_ranges);
                  return (
                    <div key={`${filename}-${i}`} className="chat-message-source">
                      <span className="chat-message-source-name">{filename}</span>
                      {pages ? <span className="chat-message-source-pages">{pages}</span> : null}
                    </div>
                  );
                })}
              </div>
            ) : null}

            {m.role === "assistant" && !isStreaming ? (
              <div className="chat-message-actions" aria-label="Message actions">
                <button
                  type="button"
                  className="chat-message-action chat-message-action-icon"
                  onClick={() => copyMessage(m)}
                  disabled={!m.content}
                  aria-label="Copy message"
                  title={copiedMessageId === m.id ? "Copied" : "Copy"}
                >
                  {copiedMessageId === m.id ? <Check className="w-1 h-1" /> : <Copy className="w-1 h-1" />}
                </button>
                <button
                  type="button"
                  className="chat-message-action chat-message-action-icon"
                  onClick={() => {
                    setBranchError(null);
                    setBranchShareDocs(true);
                    setBranchTarget({ id: m.id, role: m.role, content: m.content });
                  }}
                  disabled={!m.content}
                  aria-label="Branch to new card"
                  title="Branch to a new card"
                >
                  <GitBranch className="w-1 h-1" />
                </button>
              </div>
            ) : null}

            {branchTarget?.id === m.id ? (
              <div
                className="chat-branch-popover"
                role="dialog"
                aria-label="Branch options"
                ref={branchPopoverRef}
                onMouseDown={(e) => e.stopPropagation()}
                onPointerDown={(e) => e.stopPropagation()}
              >
                <div className="chat-branch-title">Branch to new card</div>
                <label className="chat-branch-row">
                  <input
                    type="checkbox"
                    checked={branchShareDocs}
                    onChange={(e) => setBranchShareDocs(e.target.checked)}
                    disabled={branchIsWorking}
                  />
                  <span>Share documents</span>
                </label>
                {branchError ? <div className="chat-branch-error">{branchError}</div> : null}
                <div className="chat-branch-buttons">
                  <button
                    type="button"
                    className="chat-branch-btn secondary"
                    onClick={() => setBranchTarget(null)}
                    disabled={branchIsWorking}
                  >
                    Cancel
                  </button>
                  <button
                    type="button"
                    className="chat-branch-btn"
                    onClick={branchFromMessage}
                    disabled={branchIsWorking}
                  >
                    {branchIsWorking ? "Creating…" : "Create"}
                  </button>
                </div>
              </div>
            ) : null}
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

      {toast ? (
        <div
          className="chat-toast"
          role="status"
          aria-live="polite"
          onClick={() => setToast(null)}
          title="Click to dismiss"
        >
          {toast}
        </div>
      ) : null}

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
                  <X className="w-3 h-3" />
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
                <X className="w-3 h-3" />
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
              onChange={(e) => handleInputChange(e.target.value)}
              onKeyDown={handleKeyDown}
              rows={1}
              ref={inputElRef}
              readOnly={!active}
            />
          </div>
        </div>
        <button
          className="chat-send-btn"
          onClick={isStreaming ? stopStreaming : sendMessage}
          disabled={
            !active ||
            (isStreaming && !activeRequestId) ||
            (!isStreaming && (isSending || !input.trim()))
          }
        >
          {isStreaming ? <Square className="w-5 h-5" style={{ width: "18px", height: "18px" }} /> : <SendHorizontalIcon size={24} className="" />}
        </button>
        <div className="chat-context-wrap">
          {contextStatus ? (
            (() => {
              const used = Number(contextStatus.used_tokens || 0);
              const cap = Number(contextStatus.capacity_tokens || 0);
              const pctUsed = Math.max(0, Math.min(100, Number(contextStatus.percent || 0)));
              const left = Math.max(0, cap - used);
              const inputBudget = Number(contextStatus.input_budget_tokens || 0);
              const tps =
                typeof contextStatus.last_gen_tps === "number" && Number.isFinite(contextStatus.last_gen_tps)
                  ? contextStatus.last_gen_tps
                  : null;
              const speedLine = tps ? `\nSpeed ~${tps.toFixed(1)} tok/s` : "";
              const inputLine = inputBudget
                ? `\nMax input ~${formatInt(inputBudget)} tokens`
                : "\nMax input n/a";
              const tooltip = `Used ${pctUsed}%\nRemaining ${formatInt(left)} / ${formatInt(cap)} tokens${inputLine}${speedLine}`;
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
                        <div>Max input ~{inputBudget ? formatInt(inputBudget) : "n/a"} tokens</div>
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
      </div>
    </div>
  );
}
