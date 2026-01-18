import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { listen } from "@tauri-apps/api/event";
import { Plus, List, Settings, X, Paperclip, Lock, Unlock, Palette } from "lucide-react";
import type { ChatSummary } from "../state/useSessions";
import { engine } from "../api/engine";
import { formatRelativeTime, truncateText } from "../utils/formatTime";

export type CardLayout = {
  showChat: boolean;
  showDocs: boolean;
  chatOnRight: boolean;
  splitRatio: number;
  activeFileId?: string | null;
};

export type CanvasNote = {
  chatId: string;
  title?: string;
  locked?: boolean;
  collapsed?: boolean;
  groupColor?: string;
  x: number;
  y: number;
  w: number;
  h: number;
  z: number;
  layout?: CardLayout;
};

export type CanvasLink = {
  id: string;
  fromChatId: string;
  toChatId: string;
  kind: "branch";
};

type Viewport = { x: number; y: number; scale: number };
type Board = { w: number; h: number };

type GroupStackAnim = {
  mode: "collapse" | "expand";
  active: boolean;
  items: Record<
    string,
    {
      dx: number; // world-space delta to move this card into the stack
      dy: number; // world-space delta to move this card into the stack
      scale: number;
      rot: number; // degrees
      z: number; // temporary z-index while stacking
    }
  >;
};

type Props = {
  sessions: ChatSummary[];
  activeChatId: string | null;
  notes: CanvasNote[];
  links: CanvasLink[];
  onFocusChat: (chatId: string) => void;
  onUpdateNote: (chatId: string, patch: Partial<CanvasNote>) => void;
  onOpenChat: (chatId: string) => void;
  onCreateChatAt: (pos: { x: number; y: number }) => string;
  onOpenCard: (chatId: string) => void;
  onOpenSettings: () => void;
  onDeleteChat: (chatId: string) => void;
  onDeleteChatTree: (rootChatId: string) => void;
  confirmDeleteChatId: string | null;
  onResetConfirmDelete: () => void;
  loadingSessions: boolean;
  dockVisible?: boolean;
};

const MIN_SCALE = 0.35;
const MAX_SCALE = 1.75;
const PAN_ENABLE_SCALE = 1.01;
const ZOOM_SPEED = 0.0015;
// Keep the board large enough that even at MIN_SCALE it still covers the viewport
// (avoids seeing a hard "edge" or misalignment when zoomed out).
const BOARD_MULT = 3.2;
const BOARD_MIN_W = 2200;
const BOARD_MIN_H = 1400;
const STACK_ANIM_MS = 340;
const LINK_FADE_MS = 140;
// Collapse should feel like "stacking cards behind" (not shrinking into a dot).
// Keep this close to 1 so the animation blends into the full-size stack layers.
const STACK_ANIM_SCALE = 0.92;

const STACK_TARGETS: Array<{ ox: number; oy: number; rot: number }> = [
  // Keep in sync with `.canvas-note-stack-*` transforms in `App.css`.
  { ox: 7, oy: 7, rot: 0 },
  { ox: 14, oy: 14, rot: 0 },
  { ox: 21, oy: 21, rot: 0 },
];

const GROUP_COLORS: Array<{ id: string; label: string; value: string }> = [
  { id: "blue", label: "Blue", value: "#3b82f6" },
  { id: "purple", label: "Purple", value: "#a855f7" },
  { id: "pink", label: "Pink", value: "#ec4899" },
  { id: "red", label: "Red", value: "#ef4444" },
  { id: "orange", label: "Orange", value: "#f97316" },
  { id: "yellow", label: "Yellow", value: "#eab308" },
  { id: "green", label: "Green", value: "#22c55e" },
  { id: "slate", label: "Slate", value: "#64748b" },
];

const VIEWPORT_STORAGE_KEY = "insight.canvas.viewport.v1";

function hexToRgb(hex: string): { r: number; g: number; b: number } | null {
  const m = /^#([0-9a-fA-F]{6})$/.exec((hex || "").trim());
  if (!m) return null;
  const n = parseInt(m[1], 16);
  return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255 };
}

function rgbaFromHex(hex: string, alpha: number): string | null {
  const rgb = hexToRgb(hex);
  if (!rgb) return null;
  const a = Math.max(0, Math.min(1, alpha));
  return `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${a})`;
}

function hashString(input: string): number {
  let h = 2166136261;
  for (let i = 0; i < input.length; i++) {
    h ^= input.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function loadPersistedViewport(): Viewport | null {
  try {
    const raw = localStorage.getItem(VIEWPORT_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") return null;
    const x = Number((parsed as any).x);
    const y = Number((parsed as any).y);
    const scale = Number((parsed as any).scale);
    if (![x, y, scale].every(Number.isFinite)) return null;
    return { x, y, scale };
  } catch {
    return null;
  }
}

function persistViewport(vp: Viewport) {
  try {
    localStorage.setItem(VIEWPORT_STORAGE_KEY, JSON.stringify(vp));
  } catch {
    // ignore
  }
}

export function Canvas({
  sessions,
  activeChatId,
  notes,
  links,
  onFocusChat,
  onUpdateNote,
  onOpenChat,
  onCreateChatAt,
  onOpenCard,
  onOpenSettings,
  onDeleteChat,
  confirmDeleteChatId,
  onResetConfirmDelete,
  loadingSessions,
  dockVisible = true,
}: Props) {
  const rootRef = useRef<HTMLDivElement | null>(null);
  const [vp, setVp] = useState<Viewport>({ x: 0, y: 0, scale: 1 });
  const vpRef = useRef(vp);
  const [groupStackAnim, setGroupStackAnim] = useState<Record<string, GroupStackAnim>>({});
  const groupStackAnimTimersRef = useRef<Record<string, number>>({});
  const groupStackAnimStartTimersRef = useRef<Record<string, number>>({});
  const groupStackAnimRafRef = useRef<Record<string, number>>({});
  const [board, setBoard] = useState<Board>({ w: 0, h: 0 }); // world-space board size
  const boardRef = useRef(board);
  const viewRef = useRef<{ w: number; h: number }>({ w: 0, h: 0 }); // screen-space canvas size
  const cursorRef = useRef<{ mx: number; my: number; has: boolean; at: number }>({
    mx: 0,
    my: 0,
    has: false,
    at: 0,
  });
  const [isPanning, setIsPanning] = useState(false);
  const [chatListOpen, setChatListOpen] = useState(false);
  const [filesByChatId, setFilesByChatId] = useState<Record<string, string[]>>({});
  const chatIdSetRef = useRef<Set<string>>(new Set());
  const fileReloadTimersRef = useRef<Record<string, number>>({});
  const panRef = useRef<{ startX: number; startY: number; startVpX: number; startVpY: number } | null>(
    null
  );

  const sessionById = useMemo(() => {
    const map = new Map<string, ChatSummary>();
    for (const s of sessions) map.set(s.chat_id, s);
    return map;
  }, [sessions]);

  // Enrich session data with last message info and file count from session metadata
  const enrichedChatData = useMemo(() => {
    const map = new Map<
      string,
      { lastMessageAt: string | null; lastMessageContent: string | null; fileCount: number }
    >();

    for (const note of notes) {
      const session = sessionById.get(note.chatId);
      // Use session metadata (from list_sessions API) instead of loading from chatUiStore
      const lastMessageAt = session?.last_message_at || null;
      const lastMessageContent = session?.last_message_content || null;
      const fileCount = session?.file_count || 0;

      map.set(note.chatId, {
        lastMessageAt,
        lastMessageContent,
        fileCount,
      });
    }

    return map;
  }, [notes, sessionById]);

  const noteById = useMemo(() => {
    const map = new Map<string, CanvasNote>();
    for (const n of notes) map.set(n.chatId, n);
    return map;
  }, [notes]);

  const tree = useMemo(() => {
    const parentByChild = new Map<string, string>();
    const childrenByParent = new Map<string, string[]>();
    for (const l of links) {
      if (l.kind !== "branch") continue;
      if (!parentByChild.has(l.toChatId)) parentByChild.set(l.toChatId, l.fromChatId);
      const arr = childrenByParent.get(l.fromChatId) || [];
      arr.push(l.toChatId);
      childrenByParent.set(l.fromChatId, arr);
    }
    return { parentByChild, childrenByParent };
  }, [links]);

  useEffect(() => {
    return () => {
      const timers = groupStackAnimTimersRef.current;
      for (const k of Object.keys(timers)) window.clearTimeout(timers[k]);
      const startTimers = groupStackAnimStartTimersRef.current;
      for (const k of Object.keys(startTimers)) window.clearTimeout(startTimers[k]);
      const rafs = groupStackAnimRafRef.current;
      for (const k of Object.keys(rafs)) cancelAnimationFrame(rafs[k]);
    };
  }, []);

  function listDescendants(rootChatId: string): string[] {
    const out: string[] = [];
    const seen = new Set<string>();
    const stack = [...(tree.childrenByParent.get(rootChatId) || [])];
    while (stack.length) {
      const id = stack.pop();
      if (!id) continue;
      if (seen.has(id)) continue;
      seen.add(id);
      out.push(id);
      const kids = tree.childrenByParent.get(id);
      if (kids && kids.length) stack.push(...kids);
    }
    return out;
  }

  function clearGroupColorWithDescendants(chatId: string, currentAccentColor: string | null) {
    // Clear color from this card and all descendants
    const descendants = listDescendants(chatId);
    descendants.forEach((descendantId) => {
      const descendantNote = noteById.get(descendantId);
      // Only clear if descendant doesn't have its own explicit color
      // (i.e., it's inheriting from this parent or has the same color)
      if (descendantNote && (!descendantNote.groupColor || descendantNote.groupColor === currentAccentColor)) {
        onUpdateNote(descendantId, { groupColor: undefined });
      }
    });
    onUpdateNote(chatId, { groupColor: undefined });
  }

  function startGroupStackAnimation(rootChatId: string, mode: "collapse" | "expand") {
    const root = noteById.get(rootChatId);
    if (!root) return;
    const descendants = listDescendants(rootChatId);
    if (!descendants.length) return;

    const items: GroupStackAnim["items"] = {};
    for (const id of descendants) {
      const n = noteById.get(id);
      if (!n) continue;
      const h = hashString(`${rootChatId}:${id}`);
      const target = STACK_TARGETS[h % STACK_TARGETS.length];
      const rot = target.rot;
      const stackZ = Math.max(1, root.z - 1);
      const z = Math.min(n.z, stackZ);
      items[id] = {
        dx: root.x - n.x + target.ox,
        dy: root.y - n.y + target.oy,
        scale: STACK_ANIM_SCALE,
        rot,
        z,
      };
    }

    const initialActive = mode === "expand";
    const nextActive = mode === "collapse";

    const timers = groupStackAnimTimersRef.current;
    const startTimers = groupStackAnimStartTimersRef.current;
    const rafs = groupStackAnimRafRef.current;
    if (timers[rootChatId]) window.clearTimeout(timers[rootChatId]);
    if (startTimers[rootChatId]) window.clearTimeout(startTimers[rootChatId]);
    if (rafs[rootChatId]) cancelAnimationFrame(rafs[rootChatId]);

    setGroupStackAnim((prev) => ({
      ...prev,
      [rootChatId]: { mode, active: initialActive, items },
    }));

    if (mode === "collapse") {
      startTimers[rootChatId] = window.setTimeout(() => {
        delete startTimers[rootChatId];
        setGroupStackAnim((prev) => {
          const cur = prev[rootChatId];
          if (!cur || cur.mode !== mode) return prev;
          return { ...prev, [rootChatId]: { ...cur, active: nextActive } };
        });
      }, LINK_FADE_MS);
    } else {
      rafs[rootChatId] = requestAnimationFrame(() => {
        delete rafs[rootChatId];
        setGroupStackAnim((prev) => {
          const cur = prev[rootChatId];
          if (!cur || cur.mode !== mode) return prev;
          return { ...prev, [rootChatId]: { ...cur, active: nextActive } };
        });
      });
    }

    timers[rootChatId] = window.setTimeout(() => {
      delete timers[rootChatId];
      setGroupStackAnim((prev) => {
        if (!prev[rootChatId]) return prev;
        const next = { ...prev };
        delete next[rootChatId];
        return next;
      });
    }, (mode === "collapse" ? LINK_FADE_MS : 0) + STACK_ANIM_MS + 60);
  }

  const hiddenChatIds = useMemo(() => {
    const hidden = new Set<string>();
    const collapsedRoots = notes
      .filter((n) => n.collapsed && groupStackAnim[n.chatId]?.mode !== "collapse")
      .map((n) => n.chatId);
    if (!collapsedRoots.length) return hidden;
    for (const root of collapsedRoots) {
      const stack = [...(tree.childrenByParent.get(root) || [])];
      while (stack.length) {
        const id = stack.pop();
        if (!id) continue;
        if (hidden.has(id)) continue;
        hidden.add(id);
        const kids = tree.childrenByParent.get(id);
        if (kids && kids.length) stack.push(...kids);
      }
    }
    return hidden;
  }, [notes, tree, groupStackAnim]);

  const visibleNotes = useMemo(() => {
    if (!hiddenChatIds.size) return notes;
    return notes.filter((n) => !hiddenChatIds.has(n.chatId));
  }, [notes, hiddenChatIds]);

  const stackCountByChatId = useMemo(() => {
    const out: Record<string, number> = {};
    for (const n of notes) {
      if (!n.collapsed) continue;
      const seen = new Set<string>();
      const stack = [...(tree.childrenByParent.get(n.chatId) || [])];
      while (stack.length) {
        const id = stack.pop();
        if (!id) continue;
        if (seen.has(id)) continue;
        seen.add(id);
        const kids = tree.childrenByParent.get(id);
        if (kids && kids.length) stack.push(...kids);
      }
      if (seen.size) out[n.chatId] = seen.size;
    }
    return out;
  }, [notes, tree]);

  const stackAnimByChatId = useMemo(() => {
    const out: Record<
      string,
      {
        active: boolean;
        mode: "collapse" | "expand";
        dx: number;
        dy: number;
        scale: number;
        rot: number;
        z: number;
      }
    > = {};
    for (const rootId of Object.keys(groupStackAnim)) {
      const anim = groupStackAnim[rootId];
      if (!anim) continue;
      for (const id of Object.keys(anim.items)) {
        const it = anim.items[id];
        if (!it) continue;
        out[id] = {
          active: anim.active,
          mode: anim.mode,
          dx: it.dx,
          dy: it.dy,
          scale: it.scale,
          rot: it.rot,
          z: it.z,
        };
      }
    }
    return out;
  }, [groupStackAnim]);

  const effectiveGroupColorByChatId = useMemo(() => {
    const out: Record<string, string> = {};

    const resolve = (chatId: string): string | null => {
      let cur: string | undefined = chatId;
      const seen = new Set<string>();
      while (cur && !seen.has(cur)) {
        seen.add(cur);
        const note = noteById.get(cur);
        const c = typeof note?.groupColor === "string" ? note.groupColor.trim() : "";
        if (c) return c;
        cur = tree.parentByChild.get(cur);
      }
      return null;
    };

    for (const n of notes) {
      const c = resolve(n.chatId);
      if (c) out[n.chatId] = c;
    }
    return out;
  }, [notes, noteById, tree]);

  const fadingLinkToChatIds = useMemo(() => {
    const fading = new Set<string>();
    for (const rootId of Object.keys(groupStackAnim)) {
      const anim = groupStackAnim[rootId];
      if (!anim || anim.mode !== "collapse") continue;
      for (const id of Object.keys(anim.items)) fading.add(id);
    }
    return fading;
  }, [groupStackAnim]);

  const chatIdsKey = useMemo(() => {
    const ids = [...new Set(notes.map((n) => n.chatId))].sort();
    return ids.join("|");
  }, [notes]);

  useEffect(() => {
    chatIdSetRef.current = new Set(notes.map((n) => n.chatId));
  }, [chatIdsKey, notes]);

  async function loadFileNames(chatId: string) {
    const res = await engine<{ files?: Array<{ filename?: string }> }>(
      `/files/chat/${encodeURIComponent(chatId)}`,
      undefined,
      "GET"
    );
    if (!res.ok) return;
    const rows = Array.isArray((res.data as any)?.files) ? (res.data as any).files : [];
    const names = rows
      .map((f: any) => (typeof f?.filename === "string" ? f.filename : ""))
      .filter((s: string) => s);
    setFilesByChatId((prev) => ({ ...prev, [chatId]: names }));
  }

  function scheduleLoadFileNames(chatId: string) {
    if (!chatId) return;
    const timers = fileReloadTimersRef.current;
    if (timers[chatId]) window.clearTimeout(timers[chatId]);
    timers[chatId] = window.setTimeout(() => {
      delete timers[chatId];
      loadFileNames(chatId).catch(() => {
        // ignore
      });
    }, 120);
  }

  // Load file lists for visible cards (and keep them fresh on backend events).
  useEffect(() => {
    const ids = [...new Set(notes.map((n) => n.chatId))];
    for (const id of ids) {
      if (typeof filesByChatId[id] !== "undefined") continue;
      scheduleLoadFileNames(id);
    }
    // Only depends on chat membership, not positions/sizes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chatIdsKey]);

  useEffect(() => {
    let cancelled = false;
    let unlisten: null | (() => void) = null;
    listen<any>("files-changed", (event) => {
      if (cancelled) return;
      const payload = event?.payload as any;
      const chatId = typeof payload?.chat_id === "string" ? payload.chat_id : "";
      if (!chatId) return;
      if (!chatIdSetRef.current.has(chatId)) return;
      scheduleLoadFileNames(chatId);
    })
      .then((fn) => {
        if (!cancelled) unlisten = fn;
      })
      .catch(() => {
        // ignore (non-Tauri build / event not supported)
      });
    return () => {
      cancelled = true;
      if (unlisten) unlisten();
    };
  }, []);

  function clampScale(s: number) {
    return Math.max(MIN_SCALE, Math.min(MAX_SCALE, s));
  }

  // Rehydrate viewport on startup (persistent pan/zoom).
  useEffect(() => {
    const restored = loadPersistedViewport();
    if (!restored) return;
    setVp((prev) => ({ ...prev, ...restored, scale: clampScale(restored.scale) }));
  }, []);

  useEffect(() => {
    vpRef.current = vp;
  }, [vp]);

  // Persist viewport changes (debounced).
  useEffect(() => {
    const t = window.setTimeout(() => persistViewport(vp), 200);
    return () => window.clearTimeout(t);
  }, [vp]);

  useEffect(() => {
    const root = rootRef.current;
    if (!root) return;
    const ro = new ResizeObserver(() => {
      const viewW = root.clientWidth;
      const viewH = root.clientHeight;
      viewRef.current = { w: viewW, h: viewH };
      const nextBoard = {
        w: Math.max(BOARD_MIN_W, Math.round(viewW * BOARD_MULT)),
        h: Math.max(BOARD_MIN_H, Math.round(viewH * BOARD_MULT)),
      };
      boardRef.current = nextBoard;
      setBoard(nextBoard);
      // Re-clamp viewport now that we know view/board size.
      setVpSync((prev) => prev);
    });
    ro.observe(root);
    // initial
    const viewW = root.clientWidth;
    const viewH = root.clientHeight;
    viewRef.current = { w: viewW, h: viewH };
    const initialBoard = {
      w: Math.max(BOARD_MIN_W, Math.round(viewW * BOARD_MULT)),
      h: Math.max(BOARD_MIN_H, Math.round(viewH * BOARD_MULT)),
    };
    boardRef.current = initialBoard;
    setBoard(initialBoard);
    // Re-clamp viewport now that we know initial view/board size.
    setVpSync((prev) => prev);
    return () => ro.disconnect();
  }, []);

  function clampViewport(next: Viewport): Viewport {
    const viewW = viewRef.current.w;
    const viewH = viewRef.current.h;
    const bw = boardRef.current.w;
    const bh = boardRef.current.h;
    if (!viewW || !viewH || !bw || !bh) return next;

    const boardPxW = bw * next.scale;
    const boardPxH = bh * next.scale;

    let minX = Math.min(0, viewW - boardPxW);
    let maxX = 0;
    let minY = Math.min(0, viewH - boardPxH);
    let maxY = 0;

    // If the board is smaller than the viewport on an axis, center it.
    if (boardPxW <= viewW) {
      minX = maxX = (viewW - boardPxW) / 2;
    }
    if (boardPxH <= viewH) {
      minY = maxY = (viewH - boardPxH) / 2;
    }

    const clampedX = Math.max(minX, Math.min(maxX, next.x));
    const clampedY = Math.max(minY, Math.min(maxY, next.y));
    return { ...next, x: clampedX, y: clampedY };
  }

  function setVpSync(next: Viewport | ((prev: Viewport) => Viewport)) {
    setVp((prev) => {
      const resolved = typeof next === "function" ? (next as (p: Viewport) => Viewport)(prev) : next;
      const clamped = clampViewport(resolved);
      vpRef.current = clamped;
      return clamped;
    });
  }

  function clampNoteToBoard(note: CanvasNote): CanvasNote {
    const bw = boardRef.current.w;
    const bh = boardRef.current.h;
    if (!bw || !bh) return note;
    let next = { ...note };
    if (next.w > bw) next.w = bw;
    if (next.h > bh) next.h = bh;
    if (next.x < 0) next.x = 0;
    if (next.y < 0) next.y = 0;
    if (next.x + next.w > bw) next.x = Math.max(0, bw - next.w);
    if (next.y + next.h > bh) next.y = Math.max(0, bh - next.h);
    return next;
  }

  function clampNoteToViewport(note: CanvasNote): CanvasNote {
    const viewW = viewRef.current.w;
    const viewH = viewRef.current.h;
    const { x: vpX, y: vpY, scale } = vpRef.current;
    if (!viewW || !viewH || !scale) return note;

    const worldLeft = -vpX / scale;
    const worldTop = -vpY / scale;
    const worldRight = worldLeft + viewW / scale;
    const worldBottom = worldTop + viewH / scale;

    const maxX = worldRight - note.w;
    const maxY = worldBottom - note.h;

    let nextX = note.x;
    let nextY = note.y;

    if (Number.isFinite(worldLeft) && Number.isFinite(maxX)) {
      nextX = Math.max(worldLeft, Math.min(maxX, note.x));
    }
    if (Number.isFinite(worldTop) && Number.isFinite(maxY)) {
      nextY = Math.max(worldTop, Math.min(maxY, note.y));
    }

    return nextX === note.x && nextY === note.y ? note : { ...note, x: nextX, y: nextY };
  }

  function applyUpdateNote(
    chatId: string,
    patch: Partial<CanvasNote>,
    opts?: { clampToViewport?: boolean }
  ) {
    if (!boardRef.current.w || !boardRef.current.h) {
      onUpdateNote(chatId, patch);
      return;
    }
    const current = notes.find((n) => n.chatId === chatId);
    if (!current) {
      onUpdateNote(chatId, patch);
      return;
    }

    if (typeof patch.collapsed === "boolean" && patch.collapsed !== !!current.collapsed) {
      const hasChildren = (tree.childrenByParent.get(chatId) || []).length > 0;
      if (hasChildren) {
        startGroupStackAnimation(chatId, patch.collapsed ? "collapse" : "expand");
      }
    }

    const merged = { ...current, ...patch };
    let clamped = clampNoteToBoard(merged);
    if (opts?.clampToViewport) {
      clamped = clampNoteToViewport(clamped);
      clamped = clampNoteToBoard(clamped);
    }
    const nextPatch: Partial<CanvasNote> = { ...patch };
    if (clamped.x !== merged.x) nextPatch.x = clamped.x;
    if (clamped.y !== merged.y) nextPatch.y = clamped.y;
    if (clamped.w !== merged.w) nextPatch.w = clamped.w;
    if (clamped.h !== merged.h) nextPatch.h = clamped.h;
    onUpdateNote(chatId, nextPatch);
  }

  function updateCursorFromClient(clientX: number, clientY: number) {
    const root = rootRef.current;
    if (!root) return;
    const rect = root.getBoundingClientRect();
    const mx = clientX - rect.left;
    const my = clientY - rect.top;
    cursorRef.current = { mx, my, has: true, at: Date.now() };
  }

  function pickZoomAnchor(e: WheelEvent, rect: DOMRect) {
    const now = Date.now();

    const fromWheel = { mx: e.clientX - rect.left, my: e.clientY - rect.top };
    const fromCursor = { mx: cursorRef.current.mx, my: cursorRef.current.my };
    const cursorFresh = cursorRef.current.has && now - cursorRef.current.at < 1500;

    function valid(p: { mx: number; my: number }, allowZero = true) {
      if (!Number.isFinite(p.mx) || !Number.isFinite(p.my)) return false;
      if (!allowZero && p.mx === 0 && p.my === 0) return false;
      // Permit a tiny outside margin; we'll clamp after selecting.
      return p.mx >= -2 && p.my >= -2 && p.mx <= rect.width + 2 && p.my <= rect.height + 2;
    }

    // Prefer real wheel coordinates when they look sane; many WebViews provide correct values here.
    // If wheel coordinates look bogus (often 0,0), fall back to last known pointer position.
    let anchor = fromWheel;
    if (!valid(fromWheel, false) && cursorFresh && valid(fromCursor, false)) {
      anchor = fromCursor;
    } else if (cursorFresh && valid(fromCursor, false) && valid(fromWheel, false)) {
      // If both are plausible, prefer cursor (it matches "zoom where I'm pointing" precisely).
      anchor = fromCursor;
    } else if (!valid(fromWheel, false) && !cursorFresh) {
      // As a last resort, zoom around the center instead of snapping to top-left.
      anchor = { mx: rect.width / 2, my: rect.height / 2 };
    }

    // Clamp into the canvas bounds so offsets stay stable.
    const mx = Math.max(0, Math.min(rect.width, anchor.mx));
    const my = Math.max(0, Math.min(rect.height, anchor.my));
    return { mx, my };
  }

  useEffect(() => {
    const root = rootRef.current;
    if (!root) return;

    const onMove = (e: MouseEvent | PointerEvent) => {
      if (!root.contains(e.target as Node)) return;
      updateCursorFromClient((e as MouseEvent).clientX, (e as MouseEvent).clientY);
    };

    // Use a non-passive capture listener on `window` so preventDefault reliably blocks page scroll
    // in the macOS WebView. Only handle events that originated within the canvas root.
    const onWheel = (e: WheelEvent) => {
      if (!root.contains(e.target as Node)) return;
      e.preventDefault();
      const { x, y, scale } = vpRef.current;

      // Zoom-to-cursor: scale and translate so the point under the cursor stays fixed.
      // Only zoom when a modifier is held (matches the "maps" feel and avoids accidental zoom).
      const wantsZoom = e.ctrlKey || e.metaKey;
      if (!wantsZoom) {
        // When zoomed in, allow scroll/trackpad to pan the board. When zoomed out, keep anchored.
        if (scale <= PAN_ENABLE_SCALE) return;
        setVpSync((prev) => ({ ...prev, x: prev.x - e.deltaX, y: prev.y - e.deltaY }));
        return;
      }

      const rect = root.getBoundingClientRect();
      const { mx, my } = pickZoomAnchor(e, rect);
      // Keep cursor position "fresh" even if the user is zooming without moving the mouse.
      cursorRef.current = { mx, my, has: true, at: Date.now() };

      const contentX = (mx - x) / scale;
      const contentY = (my - y) / scale;

      // Normalize wheel delta a bit across devices.
      const deltaY = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaY;
      const zoomFactor = 1 - deltaY * ZOOM_SPEED;
      const nextScale = clampScale(scale * zoomFactor);
      const nextX = mx - contentX * nextScale;
      const nextY = my - contentY * nextScale;

      // Always zoom to cursor (maps-style). Panning is simply disabled when zoomed out,
      // but we keep the current viewport so zooming out still feels centered on the cursor.
      setVpSync({ x: nextX, y: nextY, scale: nextScale });

      // Do not pan the canvas via wheel/trackpad scroll; keep the board fixed.
      // Panning is available via click-drag on the background (when zoomed in).
      return;
    };

    window.addEventListener("mousemove", onMove as any, { capture: true });
    window.addEventListener("pointermove", onMove as any, { capture: true });
    window.addEventListener("wheel", onWheel, { passive: false, capture: true });
    return () => {
      window.removeEventListener("mousemove", onMove as any, true);
      window.removeEventListener("pointermove", onMove as any, true);
      window.removeEventListener("wheel", onWheel as any, true);
    };
  }, []);

  // If the board size changes (window resize), ensure all notes remain inside.
  useEffect(() => {
    if (!board.w || !board.h) return;
    for (const n of notes) {
      const clamped = clampNoteToBoard(n);
      if (clamped.x !== n.x || clamped.y !== n.y || clamped.w !== n.w || clamped.h !== n.h) {
        onUpdateNote(n.chatId, { x: clamped.x, y: clamped.y, w: clamped.w, h: clamped.h });
      }
    }
    // Also clamp the viewport so you can't pan outside the resized board.
    setVpSync((prev) => prev);
  }, [board.w, board.h]);

  function beginPan(e: React.PointerEvent) {
    // Only pan when clicking background (not on notes).
    if (e.currentTarget !== e.target) return;
    if (e.button !== 0) return;
    // Allow click-drag panning when either:
    // - zoomed in, or
    // - the board is larger than the viewport (so panning is meaningful even at 1.0x).
    const viewW = viewRef.current.w;
    const viewH = viewRef.current.h;
    const bw = boardRef.current.w;
    const bh = boardRef.current.h;
    const boardPxW = bw * vpRef.current.scale;
    const boardPxH = bh * vpRef.current.scale;
    const canPan = vpRef.current.scale > PAN_ENABLE_SCALE || boardPxW > viewW + 1 || boardPxH > viewH + 1;
    if (!canPan) return;
    e.preventDefault();
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    setIsPanning(true);
    panRef.current = {
      startX: e.clientX,
      startY: e.clientY,
      startVpX: vpRef.current.x,
      startVpY: vpRef.current.y,
    };
    if (activeChatId) onFocusChat(activeChatId);
  }

  function movePan(e: React.PointerEvent) {
    updateCursorFromClient(e.clientX, e.clientY);
    if (!isPanning || !panRef.current) return;
    const dx = e.clientX - panRef.current.startX;
    const dy = e.clientY - panRef.current.startY;
    setVpSync((prev) => ({
      ...prev,
      x: panRef.current!.startVpX + dx,
      y: panRef.current!.startVpY + dy,
    }));
  }

  function endPan(e: React.PointerEvent) {
    if (!isPanning) return;
    setIsPanning(false);
    panRef.current = null;
    try {
      (e.currentTarget as HTMLElement).releasePointerCapture(e.pointerId);
    } catch {
      // ignore
    }
  }

  function createAtClient(clientX: number, clientY: number) {
    const root = rootRef.current;
    if (!root) return;
    const rect = root.getBoundingClientRect();
    const { x, y, scale } = vpRef.current;
    const mx = clientX - rect.left;
    const my = clientY - rect.top;
    const worldX = (mx - x) / scale;
    const worldY = (my - y) / scale;
    const created = onCreateChatAt({ x: worldX, y: worldY });
    // Creating a card from the canvas should not immediately open it.
    // The user can click the card (or open from the Cards menu) when ready.
    void created;
  }

  function createAtCenter() {
    const root = rootRef.current;
    if (!root) return;
    const rect = root.getBoundingClientRect();
    createAtClient(rect.left + rect.width / 2, rect.top + rect.height / 2);
  }

  const dpr = typeof window !== "undefined" && window.devicePixelRatio ? window.devicePixelRatio : 1;
  const renderX = Math.round(vp.x * dpr) / dpr;
  const renderY = Math.round(vp.y * dpr) / dpr;

  function pointOnRectEdge(from: CanvasNote, toward: { x: number; y: number }) {
    const cx = from.x + from.w / 2;
    const cy = from.y + from.h / 2;
    const dx = toward.x - cx;
    const dy = toward.y - cy;
    if (!dx && !dy) return { x: cx, y: cy };
    const hw = from.w / 2;
    const hh = from.h / 2;
    const t = 1 / Math.max(Math.abs(dx) / hw, Math.abs(dy) / hh);
    return { x: cx + dx * t, y: cy + dy * t };
  }

  const linkPaths = useMemo(() => {
    if (!links.length) return [];
    const out: Array<{ id: string; d: string; color: string; arrowPoints: string; fade: boolean }> =
      [];
    for (const l of links) {
      const from = noteById.get(l.fromChatId);
      const to = noteById.get(l.toChatId);
      if (!from || !to) continue;
      if (hiddenChatIds.has(from.chatId) || hiddenChatIds.has(to.chatId)) continue;
      const toCenter = { x: to.x + to.w / 2, y: to.y + to.h / 2 };
      const fromCenter = { x: from.x + from.w / 2, y: from.y + from.h / 2 };

      const sW = pointOnRectEdge(from, toCenter);
      const eW = pointOnRectEdge(to, fromCenter);

      const sx = Math.round(sW.x * vp.scale * dpr) / dpr;
      const sy = Math.round(sW.y * vp.scale * dpr) / dpr;
      const ex = Math.round(eW.x * vp.scale * dpr) / dpr;
      const ey = Math.round(eW.y * vp.scale * dpr) / dpr;

      const dx = ex - sx;
      const curve = Math.max(70, Math.min(240, Math.abs(dx) * 0.55));
      const dir = dx === 0 ? 1 : Math.sign(dx);
      const c1x = sx + curve * dir;
      const c1y = sy;
      const c2x = ex - curve * dir;
      const c2y = ey;

      const color = effectiveGroupColorByChatId[l.fromChatId] || "var(--accent)";
      const d = `M ${sx} ${sy} C ${c1x} ${c1y} ${c2x} ${c2y} ${ex} ${ey}`;
      const fade = fadingLinkToChatIds.has(l.toChatId);

      // Arrow head at the end of the curve. We avoid SVG markers so each link can be colored.
      const tx = ex - c2x;
      const ty = ey - c2y;
      const mag = Math.hypot(tx, ty) || 1;
      const ux = tx / mag;
      const uy = ty / mag;
      const arrowLen = 10;
      const arrowW = 4.5;
      const bx = ex - ux * arrowLen;
      const by = ey - uy * arrowLen;
      const px = -uy;
      const py = ux;
      const a1 = `${Math.round(ex * dpr) / dpr},${Math.round(ey * dpr) / dpr}`;
      const a2 = `${Math.round((bx + px * arrowW) * dpr) / dpr},${Math.round(
        (by + py * arrowW) * dpr
      ) / dpr}`;
      const a3 = `${Math.round((bx - px * arrowW) * dpr) / dpr},${Math.round(
        (by - py * arrowW) * dpr
      ) / dpr}`;

      out.push({ id: l.id, d, color, fade, arrowPoints: `${a1} ${a2} ${a3}` });
    }
    return out;
  }, [links, noteById, vp.scale, dpr, hiddenChatIds, effectiveGroupColorByChatId, fadingLinkToChatIds]);

  const dock = (
    <div className="canvas-dock" onPointerDown={(e) => e.stopPropagation()}>
      <div className="canvas-dock-row">
        <button
          className="canvas-dock-btn"
          onClick={(e) => {
            e.preventDefault();
            e.stopPropagation();
            createAtCenter();
          }}
          type="button"
        >
          <Plus className="w-4 h-4" />
        </button>
        <button
          className={`canvas-dock-icon ${chatListOpen ? "active" : ""}`}
          type="button"
          aria-label={chatListOpen ? "Hide chats" : "Show chats"}
          title={chatListOpen ? "Hide chats" : "Show chats"}
          onClick={(e) => {
            e.preventDefault();
            e.stopPropagation();
            setChatListOpen((v) => !v);
          }}
        >
          <List className="w-4 h-4" />
        </button>
        <button
          className="canvas-dock-icon"
          type="button"
          aria-label="Settings"
          title="Settings"
          onClick={(e) => {
            e.preventDefault();
            e.stopPropagation();
            onOpenSettings();
          }}
        >
          <Settings className="w-4 h-4" />
        </button>
      </div>

      {chatListOpen ? (
        <div className="canvas-chatlist" role="menu" aria-label="Chats">
          {loadingSessions ? <div className="canvas-chatlist-muted">Syncing…</div> : null}
          {!loadingSessions && sessions.length === 0 ? (
            <div className="canvas-chatlist-muted">No chats yet</div>
          ) : null}
          {sessions.map((s) => (
            <div key={s.chat_id} className="canvas-chatlist-row">
              <button
                type="button"
                className={`canvas-chatlist-item ${s.chat_id === activeChatId ? "active" : ""}`}
                onClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  onFocusChat(s.chat_id);
                  onOpenCard(s.chat_id);
                  setChatListOpen(false);
                }}
                title={s.chat_id}
              >
                <span className="canvas-chatlist-item-text">{noteById.get(s.chat_id)?.title || s.title || s.chat_id}</span>
                <button
                  type="button"
                  className={`canvas-chatlist-del ${confirmDeleteChatId === s.chat_id ? "confirm" : ""}`}
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    onDeleteChat(s.chat_id);
                  }}
                  title={
                    confirmDeleteChatId === s.chat_id
                      ? "Click again to confirm delete"
                      : "Delete chat"
                  }
                  aria-label={`Delete chat ${s.chat_id}`}
                >
                  {confirmDeleteChatId === s.chat_id ? "Del" : <X className="w-3 h-3" />}
                </button>
              </button>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );

  return (
    <>
      <div
        ref={rootRef}
        className="canvas-root"
        onPointerDown={beginPan}
        onPointerMove={movePan}
        onPointerUp={endPan}
        onPointerCancel={endPan}
        onMouseMove={(e) => updateCursorFromClient(e.clientX, e.clientY)}
      >
      <div
        className="canvas-viewport"
        style={{
          // Note: we intentionally avoid `scale(...)` on the whole viewport because
          // WebViews can rasterize transformed layers and make text/borders look blurry.
          // Instead, notes + the board are laid out at scaled sizes/positions.
          transform: `translate(${renderX}px, ${renderY}px)`,
        }}
        onPointerDown={(e) => {
          // Reset confirm delete state when clicking on canvas background
          // Don't reset if clicking on a button or interactive element
          const target = e.target as HTMLElement;
          if (confirmDeleteChatId && !target.closest('button')) {
            onResetConfirmDelete();
          }
        }}
      >
        {board.w && board.h ? (
          <div
            className="canvas-board"
            style={{
              width: Math.round(board.w * vp.scale * dpr) / dpr,
              height: Math.round(board.h * vp.scale * dpr) / dpr,
              backgroundSize: `${28 * vp.scale}px ${28 * vp.scale}px`,
            }}
          />
        ) : null}
        {board.w && board.h && linkPaths.length ? (
          <svg
            className="canvas-links"
            width={Math.round(board.w * vp.scale * dpr) / dpr}
            height={Math.round(board.h * vp.scale * dpr) / dpr}
            aria-hidden="true"
          >
            {linkPaths.map((p) => (
              <g key={p.id}>
                <path
                  d={p.d}
                  className="canvas-link-path"
                  style={{ stroke: p.color, opacity: p.fade ? 0 : undefined }}
                />
                <polygon
                  points={p.arrowPoints}
                  className="canvas-link-arrow"
                  style={{ fill: p.color, opacity: p.fade ? 0 : undefined }}
                />
              </g>
            ))}
          </svg>
        ) : null}
        {visibleNotes.map((n) => {
          const noteTitle = typeof n.title === "string" ? n.title.trim() : "";
          const title = noteTitle || sessionById.get(n.chatId)?.title || n.chatId;
          const isActive = n.chatId === activeChatId;
          const locked = !!n.locked;
          const collapsed = !!n.collapsed;
          const stackCount = stackCountByChatId[n.chatId] || 0;
          const accentColor = effectiveGroupColorByChatId[n.chatId] || null;
          const confirmDelete = confirmDeleteChatId === n.chatId;
          const enrichedData = enrichedChatData.get(n.chatId);
          return (
            <ChatNote
              key={n.chatId}
              chatId={n.chatId}
              title={title}
              files={filesByChatId[n.chatId] || []}
              x={n.x}
              y={n.y}
              w={n.w}
              h={n.h}
              z={n.z}
              scale={vp.scale}
              dpr={dpr}
              active={isActive}
              locked={locked}
              collapsed={collapsed}
              stackCount={stackCount}
              stackAnim={stackAnimByChatId[n.chatId] || null}
              accentColor={accentColor}
              confirmDelete={confirmDelete}
              lastMessageAt={enrichedData?.lastMessageAt || null}
              lastMessageContent={enrichedData?.lastMessageContent || null}
              fileCount={enrichedData?.fileCount || 0}
              onFocus={() => onFocusChat(n.chatId)}
              onUpdate={(patch, opts) => applyUpdateNote(n.chatId, patch, opts)}
              onOpen={() => onOpenChat(n.chatId)}
              // Default card open shows split view (docs + chat).
              onOpenCard={() => onOpenCard(n.chatId)}
              onDelete={() => onDeleteChat(n.chatId)}
              onClearGroupColor={() => clearGroupColorWithDescendants(n.chatId, accentColor)}
            />
          );
        })}
      </div>
      </div>
      {dockVisible && typeof document !== "undefined" && document.body
        ? createPortal(dock, document.body)
        : null}
    </>
  );
}

function ChatNote({
  chatId,
  title,
  files,
  x,
  y,
  w,
  h,
  z,
  scale,
  dpr,
  active,
  locked,
  collapsed,
  stackCount,
  stackAnim,
  accentColor,
  confirmDelete,
  lastMessageAt,
  lastMessageContent,
  fileCount,
  onFocus,
  onOpen,
  onOpenCard,
  onDelete,
  onClearGroupColor,
  onUpdate,
}: {
  chatId: string;
  title: string;
  files: string[];
  x: number;
  y: number;
  w: number;
  h: number;
  z: number;
  scale: number;
  dpr: number;
  active: boolean;
  locked: boolean;
  collapsed: boolean;
  stackCount: number;
  stackAnim: null | {
    active: boolean;
    mode: "collapse" | "expand";
    dx: number;
    dy: number;
    scale: number;
    rot: number;
    z: number;
  };
  accentColor: string | null;
  confirmDelete: boolean;
  lastMessageAt: string | null;
  lastMessageContent: string | null;
  fileCount: number;
  onFocus: () => void;
  onOpen: () => void;
  onOpenCard: () => void;
  onDelete: () => void;
  onClearGroupColor: () => void;
  onUpdate: (patch: Partial<CanvasNote>, opts?: { clampToViewport?: boolean }) => void;
}) {
  const dragRef = useRef<{ startX: number; startY: number; startPx: number; startPy: number } | null>(
    null
  );
  const dragMovedRef = useRef(false);
  const resizeRef = useRef<{
    dir: ResizeDir;
    startClientX: number;
    startClientY: number;
    startX: number;
    startY: number;
    startW: number;
    startH: number;
  } | null>(null);

  type ResizeDir = "n" | "s" | "e" | "w" | "ne" | "nw" | "se" | "sw";
  const minW = 200;
  const minH = 120;
  const [isRenaming, setIsRenaming] = useState(false);
  const [titleDraft, setTitleDraft] = useState(title);
  const [colorPickerOpen, setColorPickerOpen] = useState(false);
  const renameInputRef = useRef<HTMLInputElement | null>(null);
  const colorPickerRef = useRef<HTMLDivElement | null>(null);
  const paletteButtonRef = useRef<HTMLButtonElement | null>(null);
  const justClosedColorPickerRef = useRef(false);

  useEffect(() => {
    if (isRenaming) return;
    setTitleDraft(title);
  }, [title, isRenaming]);

  useEffect(() => {
    if (!isRenaming) return;
    const t = window.setTimeout(() => {
      renameInputRef.current?.focus();
      renameInputRef.current?.select();
    }, 0);
    return () => window.clearTimeout(t);
  }, [isRenaming]);

  useEffect(() => {
    if (!colorPickerOpen) return;
    const onPointerDown = (e: PointerEvent) => {
      const t = e.target as Node | null;
      if (!t) return;
      // Don't close if clicking inside the color picker itself
      if (colorPickerRef.current && colorPickerRef.current.contains(t)) return;
      // Don't close if clicking the palette button - let it handle the toggle
      if (paletteButtonRef.current && paletteButtonRef.current.contains(t)) return;
      // Close the menu on any other click
      setColorPickerOpen(false);
      justClosedColorPickerRef.current = true;
      // Reset the flag after a short delay
      setTimeout(() => {
        justClosedColorPickerRef.current = false;
      }, 100);
    };
    window.addEventListener("pointerdown", onPointerDown, { capture: true });
    return () => window.removeEventListener("pointerdown", onPointerDown, { capture: true } as any);
  }, [colorPickerOpen]);

  function commitRename(nextTitle: string) {
    const trimmed = (nextTitle || "").trim();
    onUpdate({ title: trimmed ? trimmed : undefined });
    setIsRenaming(false);
  }

  function cancelRename() {
    setTitleDraft(title);
    setIsRenaming(false);
  }

  function bringToFront() {
    onFocus();
  }

  function beginDrag(e: React.PointerEvent) {
    if (locked || isRenaming) return;
    if (e.button !== 0) return;
    if (e.detail > 1) return;
    const t = e.target as HTMLElement | null;
    if (t && (t.closest("button") || t.closest("input"))) return;
    bringToFront();
    dragRef.current = { startX: e.clientX, startY: e.clientY, startPx: x, startPy: y };
    dragMovedRef.current = false;
  }

  function moveDrag(e: React.PointerEvent) {
    if (!dragRef.current) return;
    if (e.buttons !== 1) {
      dragRef.current = null;
      dragMovedRef.current = false;
      try {
        (e.currentTarget as HTMLElement).releasePointerCapture(e.pointerId);
      } catch {
        // ignore
      }
      return;
    }
    const dx = (e.clientX - dragRef.current.startX) / scale;
    const dy = (e.clientY - dragRef.current.startY) / scale;
    if (Math.abs(dx) + Math.abs(dy) > 2) {
      if (!dragMovedRef.current) {
        dragMovedRef.current = true;
        try {
          (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
        } catch {
          // ignore
        }
      }
      e.preventDefault();
    }
    onUpdate(
      { x: dragRef.current.startPx + dx, y: dragRef.current.startPy + dy },
      { clampToViewport: true }
    );
  }

  function endDrag(e: React.PointerEvent) {
    if (!dragRef.current) return;
    dragRef.current = null;
    // Reset so future clicks can open the card reliably after a drag.
    dragMovedRef.current = false;
    try {
      (e.currentTarget as HTMLElement).releasePointerCapture(e.pointerId);
    } catch {
      // ignore
    }
  }

  function beginResize(dir: ResizeDir) {
    return (e: React.PointerEvent) => {
      if (locked) return;
      if (e.button !== 0) return;
      e.preventDefault();
      e.stopPropagation();
      bringToFront();
      resizeRef.current = {
        dir,
        startClientX: e.clientX,
        startClientY: e.clientY,
        startX: x,
        startY: y,
        startW: w,
        startH: h,
      };

      const pointerId = e.pointerId;
      const onMove = (ev: PointerEvent) => {
        if (ev.pointerId !== pointerId) return;
        const r = resizeRef.current;
        if (!r) return;
        const dx = (ev.clientX - r.startClientX) / scale;
        const dy = (ev.clientY - r.startClientY) / scale;

        let nextX = r.startX;
        let nextY = r.startY;
        let nextW = r.startW;
        let nextH = r.startH;

        const hasW = r.dir.includes("w");
        const hasE = r.dir.includes("e");
        const hasN = r.dir.includes("n");
        const hasS = r.dir.includes("s");

        if (hasE) {
          nextW = Math.max(minW, r.startW + dx);
        }
        if (hasS) {
          nextH = Math.max(minH, r.startH + dy);
        }
        if (hasW) {
          const rawW = r.startW - dx;
          nextW = Math.max(minW, rawW);
          nextX = r.startX + (r.startW - nextW);
        }
        if (hasN) {
          const rawH = r.startH - dy;
          nextH = Math.max(minH, rawH);
          nextY = r.startY + (r.startH - nextH);
        }

        onUpdate({ x: nextX, y: nextY, w: nextW, h: nextH }, { clampToViewport: true });
      };
      const onUp = (ev: PointerEvent) => {
        if (ev.pointerId !== pointerId) return;
        resizeRef.current = null;
        window.removeEventListener("pointermove", onMove);
        window.removeEventListener("pointerup", onUp);
        window.removeEventListener("pointercancel", onUp);
      };
      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", onUp);
      window.addEventListener("pointercancel", onUp);
    };
  }

  return (
    <div
      className={`canvas-note ${active ? "active" : ""} ${locked ? "locked" : ""} ${
        collapsed ? "collapsed" : ""
      } ${stackAnim ? "stack-anim" : ""}`}
      style={{
        transform: `translate(${Math.round(x * scale * dpr) / dpr}px, ${Math.round(
          y * scale * dpr
        ) / dpr}px) translate(${Math.round(((stackAnim?.active ? stackAnim.dx : 0) || 0) * scale * dpr) / dpr}px, ${Math.round(
          ((stackAnim?.active ? stackAnim.dy : 0) || 0) * scale * dpr
        ) / dpr}px) scale(${stackAnim?.active ? stackAnim.scale : 1}) rotate(${stackAnim?.active ? stackAnim.rot : 0}deg)`,
        width: `${Math.round(w * scale * dpr) / dpr}px`,
        height: `${Math.round(h * scale * dpr) / dpr}px`,
        zIndex: stackAnim && stackAnim.active ? stackAnim.z : z,
        opacity: stackAnim ? (stackAnim.active ? 0 : 1) : 1,
        pointerEvents: stackAnim ? "none" : undefined,
        ...(accentColor
          ? {
              ["--card-border" as any]: rgbaFromHex(accentColor, 0.55) ?? accentColor,
              ["--card-ring" as any]: rgbaFromHex(accentColor, 0.78) ?? accentColor,
              ["--card-stack" as any]: rgbaFromHex(accentColor, 0.48) ?? accentColor,
            }
          : {}),
      }}
      onPointerDown={() => bringToFront()}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onOpen();
        }
      }}
      tabIndex={0}
      role="group"
      aria-label={`Card ${title}`}
    >
      {collapsed && stackCount > 0 ? (
        <>
          <div className="canvas-note-stack canvas-note-stack-3" aria-hidden="true" />
          <div className="canvas-note-stack canvas-note-stack-2" aria-hidden="true" />
          <div className="canvas-note-stack canvas-note-stack-1" aria-hidden="true" />
        </>
      ) : null}
      <div className="canvas-note-inner">
        <div
          className="canvas-note-header"
          onPointerDown={beginDrag}
          onPointerMove={moveDrag}
          onPointerUp={endDrag}
          onPointerCancel={endDrag}
        >
          {isRenaming ? (
            <input
              ref={renameInputRef}
              className="canvas-note-title-input"
              value={titleDraft}
              onChange={(e) => setTitleDraft(e.target.value)}
              onPointerDown={(e) => e.stopPropagation()}
              onClick={(e) => e.stopPropagation()}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  e.stopPropagation();
                  commitRename(titleDraft);
                } else if (e.key === "Escape") {
                  e.preventDefault();
                  e.stopPropagation();
                  cancelRename();
                }
              }}
              onBlur={() => commitRename(titleDraft)}
              aria-label="Rename card"
            />
          ) : (
            <>
              <div
                className="canvas-note-title"
                title={title}
                aria-label={chatId}
                onDoubleClick={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  setTitleDraft(title);
                  setIsRenaming(true);
                }}
              >
                {title}
              </div>
              <div className="canvas-note-header-right">
                {collapsed && stackCount > 0 ? (
                  <button
                    type="button"
                    className="canvas-note-stack-badge"
                    title={`Expand ${stackCount} hidden card${stackCount === 1 ? "" : "s"}`}
                    onClick={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      onUpdate({ collapsed: false });
                    }}
                  >
                    +{stackCount}
                  </button>
                ) : null}
                {!collapsed ? (
                  <>
                    <button
                      type="button"
                      className="canvas-note-lock-btn"
                      title={locked ? "Unlock card" : "Lock card"}
                      aria-label={locked ? "Unlock card" : "Lock card"}
                      onClick={(e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        onUpdate({ locked: !locked });
                      }}
                    >
                      {locked ? <Lock className="w-3 h-3" /> : <Unlock className="w-3 h-3" />}
                    </button>
                    <button
                      ref={paletteButtonRef}
                      type="button"
                      className="canvas-note-lock-btn"
                      title="Card color"
                      aria-label="Card color"
                      data-palette-button="true"
                      onClick={(e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        setColorPickerOpen(!colorPickerOpen);
                      }}
                    >
                      <Palette className="w-3 h-3" />
                    </button>
                    <button
                      type="button"
                      className={`canvas-note-del ${confirmDelete ? "confirm" : ""}`}
                      title={confirmDelete ? "Click again to confirm delete" : "Delete card"}
                      aria-label={`Delete card ${title}`}
                      onClick={(e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        onDelete();
                      }}
                    >
                      {confirmDelete ? "Del" : <X className="w-3 h-3" />}
                    </button>
                  </>
                ) : null}
              </div>
            </>
          )}
        </div>
        <div
          className="canvas-note-body"
          onPointerDown={() => bringToFront()}
          onClick={(e) => {
            // Don't open on click if the user was dragging/resizing.
            if (dragMovedRef.current) return;
            if (resizeRef.current) return;
            // Ignore clicks on resize handles.
            const t = e.target as HTMLElement | null;
            if (t && t.closest(".canvas-note-handle")) return;
            // Don't open if we just closed the color picker
            if (justClosedColorPickerRef.current) return;
            e.preventDefault();
            e.stopPropagation();
            bringToFront();
            // If color picker is open, just close it and don't open the card
            if (colorPickerOpen) {
              setColorPickerOpen(false);
              justClosedColorPickerRef.current = true;
              setTimeout(() => {
                justClosedColorPickerRef.current = false;
              }, 100);
              return;
            }
            if (collapsed && stackCount > 0) {
              onUpdate({ collapsed: false });
              return;
            }
            onOpenCard();
          }}
          role="button"
          aria-label={`Open card ${title}`}
          tabIndex={-1}
        >
          <div className="canvas-note-preview">
            <div className="canvas-note-preview-meta">
              {lastMessageAt ? (
                <div className="canvas-note-preview-time" title="Last activity">
                  {formatRelativeTime(lastMessageAt)}
                </div>
              ) : null}
              {fileCount > 0 ? (
                <div className="canvas-note-preview-files" title={`${fileCount} file${fileCount === 1 ? "" : "s"}`}>
                  <Paperclip className="w-4 h-4" /> {fileCount}
                </div>
              ) : null}
            </div>
            {lastMessageContent ? (
              <div className="canvas-note-preview-message" title={lastMessageContent}>
                {truncateText(lastMessageContent, 80)}
              </div>
            ) : null}
            {files.length > 0 && !lastMessageContent ? (
              <div className="canvas-note-preview-files-list">
                {files.slice(0, 2).map((name, idx) => (
                  <div key={`${idx}-${name}`} className="canvas-note-preview-file" title={name}>
                    {name}
                  </div>
                ))}
                {files.length > 2 ? (
                  <div className="canvas-note-preview-sub">+{files.length - 2} more</div>
                ) : null}
              </div>
            ) : null}
            {!files.length && !lastMessageContent ? (
              <div className="canvas-note-preview-sub">New chat</div>
            ) : null}
          </div>
        </div>
      </div>
      {!locked ? (
        <>
          <div
            className="canvas-note-handle canvas-note-handle-n"
            onPointerDown={beginResize("n")}
            role="presentation"
          />
          <div
            className="canvas-note-handle canvas-note-handle-s"
            onPointerDown={beginResize("s")}
            role="presentation"
          />
          <div
            className="canvas-note-handle canvas-note-handle-e"
            onPointerDown={beginResize("e")}
            role="presentation"
          />
          <div
            className="canvas-note-handle canvas-note-handle-w"
            onPointerDown={beginResize("w")}
            role="presentation"
          />
          <div
            className="canvas-note-handle canvas-note-handle-ne"
            onPointerDown={beginResize("ne")}
            role="presentation"
          />
          <div
            className="canvas-note-handle canvas-note-handle-nw"
            onPointerDown={beginResize("nw")}
            role="presentation"
          />
          <div
            className="canvas-note-handle canvas-note-handle-se"
            onPointerDown={beginResize("se")}
            role="presentation"
          />
          <div
            className="canvas-note-handle canvas-note-handle-sw"
            onPointerDown={beginResize("sw")}
            role="presentation"
          />
        </>
      ) : null}
      {colorPickerOpen ? (
        <div
          ref={colorPickerRef}
          className="canvas-note-color-picker"
          style={{
            position: "absolute",
            top: "34px",
            right: "8px",
            zIndex: 1000,
          }}
          onPointerDown={(e) => e.stopPropagation()}
        >
          <div className="canvas-card-color-row">
            {GROUP_COLORS.map((c) => {
              const selected = accentColor === c.value;
              return (
                <button
                  key={c.id}
                  type="button"
                  className={`canvas-card-color ${selected ? "active" : ""}`}
                  title={c.label}
                  aria-label={c.label}
                  onClick={() => {
                    onUpdate({ groupColor: c.value });
                    setColorPickerOpen(false);
                  }}
                  style={{ backgroundColor: c.value }}
                />
              );
            })}
            <button
              type="button"
              className="canvas-card-color-clear"
              title="No color"
              aria-label="No color"
              onClick={() => {
                onClearGroupColor();
                setColorPickerOpen(false);
              }}
            >
              ×
            </button>
          </div>
        </div>
      ) : null}
    </div>
  );
}
