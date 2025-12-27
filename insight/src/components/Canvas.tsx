import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { listen } from "@tauri-apps/api/event";
import type { ChatSummary } from "../state/useSessions";
import { engine } from "../api/engine";

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
  x: number;
  y: number;
  w: number;
  h: number;
  z: number;
  layout?: CardLayout;
};

type Viewport = { x: number; y: number; scale: number };
type Board = { w: number; h: number };

type Props = {
  sessions: ChatSummary[];
  activeChatId: string | null;
  notes: CanvasNote[];
  onFocusChat: (chatId: string) => void;
  onUpdateNote: (chatId: string, patch: Partial<CanvasNote>) => void;
  onOpenChat: (chatId: string) => void;
  onCreateChatAt: (pos: { x: number; y: number }) => string;
  onOpenCard: (chatId: string) => void;
  onOpenSettings: () => void;
  onDeleteChat: (chatId: string) => void;
  confirmDeleteChatId: string | null;
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

const VIEWPORT_STORAGE_KEY = "insight.canvas.viewport.v1";

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
  onFocusChat,
  onUpdateNote,
  onOpenChat,
  onCreateChatAt,
  onOpenCard,
  onOpenSettings,
  onDeleteChat,
  confirmDeleteChatId,
  loadingSessions,
  dockVisible = true,
}: Props) {
  const rootRef = useRef<HTMLDivElement | null>(null);
  const [vp, setVp] = useState<Viewport>({ x: 0, y: 0, scale: 1 });
  const vpRef = useRef(vp);
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

  const noteById = useMemo(() => {
    const map = new Map<string, CanvasNote>();
    for (const n of notes) map.set(n.chatId, n);
    return map;
  }, [notes]);

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
          + Card
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
          ☰
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
          ⚙
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
                {noteById.get(s.chat_id)?.title || s.title || s.chat_id}
              </button>
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
                {confirmDeleteChatId === s.chat_id ? "Del" : "×"}
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
        {notes.map((n) => {
          const noteTitle = typeof n.title === "string" ? n.title.trim() : "";
          const title = noteTitle || sessionById.get(n.chatId)?.title || n.chatId;
          const isActive = n.chatId === activeChatId;
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
              onFocus={() => onFocusChat(n.chatId)}
              onUpdate={(patch, opts) => applyUpdateNote(n.chatId, patch, opts)}
              onOpen={() => onOpenChat(n.chatId)}
              // Default card open shows split view (docs + chat).
              onOpenCard={() => onOpenCard(n.chatId)}
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
  onFocus,
  onOpen,
  onOpenCard,
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
  onFocus: () => void;
  onOpen: () => void;
  onOpenCard: () => void;
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
  const renameInputRef = useRef<HTMLInputElement | null>(null);

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
    if (e.button !== 0) return;
    const t = e.target as HTMLElement | null;
    if (t && (t.closest("button") || t.closest("input"))) return;
    bringToFront();
    dragRef.current = { startX: e.clientX, startY: e.clientY, startPx: x, startPy: y };
    dragMovedRef.current = false;
  }

  function moveDrag(e: React.PointerEvent) {
    if (!dragRef.current) return;
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
      className={`canvas-note ${active ? "active" : ""}`}
      style={{
        transform: `translate(${Math.round(x * scale * dpr) / dpr}px, ${Math.round(
          y * scale * dpr
        ) / dpr}px)`,
        width: `${Math.round(w * scale * dpr) / dpr}px`,
        height: `${Math.round(h * scale * dpr) / dpr}px`,
        zIndex: z,
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
            e.preventDefault();
            e.stopPropagation();
            bringToFront();
            onOpenCard();
          }}
          role="button"
          aria-label={`Open card ${title}`}
          tabIndex={-1}
        >
          <div className="canvas-note-preview">
            {files.length ? (
              <>
                {files.slice(0, 3).map((name, idx) => (
                  <div key={`${idx}-${name}`} className="canvas-note-preview-file" title={name}>
                    {name}
                  </div>
                ))}
                {files.length > 3 ? (
                  <div className="canvas-note-preview-sub">+{files.length - 3} more</div>
                ) : null}
              </>
            ) : (
              <>
                <div className="canvas-note-preview-sub">No files</div>
              </>
            )}
          </div>
        </div>
      </div>
      <div className="canvas-note-handle canvas-note-handle-n" onPointerDown={beginResize("n")} role="presentation" />
      <div className="canvas-note-handle canvas-note-handle-s" onPointerDown={beginResize("s")} role="presentation" />
      <div className="canvas-note-handle canvas-note-handle-e" onPointerDown={beginResize("e")} role="presentation" />
      <div className="canvas-note-handle canvas-note-handle-w" onPointerDown={beginResize("w")} role="presentation" />
      <div className="canvas-note-handle canvas-note-handle-ne" onPointerDown={beginResize("ne")} role="presentation" />
      <div className="canvas-note-handle canvas-note-handle-nw" onPointerDown={beginResize("nw")} role="presentation" />
      <div className="canvas-note-handle canvas-note-handle-se" onPointerDown={beginResize("se")} role="presentation" />
      <div className="canvas-note-handle canvas-note-handle-sw" onPointerDown={beginResize("sw")} role="presentation" />
    </div>
  );
}
