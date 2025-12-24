import { useEffect, useMemo, useRef, useState } from "react";
import type { ChatSummary } from "../state/useSessions";

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
  onCreateChatAt: (pos: { x: number; y: number }) => void;
  onOpenCard: (chatId: string) => void;
  onDeleteChat: (chatId: string) => void;
  confirmDeleteChatId: string | null;
  loadingSessions: boolean;
};

const MIN_SCALE = 0.35;
const MAX_SCALE = 1.75;
const PAN_ENABLE_SCALE = 1.01;
const ZOOM_SPEED = 0.0015;
const BOARD_MULT = 2.6;
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
  onDeleteChat,
  confirmDeleteChatId,
  loadingSessions,
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
  const panRef = useRef<{ startX: number; startY: number; startVpX: number; startVpY: number } | null>(
    null
  );

  const sessionById = useMemo(() => {
    const map = new Map<string, ChatSummary>();
    for (const s of sessions) map.set(s.chat_id, s);
    return map;
  }, [sessions]);

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

  function applyUpdateNote(chatId: string, patch: Partial<CanvasNote>) {
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
    const clamped = clampNoteToBoard(merged);
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
    // Only allow click-drag panning when zoomed in.
    if (vpRef.current.scale <= PAN_ENABLE_SCALE) return;
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
    if (vpRef.current.scale <= PAN_ENABLE_SCALE) return;
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
    onCreateChatAt({ x: worldX, y: worldY });
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

  return (
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
              width: board.w * vp.scale,
              height: board.h * vp.scale,
              borderRadius: `${18 * vp.scale}px`,
              backgroundSize: `${28 * vp.scale}px ${28 * vp.scale}px`,
            }}
          />
        ) : null}
      {notes.map((n) => {
          const title = sessionById.get(n.chatId)?.title || n.title || n.chatId;
          const isActive = n.chatId === activeChatId;
          return (
            <ChatNote
              key={n.chatId}
              chatId={n.chatId}
              title={title}
              x={n.x}
              y={n.y}
              w={n.w}
              h={n.h}
              z={n.z}
              scale={vp.scale}
              dpr={dpr}
              active={isActive}
              onFocus={() => onFocusChat(n.chatId)}
              onUpdate={(patch) => applyUpdateNote(n.chatId, patch)}
              onOpen={() => onOpenChat(n.chatId)}
              // Default card open shows split view (docs + chat).
              onOpenCard={() => onOpenCard(n.chatId)}
            />
          );
        })}
      </div>

      <div
        className="canvas-dock"
        onPointerDown={(e) => e.stopPropagation()}
        onPointerMove={(e) => e.stopPropagation()}
        onPointerUp={(e) => e.stopPropagation()}
        onWheel={(e) => e.stopPropagation()}
      >
        <div className="canvas-dock-row">
          <button className="canvas-dock-btn" onClick={createAtCenter} type="button">
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
        </div>

        {chatListOpen ? (
          <div
            className="canvas-chatlist"
            role="menu"
            aria-label="Chats"
            onPointerDown={(e) => e.stopPropagation()}
          >
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
                    setChatListOpen(false);
                  }}
                  title={s.chat_id}
                >
                  {s.title || s.chat_id}
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
    </div>
  );
}

function ChatNote({
  chatId,
  title,
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
  onUpdate: (patch: Partial<CanvasNote>) => void;
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

  function bringToFront() {
    onFocus();
  }

  function beginDrag(e: React.PointerEvent) {
    if (e.button !== 0) return;
    e.preventDefault();
    bringToFront();
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    dragRef.current = { startX: e.clientX, startY: e.clientY, startPx: x, startPy: y };
    dragMovedRef.current = false;
  }

  function moveDrag(e: React.PointerEvent) {
    if (!dragRef.current) return;
    const dx = (e.clientX - dragRef.current.startX) / scale;
    const dy = (e.clientY - dragRef.current.startY) / scale;
    if (Math.abs(dx) + Math.abs(dy) > 2) dragMovedRef.current = true;
    onUpdate({ x: dragRef.current.startPx + dx, y: dragRef.current.startPy + dy });
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

        onUpdate({ x: nextX, y: nextY, w: nextW, h: nextH });
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
          <div className="canvas-note-title" title={title} aria-label={chatId}>
            {title}
          </div>
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
            <div className="canvas-note-preview-line">Documents</div>
            <div className="canvas-note-preview-sub">Drop files here</div>
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
