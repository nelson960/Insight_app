import React, { useEffect, useMemo, useRef, useState } from "react";
import { engine } from "../api/engine";
import { listen } from "@tauri-apps/api/event";

type ChatFile = {
  file_id: string;
  filename: string;
  mime: string;
  size_bytes: number;
  status?: string;
  pages?: number | null;
};

type ExtractedBlock = {
  kind: string;
  text: string;
  metadata?: any;
};

type ExtractedView = {
  file_id: string;
  filename: string;
  mime: string;
  status: string;
  pages: number | null;
  blocks: ExtractedBlock[];
  plain_text: string;
};

type Props = {
  chatId: string;
  refreshSeq?: number;
  activeFileId?: string | null;
  onActiveFileIdChange?: (fileId: string | null) => void;
  onSelectionChange?: (sel: { file_id: string; text: string } | null) => void;
  onAskSelection?: () => void;
};

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

function fallbackBlocksFromPlainText(text: string): ExtractedBlock[] {
  const parts = (text || "")
    .split("\n\n")
    .map((p) => p.trim())
    .filter(Boolean);
  return parts.map((p) => ({ kind: "paragraph", text: p }));
}

export function DocumentsPane({
  chatId,
  refreshSeq = 0,
  activeFileId,
  onActiveFileIdChange,
  onSelectionChange,
  onAskSelection,
}: Props) {
  const [files, setFiles] = useState<ChatFile[]>([]);
  const [pendingUploads, setPendingUploads] = useState<string[]>([]);
  const [internalActiveFileId, setInternalActiveFileId] = useState<string | null>(null);
  const [view, setView] = useState<ExtractedView | null>(null);
  const [isLoadingFiles, setIsLoadingFiles] = useState(false);
  const [isLoadingView, setIsLoadingView] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const bodyRef = useRef<HTMLDivElement | null>(null);
  const [selectionText, setSelectionText] = useState<string>("");
  const [selectionPos, setSelectionPos] = useState<{ x: number; y: number } | null>(null);
  const [pendingSelection, setPendingSelection] = useState<{ file_id: string; text: string } | null>(null);
  const popoverTimerRef = useRef<number | null>(null);
  const [isDropHover, setIsDropHover] = useState(false);
  const dropCounterRef = useRef(0);
  const ingestInFlightRef = useRef(false);
  const activeFileIdRef = useRef<string | null>(null);
  const latestFilesChangedRef = useRef<string | null>(null);

  const isControlled = typeof activeFileId !== "undefined";
  const effectiveActiveFileId = isControlled ? (activeFileId ?? null) : internalActiveFileId;

  function setActiveFileId(nextId: string | null) {
    if (onActiveFileIdChange) onActiveFileIdChange(nextId);
    if (!isControlled) setInternalActiveFileId(nextId);
  }

  useEffect(() => {
    setError(null);
    setFiles([]);
    setPendingUploads([]);
    setView(null);
    setIsLoadingFiles(false);
    setIsLoadingView(false);
    setSelectionText("");
    setSelectionPos(null);
    setPendingSelection(null);
    setIsDropHover(false);
    dropCounterRef.current = 0;
    ingestInFlightRef.current = false;
    if (!isControlled) setInternalActiveFileId(null);
  }, [chatId]);

  useEffect(() => {
    function onPending(e: Event) {
      const ce = e as CustomEvent;
      const targetChatId = ce?.detail?.chatId;
      if (typeof targetChatId !== "string" || targetChatId !== chatId) return;
      const names = Array.isArray(ce?.detail?.names)
        ? (ce.detail.names as any[]).filter((x) => typeof x === "string" && x)
        : [];
      if (!names.length) return;
      setPendingUploads(names.slice(0, 6));
    }
    window.addEventListener("insight:docs-pending", onPending as any);
    return () => window.removeEventListener("insight:docs-pending", onPending as any);
  }, [chatId]);

  useEffect(() => {
    activeFileIdRef.current = effectiveActiveFileId;
  }, [effectiveActiveFileId]);

  const dropEnabled = !files.length && !pendingUploads.length && !ingestInFlightRef.current;

  async function reloadFiles() {
    const res = await engine<{ files: ChatFile[] }>(
      `/files/chat/${encodeURIComponent(chatId)}`,
      undefined,
      "GET"
    );
    if (!res.ok) {
      setError(res.error || `Failed to load files (${res.status})`);
      return [];
    }
    const next = Array.isArray(res.data?.files) ? res.data.files : [];
    setFiles(next);
    if (next.length) {
      setPendingUploads((prev) => (prev.length ? [] : prev));
    }
    return next;
  }

  // Backend-driven, out-of-band events (no polling):
  // - files-changed: file registered/attached to this chat
  // - file-text-ready: extracted view persisted, safe to render
  // - file-status: ingestion progress update
  useEffect(() => {
    let cancelled = false;
    const unlistenFns: Array<() => void> = [];

    function shouldHandle(payload: any): boolean {
      return typeof payload?.chat_id === "string" && payload.chat_id === chatId;
    }

    function safeReloadFilesAndMaybeSelect(payload: any, reason?: string) {
      reloadFiles()
        .then((next) => {
          const fidFromEvent =
            typeof payload?.file_id === "string" ? payload.file_id : null;

          // When a new file is attached, we want to immediately switch the active
          // document to that file (even if another file was already selected).
          // Do this after the list reload so the tab definitely exists.
          if (reason === "files-changed" && fidFromEvent) {
            // Multiple files can be registered in quick succession (multi-attach).
            // Only apply the selection for the latest files-changed event we've seen,
            // otherwise slower reloads can "snap" the UI back to an earlier file.
            if (
              latestFilesChangedRef.current === fidFromEvent &&
              next.some((f) => f.file_id === fidFromEvent)
            ) {
              setActiveFileId(fidFromEvent);
              return;
            }
          }

          const active = activeFileIdRef.current;
          const activeStillExists = active ? next.some((f) => f.file_id === active) : false;
          if (active && !activeStillExists) {
            setActiveFileId(next.length ? next[0].file_id : null);
            return;
          }
          if (!active && next.length) {
            const preferred =
              typeof payload?.file_id === "string"
                ? next.find((f) => f.file_id === payload.file_id)?.file_id
                : null;
            setActiveFileId(preferred || next[0].file_id);
          }
        })
        .catch(() => {
          // ignore; UI can recover on next event/user action
        });
    }

    const eventNames = ["files-changed", "file-text-ready", "file-status"];
    for (const name of eventNames) {
      listen<any>(name, (event) => {
        if (cancelled) return;
        const payload = event?.payload as any;
        if (!shouldHandle(payload)) return;

        if (name === "files-changed") {
          const fid = typeof payload?.file_id === "string" ? payload.file_id : null;
          if (fid) latestFilesChangedRef.current = fid;
        }

        // Always refresh the file list promptly.
        safeReloadFilesAndMaybeSelect(payload, name);

        // If the currently selected file finished extracting, refresh the view.
        if (name === "file-text-ready") {
          const fid = typeof payload?.file_id === "string" ? payload.file_id : null;
          const activeFid = activeFileIdRef.current;
          if (fid) {
            // If we didn't have an active file yet, auto-select this one.
            if (!activeFid) {
              setActiveFileId(fid);
            } else if (fid === activeFid) {
              loadView(fid).catch(() => {
                // ignore
              });
            }
          }
        }
      })
        .then((fn) => {
          if (!cancelled) unlistenFns.push(fn);
        })
        .catch(() => {
          // ignore (non-Tauri build / event not supported)
        });
    }

    return () => {
      cancelled = true;
      for (const fn of unlistenFns) fn();
    };
  }, [chatId]);

  useEffect(() => {
    let cancelled = false;
    setError(null);
    setIsLoadingFiles(true);

    async function load() {
      if (cancelled) return;
      const next = await reloadFiles();
      setIsLoadingFiles(false);

      const current = isControlled ? (activeFileId ?? null) : internalActiveFileId;
      const currentStillExists = current ? next.some((f) => f.file_id === current) : false;
      if (!currentStillExists) setActiveFileId(next.length ? next[0].file_id : null);
      if (!current && next.length) setActiveFileId(next[0].file_id);
    }

    load().catch((e) => {
      if (!cancelled) {
        setIsLoadingFiles(false);
        setError(e?.message ?? String(e));
      }
    });

    return () => {
      cancelled = true;
    };
  }, [chatId, refreshSeq]);

  async function ingestDroppedPaths(paths: string[]) {
    if (!paths.length) return;
    if (ingestInFlightRef.current) return;
    ingestInFlightRef.current = true;
    setPendingUploads(
      paths
        .map((p) => String(p).split(/[\\/]/g).pop() || String(p))
        .filter(Boolean)
        .slice(0, 6)
    );
    setError(null);
    setIsLoadingFiles(true);
    setIsDropHover(false);
    dropCounterRef.current = 0;
    try {
      const res = await engine<{ files: { file_id: string; filename: string; job_id: string }[] }>(
        "/files/ingest_path",
        { chat_id: chatId, paths },
        "POST"
      );
      setIsLoadingFiles(false);
      if (!res.ok) {
        setError(res.error || `Failed to ingest files (${res.status})`);
        return;
      }
      // Refresh list; newly-ingested files may still be processing but should appear immediately.
      const next = await reloadFiles();
      const first = Array.isArray(res.data?.files) && res.data.files.length ? res.data.files[0] : null;
      if (first?.file_id) setActiveFileId(first.file_id);
      else if (next.length) setActiveFileId(next[0].file_id);
    } finally {
      ingestInFlightRef.current = false;
      setIsLoadingFiles(false);
    }
  }

  // Tauri-native file drop (gives us absolute paths). Only enabled when the pane is empty.
  useEffect(() => {
    if (!dropEnabled) return;
    const unlistenFns: Array<() => void> = [];
    let cancelled = false;
    let lastDropKey = "";
    let lastDropAt = 0;

    function extractPaths(payload: any): string[] {
      const paths = Array.isArray(payload)
        ? payload
        : Array.isArray(payload?.paths)
          ? payload.paths
          : [];
      return paths.filter((p: any) => typeof p === "string" && p);
    }

    function maybeHandleDrop(paths: string[]) {
      const cleaned = paths.filter(Boolean);
      if (!cleaned.length) return;
      const now = Date.now();
      const key = cleaned[0] || "";
      if (key && key === lastDropKey && now - lastDropAt < 800) return;
      lastDropKey = key;
      lastDropAt = now;
      ingestDroppedPaths(cleaned).catch((e) => setError(e?.message ?? String(e)));
    }

    const eventNames = ["tauri://file-drop", "tauri://drag-drop"];
    for (const name of eventNames) {
      listen<any>(name, (event) => {
        if (cancelled) return;
        const payload = event?.payload as any;
        const paths = extractPaths(payload);
        maybeHandleDrop(paths);
      })
        .then((fn) => {
          if (!cancelled) unlistenFns.push(fn);
        })
        .catch(() => {
          // ignore (non-Tauri build / event not supported)
        });
    }
    return () => {
      cancelled = true;
      for (const fn of unlistenFns) fn();
    };
  }, [chatId, dropEnabled]);

  async function loadView(fileId: string) {
    setError(null);
    setIsLoadingView(true);
    setView(null);
    const res = await engine<ExtractedView>(`/files/extracted/${encodeURIComponent(fileId)}`, undefined, "GET");
    setIsLoadingView(false);
    if (!res.ok) {
      setError(res.error || `Failed to load extracted view (${res.status})`);
      return;
    }
    setView(res.data as any);
  }

  useEffect(() => {
    let cancelled = false;
    const id = effectiveActiveFileId;
    if (!id) {
      setView(null);
      return;
    }
    (async () => {
      try {
        setError(null);
        setIsLoadingView(true);
        setView(null);
        const res = await engine<ExtractedView>(`/files/extracted/${encodeURIComponent(id)}`, undefined, "GET");
        if (cancelled) return;
        setIsLoadingView(false);
        if (!res.ok) {
          setError(res.error || `Failed to load extracted view (${res.status})`);
          return;
        }
        setView(res.data as any);
      } catch (e: any) {
        if (!cancelled) {
          setIsLoadingView(false);
          setError(e?.message ?? String(e));
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [effectiveActiveFileId]);

  function readSelectionFromWindow(opts?: { showPopover?: boolean }) {
    if (!effectiveActiveFileId) return;
    const sel = window.getSelection?.();
    const txt = (sel && typeof sel.toString === "function" ? sel.toString() : "") || "";
    const cleaned = txt.replace(/\s+/g, " ").trim();
    if (!cleaned) {
      setSelectionText("");
      setSelectionPos(null);
      setPendingSelection(null);
      return;
    }
    // Only accept selection that happens inside our document viewer.
    try {
      const anchorNode = sel?.anchorNode;
      const focusNode = sel?.focusNode;
      const body = bodyRef.current;
      if (body) {
        const anchorOk = anchorNode ? body.contains(anchorNode) : false;
        const focusOk = focusNode ? body.contains(focusNode) : false;
        if (!anchorOk && !focusOk) {
          setSelectionText("");
          setSelectionPos(null);
          setPendingSelection(null);
          return;
        }
      }
    } catch {
      // ignore
    }
    const capped = cleaned.length > 2000 ? cleaned.slice(0, 2000) + "…" : cleaned;
    setSelectionText(capped);
    setPendingSelection({ file_id: effectiveActiveFileId, text: capped });
    try {
      if (!opts?.showPopover) return;
      if (!sel || sel.rangeCount === 0) return;
      const range = sel.getRangeAt(0);
      const rect = range.getBoundingClientRect();
      const body = bodyRef.current;
      if (!body) return;
      const bodyRect = body.getBoundingClientRect();
      // Anchor to selection center, show popover above it.
      const xRaw = rect.left - bodyRect.left + body.scrollLeft + rect.width / 2;
      const minX = body.scrollLeft + 28;
      const maxX = body.scrollLeft + body.clientWidth - 28;
      const x = Math.max(minX, Math.min(maxX, xRaw));

      const yRaw = rect.top - bodyRect.top + body.scrollTop - 48;
      const minY = body.scrollTop + 8;
      const y = Math.max(minY, yRaw);
      // IMPORTANT: Delay popover render to the next tick so the mouseup/click that
      // ended the selection can't accidentally click the newly-rendered Ask button.
      if (popoverTimerRef.current != null) {
        window.clearTimeout(popoverTimerRef.current);
      }
      popoverTimerRef.current = window.setTimeout(() => {
        popoverTimerRef.current = null;
        setSelectionPos({ x: Math.max(8, x), y: Math.max(8, y) });
      }, 0);
    } catch {
      // ignore
    }
  }

  useEffect(() => {
    // More reliable than relying on mouseup within the element:
    // selection often ends outside the viewer, especially with drag-to-select.
    function onSelectionChangeEvent() {
      readSelectionFromWindow({ showPopover: false });
    }
    document.addEventListener("selectionchange", onSelectionChangeEvent);
    return () => document.removeEventListener("selectionchange", onSelectionChangeEvent);
  }, [effectiveActiveFileId]);

  function commitSelectionToChat() {
    if (!pendingSelection) return;
    if (onSelectionChange) onSelectionChange(pendingSelection);
    onAskSelection?.();
    setSelectionText("");
    setPendingSelection(null);
    setSelectionPos(null);
  }

  const activeFile = useMemo(
    () => files.find((f) => f.file_id === effectiveActiveFileId) || null,
    [files, effectiveActiveFileId]
  );

  const blocks = useMemo(() => {
    const b = Array.isArray(view?.blocks) ? view!.blocks : [];
    if (b.length) return b;
    if (view?.plain_text) return fallbackBlocksFromPlainText(view.plain_text);
    return [];
  }, [view]);

  function renderBlock(block: ExtractedBlock, idx: number) {
    const kind = (block.kind || "paragraph").toLowerCase();
    if (kind === "page_break") {
      const label = block.text || `Page ${block?.metadata?.page ?? ""}`.trim();
      return (
        <div key={idx} className="docs-block docs-block-page">
          <div className="docs-block-page-line" />
          <div className="docs-block-page-label">{label}</div>
          <div className="docs-block-page-line" />
        </div>
      );
    }

    if (kind === "heading") {
      const level = Number(block?.metadata?.level || 2);
      const cls = `docs-block docs-block-heading h${Math.min(6, Math.max(1, level))}`;
      return (
        <div key={idx} className={cls}>
          {block.text}
        </div>
      );
    }

    if (kind === "list") {
      const items = Array.isArray(block?.metadata?.items)
        ? block.metadata.items
        : (block.text || "")
            .split("\n")
            .map((x: string) => x.trim())
            .filter(Boolean);
      return (
        <ul key={idx} className="docs-block docs-block-list">
          {items.map((it: string, i: number) => (
            <li key={i}>{it}</li>
          ))}
        </ul>
      );
    }

    if (kind === "code") {
      return (
        <pre key={idx} className="docs-block docs-block-code">
          {block.text}
        </pre>
      );
    }

    return (
      <p key={idx} className="docs-block docs-block-paragraph">
        {block.text}
      </p>
    );
  }

  return (
    <div className="docs-pane">
      <div className="docs-pane-body">
        <div className="docs-tabs" role="tablist" aria-label="Files">
          {files.map((f) => {
            const active = f.file_id === effectiveActiveFileId;
            return (
              <button
                key={f.file_id}
                className={`docs-tab ${active ? "active" : ""}`}
                type="button"
                role="tab"
                aria-selected={active}
                title={`${f.filename} · ${formatBytes(Number(f.size_bytes || 0))}`}
                onClick={() => setActiveFileId(f.file_id)}
              >
                <span className="docs-tab-title">{f.filename}</span>
              </button>
            );
          })}
          {!files.length && pendingUploads.length ? (
            <>
              {pendingUploads.map((name, idx) => (
                <button
                  key={`${name}-${idx}`}
                  className="docs-tab pending"
                  type="button"
                  disabled
                  aria-disabled="true"
                  title="Uploading…"
                >
                  <span className="docs-tab-spinner" aria-hidden="true" />
                  <span className="docs-tab-title">{name}</span>
                </button>
              ))}
            </>
          ) : null}
          {!files.length && !pendingUploads.length ? (
            <div className="docs-tabs-empty">No documents yet</div>
          ) : null}
        </div>

        <div
          className={`docs-pane-viewer ${dropEnabled ? "docs-pane-drop-enabled" : ""} ${isDropHover ? "drag-over" : ""}`}
          aria-label="Document text"
          onDragEnter={(e) => {
            e.preventDefault();
            e.stopPropagation();
            if (!dropEnabled) return;
            dropCounterRef.current += 1;
            setIsDropHover(true);
          }}
          onDragLeave={(e) => {
            e.preventDefault();
            e.stopPropagation();
            if (!dropEnabled) return;
            dropCounterRef.current = Math.max(0, dropCounterRef.current - 1);
            if (dropCounterRef.current === 0) setIsDropHover(false);
          }}
          onDragOver={(e) => {
            e.preventDefault();
            e.stopPropagation();
            // Required to allow drop.
            if (!dropEnabled) return;
            setIsDropHover(true);
          }}
          onDrop={(e) => {
            e.preventDefault();
            e.stopPropagation();
            setIsDropHover(false);
            dropCounterRef.current = 0;
            if (!dropEnabled) return;

            const files = Array.from(e.dataTransfer?.files || []);
            const paths = files
              .map((f) => (f as any)?.path as string)
              .filter((p) => typeof p === "string" && p);
            if (!paths.length) {
              setError("Drag-and-drop didn't provide file paths. Use the attach button in chat.");
              return;
            }
            ingestDroppedPaths(paths).catch((err) => setError(err?.message ?? String(err)));
          }}
        >
          {error ? <div className="docs-pane-error">{error}</div> : null}

          {!files.length ? (
            <div className="docs-pane-empty">
              {pendingUploads.length ? (
                <div className="docs-dropzone">
                  <div className="docs-dropzone-title">Uploading…</div>
                  <div className="docs-dropzone-sub">
                    Extracted text will appear here once ready.
                  </div>
                </div>
              ) : (
                <div className="docs-dropzone">
                  <div className="docs-dropzone-title">Drop a document here</div>
                  <div className="docs-dropzone-sub">It will be attached to this card.</div>
                </div>
              )}
            </div>
          ) : null}

          {activeFile ? (
            <div className="docs-reader">
              <div className="docs-reader-header">
                <div className="docs-reader-title">{activeFile.filename}</div>
                <div className="docs-reader-meta">
                  {activeFile.pages ? <span>{activeFile.pages} pages</span> : null}
                  {activeFile.mime ? <span>{activeFile.mime}</span> : null}
                  {activeFile.status ? <span>{activeFile.status}</span> : null}
                </div>
                <button
                  className="docs-reader-refresh"
                  type="button"
                  onClick={() => (effectiveActiveFileId ? loadView(effectiveActiveFileId) : undefined)}
                  disabled={!effectiveActiveFileId || isLoadingView}
                >
                  Refresh
                </button>
              </div>

              {isLoadingFiles || isLoadingView ? (
                <div className="docs-reader-loading">Loading…</div>
              ) : null}

              {!isLoadingView && view && view.status && view.status !== "completed" && !blocks.length ? (
                <div className="docs-reader-loading">
                  Processing… extracted text will appear here once ready.
                </div>
              ) : null}

              {!isLoadingView && blocks.length ? (
                <div
                  className="docs-reader-body"
                  ref={bodyRef}
                  onMouseUp={() => readSelectionFromWindow({ showPopover: true })}
                  onKeyUp={() => readSelectionFromWindow({ showPopover: true })}
                  onScroll={() => {
                    // Hide the popover while scrolling; selection remains available for chat.
                    if (selectionPos) setSelectionPos(null);
                  }}
                >
                  {selectionText && selectionPos ? (
                    <button
                      type="button"
                      className="docs-selection-ask docs-selection-popover"
                      style={{ left: selectionPos.x, top: selectionPos.y }}
                      onClick={commitSelectionToChat}
                      title="Ask about this selection"
                      onMouseDown={(e) => e.stopPropagation()}
                      onPointerDown={(e) => e.stopPropagation()}
                    >
                      Ask
                    </button>
                  ) : null}
                  {blocks.map(renderBlock)}
                </div>
              ) : null}

              {!isLoadingView && view && !blocks.length && view.status === "completed" ? (
                <div className="docs-pane-empty">No extractable text found for this file.</div>
              ) : null}
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
