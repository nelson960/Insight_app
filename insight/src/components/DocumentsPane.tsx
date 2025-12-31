import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";

import { engine } from "../api/engine";
import { DocEditor } from "../editor/DocEditor";
import { proseMirrorDocToHtml, proseMirrorDocToMarkdown, proseMirrorDocToPlainText } from "../editor/docExport";
import { loadDocPage, saveDocPage } from "../editor/docPages";

type ChatFile = {
  file_id: string;
  filename: string;
  mime: string;
  size_bytes: number;
  status?: string;
  pages?: number | null;
  created_at?: string | null;
  updated_at?: string | null;
  source?: string;
};

type DocSearchMatch = {
  id: string;
  from: number;
  to: number;
  snippet: string;
};

type DocSearchResponse = {
  chat_id: string;
  file_id: string;
  q: string;
  matches: Array<{ from: number; to: number; snippet: string }>;
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

function formatTimestamp(ts: string | null | undefined) {
  if (!ts) return "";
  const d = new Date(ts);
  if (!Number.isFinite(d.getTime())) return String(ts);
  try {
    return d.toLocaleString();
  } catch {
    return d.toISOString();
  }
}

function countWords(text: string) {
  const cleaned = (text || "").trim();
  if (!cleaned) return 0;
  return cleaned.split(/\s+/g).filter(Boolean).length;
}

function buildExportPdfHtml(bodyHtml: string): string {
  const css = `
    @page { margin: 18mm; }
    html, body { margin: 0; padding: 0; background: #ffffff; color: #0f172a; }
    body { box-sizing: border-box; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji"; }
    .doc-root { max-width: 900px; margin: 0 auto; font-size: 14px; line-height: 1.6; }
    .doc-root h1 { font-size: 26px; line-height: 1.15; margin: 22px 0 10px; }
    .doc-root h2 { font-size: 20px; line-height: 1.22; margin: 18px 0 8px; }
    .doc-root h3 { font-size: 17px; line-height: 1.3; margin: 14px 0 6px; }
    .doc-root h4 { font-size: 15px; line-height: 1.4; margin: 12px 0 6px; }
    .doc-root p { margin: 0 0 8px; }
    .doc-root ul, .doc-root ol { margin: 0 0 10px; padding-left: 1.25em; }
    .doc-root li { margin: 4px 0; }
    .doc-root blockquote { margin: 12px 0; padding: 6px 0 6px 12px; border-left: 2px solid rgba(2, 132, 199, 0.35); color: rgba(15, 23, 42, 0.92); }
    .doc-root hr { border: none; border-top: 1px solid rgba(15, 23, 42, 0.15); margin: 14px 0; }
    .doc-root code { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; font-size: 0.92em; background: rgba(2, 132, 199, 0.08); padding: 0.12em 0.32em; border-radius: 6px; }
    .doc-root pre { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; font-size: 12px; line-height: 1.5; background: rgba(15, 23, 42, 0.06); padding: 10px 12px; border-radius: 10px; overflow: auto; margin: 12px 0; }
    .doc-root pre code { background: transparent; padding: 0; border-radius: 0; }
    .doc-root table { border-collapse: collapse; width: 100%; margin: 12px 0; }
    .doc-root th, .doc-root td { border: 1px solid rgba(15, 23, 42, 0.15); padding: 7px 9px; vertical-align: top; }
    .doc-root th { font-weight: 650; background: rgba(15, 23, 42, 0.04); }
    .doc-root a { color: #0369a1; text-decoration: underline; }
  `;
  return `<!doctype html><html><head><meta charset="utf-8" /><style>${css}</style></head><body><div class="doc-root">${bodyHtml}</div></body></html>`;
}

function plainTextFromProseMirror(doc: any): string {
  const parts: string[] = [];
  function walk(node: any) {
    if (!node || typeof node !== "object") return;
    const type = String((node as any).type || "").toLowerCase();
    if (type === "text") {
      parts.push(String((node as any).text || ""));
      return;
    }
    if (type === "hardbreak") {
      parts.push("\n");
      return;
    }
    const content = (node as any).content;
    if (Array.isArray(content)) {
      for (const child of content) walk(child);
      if (type === "paragraph" || type === "heading" || type === "codeblock") {
        parts.push("\n\n");
      }
    }
  }
  walk(doc);
  return parts.join("").trim();
}

type IndexedBlock = {
  segments: Array<[number, string]>;
  text: string;
  text_ci: string;
};

function snippet(text: string, start: number, end: number, radius = 42) {
  const left = Math.max(0, start - radius);
  const right = Math.min(text.length, end + radius);
  let out = text.slice(left, right);
  out = out.split(/\s+/g).filter(Boolean).join(" ");
  if (left > 0) out = `…${out}`;
  if (right < text.length) out = `${out}…`;
  return out;
}

function offsetToPos(segments: Array<[number, string]>, offset: number) {
  let cursor = 0;
  let lastPos = 0;
  const safeOffset = Math.max(0, Number.isFinite(offset) ? Math.trunc(offset) : 0);
  for (const [pos, raw] of segments) {
    const text = raw ?? "";
    lastPos = pos + text.length;
    const nextCursor = cursor + text.length;
    if (safeOffset <= nextCursor) {
      return pos + (safeOffset - cursor);
    }
    cursor = nextCursor;
  }
  return lastPos;
}

function buildIndexedBlocks(doc: any): IndexedBlock[] {
  const blocks: IndexedBlock[] = [];
  const blockStack: Array<{ segments: Array<[number, string]> }> = [];

  function pushBlock() {
    blockStack.push({ segments: [] });
  }

  function popBlock() {
    const blk = blockStack.pop();
    if (!blk) return;
    const segments = blk.segments
      .map(([p, t]) => [Number(p), String(t ?? "")] as [number, string])
      .filter(([, t]) => t);
    if (!segments.length) return;
    const text = segments.map(([, t]) => t).join("");
    if (!text) return;
    blocks.push({ segments, text, text_ci: text.toLowerCase() });
  }

  const blockTypes = new Set(["paragraph", "heading", "codeblock"]);
  const leafOneTypes = new Set(["hardbreak", "horizontalrule"]);

  function walk(node: any, pos: number): number {
    if (!node || typeof node !== "object") return 0;
    const nodeType = String((node as any).type || "").toLowerCase();

    if (nodeType === "text") {
      const text = String((node as any).text || "");
      if (blockStack.length && text) {
        blockStack[blockStack.length - 1].segments.push([pos, text]);
      }
      return text.length;
    }

    if (leafOneTypes.has(nodeType)) {
      if (nodeType === "hardbreak" && blockStack.length) {
        blockStack[blockStack.length - 1].segments.push([pos, "\n"]);
      }
      return 1;
    }

    const pushed = blockTypes.has(nodeType);
    if (pushed) pushBlock();

    const content = (node as any).content;
    let totalChild = 0;
    if (Array.isArray(content)) {
      let childPos = nodeType === "doc" ? pos : pos + 1;
      for (const child of content) {
        const childSize = walk(child, childPos);
        childPos += childSize;
        totalChild += childSize;
      }
    }

    if (pushed) popBlock();
    if (nodeType === "doc") return totalChild;
    return 2 + totalChild;
  }

  walk(doc, 0);
  return blocks;
}

function searchProseMirrorDoc(doc: any, qRaw: string, opts?: { limit?: number }): DocSearchMatch[] {
  const q = (qRaw || "").trim();
  if (!q) return [];
  if (!doc || typeof doc !== "object") return [];

  const limit = Math.max(1, Math.min(Number(opts?.limit ?? 400) || 400, 1000));
  const needle = q.toLowerCase();
  const blocks = buildIndexedBlocks(doc);
  const matches: DocSearchMatch[] = [];
  for (const block of blocks) {
    const hay = block.text_ci;
    if (!hay) continue;
    let startAt = 0;
    while (true) {
      const found = hay.indexOf(needle, startAt);
      if (found < 0) break;
      const end = found + needle.length;
      const from = offsetToPos(block.segments, found);
      const to = offsetToPos(block.segments, end);
      if (Number.isFinite(from) && Number.isFinite(to) && to > from) {
        matches.push({
          id: `${from}:${to}:${matches.length}`,
          from,
          to,
          snippet: snippet(block.text, found, end),
        });
        if (matches.length >= limit) return matches;
      }
      startAt = end > found ? end : found + 1;
    }
  }
  return matches;
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
  const [isLoadingFiles, setIsLoadingFiles] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [doc, setDoc] = useState<any | null>(null);
  const [isLoadingDoc, setIsLoadingDoc] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const editingFileIdRef = useRef<string | null>(null);
  const docRef = useRef<any | null>(null);
  const saveTimerRef = useRef<number | null>(null);

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

  const [searchQuery, setSearchQuery] = useState("");
  const [searchMatches, setSearchMatches] = useState<DocSearchMatch[]>([]);
  const [activeMatchIndex, setActiveMatchIndex] = useState(0);
  const [isSearching, setIsSearching] = useState(false);
  const [isSearchOpen, setIsSearchOpen] = useState(false);
  const [replaceQuery, setReplaceQuery] = useState("");
  const searchInputRef = useRef<HTMLInputElement | null>(null);
  const editorRef = useRef<any | null>(null);
  const lastSearchKeyRef = useRef<string>("");
  const [isReplaceMenuOpen, setIsReplaceMenuOpen] = useState(false);
  const replaceMenuRef = useRef<HTMLDivElement | null>(null);
  const handleEditorReady = useCallback((editor: any | null) => {
    editorRef.current = editor;
  }, []);

  const [isExportMenuOpen, setIsExportMenuOpen] = useState(false);
  const exportMenuRef = useRef<HTMLDivElement | null>(null);

  const infoPopoverRef = useRef<HTMLDivElement | null>(null);
  const infoAnchorRef = useRef<HTMLElement | null>(null);
  const [infoPos, setInfoPos] = useState<{ x: number; y: number } | null>(null);
  const [confirmDeleteFileId, setConfirmDeleteFileId] = useState<string | null>(null);

  const isControlled = typeof activeFileId !== "undefined";
  const effectiveActiveFileId = isControlled ? (activeFileId ?? null) : internalActiveFileId;

  function setActiveFileId(nextId: string | null) {
    setConfirmDeleteFileId(null);
    onActiveFileIdChange?.(nextId);
    if (!isControlled) setInternalActiveFileId(nextId);
  }

  useEffect(() => {
    docRef.current = doc;
  }, [doc]);

  useEffect(() => {
    activeFileIdRef.current = effectiveActiveFileId;
  }, [effectiveActiveFileId]);

  useEffect(() => {
    setError(null);
    setFiles([]);
    setPendingUploads([]);
    setIsLoadingFiles(false);
    setIsDropHover(false);
    dropCounterRef.current = 0;
    ingestInFlightRef.current = false;

    setDoc(null);
    setIsLoadingDoc(false);
    setIsEditing(false);
    editingFileIdRef.current = null;
    docRef.current = null;
    if (saveTimerRef.current != null) {
      window.clearTimeout(saveTimerRef.current);
      saveTimerRef.current = null;
    }

    setSelectionText("");
    setSelectionPos(null);
    setPendingSelection(null);
    if (popoverTimerRef.current != null) {
      window.clearTimeout(popoverTimerRef.current);
      popoverTimerRef.current = null;
    }

    setSearchQuery("");
    setSearchMatches([]);
    setActiveMatchIndex(0);
    setIsSearching(false);
    setIsSearchOpen(false);
    setReplaceQuery("");
    setIsReplaceMenuOpen(false);
    setIsExportMenuOpen(false);
    setInfoPos(null);
    infoAnchorRef.current = null;

    latestFilesChangedRef.current = null;
    if (!isControlled) setInternalActiveFileId(null);
  }, [chatId, isControlled]);

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

  async function flushDocSaveNow() {
    const fileId = editingFileIdRef.current;
    if (!fileId) return;
    const currentDoc = docRef.current;
    if (!currentDoc) return;
    if (saveTimerRef.current != null) {
      window.clearTimeout(saveTimerRef.current);
      saveTimerRef.current = null;
    }
    try {
      await saveDocPage(chatId, fileId, { doc: currentDoc });
    } catch {
      // ignore
    }
  }

  async function reloadFiles(): Promise<ChatFile[]> {
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

  async function reloadActiveDoc(fileId: string) {
    setIsLoadingDoc(true);
    const res = await loadDocPage(chatId, fileId);
    setIsLoadingDoc(false);
    if (!res.ok) {
      setError(res.error || `Failed to load document (${res.status})`);
      setDoc(null);
      return;
    }
    setDoc((res.data as any)?.doc ?? null);
  }

  const dropEnabled = !files.length && !pendingUploads.length && !ingestInFlightRef.current;

  // Backend-driven events:
  // - files-changed: file registered/attached to this chat
  // - file-text-ready: extracted representation persisted (doc page can bootstrap/update)
  // - file-status: ingestion status update
  useEffect(() => {
    let cancelled = false;
    const unlistenFns: Array<() => void> = [];

    function shouldHandle(payload: any): boolean {
      return typeof payload?.chat_id === "string" && payload.chat_id === chatId;
    }

    function safeReloadFilesAndMaybeSelect(payload: any, reason?: string) {
      reloadFiles()
        .then((next) => {
          const fidFromEvent = typeof payload?.file_id === "string" ? payload.file_id : null;

          if (reason === "files-changed" && fidFromEvent) {
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

        safeReloadFilesAndMaybeSelect(payload, name);

        if (name === "file-text-ready") {
          const fid = typeof payload?.file_id === "string" ? payload.file_id : null;
          const activeFid = activeFileIdRef.current;
          if (!fid) return;

          if (!activeFid) {
            setActiveFileId(fid);
            return;
          }
          if (fid === activeFid && !editingFileIdRef.current) {
            reloadActiveDoc(fid).catch(() => {
              // ignore
            });
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
      const next = await reloadFiles();
      if (cancelled) return;
      setIsLoadingFiles(false);

      const current = isControlled ? (activeFileId ?? null) : internalActiveFileId;
      const currentStillExists = current ? next.some((f) => f.file_id === current) : false;
      if (!currentStillExists) setActiveFileId(next.length ? next[0].file_id : null);
      if (!current && next.length) setActiveFileId(next[0].file_id);
    }

    load().catch((e) => {
      if (cancelled) return;
      setIsLoadingFiles(false);
      setError(e?.message ?? String(e));
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
    try {
      const res = await engine<{ files: { file_id: string; filename: string; job_id: string }[] }>(
        "/files/ingest_path",
        { chat_id: chatId, paths },
        "POST"
      );
      if (!res.ok) {
        setError(res.error || `Failed to ingest files (${res.status})`);
        return;
      }
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

  useEffect(() => {
    // Reset view state when switching documents.
    setInfoPos(null);
    infoAnchorRef.current = null;
    setSelectionText("");
    setSelectionPos(null);
    setPendingSelection(null);
    if (popoverTimerRef.current != null) {
      window.clearTimeout(popoverTimerRef.current);
      popoverTimerRef.current = null;
    }

    if (isEditing) {
      void flushDocSaveNow();
    }
    setIsEditing(false);
    editingFileIdRef.current = null;

    if (saveTimerRef.current != null) {
      window.clearTimeout(saveTimerRef.current);
      saveTimerRef.current = null;
    }
  }, [effectiveActiveFileId]);

  useEffect(() => {
    let cancelled = false;
    const id = effectiveActiveFileId;
    if (!id) {
      setDoc(null);
      setIsLoadingDoc(false);
      return;
    }
    setIsLoadingDoc(true);
    setDoc(null);
    loadDocPage(chatId, id)
      .then((res) => {
        if (cancelled) return;
        setIsLoadingDoc(false);
        if (!res.ok) {
          setError(res.error || `Failed to load document (${res.status})`);
          setDoc(null);
          return;
        }
        setDoc((res.data as any)?.doc ?? null);
      })
      .catch((e: any) => {
        if (cancelled) return;
        setIsLoadingDoc(false);
        setError(e?.message ?? String(e));
        setDoc(null);
      });
    return () => {
      cancelled = true;
    };
  }, [effectiveActiveFileId, chatId]);

  function scheduleSave(nextDoc: any) {
    const fileId = editingFileIdRef.current || effectiveActiveFileId;
    if (!fileId) return;
    if (!isEditing) return;
    if (saveTimerRef.current != null) window.clearTimeout(saveTimerRef.current);
    saveTimerRef.current = window.setTimeout(() => {
      saveTimerRef.current = null;
      saveDocPage(chatId, fileId, { doc: nextDoc }).catch(() => {
        // ignore; next edit will retry
      });
    }, 500);
  }

  function readSelectionFromWindow(opts?: { showPopover?: boolean }) {
    if (isEditing) return;
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
      const xRaw = rect.left - bodyRect.left + body.scrollLeft + rect.width / 2;
      const minX = body.scrollLeft + 28;
      const maxX = body.scrollLeft + body.clientWidth - 28;
      const x = Math.max(minX, Math.min(maxX, xRaw));

      const yRaw = rect.top - bodyRect.top + body.scrollTop - 48;
      const minY = body.scrollTop + 8;
      const y = Math.max(minY, yRaw);
      if (popoverTimerRef.current != null) window.clearTimeout(popoverTimerRef.current);
      popoverTimerRef.current = window.setTimeout(() => {
        popoverTimerRef.current = null;
        setSelectionPos({ x: Math.max(8, x), y: Math.max(8, y) });
      }, 0);
    } catch {
      // ignore
    }
  }

  useEffect(() => {
    if (isEditing) return;
    function onSelectionChangeEvent() {
      readSelectionFromWindow({ showPopover: false });
    }
    document.addEventListener("selectionchange", onSelectionChangeEvent);
    return () => document.removeEventListener("selectionchange", onSelectionChangeEvent);
  }, [effectiveActiveFileId, isEditing]);

  function commitSelectionToChat() {
    if (!pendingSelection) return;
    onSelectionChange?.(pendingSelection);
    onAskSelection?.();
    setSelectionText("");
    setPendingSelection(null);
    setSelectionPos(null);
  }

  const activeFile = useMemo(
    () => files.find((f) => f.file_id === effectiveActiveFileId) || null,
    [files, effectiveActiveFileId]
  );

  const wordCount = useMemo(() => {
    const text = plainTextFromProseMirror(doc);
    const n = countWords(text);
    return n > 0 ? n : null;
  }, [doc]);

  const effectiveSearchOpen = isSearchOpen || !!searchQuery.trim() || (isEditing && !!replaceQuery.trim());

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (!effectiveActiveFileId) return;
      if (!e.ctrlKey && !e.metaKey) return;
      if (e.key.toLowerCase() !== "f") return;
      e.preventDefault();
      setIsSearchOpen(true);
      searchInputRef.current?.focus();
      searchInputRef.current?.select();
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [effectiveActiveFileId]);

  useEffect(() => {
    let cancelled = false;
    const fileId = effectiveActiveFileId;
    const q = searchQuery.trim();
    if (!fileId || !q) {
      setIsSearching(false);
      setSearchMatches([]);
      setActiveMatchIndex(0);
      return;
    }

    setIsSearching(true);
    const searchKey = `${fileId}|${q}|${isEditing ? "edit" : "view"}`;
    const shouldResetIndex = lastSearchKeyRef.current !== searchKey;
    lastSearchKeyRef.current = searchKey;
    const timer = window.setTimeout(() => {
      if (isEditing) {
        try {
          const matches = searchProseMirrorDoc(docRef.current, q, { limit: 400 });
          if (cancelled) return;
          setIsSearching(false);
          setSearchMatches(matches);
          setActiveMatchIndex((prev) => {
            if (!matches.length) return 0;
            if (shouldResetIndex) return 0;
            return Math.min(Math.max(0, prev), matches.length - 1);
          });
        } catch {
          if (cancelled) return;
          setIsSearching(false);
          setSearchMatches([]);
          setActiveMatchIndex(0);
        }
        return;
      }
      const endpoint = `/search/doc/${encodeURIComponent(chatId)}/${encodeURIComponent(fileId)}?q=${encodeURIComponent(q)}&limit=400`;
      engine<DocSearchResponse>(endpoint, undefined, "GET")
        .then((res) => {
          if (cancelled) return;
          setIsSearching(false);
          if (!res.ok) {
            setSearchMatches([]);
            setActiveMatchIndex(0);
            return;
          }
          const raw = Array.isArray((res.data as any)?.matches)
            ? ((res.data as any).matches as any[])
            : [];
          const matches = raw
            .map((m, i) => {
              const from = Number(m?.from ?? m?.from_pos ?? m?.fromPos ?? -1);
              const to = Number(m?.to ?? m?.to_pos ?? m?.toPos ?? -1);
              if (!Number.isFinite(from) || from < 0) return null;
              if (!Number.isFinite(to) || to <= from) return null;
              return {
                id: `${from}:${to}:${i}`,
                from,
                to,
                snippet: String(m?.snippet ?? ""),
              } satisfies DocSearchMatch;
            })
            .filter(Boolean) as DocSearchMatch[];
          setSearchMatches(matches);
          setActiveMatchIndex((prev) => {
            if (!matches.length) return 0;
            if (shouldResetIndex) return 0;
            return Math.min(Math.max(0, prev), matches.length - 1);
          });
        })
        .catch(() => {
          if (cancelled) return;
          setIsSearching(false);
          setSearchMatches([]);
          setActiveMatchIndex(0);
        });
    }, 180);

    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [effectiveActiveFileId, searchQuery, chatId, isEditing, doc]);

  function replaceActiveMatch() {
    if (!isEditing) return;
    const editor = editorRef.current;
    if (!editor) return;
    const q = searchQuery.trim();
    if (!q) return;
    const len = searchMatches.length;
    if (!len) return;
    const idx = Math.min(Math.max(0, activeMatchIndex), len - 1);
    const m = searchMatches[idx];
    if (!m) return;
    const from = Number(m.from);
    const to = Number(m.to);
    if (!Number.isFinite(from) || !Number.isFinite(to) || to <= from) return;
    const replacement = String(replaceQuery ?? "");
    editor.view.dispatch(editor.state.tr.insertText(replacement, from, to));
    editor.commands.focus();
  }

  function replaceAllMatches() {
    if (!isEditing) return;
    const editor = editorRef.current;
    if (!editor) return;
    const q = searchQuery.trim();
    if (!q) return;
    const replacement = String(replaceQuery ?? "");

    const matches = [...searchMatches]
      .map((m) => ({ from: Number(m.from), to: Number(m.to) }))
      .filter((m) => Number.isFinite(m.from) && Number.isFinite(m.to) && m.to > m.from)
      .sort((a, b) => b.from - a.from);
    if (!matches.length) return;

    let tr = editor.state.tr;
    for (const m of matches) {
      tr = tr.insertText(replacement, m.from, m.to);
    }
    editor.view.dispatch(tr);
    editor.commands.focus();
  }

  function toggleInfoPopover(anchor: HTMLElement) {
    setInfoPos((prev) => {
      if (prev) return null;
      const rect = anchor.getBoundingClientRect();
      infoAnchorRef.current = anchor;
      return { x: rect.right, y: rect.bottom };
    });
  }

  useEffect(() => {
    if (!isReplaceMenuOpen) return;
    function onPointerDown(e: Event) {
      const target = e.target as Node | null;
      const el = replaceMenuRef.current;
      if (!el || !target) return;
      if (el.contains(target)) return;
      setIsReplaceMenuOpen(false);
    }
    window.addEventListener("pointerdown", onPointerDown, true);
    return () => window.removeEventListener("pointerdown", onPointerDown, true);
  }, [isReplaceMenuOpen]);

  useEffect(() => {
    if (!isEditing) setIsReplaceMenuOpen(false);
  }, [isEditing]);

  useEffect(() => {
    if (!isExportMenuOpen) return;
    function onPointerDown(e: Event) {
      const target = e.target as Node | null;
      const el = exportMenuRef.current;
      if (!el || !target) return;
      if (el.contains(target)) return;
      setIsExportMenuOpen(false);
    }
    window.addEventListener("pointerdown", onPointerDown, true);
    return () => window.removeEventListener("pointerdown", onPointerDown, true);
  }, [isExportMenuOpen]);

  useEffect(() => {
    if (!isEditing) setIsExportMenuOpen(false);
  }, [isEditing]);

  useEffect(() => {
    if (!infoPos) return;
    function onPointerDown(e: Event) {
      const target = e.target as Node | null;
      const popover = infoPopoverRef.current;
      const anchor = infoAnchorRef.current;
      if (!target) return;
      if (popover && popover.contains(target)) return;
      if (anchor && anchor.contains(target)) return;
      setInfoPos(null);
    }
    window.addEventListener("pointerdown", onPointerDown, true);
    return () => window.removeEventListener("pointerdown", onPointerDown, true);
  }, [infoPos]);

  useEffect(() => {
    if (!infoPos) return;
    const popover = infoPopoverRef.current;
    const anchor = infoAnchorRef.current;
    if (!popover || !anchor) return;

    const raf = window.requestAnimationFrame(() => {
      const popRect = popover.getBoundingClientRect();
      const boundaryEl = anchor.closest(".canvas-root") as HTMLElement | null;
      const boundaryRect = boundaryEl
        ? boundaryEl.getBoundingClientRect()
        : new DOMRect(0, 0, window.innerWidth, window.innerHeight);

      const pad = 14;
      const minX = boundaryRect.left + pad;
      const maxX = boundaryRect.right - pad;
      const minY = boundaryRect.top + pad;
      const maxY = boundaryRect.bottom - pad;

      let nextX = infoPos.x;
      let nextY = infoPos.y;

      if (popRect.left < minX) nextX += minX - popRect.left;
      else if (popRect.right > maxX) nextX -= popRect.right - maxX;

      if (popRect.top < minY) nextY += minY - popRect.top;
      else if (popRect.bottom > maxY) nextY -= popRect.bottom - maxY;

      nextX = Math.round(nextX);
      nextY = Math.round(nextY);

      if (nextX !== infoPos.x || nextY !== infoPos.y) {
        setInfoPos({ x: nextX, y: nextY });
      }
    });

    return () => window.cancelAnimationFrame(raf);
  }, [infoPos]);

  useEffect(() => {
    setInfoPos(null);
    infoAnchorRef.current = null;
  }, [effectiveActiveFileId]);

  function exportBaseNameFromFilename(name: string): string {
    const cleaned = (name || "").trim();
    if (!cleaned) return "document";
    const base = cleaned.replace(/\.[^/.]+$/, "");
    return base.trim() || "document";
  }

  async function saveExportFile(opts: { defaultName: string; content: string }) {
    try {
      const res = await invoke<{ path?: string | null; cancelled?: boolean }>("save_export_file", {
        defaultName: opts.defaultName,
        content: opts.content,
      });
      if (res?.path) setError(null);
    } catch (e: any) {
      setError(e?.message ?? String(e));
    }
  }

  function exportMarkdown() {
    if (!activeFile) return;
    const d = docRef.current;
    if (!d) {
      setError("No document content to export.");
      return;
    }
    const base = exportBaseNameFromFilename(activeFile.filename);
    const md = proseMirrorDocToMarkdown(d);
    void saveExportFile({ defaultName: `${base}.md`, content: md });
  }

  function exportText() {
    if (!activeFile) return;
    const d = docRef.current;
    if (!d) {
      setError("No document content to export.");
      return;
    }
    const base = exportBaseNameFromFilename(activeFile.filename);
    const txt = proseMirrorDocToPlainText(d);
    void saveExportFile({ defaultName: `${base}.txt`, content: txt });
  }

  function exportPdf() {
    if (!activeFile) return;
    const d = docRef.current;
    if (!d) {
      setError("No document content to export.");
      return;
    }
    const base = exportBaseNameFromFilename(activeFile.filename);
    const bodyHtml = proseMirrorDocToHtml(d);
    const html = buildExportPdfHtml(bodyHtml);
    invoke<{ path?: string | null; cancelled?: boolean }>("export_pdf_file", {
      defaultName: `${base}.pdf`,
      html,
    })
      .then((res) => {
        if (res?.path) setError(null);
      })
      .catch((e: any) => setError(e?.message ?? String(e)));
  }

  return (
    <div className="docs-pane">
      <div className="docs-pane-body">
        <div className="docs-tabs" role="tablist" aria-label="Files">
          {files.map((f) => {
            const active = f.file_id === effectiveActiveFileId;
            return (
              <div key={f.file_id} className={`docs-tab ${active ? "active" : ""}`}>
                <button
                  className="docs-tab-main"
                  type="button"
                  role="tab"
                  aria-selected={active}
                  title={`${f.filename} · ${formatBytes(Number(f.size_bytes || 0))}`}
                  onClick={() => setActiveFileId(f.file_id)}
                >
                  <span className="docs-tab-title">{f.filename}</span>
                </button>
                {active ? (
                  <button
                    type="button"
                    className="docs-tab-info"
                    aria-label="Document info"
                    title="Info"
                    onClick={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      toggleInfoPopover(e.currentTarget);
                    }}
                  >
                    i
                  </button>
                ) : null}
                {active ? (
                  <button
                    type="button"
                    className={`docs-tab-delete ${
                      confirmDeleteFileId === f.file_id ? "confirm" : ""
                    }`}
                    aria-label="Delete document"
                    title={
                      confirmDeleteFileId === f.file_id
                        ? "Click again to delete"
                        : "Delete"
                    }
                    onClick={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      if (confirmDeleteFileId !== f.file_id) {
                        setConfirmDeleteFileId(f.file_id);
                        return;
                      }
                      setConfirmDeleteFileId(null);
                      setInfoPos(null);
                      setError(null);
                      engine(
                        `/files/chat/${encodeURIComponent(chatId)}/${encodeURIComponent(
                          f.file_id
                        )}`,
                        undefined,
                        "DELETE"
                      )
                        .then(async (res) => {
                          if (!res.ok) {
                            setError(res.error || `Failed to delete file (${res.status})`);
                            return;
                          }
                          const prevFiles = files;
                          const deletedIdx = prevFiles.findIndex((x) => x.file_id === f.file_id);
                          const next = await reloadFiles();
                          const currentActive = activeFileIdRef.current;
                          if (currentActive === f.file_id) {
                            const pick =
                              (deletedIdx >= 0 && next[deletedIdx]) ||
                              (deletedIdx > 0 && next[deletedIdx - 1]) ||
                              (next.length ? next[0] : null);
                            setActiveFileId(pick ? pick.file_id : null);
                            setDoc(null);
                            setIsEditing(false);
                            editingFileIdRef.current = null;
                            docRef.current = null;
                            setSelectionText("");
                            setSelectionPos(null);
                            setPendingSelection(null);
                          }
                        })
                        .catch((err: any) => {
                          setError(err?.message ?? String(err));
                        });
                    }}
                  >
                    {confirmDeleteFileId === f.file_id ? "Del" : "×"}
                  </button>
                ) : null}
              </div>
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

        {infoPos && activeFile ? (
          <div
            ref={infoPopoverRef}
            className="docs-tab-info-popover"
            role="dialog"
            aria-label="Document info"
            style={{ left: infoPos.x, top: infoPos.y }}
          >
            <div className="docs-reader-info-panel">
              <div className="docs-reader-info-row">
                <div className="docs-reader-info-label">Type</div>
                <div className="docs-reader-info-value">{activeFile.mime || "—"}</div>
              </div>
              <div className="docs-reader-info-row">
                <div className="docs-reader-info-label">Pages</div>
                <div className="docs-reader-info-value">{typeof activeFile.pages === "number" ? activeFile.pages : "—"}</div>
              </div>
              <div className="docs-reader-info-row">
                <div className="docs-reader-info-label">Uploaded</div>
                <div className="docs-reader-info-value">{formatTimestamp(activeFile.created_at) || "—"}</div>
              </div>
              <div className="docs-reader-info-row">
                <div className="docs-reader-info-label">Size</div>
                <div className="docs-reader-info-value">{formatBytes(Number(activeFile.size_bytes || 0))}</div>
              </div>
              <div className="docs-reader-info-row">
                <div className="docs-reader-info-label">Words</div>
                <div className="docs-reader-info-value">{typeof wordCount === "number" ? wordCount.toLocaleString() : "—"}</div>
              </div>
            </div>
          </div>
        ) : null}

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
            if (!dropEnabled) return;
            setIsDropHover(true);
          }}
          onDrop={(e) => {
            e.preventDefault();
            e.stopPropagation();
            setIsDropHover(false);
            dropCounterRef.current = 0;
            if (!dropEnabled) return;

            const droppedFiles = Array.from(e.dataTransfer?.files || []);
            const paths = droppedFiles
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
                  <div className="docs-dropzone-sub">Extracted text will appear here once ready.</div>
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
		                <div className="docs-reader-header-row">
		                  <button
		                    type="button"
		                    className="docs-reader-search-toggle"
		                    aria-label={isEditing ? "Search and replace" : "Search in document"}
		                    title={isEditing ? "Search & Replace" : "Search"}
		                    onClick={() => {
		                      setIsSearchOpen((prev) => {
		                        const next = !prev;
		                        if (!next) {
		                          setReplaceQuery("");
		                          setIsReplaceMenuOpen(false);
		                        }
		                        if (next) {
		                          window.setTimeout(() => {
		                            searchInputRef.current?.focus();
		                            searchInputRef.current?.select();
		                          }, 0);
		                        }
		                        return next;
		                      });
		                    }}
			                  >
			                    ⌕
			                  </button>

		                  <div className="docs-reader-header-actions">
		                    {isEditing ? (
		                      <div className="docs-reader-export-actions" ref={exportMenuRef}>
		                        <button
		                          type="button"
		                          className="docs-reader-export-toggle"
		                          aria-label="Export document"
		                          title="Export"
		                          onClick={() => setIsExportMenuOpen((prev) => !prev)}
		                        >
		                          ⤓
		                        </button>
		                        {isExportMenuOpen ? (
		                          <div className="docs-reader-export-menu" role="menu" aria-label="Export options">
		                            <button
		                              type="button"
		                              className="docs-reader-export-menu-item"
		                              role="menuitem"
		                              onClick={() => {
		                                setIsExportMenuOpen(false);
		                                exportPdf();
		                              }}
		                            >
		                              PDF (.pdf)
		                            </button>
		                            <button
		                              type="button"
		                              className="docs-reader-export-menu-item"
		                              role="menuitem"
		                              onClick={() => {
		                                setIsExportMenuOpen(false);
		                                exportMarkdown();
		                              }}
		                            >
		                              Markdown (.md)
		                            </button>
		                            <button
		                              type="button"
		                              className="docs-reader-export-menu-item"
		                              role="menuitem"
		                              onClick={() => {
		                                setIsExportMenuOpen(false);
		                                exportText();
		                              }}
		                            >
		                              Text (.txt)
		                            </button>
		                          </div>
		                        ) : null}
		                      </div>
		                    ) : null}

		                    <button
		                      type="button"
		                      className="docs-reader-edit-toggle"
		                      aria-label={isEditing ? "Done editing" : "Edit document"}
		                      title={isEditing ? "Done" : "Edit"}
		                      onClick={() => {
		                        setIsExportMenuOpen(false);
		                        if (isEditing) {
		                          void flushDocSaveNow().finally(() => {
		                            editingFileIdRef.current = null;
		                            setIsEditing(false);
		                          });
		                          return;
		                        }
		                        editingFileIdRef.current = effectiveActiveFileId;
		                        setIsEditing(true);
		                      }}
		                    >
		                      <span className="docs-reader-edit-icon">{isEditing ? "✓" : "✎"}</span>
		                    </button>
		                  </div>
			                </div>

                <div className={`docs-reader-search-wrap ${effectiveSearchOpen ? "open" : ""}`}>
                  <div className="docs-reader-search">
                    <span className="docs-reader-search-icon" aria-hidden="true">
                      ⌕
                    </span>
                    <input
                      ref={searchInputRef}
                      className="docs-reader-search-input"
                      placeholder="Search in document…"
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      onKeyDown={(e) => {
                        const len = searchMatches.length;
                        const navDown = e.key === "Enter" || e.key === "ArrowDown";
                        const navUp = e.key === "ArrowUp";
                        if (navDown || navUp) {
                          if (!len) return;
                          e.preventDefault();
                          const delta = navUp || (e.key === "Enter" && e.shiftKey) ? -1 : 1;
                          setActiveMatchIndex((prev) => (prev + delta + len) % len);
                          return;
                        }
                        if (e.key === "Escape") {
                          e.preventDefault();
                          setSearchQuery("");
                          setSearchMatches([]);
                          setActiveMatchIndex(0);
                          setIsSearchOpen(false);
                          setReplaceQuery("");
                          setIsReplaceMenuOpen(false);
                        }
                      }}
                    />
                    {searchQuery.trim() ? (
                      <button
                        type="button"
                        className="docs-reader-search-clear"
                        onClick={() => {
                          setSearchQuery("");
                          setSearchMatches([]);
                          setActiveMatchIndex(0);
                          searchInputRef.current?.focus();
                        }}
                        aria-label="Clear search"
                        title="Clear"
                      >
                        ×
                      </button>
                    ) : null}
                    <div className="docs-reader-search-count" aria-label="Search matches">
                      {isSearching
                        ? "…"
                        : searchQuery.trim()
                          ? `${searchMatches.length ? activeMatchIndex + 1 : 0}/${searchMatches.length}`
                          : ""}
                    </div>
                  </div>

                  {isEditing ? (
                    <div className="docs-reader-replace">
                      <span className="docs-reader-replace-icon" aria-hidden="true">
                        ↺
                      </span>
                      <input
                        className="docs-reader-replace-input"
                        placeholder="Replace…"
                        value={replaceQuery}
                        onChange={(e) => setReplaceQuery(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") {
                            e.preventDefault();
                            replaceActiveMatch();
                          }
                          if (e.key === "Escape") {
                            e.preventDefault();
                            setReplaceQuery("");
                            setIsReplaceMenuOpen(false);
                          }
                        }}
                      />
                      <div className="docs-reader-replace-actions" ref={replaceMenuRef}>
                        <button
                          type="button"
                          className="docs-reader-replace-btn"
                          disabled={!searchQuery.trim() || !searchMatches.length}
                          onClick={() => setIsReplaceMenuOpen((prev) => !prev)}
                          title="Replace…"
                        >
                          Replace ▾
                        </button>
                        {isReplaceMenuOpen ? (
                          <div className="docs-reader-replace-menu" role="menu" aria-label="Replace options">
                            <button
                              type="button"
                              className="docs-reader-replace-menu-item"
                              role="menuitem"
                              onClick={() => {
                                setIsReplaceMenuOpen(false);
                                replaceActiveMatch();
                              }}
                            >
                              Replace (selection)
                            </button>
                            <button
                              type="button"
                              className="docs-reader-replace-menu-item"
                              role="menuitem"
                              onClick={() => {
                                setIsReplaceMenuOpen(false);
                                replaceAllMatches();
                              }}
                            >
                              Replace all
                            </button>
                          </div>
                        ) : null}
                      </div>
                    </div>
                  ) : null}
                </div>

	                {/* Info popover is rendered next to the info button for both view and edit modes. */}
	              </div>

              {isLoadingFiles ? <div className="docs-reader-loading">Loading…</div> : null}

              {!isLoadingDoc && !doc && activeFile.status && activeFile.status !== "completed" ? (
                <div className="docs-reader-loading">Processing… extracted text will appear here once ready.</div>
              ) : null}

              <div
                className="docs-reader-body"
                ref={bodyRef}
                onMouseUp={isEditing ? undefined : () => readSelectionFromWindow({ showPopover: true })}
                onKeyUp={isEditing ? undefined : () => readSelectionFromWindow({ showPopover: true })}
                onScroll={
                  isEditing
                    ? undefined
                    : () => {
                        if (selectionPos) setSelectionPos(null);
                      }
                }
              >
                {!isLoadingDoc && !doc && activeFile.status === "completed" ? (
                  <div className="docs-reader-loading">No extractable text found for this file.</div>
                ) : null}

                {isLoadingDoc ? <div className="docs-reader-loading">Loading…</div> : null}

                {doc ? (
                  <>
                    {selectionText && selectionPos && !isEditing ? (
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
	                    <DocEditor
	                      doc={doc}
	                      editable={isEditing}
	                      searchMatches={searchMatches}
	                      activeMatchIndex={activeMatchIndex}
	                      onEditorReady={handleEditorReady}
	                      onDocChange={(next) => {
	                        setDoc(next);
	                        docRef.current = next;
	                        scheduleSave(next);
	                      }}
	                    />
                  </>
                ) : null}
              </div>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
