import { useEffect, useMemo } from "react";
import { EditorContent, useEditor } from "@tiptap/react";
import StarterKit from "@tiptap/starter-kit";
import { Extension } from "@tiptap/core";
import { Plugin, PluginKey, TextSelection } from "@tiptap/pm/state";
import { Decoration, DecorationSet } from "@tiptap/pm/view";
import Link from "@tiptap/extension-link";
import Underline from "@tiptap/extension-underline";
import Highlight from "@tiptap/extension-highlight";
import Table from "@tiptap/extension-table";
import TableRow from "@tiptap/extension-table-row";
import TableCell from "@tiptap/extension-table-cell";
import TableHeader from "@tiptap/extension-table-header";
import TaskList from "@tiptap/extension-task-list";
import TaskItem from "@tiptap/extension-task-item";
import { CodeBlockCopyButton, useEditorRadialTools } from "./EditorTools";

type SearchMatch = { from: number; to: number };

type HighlightMeta = {
  matches?: SearchMatch[];
  activeIndex?: number;
};

type HighlightPluginState = {
  matches: SearchMatch[];
  activeIndex: number;
  decorations: DecorationSet;
};

const highlightKey = new PluginKey<HighlightPluginState>("insightSearchHighlights");

function clampInt(n: number, min: number, max: number) {
  if (!Number.isFinite(n)) return min;
  return Math.max(min, Math.min(max, Math.trunc(n)));
}

function normalizeMatches(doc: any, matches: SearchMatch[]): SearchMatch[] {
  const maxPos = clampInt((doc as any)?.content?.size ?? 0, 0, Number.MAX_SAFE_INTEGER);
  const out: SearchMatch[] = [];
  for (const m of matches) {
    const from = clampInt(m.from, 0, maxPos);
    const to = clampInt(m.to, 0, maxPos);
    if (to > from) out.push({ from, to });
    if (out.length >= 1000) break;
  }
  return out;
}

function buildDecorations(doc: any, matches: SearchMatch[], activeIndex: number): DecorationSet {
  if (!matches.length) return DecorationSet.empty;
  const safeIndex = Math.min(Math.max(0, activeIndex), matches.length - 1);
  const decos: Decoration[] = [];
  for (let i = 0; i < matches.length; i += 1) {
    const m = matches[i];
    const cls = i === safeIndex ? "docs-search-hit active" : "docs-search-hit";
    decos.push(Decoration.inline(m.from, m.to, { class: cls }));
  }
  return DecorationSet.create(doc, decos);
}

function scrollRangeIntoView(opts: {
  editor: any;
  from: number;
  to: number;
  marginPx?: number;
}) {
  const { editor, from, to, marginPx = 48 } = opts;
  const viewDom = editor?.view?.dom as HTMLElement | undefined;
  if (!viewDom) return;

  // The ProseMirror `scrollIntoView()` transaction helper tends to scroll the wrong ancestor
  // in the Tauri WebView (selection updates but the overflow container does not scroll).
  // We manually scroll the nearest scroll container for the documents pane.
  const scroller = viewDom.closest(".docs-reader-body") as HTMLElement | null;
  if (!scroller) return;

  let top = 0;
  let bottom = 0;
  try {
    const a = editor.view.coordsAtPos(from);
    top = a.top;
    bottom = a.bottom;
  } catch {
    return;
  }
  try {
    const b = editor.view.coordsAtPos(to);
    top = Math.min(top, b.top);
    bottom = Math.max(bottom, b.bottom);
  } catch {
    // ignore
  }

  const rect = scroller.getBoundingClientRect();
  const posTop = top - rect.top + scroller.scrollTop;
  const posBottom = bottom - rect.top + scroller.scrollTop;

  const viewTop = scroller.scrollTop;
  const viewBottom = viewTop + rect.height;

  let next = viewTop;
  if (posTop < viewTop + marginPx) {
    next = Math.max(0, posTop - marginPx);
  } else if (posBottom > viewBottom - marginPx) {
    next = posBottom - (rect.height - marginPx);
  } else {
    return;
  }

  const maxTop = Math.max(0, scroller.scrollHeight - scroller.clientHeight);
  next = Math.max(0, Math.min(maxTop, next));
  if (Math.abs(next - scroller.scrollTop) < 1) return;
  // WKWebView in Tauri can ignore `Element.scrollTo({ top })` for nested overflow containers.
  // Directly setting `scrollTop` is the most reliable cross-WebView way to ensure the active hit is visible.
  scroller.scrollTop = next;
}

const SearchHighlights = Extension.create({
  name: "insightSearchHighlights",
  addProseMirrorPlugins() {
    return [
      new Plugin<HighlightPluginState>({
        key: highlightKey,
        state: {
          init: () => ({
            matches: [],
            activeIndex: 0,
            decorations: DecorationSet.empty,
          }),
          apply: (tr, prev, _oldState, newState) => {
            let matches = prev.matches;
            let activeIndex = prev.activeIndex;
            const meta = tr.getMeta(highlightKey) as HighlightMeta | undefined;
            if (meta) {
              if (Array.isArray(meta.matches)) matches = meta.matches;
              if (typeof meta.activeIndex === "number") activeIndex = meta.activeIndex;
            }
            if (tr.docChanged || meta) {
              const normalized = normalizeMatches(newState.doc, matches);
              return {
                matches: normalized,
                activeIndex,
                decorations: buildDecorations(newState.doc, normalized, activeIndex),
              };
            }
            return prev;
          },
        },
        props: {
          decorations(state) {
            return highlightKey.getState(state)?.decorations ?? null;
          },
        },
      }),
    ];
  },
});

type Props = {
  doc: any;
  editable: boolean;
  onDocChange?: (doc: any) => void;
  onEditorReady?: (editor: any | null) => void;
  searchMatches?: Array<{ from: number; to: number }>;
  activeMatchIndex?: number;
};

export function DocEditor({
  doc,
  editable,
  onDocChange,
  onEditorReady,
  searchMatches = [],
  activeMatchIndex = 0,
}: Props) {
  const editor = useEditor({
    extensions: [
      StarterKit,
      Underline,
      Highlight.configure({ multicolor: false }),
      Link.configure({ openOnClick: false }),
      Table.configure({ resizable: true }),
      TableRow,
      TableCell,
      TableHeader,
      TaskList,
      TaskItem.configure({ nested: true }),
      CodeBlockCopyButton,
      SearchHighlights,
    ],
    content: doc,
    editable,
    onUpdate: ({ editor }) => {
      if (!onDocChange) return;
      if (!editor.isEditable) return;
      onDocChange(editor.getJSON());
    },
  });

  const { onContextMenu, overlay } = useEditorRadialTools({ editor: (editor as any) ?? null, enabled: editable });

  useEffect(() => {
    if (!editor) return;
    editor.setEditable(editable);
  }, [editor, editable]);

  useEffect(() => {
    if (!onEditorReady) return;
    onEditorReady(editor ?? null);
    return () => onEditorReady(null);
  }, [editor, onEditorReady]);

  useEffect(() => {
    if (!editor) return;
    if (!doc) return;
    if (editable) return;
    editor.commands.setContent(doc, false);
  }, [editor, doc, editable]);

  const normalizedSearch = useMemo(() => {
    return (Array.isArray(searchMatches) ? searchMatches : [])
      .map((m) => ({ from: Number(m.from), to: Number(m.to) }))
      .filter((m) => Number.isFinite(m.from) && Number.isFinite(m.to) && m.to > m.from);
  }, [searchMatches]);

  useEffect(() => {
    if (!editor) return;
    const tr = editor.state.tr.setMeta(highlightKey, {
      matches: normalizedSearch,
      activeIndex: activeMatchIndex,
    } satisfies HighlightMeta);
    editor.view.dispatch(tr);
  }, [editor, normalizedSearch, activeMatchIndex]);

  useEffect(() => {
    if (!editor) return;
    if (!normalizedSearch.length) return;
    const idx = Math.min(Math.max(0, activeMatchIndex), normalizedSearch.length - 1);
    const m = normalizedSearch[idx];
    if (!m) return;
    const maxPos = editor.state.doc.content.size;
    const from = clampInt(m.from, 0, maxPos);
    const to = clampInt(m.to, 0, maxPos);
    if (to <= from) return;
    const tr = editor.state.tr.setSelection(TextSelection.create(editor.state.doc, from, to));
    editor.view.dispatch(tr);

    // Defer until after the selection is applied so coords are stable.
    window.requestAnimationFrame(() => {
      scrollRangeIntoView({ editor, from, to });
    });
  }, [editor, normalizedSearch, activeMatchIndex]);

  return (
    <div className="insight-editor-wrap" onContextMenu={onContextMenu}>
      {overlay}
      <EditorContent editor={editor} className="insight-editor" />
    </div>
  );
}
