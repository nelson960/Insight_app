import { useCallback, useEffect, useMemo, useState, type CSSProperties } from "react";
import { Extension } from "@tiptap/core";
import { Plugin, PluginKey, TextSelection } from "@tiptap/pm/state";
import { Decoration, DecorationSet } from "@tiptap/pm/view";

type EditorLike = {
  chain: () => any;
  commands: any;
  state: any;
  view: any;
  isEditable: boolean;
  isActive: (name: string, attrs?: any) => boolean;
  getAttributes: (name: string) => any;
};

type MenuKind =
  | "format"
  | "format_more"
  | "convert"
  | "convert_heading"
  | "convert_list"
  | "insert"
  | "heading"
  | "list"
  | "table_ops"
  | "code_ops";

type PanelState =
  | {
      kind: "link";
      href: string;
      level: number;
      index: number;
    }
  | null;

type MenuState = {
  x: number;
  y: number;
  stack: MenuKind[];
  activeKeys: string[];
  activeIndices: number[];
  panel: PanelState;
};

type MenuItemSelectContext = {
  level: number;
  index: number;
};

type MenuItem = {
  key: string;
  label: string;
  icon: string;
  submenu?: MenuKind;
  onSelect?: (ctx?: MenuItemSelectContext) => void;
  disabled?: boolean;
  keepOpen?: boolean;
};

function inAncestorFrom($from: any, names: string[]): boolean {
  try {
    for (let depth = $from.depth; depth >= 0; depth -= 1) {
      const node = $from.node(depth);
      const name = String(node?.type?.name || "");
      if (names.includes(name)) return true;
    }
  } catch {
    // ignore
  }
  return false;
}

function pickMenuAtPoint(opts: {
  editor: EditorLike;
  clientX: number;
  clientY: number;
}): { kind: MenuKind; setCaretPos: number | null } {
  const { editor, clientX, clientY } = opts;
  const view = editor.view;
  const state = (view?.state as any) ?? editor.state;
  const sel = state?.selection;
  const doc = state?.doc;
  if (!sel || !doc || !view) return { kind: "insert", setCaretPos: null };

  let clickPos: number | null = null;
  try {
    const found = view.posAtCoords?.({ left: clientX, top: clientY });
    if (found && typeof found.pos === "number") clickPos = found.pos;
  } catch {
    clickPos = null;
  }

  const hasSelection = !sel.empty;
  if (clickPos == null) {
    // If we can't resolve the click into a document position, avoid using the existing caret's
    // ancestor context (it makes the menu feel "stuck" in code/table mode). Treat as a normal
    // selection/caret menu and let the user click to place the cursor explicitly.
    return { kind: hasSelection ? "format" : "insert", setCaretPos: null };
  }

  let effectiveHasSelection = hasSelection;
  let $from = sel.$from;
  let setCaretPos: number | null = null;

  if (hasSelection) {
    const lo = Math.min(sel.from, sel.to);
    const hi = Math.max(sel.from, sel.to);
    const inside = clickPos >= lo && clickPos <= hi;
    if (!inside) {
      effectiveHasSelection = false;
      $from = doc.resolve(clickPos);
      setCaretPos = clickPos;
    }
  } else {
    $from = doc.resolve(clickPos);
    setCaretPos = clickPos;
  }

  if (effectiveHasSelection) return { kind: "format", setCaretPos };
  if (inAncestorFrom($from, ["codeBlock"])) return { kind: "code_ops", setCaretPos };
  if (inAncestorFrom($from, ["table", "tableRow", "tableCell", "tableHeader"]))
    return { kind: "table_ops", setCaretPos };
  return { kind: "insert", setCaretPos };
}

async function copyToClipboard(text: string) {
  const cleaned = String(text || "");
  if (!cleaned) return;
  try {
    await navigator.clipboard.writeText(cleaned);
  } catch {
    // ignore (best-effort)
  }
}

const codeCopyKey = new PluginKey("insightCodeCopyButton");

export const CodeBlockCopyButton = Extension.create({
  name: "insightCodeCopyButton",
  addProseMirrorPlugins() {
    const editor = this.editor as unknown as EditorLike;
    return [
      new Plugin({
        key: codeCopyKey,
        props: {
          decorations(state) {
            if (!editor?.isEditable) return null;
            const sel = state.selection;
            if (!sel || !sel.empty) return null;
            const $from = sel.$from;
            for (let depth = $from.depth; depth >= 0; depth -= 1) {
              const node = $from.node(depth);
              if (String(node?.type?.name || "") !== "codeBlock") continue;
              const pos = $from.start(depth);
              const text = String(node?.textContent || "");
              const btn = document.createElement("button");
              btn.type = "button";
              btn.className = "insight-code-copy-btn";
              btn.textContent = "Copy";
              btn.setAttribute("contenteditable", "false");
              btn.addEventListener("pointerdown", (e) => e.stopPropagation());
              btn.addEventListener("click", (e) => {
                e.preventDefault();
                e.stopPropagation();
                void copyToClipboard(text);
              });
              const deco = Decoration.widget(pos + 1, btn, {
                side: -1,
                stopEvent: () => true,
              });
              return DecorationSet.create(state.doc, [deco]);
            }
            return null;
          },
        },
      }),
    ];
  },
});

function deleteCurrentCodeBlock(editor: EditorLike) {
  const view = editor.view;
  const state = editor.state;
  const sel = state.selection;
  const $from = sel.$from;
  for (let depth = $from.depth; depth >= 0; depth -= 1) {
    const node = $from.node(depth);
    if (String(node?.type?.name || "") !== "codeBlock") continue;
    const from = $from.before(depth);
    const to = $from.after(depth);
    const tr = state.tr.delete(from, to);
    view.dispatch(tr);
    editor.chain().focus().run();
    return;
  }
}

function getCodeBlockText(editor: EditorLike): string {
  const state = editor.state;
  const sel = state.selection;
  const $from = sel.$from;
  for (let depth = $from.depth; depth >= 0; depth -= 1) {
    const node = $from.node(depth);
    if (String(node?.type?.name || "") === "codeBlock") {
      return String(node?.textContent || "");
    }
  }
  return "";
}

function ensureFreshEmptyLine(editor: EditorLike): boolean {
  try {
    const view = editor.view;
    const state = view?.state;
    const doc = state?.doc;
    const sel = state?.selection;
    const paragraph = state?.schema?.nodes?.paragraph;
    if (!view || !doc || !sel || !paragraph) return false;

    const $from = sel.$from;
    let depth = $from.depth;
    while (depth > 0 && !$from.node(depth).isTextblock) depth -= 1;
    if (depth <= 0) return false;

    const node = $from.node(depth);
    const isEmpty = String(node?.textContent || "").trim().length === 0;
    if (isEmpty) return true;

    const insertPos = $from.after(depth);
    let tr = state.tr.insert(insertPos, paragraph.create());
    tr = tr.setSelection(TextSelection.near(tr.doc.resolve(insertPos + 1), 1));
    view.dispatch(tr);
    view.focus();
    return true;
  } catch {
    return false;
  }
}

const MENU_WIDTH = 210;
const MENU_ROW_H = 32;
const MENU_PAD = 6;
const MENU_GAP = 6;
const MENU_MARGIN = 10;

function clamp(n: number, lo: number, hi: number): number {
  if (!Number.isFinite(n)) return lo;
  return Math.max(lo, Math.min(hi, n));
}

function clampInt(n: number, lo: number, hi: number): number {
  return Math.trunc(clamp(n, lo, hi));
}

function menuHeight(items: MenuItem[]): number {
  const count = Math.max(0, items.length);
  return MENU_PAD * 2 + count * MENU_ROW_H;
}

function getMenuBounds(editor: EditorLike): DOMRect {
  const viewDom = editor?.view?.dom as HTMLElement | undefined;
  const scroller =
    (viewDom?.closest?.(".docs-reader-body") as HTMLElement | null) ??
    (viewDom?.closest?.(".docs-pane") as HTMLElement | null) ??
    null;
  if (scroller) return scroller.getBoundingClientRect();
  if (viewDom?.getBoundingClientRect) return viewDom.getBoundingClientRect();
  return new DOMRect(0, 0, window.innerWidth, window.innerHeight);
}

function selectedText(editor: EditorLike): string {
  try {
    const sel = editor?.state?.selection;
    if (!sel || sel.empty) return "";
    return editor.state.doc.textBetween(sel.from, sel.to, "\n\n", "\n\n");
  } catch {
    return "";
  }
}

function LinkPanel(props: {
  href: string;
  onChangeHref: (v: string) => void;
  onApply: () => void;
  onRemove: () => void;
}) {
  return (
    <div className="insight-tools-panel" role="dialog" aria-label="Link">
      <div className="insight-tools-panel-title">Link</div>
      <input
        className="insight-tools-panel-input"
        placeholder="https://…"
        value={props.href}
        onChange={(e) => props.onChangeHref(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            e.preventDefault();
            props.onApply();
          }
        }}
      />
      <div className="insight-tools-panel-actions">
        <button type="button" className="insight-tools-panel-btn" onClick={props.onApply}>
          Apply
        </button>
        <button type="button" className="insight-tools-panel-btn subtle" onClick={props.onRemove}>
          Remove
        </button>
      </div>
    </div>
  );
}

export function useEditorRadialTools(opts: { editor: EditorLike | null; enabled: boolean }) {
  const { editor, enabled } = opts;
  const [menu, setMenu] = useState<MenuState | null>(null);

  const close = useCallback(() => setMenu(null), []);

  useEffect(() => {
    if (!menu) return;
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") {
        e.preventDefault();
        close();
      }
    }
    window.addEventListener("keydown", onKeyDown, true);
    return () => window.removeEventListener("keydown", onKeyDown, true);
  }, [menu, close]);

  const onContextMenu = useCallback(
    (e: any) => {
      if (!enabled) return;
      if (!editor) return;
      e.preventDefault();
      e.stopPropagation();
      editor.chain().focus().run();
      const { kind, setCaretPos } = pickMenuAtPoint({
        editor,
        clientX: Number(e.clientX) || 0,
        clientY: Number(e.clientY) || 0,
      });
      if (typeof setCaretPos === "number") {
        try {
          const state = editor.view.state;
          const resolved = state.doc.resolve(setCaretPos);
          const nextSel = TextSelection.near(resolved, 1);
          const tr = state.tr.setSelection(nextSel);
          editor.view.dispatch(tr);
        } catch {
          // ignore (best-effort)
        }
      }
      setMenu({
        x: Number(e.clientX) || 0,
        y: Number(e.clientY) || 0,
        stack: [kind],
        activeKeys: [],
        activeIndices: [],
        panel: null,
      });
    },
    [editor, enabled]
  );

  const getItems = useCallback(
    (kind: MenuKind): MenuItem[] => {
      if (!editor) return [];
      const ch = editor.chain().focus();

      if (kind === "format") {
        const currentHref = String(editor.getAttributes("link")?.href || "");
        return [
          { key: "bold", label: "Bold", icon: "B", onSelect: () => ch.toggleBold().run() },
          { key: "italic", label: "Italic", icon: "I", onSelect: () => ch.toggleItalic().run() },
          { key: "underline", label: "Underline", icon: "U", onSelect: () => ch.toggleUnderline?.().run?.() },
          { key: "highlight", label: "Highlight", icon: "▧", onSelect: () => ch.toggleHighlight?.().run?.() },
          {
            key: "link",
            label: "Link…",
            icon: "🔗",
            keepOpen: true,
            onSelect: (ctx) => {
              setMenu((prev) => {
                if (!prev) return prev;
                return {
                  ...prev,
                  panel: {
                    kind: "link",
                    href: currentHref,
                    level: ctx?.level ?? 0,
                    index: ctx?.index ?? 0,
                  },
                };
              });
            },
          },
          { key: "more", label: "More", icon: "⋯", submenu: "format_more", keepOpen: true },
        ];
      }

      if (kind === "format_more") {
        const canCopy = selectedText(editor).trim().length > 0;
        return [
          { key: "strike", label: "Strikethrough", icon: "S", onSelect: () => ch.toggleStrike().run() },
          { key: "code", label: "Inline code", icon: "<>", onSelect: () => ch.toggleCode().run() },
          { key: "clear", label: "Clear formatting", icon: "×", onSelect: () => ch.unsetAllMarks().run() },
          { key: "convert", label: "Convert block", icon: "¶", submenu: "convert", keepOpen: true },
          {
            key: "copy",
            label: "Copy",
            icon: "⧉",
            disabled: !canCopy,
            onSelect: () => void copyToClipboard(selectedText(editor)),
          },
        ];
      }

      if (kind === "convert") {
        return [
          { key: "p", label: "Paragraph", icon: "¶", onSelect: () => ch.setParagraph().run() },
          { key: "heading", label: "Heading", icon: "H", submenu: "convert_heading", keepOpen: true },
          { key: "list", label: "List", icon: "≡", submenu: "convert_list", keepOpen: true },
          { key: "quote", label: "Quote", icon: "❝", onSelect: () => ch.toggleBlockquote().run() },
          { key: "code", label: "Code block", icon: "</>", onSelect: () => ch.toggleCodeBlock().run() },
        ];
      }

      if (kind === "convert_heading") {
        return [
          { key: "h1", label: "H1", icon: "H1", onSelect: () => ch.toggleHeading({ level: 1 }).run() },
          { key: "h2", label: "H2", icon: "H2", onSelect: () => ch.toggleHeading({ level: 2 }).run() },
          { key: "h3", label: "H3", icon: "H3", onSelect: () => ch.toggleHeading({ level: 3 }).run() },
          { key: "h4", label: "H4", icon: "H4", onSelect: () => ch.toggleHeading({ level: 4 }).run() },
        ];
      }

      if (kind === "convert_list") {
        return [
          { key: "bullets", label: "Bullets", icon: "•", onSelect: () => ch.toggleBulletList().run() },
          { key: "numbered", label: "Numbered", icon: "1.", onSelect: () => ch.toggleOrderedList().run() },
          { key: "task", label: "Task", icon: "☐", onSelect: () => ch.toggleTaskList?.().run?.() },
        ];
      }

      if (kind === "insert") {
        return [
          {
            key: "text",
            label: "Text",
            icon: "T",
            onSelect: () => {
              ensureFreshEmptyLine(editor);
              editor.chain().focus().setParagraph().run();
            },
          },
          { key: "heading", label: "Heading", icon: "H", submenu: "heading", keepOpen: true },
          { key: "list", label: "List", icon: "≡", submenu: "list", keepOpen: true },
          {
            key: "table",
            label: "Table",
            icon: "▦",
            onSelect: () => {
              ensureFreshEmptyLine(editor);
              editor.chain().focus().insertTable?.({ rows: 3, cols: 3, withHeaderRow: true })?.run?.();
            },
          },
          {
            key: "code",
            label: "Code block",
            icon: "</>",
            onSelect: () => {
              ensureFreshEmptyLine(editor);
              editor.chain().focus().toggleCodeBlock().run();
            },
          },
          {
            key: "quote",
            label: "Quote",
            icon: "❝",
            onSelect: () => {
              ensureFreshEmptyLine(editor);
              editor.chain().focus().toggleBlockquote().run();
            },
          },
        ];
      }

      if (kind === "heading") {
        return [
          {
            key: "h1",
            label: "H1",
            icon: "H1",
            onSelect: () => {
              ensureFreshEmptyLine(editor);
              editor.chain().focus().toggleHeading({ level: 1 }).run();
            },
          },
          {
            key: "h2",
            label: "H2",
            icon: "H2",
            onSelect: () => {
              ensureFreshEmptyLine(editor);
              editor.chain().focus().toggleHeading({ level: 2 }).run();
            },
          },
          {
            key: "h3",
            label: "H3",
            icon: "H3",
            onSelect: () => {
              ensureFreshEmptyLine(editor);
              editor.chain().focus().toggleHeading({ level: 3 }).run();
            },
          },
          {
            key: "h4",
            label: "H4",
            icon: "H4",
            onSelect: () => {
              ensureFreshEmptyLine(editor);
              editor.chain().focus().toggleHeading({ level: 4 }).run();
            },
          },
        ];
      }

      if (kind === "list") {
        return [
          {
            key: "bullets",
            label: "Bullets",
            icon: "•",
            onSelect: () => {
              ensureFreshEmptyLine(editor);
              editor.chain().focus().toggleBulletList().run();
            },
          },
          {
            key: "numbered",
            label: "Numbered",
            icon: "1.",
            onSelect: () => {
              ensureFreshEmptyLine(editor);
              editor.chain().focus().toggleOrderedList().run();
            },
          },
          {
            key: "task",
            label: "Task",
            icon: "☐",
            onSelect: () => {
              ensureFreshEmptyLine(editor);
              editor.chain().focus().toggleTaskList?.().run?.();
            },
          },
        ];
      }

      if (kind === "table_ops") {
        return [
          { key: "row_add", label: "Add row", icon: "+R", onSelect: () => ch.addRowAfter?.().run?.() },
          { key: "row_del", label: "Remove row", icon: "−R", onSelect: () => ch.deleteRow?.().run?.() },
          { key: "col_add", label: "Add column", icon: "+C", onSelect: () => ch.addColumnAfter?.().run?.() },
          { key: "col_del", label: "Remove column", icon: "−C", onSelect: () => ch.deleteColumn?.().run?.() },
          { key: "hdr", label: "Header row", icon: "H", onSelect: () => ch.toggleHeaderRow?.().run?.() },
          { key: "tbl_del", label: "Delete table", icon: "🗑", onSelect: () => ch.deleteTable?.().run?.() },
        ];
      }

      if (kind === "code_ops") {
        const text = getCodeBlockText(editor);
        return [
          { key: "copy", label: "Copy code", icon: "⧉", onSelect: () => void copyToClipboard(text) },
          { key: "to_p", label: "Convert to text", icon: "T", onSelect: () => ch.toggleCodeBlock().run() },
          { key: "del", label: "Delete block", icon: "🗑", onSelect: () => deleteCurrentCodeBlock(editor) },
        ];
      }

      return [];
    },
    [editor]
  );

  const overlay = useMemo(() => {
    if (!menu || !editor || !enabled) return null;
    const bounds = getMenuBounds(editor);

    const columns = (() => {
      const stack = Array.isArray(menu.stack) ? menu.stack : [];
      const out: Array<{ kind: MenuKind; x: number; y: number; items: MenuItem[] }> = [];
      if (!stack.length) return out;

      const rootItems = getItems(stack[0]);
      const rootH = menuHeight(rootItems);
      const maxRootX = bounds.right - MENU_MARGIN - MENU_WIDTH;
      const maxRootY = bounds.bottom - MENU_MARGIN - rootH;
      let x = clamp(menu.x, bounds.left + MENU_MARGIN, maxRootX);
      let y = clamp(menu.y, bounds.top + MENU_MARGIN, maxRootY);
      out.push({ kind: stack[0], x, y, items: rootItems });

      for (let level = 1; level < stack.length; level += 1) {
        const prev = out[level - 1];
        const prevItems = prev.items;
        const idxRaw = menu.activeIndices[level - 1] ?? 0;
        const idx = clampInt(idxRaw, 0, Math.max(0, prevItems.length - 1));
        const items = getItems(stack[level]);
        const h = menuHeight(items);

        const rightX = prev.x + MENU_WIDTH + MENU_GAP;
        const leftX = prev.x - MENU_WIDTH - MENU_GAP;
        const canRight = rightX + MENU_WIDTH <= bounds.right - MENU_MARGIN;
        const canLeft = leftX >= bounds.left + MENU_MARGIN;
        let colX = canRight ? rightX : canLeft ? leftX : rightX;
        colX = clamp(colX, bounds.left + MENU_MARGIN, bounds.right - MENU_MARGIN - MENU_WIDTH);

        let colY = prev.y + idx * MENU_ROW_H;
        colY = clamp(colY, bounds.top + MENU_MARGIN, bounds.bottom - MENU_MARGIN - h);

        out.push({ kind: stack[level], x: colX, y: colY, items });
      }

      return out;
    })();

    const linkPanelPos = (() => {
      if (menu.panel?.kind !== "link") return null;
      const level = clampInt(menu.panel.level, 0, Math.max(0, columns.length - 1));
      const col = columns[level];
      if (!col) return null;

      const panelW = 260;
      const panelH = 140;
      const rightX = col.x + MENU_WIDTH + MENU_GAP;
      const leftX = col.x - panelW - MENU_GAP;
      const canRight = rightX + panelW <= bounds.right - MENU_MARGIN;
      const canLeft = leftX >= bounds.left + MENU_MARGIN;
      let x = canRight ? rightX : canLeft ? leftX : rightX;
      x = clamp(x, bounds.left + MENU_MARGIN, bounds.right - MENU_MARGIN - panelW);

      const idx = clampInt(menu.panel.index, 0, Math.max(0, col.items.length - 1));
      let y = col.y + idx * MENU_ROW_H;
      y = clamp(y, bounds.top + MENU_MARGIN, bounds.bottom - MENU_MARGIN - panelH);
      return { x, y, w: panelW };
    })();

    return (
      <div
        className="insight-menu-overlay"
        role="presentation"
        onPointerDown={() => setMenu(null)}
        onContextMenu={(e) => {
          e.preventDefault();
          e.stopPropagation();
          editor.chain().focus().run();
          const { kind, setCaretPos } = pickMenuAtPoint({
            editor,
            clientX: Number(e.clientX) || 0,
            clientY: Number(e.clientY) || 0,
          });
          if (typeof setCaretPos === "number") {
            try {
              const state = editor.view.state;
              const resolved = state.doc.resolve(setCaretPos);
              const nextSel = TextSelection.near(resolved, 1);
              const tr = state.tr.setSelection(nextSel);
              editor.view.dispatch(tr);
            } catch {
              // ignore
            }
          }
          setMenu({
            x: e.clientX,
            y: e.clientY,
            stack: [kind],
            activeKeys: [],
            activeIndices: [],
            panel: null,
          });
        }}
      >
        {columns.map((col, level) => (
          <div
            key={`${col.kind}-${level}`}
            className="insight-menu-col"
            style={{ left: col.x, top: col.y } as CSSProperties}
            onPointerDown={(e) => e.stopPropagation()}
          >
            {col.items.map((item, index) => {
              const isDisabled = !!item.disabled || (!item.onSelect && !item.submenu);
              const isActive = (menu.activeKeys[level] ?? "") === item.key;
              const keepOpen = item.keepOpen ?? !!item.submenu;

              return (
                <button
                  key={item.key}
                  type="button"
                  className={`insight-menu-row${isActive ? " active" : ""}${isDisabled ? " disabled" : ""}`}
                  onMouseEnter={() => {
                    setMenu((prev) => {
                      if (!prev) return prev;
                      const next: MenuState = { ...prev };
                      next.activeKeys = [...(prev.activeKeys ?? [])];
                      next.activeIndices = [...(prev.activeIndices ?? [])];
                      next.activeKeys[level] = item.key;
                      next.activeIndices[level] = index;

                      // Hover never opens submenus (click-only). It can, however, close deeper stacks so
                      // the visible submenu doesn't mismatch the highlighted parent row.
                      if (prev.stack.length > level + 1) {
                        next.stack = prev.stack.slice(0, level + 1);
                        next.activeKeys = next.activeKeys.slice(0, level + 1);
                        next.activeIndices = next.activeIndices.slice(0, level + 1);
                        if (next.panel && next.panel.level >= next.stack.length) next.panel = null;
                      }
                      return next;
                    });
                  }}
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    if (isDisabled) return;

                    if (item.submenu) {
                      setMenu((prev) => {
                        if (!prev) return prev;
                        return {
                          ...prev,
                          stack: [...prev.stack.slice(0, level + 1), item.submenu!],
                          activeKeys: [...prev.activeKeys.slice(0, level), item.key],
                          activeIndices: [...prev.activeIndices.slice(0, level), index],
                          panel: null,
                        };
                      });
                      return;
                    }

                    item.onSelect?.({ level, index });
                    if (!keepOpen) setMenu(null);
                  }}
                  disabled={isDisabled}
                >
                  <span className="insight-menu-icon" aria-hidden="true">
                    {item.icon}
                  </span>
                  <span className="insight-menu-label">{item.label}</span>
                  {item.submenu ? (
                    <span className="insight-menu-arrow" aria-hidden="true">
                      ›
                    </span>
                  ) : null}
                </button>
              );
            })}
          </div>
        ))}

        {menu.panel?.kind === "link" && linkPanelPos ? (
          <div
            className="insight-tools-panel-anchor"
            style={{ left: linkPanelPos.x, top: linkPanelPos.y, transform: "none" } as CSSProperties}
            onPointerDown={(e) => e.stopPropagation()}
          >
            <div style={{ width: linkPanelPos.w } as CSSProperties}>
              <LinkPanel
                href={menu.panel.href}
                onChangeHref={(v) => {
                  setMenu((prev) => {
                    if (!prev || prev.panel?.kind !== "link") return prev;
                    return { ...prev, panel: { ...prev.panel, href: v } };
                  });
                }}
                onApply={() => {
                  const href = menu.panel?.kind === "link" ? menu.panel.href.trim() : "";
                  if (href) {
                    editor.chain().focus().extendMarkRange("link").setLink({ href }).run();
                  }
                  setMenu(null);
                }}
                onRemove={() => {
                  editor.chain().focus().extendMarkRange("link").unsetLink().run();
                  setMenu(null);
                }}
              />
            </div>
          </div>
        ) : null}
      </div>
    );
  }, [menu, editor, enabled, getItems]);

  return { onContextMenu, overlay };
}
