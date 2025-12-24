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
    }
  | null;

type MenuState = {
  x: number;
  y: number;
  stack: MenuKind[];
  panel: PanelState;
};

type RadialItem = {
  key: string;
  label: string;
  icon: string;
  onSelect?: () => void;
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

function degToRad(deg: number): number {
  return (deg * Math.PI) / 180;
}

function polar(opts: { cx: number; cy: number; r: number; deg: number }) {
  const rad = degToRad(opts.deg);
  return {
    x: opts.cx + Math.cos(rad) * opts.r,
    y: opts.cy + Math.sin(rad) * opts.r,
  };
}

function annularSectorPath(opts: {
  cx: number;
  cy: number;
  rInner: number;
  rOuter: number;
  startDeg: number;
  endDeg: number;
}): string {
  const sweepDeg = Math.abs(opts.endDeg - opts.startDeg);
  const largeArc = sweepDeg > 180 ? 1 : 0;
  const p0 = polar({ cx: opts.cx, cy: opts.cy, r: opts.rOuter, deg: opts.startDeg });
  const p1 = polar({ cx: opts.cx, cy: opts.cy, r: opts.rOuter, deg: opts.endDeg });
  const p2 = polar({ cx: opts.cx, cy: opts.cy, r: opts.rInner, deg: opts.endDeg });
  const p3 = polar({ cx: opts.cx, cy: opts.cy, r: opts.rInner, deg: opts.startDeg });
  return [
    `M ${p0.x.toFixed(3)} ${p0.y.toFixed(3)}`,
    `A ${opts.rOuter} ${opts.rOuter} 0 ${largeArc} 1 ${p1.x.toFixed(3)} ${p1.y.toFixed(3)}`,
    `L ${p2.x.toFixed(3)} ${p2.y.toFixed(3)}`,
    `A ${opts.rInner} ${opts.rInner} 0 ${largeArc} 0 ${p3.x.toFixed(3)} ${p3.y.toFixed(3)}`,
    "Z",
  ].join(" ");
}

function RadialMenu(props: { items: RadialItem[]; onClose: () => void; onCenter: () => void }) {
  const visible = useMemo(() => props.items.slice(0, 6), [props.items]);
  const geom = useMemo(() => {
    const n = Math.max(0, Math.min(6, visible.length));
    const size = 280;
    const cx = size / 2;
    const cy = size / 2;
    const rOuter = 132;
    const rInner = 64;
    const stepDeg = n > 0 ? 360 / n : 360;
    const startDeg = -90 - stepDeg / 2;
    return { n, size, cx, cy, rOuter, rInner, stepDeg, startDeg };
  }, [visible.length]);

  return (
    <div className="insight-radial">
      <svg
        className="insight-pie-svg"
        viewBox={`0 0 ${geom.size} ${geom.size}`}
        width={geom.size}
        height={geom.size}
        role="presentation"
      >
        {visible.map((item, idx) => {
          const n = Math.max(1, geom.n);
          const step = geom.stepDeg || 360 / n;
          const start = geom.startDeg;
          const startDeg = start + idx * step;
          const endDeg = start + (idx + 1) * step;
          const safeEnd = geom.n === 1 ? startDeg + 359.999 : endDeg;
          const d = annularSectorPath({
            cx: geom.cx,
            cy: geom.cy,
            rInner: geom.rInner,
            rOuter: geom.rOuter,
            startDeg,
            endDeg: safeEnd,
          });

          const mid = (startDeg + safeEnd) / 2;
          const centerR = (geom.rInner + geom.rOuter) / 2;
          const p = polar({ cx: geom.cx, cy: geom.cy, r: centerR, deg: mid });

        const disabled = !!item.disabled || !item.onSelect;
        const style = {
          ["--delay" as any]: `${idx * 14}ms`,
        } as CSSProperties;
        return (
          <g
            key={item.key}
            className={`insight-pie-slice-group ${disabled ? "disabled" : ""}`}
            style={style}
            onClick={() => {
              if (disabled) return;
              item.onSelect?.();
              if (!item.keepOpen) props.onClose();
            }}
          >
            <path className="insight-pie-slice" d={d} />
            <text
              className="insight-radial-icon"
              x={p.x}
              y={p.y - 6}
              textAnchor="middle"
              dominantBaseline="middle"
            >
              {item.icon}
            </text>
            <text
              className="insight-radial-label"
              x={p.x}
              y={p.y + 12}
              textAnchor="middle"
              dominantBaseline="middle"
            >
              {item.label}
            </text>
          </g>
        );
      })}
      </svg>

      <button
        type="button"
        className="insight-radial-center-btn"
        onClick={props.onCenter}
        aria-label="Back"
        title="Back"
      >
        ↩
      </button>
    </div>
  );
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
        panel: null,
      });
    },
    [editor, enabled]
  );

  const currentKind = menu?.stack[menu.stack.length - 1] ?? null;

  const items = useMemo<RadialItem[]>(() => {
    if (!editor || !currentKind) return [];
    const ch = editor.chain().focus();
    if (currentKind === "format") {
      const currentHref = String(editor.getAttributes("link")?.href || "");
      return [
        { key: "bold", label: "Bold", icon: "B", onSelect: () => ch.toggleBold().run() },
        { key: "italic", label: "Italic", icon: "I", onSelect: () => ch.toggleItalic().run() },
        { key: "underline", label: "Underline", icon: "U", onSelect: () => ch.toggleUnderline?.().run?.() },
        { key: "highlight", label: "Highlight", icon: "▧", onSelect: () => ch.toggleHighlight?.().run?.() },
        {
          key: "link",
          label: "Link",
          icon: "🔗",
          keepOpen: true,
          onSelect: () => {
            setMenu((prev) => {
              if (!prev) return prev;
              return { ...prev, panel: { kind: "link", href: currentHref } };
            });
          },
        },
        {
          key: "more",
          label: "More",
          icon: "⋯",
          keepOpen: true,
          onSelect: () =>
            setMenu((prev) => (prev ? { ...prev, stack: [...prev.stack, "format_more"] } : prev)),
        },
      ];
    }

    if (currentKind === "format_more") {
      return [
        { key: "strike", label: "Strike", icon: "S", onSelect: () => ch.toggleStrike().run() },
        { key: "code", label: "Inline code", icon: "<>", onSelect: () => ch.toggleCode().run() },
        {
          key: "block",
          label: "Block",
          icon: "¶",
          keepOpen: true,
          onSelect: () => setMenu((prev) => (prev ? { ...prev, stack: [...prev.stack, "convert"] } : prev)),
        },
        { key: "clear", label: "Clear", icon: "×", onSelect: () => ch.unsetAllMarks().run() },
      ];
    }

    if (currentKind === "convert") {
      return [
        { key: "p", label: "Paragraph", icon: "¶", onSelect: () => ch.setParagraph().run() },
        {
          key: "heading",
          label: "Heading",
          icon: "H",
          keepOpen: true,
          onSelect: () =>
            setMenu((prev) => (prev ? { ...prev, stack: [...prev.stack, "convert_heading"] } : prev)),
        },
        {
          key: "list",
          label: "List",
          icon: "≡",
          keepOpen: true,
          onSelect: () =>
            setMenu((prev) => (prev ? { ...prev, stack: [...prev.stack, "convert_list"] } : prev)),
        },
        { key: "code", label: "Code block", icon: "</>", onSelect: () => ch.toggleCodeBlock().run() },
        { key: "quote", label: "Quote", icon: "❝", onSelect: () => ch.toggleBlockquote().run() },
      ];
    }

    if (currentKind === "convert_heading") {
      return [
        { key: "h1", label: "H1", icon: "H1", onSelect: () => ch.toggleHeading({ level: 1 }).run() },
        { key: "h2", label: "H2", icon: "H2", onSelect: () => ch.toggleHeading({ level: 2 }).run() },
        { key: "h3", label: "H3", icon: "H3", onSelect: () => ch.toggleHeading({ level: 3 }).run() },
        { key: "h4", label: "H4", icon: "H4", onSelect: () => ch.toggleHeading({ level: 4 }).run() },
      ];
    }

    if (currentKind === "convert_list") {
      return [
        { key: "bullets", label: "Bullets", icon: "•", onSelect: () => ch.toggleBulletList().run() },
        { key: "numbered", label: "Numbered", icon: "1.", onSelect: () => ch.toggleOrderedList().run() },
        { key: "task", label: "Task", icon: "☐", onSelect: () => ch.toggleTaskList?.().run?.() },
      ];
    }

    if (currentKind === "insert") {
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
        {
          key: "heading",
          label: "Heading",
          icon: "H",
          keepOpen: true,
          onSelect: () => setMenu((prev) => (prev ? { ...prev, stack: [...prev.stack, "heading"] } : prev)),
        },
        {
          key: "list",
          label: "List",
          icon: "≡",
          keepOpen: true,
          onSelect: () => setMenu((prev) => (prev ? { ...prev, stack: [...prev.stack, "list"] } : prev)),
        },
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
          label: "Code",
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

    if (currentKind === "heading") {
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

    if (currentKind === "list") {
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

    if (currentKind === "table_ops") {
      return [
        { key: "row_add", label: "Add row", icon: "+R", onSelect: () => ch.addRowAfter?.().run?.() },
        { key: "row_del", label: "Remove row", icon: "−R", onSelect: () => ch.deleteRow?.().run?.() },
        { key: "col_add", label: "Add column", icon: "+C", onSelect: () => ch.addColumnAfter?.().run?.() },
        { key: "col_del", label: "Remove column", icon: "−C", onSelect: () => ch.deleteColumn?.().run?.() },
        { key: "hdr", label: "Header row", icon: "H", onSelect: () => ch.toggleHeaderRow?.().run?.() },
        { key: "tbl_del", label: "Delete table", icon: "🗑", onSelect: () => ch.deleteTable?.().run?.() },
      ];
    }

    if (currentKind === "code_ops") {
      const text = getCodeBlockText(editor);
      return [
        { key: "copy", label: "Copy", icon: "⧉", onSelect: () => void copyToClipboard(text) },
        { key: "to_p", label: "To text", icon: "T", onSelect: () => ch.toggleCodeBlock().run() },
        { key: "del", label: "Delete", icon: "🗑", onSelect: () => deleteCurrentCodeBlock(editor) },
      ];
    }

    return [];
  }, [editor, currentKind]);

  const overlay = useMemo(() => {
    if (!menu || !editor || !enabled) return null;
    return (
      <div
        className="insight-radial-overlay"
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
          setMenu({ x: e.clientX, y: e.clientY, stack: [kind], panel: null });
        }}
      >
        <div
          className="insight-radial-anchor"
          style={{ left: menu.x, top: menu.y }}
          onPointerDown={(e) => e.stopPropagation()}
        >
          <RadialMenu
            items={items}
            onClose={() => setMenu(null)}
            onCenter={() => {
              setMenu((prev) => {
                if (!prev) return prev;
                if (prev.panel) return { ...prev, panel: null };
                if (prev.stack.length > 1) return { ...prev, stack: prev.stack.slice(0, -1) };
                return null;
              });
            }}
          />
          {menu.panel?.kind === "link" ? (
            <div className="insight-tools-panel-anchor" onPointerDown={(e) => e.stopPropagation()}>
              <LinkPanel
                href={menu.panel.href}
                onChangeHref={(v) => {
                  setMenu((prev) => {
                    if (!prev || prev.panel?.kind !== "link") return prev;
                    return { ...prev, panel: { kind: "link", href: v } };
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
          ) : null}
        </div>
      </div>
    );
  }, [menu, editor, enabled, items]);

  return { onContextMenu, overlay };
}
