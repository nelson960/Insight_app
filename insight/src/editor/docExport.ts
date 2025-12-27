type PMMark = {
  type?: string;
  attrs?: Record<string, any>;
};

type PMNode = {
  type?: string;
  attrs?: Record<string, any>;
  text?: string;
  marks?: PMMark[];
  content?: PMNode[];
};

function normalizeType(t: unknown): string {
  return String(t ?? "").trim();
}

function escapeHtml(text: string): string {
  return String(text ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function escapeMarkdownText(text: string): string {
  // Minimal escaping for common Markdown special chars in plain text.
  // Leave newlines as-is (block serializer handles spacing).
  return String(text ?? "").replace(/[\\`*_{}\[\]()#+\-.!|>]/g, "\\$&");
}

function escapeMarkdownTableCell(text: string): string {
  return String(text ?? "").replace(/\|/g, "\\|").replace(/\n/g, " ").replace(/\r/g, " ");
}

function getMark(marks: PMMark[] | undefined, name: string): PMMark | null {
  if (!Array.isArray(marks)) return null;
  for (const m of marks) {
    if (normalizeType(m?.type) === name) return m;
  }
  return null;
}

function renderInlineTextMarkdown(node: PMNode): string {
  const raw = String(node.text ?? "");
  const marks = Array.isArray(node.marks) ? node.marks : [];

  const code = getMark(marks, "code");
  if (code) {
    const inner = raw.replace(/`/g, "\\`");
    return `\`${inner}\``;
  }

  let out = escapeMarkdownText(raw);

  const bold = !!getMark(marks, "bold");
  const italic = !!getMark(marks, "italic");
  const strike = !!getMark(marks, "strike");
  const underline = !!getMark(marks, "underline");
  const highlight = !!getMark(marks, "highlight");
  const link = getMark(marks, "link");

  if (bold) out = `**${out}**`;
  if (italic) out = `*${out}*`;
  if (strike) out = `~~${out}~~`;
  if (underline) out = `<u>${out}</u>`;
  if (highlight) out = `<mark>${out}</mark>`;
  if (link) {
    const href = String(link.attrs?.href ?? "").trim();
    if (href) out = `[${out}](${href})`;
  }

  return out;
}

function renderInlineTextPlain(node: PMNode): string {
  return String(node.text ?? "");
}

function renderInlineMarkdown(nodes: PMNode[] | undefined): string {
  if (!Array.isArray(nodes)) return "";
  const parts: string[] = [];
  for (const n of nodes) {
    const t = normalizeType(n?.type).toLowerCase();
    if (t === "text") {
      parts.push(renderInlineTextMarkdown(n));
      continue;
    }
    if (t === "hardbreak") {
      parts.push("  \n");
      continue;
    }
    // Fallback: inline content
    if (Array.isArray(n?.content)) parts.push(renderInlineMarkdown(n.content));
  }
  return parts.join("");
}

function renderInlinePlain(nodes: PMNode[] | undefined): string {
  if (!Array.isArray(nodes)) return "";
  const parts: string[] = [];
  for (const n of nodes) {
    const t = normalizeType(n?.type).toLowerCase();
    if (t === "text") {
      parts.push(renderInlineTextPlain(n));
      continue;
    }
    if (t === "hardbreak") {
      parts.push("\n");
      continue;
    }
    if (Array.isArray(n?.content)) parts.push(renderInlinePlain(n.content));
  }
  return parts.join("");
}

type BlockCtx = {
  indent: string;
  orderedIndex?: number;
  inBlockquote?: boolean;
};

function prefixLines(lines: string[], prefix: string): string[] {
  if (!prefix) return lines;
  return lines.map((l) => (l ? `${prefix}${l}` : prefix.trimEnd()));
}

function trimTrailingBlanks(lines: string[]): string[] {
  const out = [...lines];
  while (out.length && out[out.length - 1].trim() === "") out.pop();
  return out;
}

function blockToMarkdownLines(node: PMNode, ctx: BlockCtx): string[] {
  const t = normalizeType(node?.type).toLowerCase();
  const indent = ctx.indent ?? "";

  if (t === "paragraph") {
    const line = renderInlineMarkdown(node.content).trimEnd();
    if (!line) return [];
    return prefixLines([line], indent);
  }

  if (t === "heading") {
    const level = Number(node.attrs?.level ?? 1);
    const safeLevel = Number.isFinite(level) ? Math.min(Math.max(1, level), 6) : 1;
    const hashes = "#".repeat(safeLevel);
    const line = `${hashes} ${renderInlineMarkdown(node.content).trim()}`.trimEnd();
    return prefixLines([line], indent);
  }

  if (t === "blockquote") {
    const inner = blocksToMarkdownLines(node.content, { ...ctx, indent: "", inBlockquote: true });
    const prefixed = inner.map((l) => (l ? `> ${l}` : ">"));
    return prefixLines(prefixed, indent);
  }

  if (t === "horizontalrule") {
    return prefixLines(["---"], indent);
  }

  if (t === "codeblock") {
    const lang = String(node.attrs?.language ?? "").trim();
    const fence = lang ? `\`\`\`${lang}` : "```";
    const text = renderInlinePlain(node.content).replace(/\r\n/g, "\n").replace(/\r/g, "\n");
    const body = text.endsWith("\n") ? text.slice(0, -1) : text;
    const lines = [fence, ...body.split("\n"), "```"];
    return prefixLines(lines, indent);
  }

  if (t === "bulletlist" || t === "orderedlist" || t === "tasklist") {
    const items = Array.isArray(node.content) ? node.content : [];
    const orderedStart = Number(node.attrs?.start ?? 1);
    let ordinal = Number.isFinite(orderedStart) ? orderedStart : 1;
    const out: string[] = [];

    for (const it of items) {
      const itType = normalizeType(it?.type).toLowerCase();
      if (itType !== "listitem" && itType !== "taskitem") continue;

      const checked = itType === "taskitem" ? !!it.attrs?.checked : false;
      const marker =
        t === "orderedlist"
          ? `${ordinal}. `
          : t === "tasklist"
            ? `- [${checked ? "x" : " "}] `
            : "- ";
      if (t === "orderedlist") ordinal += 1;

      const continuationIndent = indent + " ".repeat(marker.length);
      const nestedIndent = continuationIndent + "  ";

      const children = Array.isArray(it.content) ? it.content : [];
      const childBlocks: string[][] = [];
      for (const c of children) {
        const ct = normalizeType(c?.type).toLowerCase();
        if (ct === "bulletlist" || ct === "orderedlist" || ct === "tasklist") {
          const nestedLines = blockToMarkdownLines(c, { ...ctx, indent: nestedIndent });
          if (nestedLines.length) childBlocks.push(nestedLines);
          continue;
        }
        const blk = blockToMarkdownLines(c, { ...ctx, indent: continuationIndent });
        if (blk.length) childBlocks.push(blk);
      }

      const flat = childBlocks.flat();
      if (!flat.length) {
        out.push(`${indent}${marker}`.trimEnd());
        continue;
      }

      // First non-empty line becomes the marker line.
      let firstIdx = flat.findIndex((l) => l.trim().length > 0);
      if (firstIdx < 0) firstIdx = 0;
      const firstLine = flat[firstIdx].trimStart();
      out.push(`${indent}${marker}${firstLine}`.trimEnd());
      for (let i = firstIdx + 1; i < flat.length; i += 1) {
        const l = flat[i];
        out.push(l ? `${continuationIndent}${l.trimStart()}` : "");
      }
    }

    return trimTrailingBlanks(out);
  }

  if (t === "table") {
    const rows = Array.isArray(node.content) ? node.content : [];
    const grid: string[][] = [];
    let maxCols = 0;
    for (const r of rows) {
      if (normalizeType(r?.type).toLowerCase() !== "tablerow") continue;
      const cells = Array.isArray(r.content) ? r.content : [];
      const row: string[] = [];
      for (const c of cells) {
        const ct = normalizeType(c?.type).toLowerCase();
        if (ct !== "tablecell" && ct !== "tableheader") continue;
        const cellText = renderInlinePlain(c.content).replace(/\s+/g, " ").trim();
        row.push(escapeMarkdownTableCell(cellText));
      }
      if (row.length) {
        maxCols = Math.max(maxCols, row.length);
        grid.push(row);
      }
    }
    if (!grid.length) return [];
    maxCols = Math.max(maxCols, 1);
    const normalized = grid.map((r) => {
      const row = [...r];
      while (row.length < maxCols) row.push("");
      return row;
    });
    const header = normalized[0];
    const sep = header.map(() => "---");
    const out: string[] = [];
    out.push(`${indent}| ${header.join(" | ")} |`);
    out.push(`${indent}| ${sep.join(" | ")} |`);
    for (let i = 1; i < normalized.length; i += 1) {
      out.push(`${indent}| ${normalized[i].join(" | ")} |`);
    }
    return out;
  }

  // Fallback: treat as a block container.
  if (Array.isArray(node.content)) return blocksToMarkdownLines(node.content, ctx);
  return [];
}

function blocksToMarkdownLines(nodes: PMNode[] | undefined, ctx: BlockCtx): string[] {
  if (!Array.isArray(nodes)) return [];
  const out: string[] = [];
  for (const n of nodes) {
    const block = blockToMarkdownLines(n, ctx);
    if (!block.length) continue;
    if (out.length && out[out.length - 1].trim() !== "") out.push("");
    out.push(...block);
  }
  return trimTrailingBlanks(out);
}

export function proseMirrorDocToMarkdown(doc: any): string {
  if (!doc || typeof doc !== "object") return "";
  const root = doc as PMNode;
  const lines = blocksToMarkdownLines(root.content, { indent: "" });
  return lines.join("\n").trim() + "\n";
}

function blockToPlainLines(node: PMNode, ctx: BlockCtx): string[] {
  const t = normalizeType(node?.type).toLowerCase();
  const indent = ctx.indent ?? "";

  if (t === "paragraph") {
    const line = renderInlinePlain(node.content).trimEnd();
    if (!line) return [];
    return prefixLines([line], indent);
  }

  if (t === "heading") {
    const line = renderInlinePlain(node.content).trimEnd();
    if (!line) return [];
    return prefixLines([line], indent);
  }

  if (t === "blockquote") {
    const inner = blocksToPlainLines(node.content, { ...ctx, indent: "" });
    const prefixed = inner.map((l) => (l ? `> ${l}` : ">"));
    return prefixLines(prefixed, indent);
  }

  if (t === "horizontalrule") {
    return prefixLines(["—".repeat(32)], indent);
  }

  if (t === "codeblock") {
    const text = renderInlinePlain(node.content).replace(/\r\n/g, "\n").replace(/\r/g, "\n");
    const body = text.endsWith("\n") ? text.slice(0, -1) : text;
    const lines = body.split("\n").map((l: string) => `    ${l}`);
    return prefixLines(lines, indent);
  }

  if (t === "bulletlist" || t === "orderedlist" || t === "tasklist") {
    const items = Array.isArray(node.content) ? node.content : [];
    const orderedStart = Number(node.attrs?.start ?? 1);
    let ordinal = Number.isFinite(orderedStart) ? orderedStart : 1;
    const out: string[] = [];

    for (const it of items) {
      const itType = normalizeType(it?.type).toLowerCase();
      if (itType !== "listitem" && itType !== "taskitem") continue;

      const checked = itType === "taskitem" ? !!it.attrs?.checked : false;
      const marker =
        t === "orderedlist"
          ? `${ordinal}. `
          : t === "tasklist"
            ? `- [${checked ? "x" : " "}] `
            : "- ";
      if (t === "orderedlist") ordinal += 1;

      const continuationIndent = indent + " ".repeat(marker.length);
      const nestedIndent = continuationIndent + "  ";

      const children = Array.isArray(it.content) ? it.content : [];
      const childLines: string[] = [];
      for (const c of children) {
        const ct = normalizeType(c?.type).toLowerCase();
        if (ct === "bulletlist" || ct === "orderedlist" || ct === "tasklist") {
          childLines.push(...blockToPlainLines(c, { ...ctx, indent: nestedIndent }));
          continue;
        }
        childLines.push(...blockToPlainLines(c, { ...ctx, indent: continuationIndent }));
      }

      if (!childLines.length) {
        out.push(`${indent}${marker}`.trimEnd());
        continue;
      }

      let firstIdx = childLines.findIndex((l) => l.trim().length > 0);
      if (firstIdx < 0) firstIdx = 0;
      const firstLine = childLines[firstIdx].trimStart();
      out.push(`${indent}${marker}${firstLine}`.trimEnd());
      for (let i = firstIdx + 1; i < childLines.length; i += 1) {
        const l = childLines[i];
        out.push(l ? `${continuationIndent}${l.trimStart()}` : "");
      }
    }
    return trimTrailingBlanks(out);
  }

  if (t === "table") {
    const rows = Array.isArray(node.content) ? node.content : [];
    const out: string[] = [];
    for (const r of rows) {
      if (normalizeType(r?.type).toLowerCase() !== "tablerow") continue;
      const cells = Array.isArray(r.content) ? r.content : [];
      const row: string[] = [];
      for (const c of cells) {
        const ct = normalizeType(c?.type).toLowerCase();
        if (ct !== "tablecell" && ct !== "tableheader") continue;
        row.push(renderInlinePlain(c.content).replace(/\s+/g, " ").trim());
      }
      if (row.length) out.push(prefixLines([row.join("\t")], indent)[0]);
    }
    return out;
  }

  if (Array.isArray(node.content)) return blocksToPlainLines(node.content, ctx);
  return [];
}

function blocksToPlainLines(nodes: PMNode[] | undefined, ctx: BlockCtx): string[] {
  if (!Array.isArray(nodes)) return [];
  const out: string[] = [];
  for (const n of nodes) {
    const block = blockToPlainLines(n, ctx);
    if (!block.length) continue;
    if (out.length && out[out.length - 1].trim() !== "") out.push("");
    out.push(...block);
  }
  return trimTrailingBlanks(out);
}

export function proseMirrorDocToPlainText(doc: any): string {
  if (!doc || typeof doc !== "object") return "";
  const root = doc as PMNode;
  const lines = blocksToPlainLines(root.content, { indent: "" });
  return lines.join("\n").trim() + "\n";
}

function renderInlineHtml(nodes: PMNode[] | undefined): string {
  if (!Array.isArray(nodes)) return "";
  const parts: string[] = [];
  for (const n of nodes) {
    const t = normalizeType(n?.type).toLowerCase();
    if (t === "text") {
      const marks = Array.isArray(n.marks) ? n.marks : [];
      let out = escapeHtml(String(n.text ?? ""));
      const code = !!getMark(marks, "code");
      if (code) {
        parts.push(`<code>${out}</code>`);
        continue;
      }
      if (getMark(marks, "bold")) out = `<strong>${out}</strong>`;
      if (getMark(marks, "italic")) out = `<em>${out}</em>`;
      if (getMark(marks, "strike")) out = `<s>${out}</s>`;
      if (getMark(marks, "underline")) out = `<u>${out}</u>`;
      if (getMark(marks, "highlight")) out = `<mark>${out}</mark>`;
      const link = getMark(marks, "link");
      if (link) {
        const href = escapeHtml(String(link.attrs?.href ?? "").trim());
        if (href) out = `<a href="${href}">${out}</a>`;
      }
      parts.push(out);
      continue;
    }
    if (t === "hardbreak") {
      parts.push("<br/>");
      continue;
    }
    if (Array.isArray(n.content)) parts.push(renderInlineHtml(n.content));
  }
  return parts.join("");
}

function nodeToHtml(node: PMNode): string {
  const t = normalizeType(node?.type).toLowerCase();
  if (t === "paragraph") return `<p>${renderInlineHtml(node.content)}</p>`;
  if (t === "heading") {
    const level = Number(node.attrs?.level ?? 1);
    const safe = Number.isFinite(level) ? Math.min(Math.max(1, level), 6) : 1;
    return `<h${safe}>${renderInlineHtml(node.content)}</h${safe}>`;
  }
  if (t === "blockquote") return `<blockquote>${nodesToHtml(node.content)}</blockquote>`;
  if (t === "horizontalrule") return "<hr/>";
  if (t === "codeblock") {
    const lang = escapeHtml(String(node.attrs?.language ?? "").trim());
    const cls = lang ? ` class="language-${lang}"` : "";
    const text = escapeHtml(renderInlinePlain(node.content));
    return `<pre><code${cls}>${text}</code></pre>`;
  }
  if (t === "bulletlist") return `<ul>${nodesToHtml(node.content)}</ul>`;
  if (t === "orderedlist") return `<ol>${nodesToHtml(node.content)}</ol>`;
  if (t === "listitem") return `<li>${nodesToHtml(node.content)}</li>`;
  if (t === "tasklist") return `<ul>${nodesToHtml(node.content)}</ul>`;
  if (t === "taskitem") {
    const checked = !!node.attrs?.checked;
    const box = `<input type="checkbox" disabled${checked ? " checked" : ""}/>`;
    return `<li>${box} ${nodesToHtml(node.content)}</li>`;
  }
  if (t === "table") return `<table>${nodesToHtml(node.content)}</table>`;
  if (t === "tablerow") return `<tr>${nodesToHtml(node.content)}</tr>`;
  if (t === "tablecell") return `<td>${nodesToHtml(node.content)}</td>`;
  if (t === "tableheader") return `<th>${nodesToHtml(node.content)}</th>`;
  if (Array.isArray(node.content)) return nodesToHtml(node.content);
  return "";
}

function nodesToHtml(nodes: PMNode[] | undefined): string {
  if (!Array.isArray(nodes)) return "";
  return nodes.map((n) => nodeToHtml(n)).join("");
}

export function proseMirrorDocToHtml(doc: any): string {
  if (!doc || typeof doc !== "object") return "";
  const root = doc as PMNode;
  return nodesToHtml(root.content);
}
