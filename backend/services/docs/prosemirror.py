from __future__ import annotations

from typing import Any, Dict, List, Sequence


def _text_node(text: str) -> dict[str, object]:
    return {"type": "text", "text": text}


def _paragraph(text: str) -> dict[str, object]:
    node: dict[str, object] = {"type": "paragraph"}
    if text:
        node["content"] = [_text_node(text)]
    return node


def _heading(text: str, *, level: int) -> dict[str, object]:
    lvl = max(1, min(int(level or 1), 6))
    node: dict[str, object] = {"type": "heading", "attrs": {"level": lvl}}
    if text:
        node["content"] = [_text_node(text)]
    return node


def _code_block(text: str) -> dict[str, object]:
    node: dict[str, object] = {"type": "codeBlock"}
    if text:
        node["content"] = [_text_node(text)]
    return node


def _bullet_list(items: Sequence[str]) -> dict[str, object]:
    content: list[dict[str, object]] = []
    for raw in items:
        t = (raw or "").strip()
        if not t:
            continue
        content.append(
            {
                "type": "listItem",
                "content": [
                    {
                        "type": "paragraph",
                        "content": [_text_node(t)],
                    }
                ],
            }
        )
    if not content:
        return _paragraph("")
    return {"type": "bulletList", "content": content}


def blocks_to_prosemirror_doc(blocks: Sequence[dict[str, Any]] | None) -> dict[str, object]:
    """
    Convert extracted display blocks (stored in SQLite `file_text.blocks_json`) into a
    ProseMirror JSON document compatible with TipTap StarterKit.

    This is intentionally conservative: keep content readable and stable without trying
    to preserve every layout detail of the source file.
    """
    out: list[dict[str, object]] = []
    for block in blocks or []:
        if not isinstance(block, dict):
            continue
        kind = str(block.get("kind") or "").lower()
        text = str(block.get("text") or "").strip("\n")
        meta = block.get("metadata") if isinstance(block.get("metadata"), dict) else {}

        if kind == "heading":
            level = meta.get("level", 2)
            out.append(_heading(text, level=int(level or 2)))
            continue

        if kind == "list":
            items = meta.get("items")
            if isinstance(items, list):
                out.append(_bullet_list([str(x) for x in items]))
            else:
                out.append(_bullet_list([ln for ln in text.splitlines() if ln.strip()]))
            continue

        if kind == "code":
            out.append(_code_block(text))
            continue

        if kind == "page_break":
            page = meta.get("page")
            label = text or (f"Page {page}" if isinstance(page, int) else "Page")
            out.append(_paragraph(label))
            continue

        # Default to paragraph.
        if text:
            out.append(_paragraph(text))

    if not out:
        out.append(_paragraph(""))
    return {"type": "doc", "content": out}


def _flatten_text(node: dict[str, Any]) -> str:
    node_type = str(node.get("type") or "")
    if node_type == "text":
        return str(node.get("text") or "")
    if node_type == "hardBreak":
        return "\n"
    content = node.get("content")
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for child in content:
        if isinstance(child, dict):
            parts.append(_flatten_text(child))
    return "".join(parts)


def prosemirror_doc_to_blocks(doc: dict[str, Any] | None) -> list[dict[str, Any]]:
    """
    Convert a ProseMirror JSON document (TipTap StarterKit compatible) into our extracted
    block representation used by the Documents pane and backend search.

    This is intentionally lossy for rich formatting (marks). It preserves the main
    block structure (headings, lists, code, paragraphs) for navigation/search.
    """
    if not isinstance(doc, dict):
        return []
    content = doc.get("content")
    if not isinstance(content, list):
        return []

    blocks: list[dict[str, Any]] = []
    for node in content:
        if not isinstance(node, dict):
            continue
        node_type = str(node.get("type") or "").lower()

        if node_type == "heading":
            attrs = node.get("attrs") if isinstance(node.get("attrs"), dict) else {}
            level = attrs.get("level", 2)
            try:
                level = int(level)
            except Exception:
                level = 2
            level = max(1, min(level, 6))
            blocks.append(
                {
                    "kind": "heading",
                    "text": _flatten_text(node).strip("\n"),
                    "metadata": {"level": level},
                }
            )
            continue

        if node_type == "paragraph":
            text = _flatten_text(node).strip("\n")
            if text:
                blocks.append({"kind": "paragraph", "text": text, "metadata": {}})
            continue

        if node_type in ("bulletlist", "orderedlist"):
            items: list[str] = []
            node_items = node.get("content")
            if isinstance(node_items, list):
                for li in node_items:
                    if not isinstance(li, dict) or str(li.get("type") or "").lower() != "listitem":
                        continue
                    t = _flatten_text(li).strip()
                    if t:
                        items.append(t)
            if items:
                blocks.append(
                    {
                        "kind": "list",
                        "text": "\n".join(items),
                        "metadata": {"items": items, "ordered": node_type == "orderedlist"},
                    }
                )
            continue

        if node_type == "codeblock":
            text = _flatten_text(node).strip("\n")
            blocks.append({"kind": "code", "text": text, "metadata": {}})
            continue

        # Fallback: flatten to a paragraph.
        text = _flatten_text(node).strip("\n")
        if text:
            blocks.append({"kind": "paragraph", "text": text, "metadata": {}})

    return blocks


def blocks_to_plain_text(blocks: Sequence[dict[str, Any]] | None) -> str:
    parts: list[str] = []
    for block in blocks or []:
        if not isinstance(block, dict):
            continue
        kind = str(block.get("kind") or "").lower()
        text = str(block.get("text") or "").strip()
        meta = block.get("metadata") if isinstance(block.get("metadata"), dict) else {}
        if kind == "list" and isinstance(meta.get("items"), list):
            items = [str(x).strip() for x in meta["items"] if str(x).strip()]
            if items:
                parts.append("\n".join(items))
                continue
        if text:
            parts.append(text)
    return "\n\n".join(parts).strip()


__all__ = ["blocks_to_prosemirror_doc", "prosemirror_doc_to_blocks", "blocks_to_plain_text"]
