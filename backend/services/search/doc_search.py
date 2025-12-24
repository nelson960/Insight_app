from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional

from backend.services.docs import blocks_to_prosemirror_doc
from backend.services.storage.sqlite_store import SQLiteMetadataStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DocSearchMatch:
    from_pos: int
    to_pos: int
    snippet: str


@dataclass
class _IndexedBlock:
    segments: list[tuple[int, str]]  # (pos_start, text)
    text: str
    text_ci: str


@dataclass
class _DocIndex:
    updated_at: str
    blocks: list[_IndexedBlock]


def _is_word_char(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


def _is_whole_word(text: str, start: int, end: int) -> bool:
    before = text[start - 1] if start > 0 else ""
    after = text[end] if end < len(text) else ""
    if before and _is_word_char(before):
        return False
    if after and _is_word_char(after):
        return False
    return True


def _snippet(text: str, start: int, end: int, *, radius: int = 42) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    snippet = text[left:right]
    snippet = " ".join(snippet.split())
    if left > 0:
        snippet = "…" + snippet
    if right < len(text):
        snippet = snippet + "…"
    return snippet


def _offset_to_pos(segments: list[tuple[int, str]], offset: int) -> int:
    offset = max(0, int(offset))
    cursor = 0
    last_pos = 0
    for pos, text in segments:
        t = text or ""
        last_pos = pos + len(t)
        next_cursor = cursor + len(t)
        if offset <= next_cursor:
            return pos + (offset - cursor)
        cursor = next_cursor
    return last_pos


def _build_indexed_blocks(doc: dict[str, Any]) -> list[_IndexedBlock]:
    """
    Build searchable blocks from a ProseMirror JSON document.

    We treat paragraph/heading/codeBlock nodes as "blocks" and index their inline text segments with
    ProseMirror positions. This enables precise highlight ranges in the frontend.
    """

    blocks: list[_IndexedBlock] = []
    block_stack: list[dict[str, Any]] = []

    def push_block() -> None:
        block_stack.append({"segments": []})

    def pop_block() -> None:
        blk = block_stack.pop()
        segments = [(int(p), str(t)) for (p, t) in (blk.get("segments") or []) if str(t)]
        if not segments:
            return
        text = "".join(t for _, t in segments)
        if not text:
            return
        blocks.append(_IndexedBlock(segments=segments, text=text, text_ci=text.casefold()))

    block_types = {"paragraph", "heading", "codeblock"}
    leaf_one_types = {"hardbreak", "horizontalrule"}

    def walk(node: Any, pos: int) -> int:
        if not isinstance(node, dict):
            return 0
        node_type = str(node.get("type") or "").lower()

        if node_type == "text":
            text = str(node.get("text") or "")
            if block_stack and text:
                block_stack[-1]["segments"].append((pos, text))
            return len(text)

        if node_type in leaf_one_types:
            if node_type == "hardbreak" and block_stack:
                block_stack[-1]["segments"].append((pos, "\n"))
            return 1

        pushed = False
        if node_type in block_types:
            push_block()
            pushed = True

        content = node.get("content")
        total_child = 0
        if isinstance(content, list):
            child_pos = pos if node_type == "doc" else pos + 1
            for child in content:
                child_size = walk(child, child_pos)
                child_pos += child_size
                total_child += child_size

        if pushed:
            pop_block()

        # Root doc is special in ProseMirror: its content starts at pos=0 and its size is content.size.
        if node_type == "doc":
            return total_child
        return 2 + total_child

    walk(doc, 0)
    return blocks


class DocSearchService:
    """
    Backend search over ProseMirror docs (doc_pages), returning ProseMirror positions (from/to).
    """

    def __init__(self, metadata_store: SQLiteMetadataStore, *, max_cached_files: int = 16) -> None:
        self._store = metadata_store
        self._max_cached_files = max(1, int(max_cached_files))
        self._cache: "OrderedDict[str, _DocIndex]" = OrderedDict()
        self._lock = threading.Lock()

    def search_doc_page(
        self,
        chat_id: str,
        file_id: str,
        *,
        query: str,
        limit: int = 200,
        case_sensitive: bool = False,
        whole_word: bool = False,
    ) -> list[DocSearchMatch]:
        q = (query or "").strip()
        if not q:
            return []
        limit = max(1, min(int(limit or 200), 1000))

        index = self._get_index(chat_id, file_id)
        if index is None:
            return []

        needle = q if case_sensitive else q.casefold()
        matches: list[DocSearchMatch] = []

        for block in index.blocks:
            haystack = block.text if case_sensitive else block.text_ci
            if not haystack:
                continue
            start_at = 0
            while True:
                found = haystack.find(needle, start_at)
                if found < 0:
                    break
                end = found + len(needle)
                if whole_word and not _is_whole_word(block.text, found, end):
                    start_at = found + 1
                    continue
                matches.append(
                    DocSearchMatch(
                        from_pos=_offset_to_pos(block.segments, found),
                        to_pos=_offset_to_pos(block.segments, end),
                        snippet=_snippet(block.text, found, end),
                    )
                )
                if len(matches) >= limit:
                    return matches
                start_at = end if end > found else found + 1

        return matches

    def invalidate(self, file_id: str) -> None:
        with self._lock:
            self._cache.pop(file_id, None)

    def _get_index(self, chat_id: str, file_id: str) -> Optional[_DocIndex]:
        doc_page = self._store.get_doc_page(chat_id, file_id)
        doc = doc_page.get("doc") if doc_page else None
        updated_at = doc_page.get("updated_at") if doc_page else None

        if not isinstance(updated_at, str) or not updated_at:
            # Fall back to bootstrapping from extractor output if available.
            file_text = self._store.get_file_text(file_id) or {}
            blocks = file_text.get("blocks") if isinstance(file_text.get("blocks"), list) else []
            if not blocks:
                return None
            doc = blocks_to_prosemirror_doc(blocks)
            updated_at = file_text.get("updated_at")

        if not isinstance(doc, dict) or not isinstance(updated_at, str) or not updated_at:
            return None

        with self._lock:
            existing = self._cache.get(file_id)
            if existing and existing.updated_at == updated_at:
                self._cache.move_to_end(file_id, last=True)
                return existing

        blocks = _build_indexed_blocks(doc)
        index = _DocIndex(updated_at=updated_at, blocks=blocks)
        with self._lock:
            self._cache[file_id] = index
            self._cache.move_to_end(file_id, last=True)
            while len(self._cache) > self._max_cached_files:
                self._cache.popitem(last=False)
        logger.debug("Built doc search index file=%s blocks=%d", file_id, len(blocks))
        return index


__all__ = ["DocSearchMatch", "DocSearchService"]

