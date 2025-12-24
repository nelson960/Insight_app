from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

from backend.services.docs import prosemirror_doc_to_blocks
from backend.services.storage.sqlite_store import SQLiteMetadataStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FileSearchMatch:
    block_index: int
    start: int
    end: int
    snippet: str
    page: int | None = None


@dataclass
class _FileTextIndex:
    updated_at: str
    block_texts: list[str]
    block_texts_ci: list[str]
    pages: list[int | None]


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


class FileSearchService:
    """
    Fast, literal text search over extracted file blocks stored in SQLite.

    Designed to back a "Ctrl+F" UI and later be reused for chat/global search.
    """

    def __init__(self, metadata_store: SQLiteMetadataStore, *, max_cached_files: int = 16) -> None:
        self._store = metadata_store
        self._max_cached_files = max(1, int(max_cached_files))
        self._cache: "OrderedDict[str, _FileTextIndex]" = OrderedDict()
        self._lock = threading.Lock()

    def search_file(
        self,
        file_id: str,
        *,
        query: str,
        limit: int = 200,
        case_sensitive: bool = False,
        whole_word: bool = False,
    ) -> list[FileSearchMatch]:
        q = (query or "").strip()
        if not q:
            return []
        limit = max(1, min(int(limit or 200), 1000))

        index = self._get_index(file_id)
        if index is None:
            return []

        needle = q if case_sensitive else q.casefold()
        matches: list[FileSearchMatch] = []

        for block_index, text in enumerate(index.block_texts):
            haystack = text if case_sensitive else index.block_texts_ci[block_index]
            if not haystack:
                continue
            start_at = 0
            while True:
                found = haystack.find(needle, start_at)
                if found < 0:
                    break
                end = found + len(needle)
                if whole_word and not _is_whole_word(text, found, end):
                    start_at = found + 1
                    continue
                matches.append(
                    FileSearchMatch(
                        block_index=block_index,
                        start=found,
                        end=end,
                        snippet=_snippet(text, found, end),
                        page=index.pages[block_index],
                    )
                )
                if len(matches) >= limit:
                    return matches
                start_at = end if end > found else found + 1

        return matches

    def invalidate(self, file_id: str) -> None:
        with self._lock:
            self._cache.pop(file_id, None)

    def _get_index(self, file_id: str) -> Optional[_FileTextIndex]:
        record = self._store.get_file(file_id) or {}
        chat_id = record.get("chat_id") if isinstance(record.get("chat_id"), str) else None
        if chat_id:
            doc_page = self._store.get_doc_page(chat_id, file_id)
            if doc_page and bool(doc_page.get("is_user_edited")) and isinstance(doc_page.get("doc"), dict):
                updated_at = doc_page.get("updated_at")
                if isinstance(updated_at, str) and updated_at:
                    with self._lock:
                        existing = self._cache.get(file_id)
                        if existing and existing.updated_at == updated_at:
                            self._cache.move_to_end(file_id, last=True)
                            return existing

                    blocks = prosemirror_doc_to_blocks(doc_page["doc"])
                    texts: list[str] = []
                    texts_ci: list[str] = []
                    pages: list[int | None] = []
                    for block in blocks:
                        if not isinstance(block, dict):
                            texts.append("")
                            texts_ci.append("")
                            pages.append(None)
                            continue
                        text = str(block.get("text") or "")
                        texts.append(text)
                        texts_ci.append(text.casefold())
                        pages.append(None)

                    index = _FileTextIndex(
                        updated_at=str(updated_at),
                        block_texts=texts,
                        block_texts_ci=texts_ci,
                        pages=pages,
                    )
                    with self._lock:
                        self._cache[file_id] = index
                        self._cache.move_to_end(file_id, last=True)
                        while len(self._cache) > self._max_cached_files:
                            self._cache.popitem(last=False)
                    logger.debug(
                        "Built search index for doc_page file %s blocks=%d", file_id, len(texts)
                    )
                    return index

        updated_at = self._store.get_file_text_updated_at(file_id)
        if not updated_at:
            return None

        with self._lock:
            existing = self._cache.get(file_id)
            if existing and existing.updated_at == updated_at:
                self._cache.move_to_end(file_id, last=True)
                return existing

        file_text = self._store.get_file_text(file_id)
        if not file_text:
            with self._lock:
                self._cache.pop(file_id, None)
            return None

        blocks = file_text.get("blocks") or []
        if not isinstance(blocks, list):
            blocks = []

        texts: list[str] = []
        texts_ci: list[str] = []
        pages: list[int | None] = []
        current_page: int | None = None

        for block in blocks:
            if not isinstance(block, dict):
                texts.append("")
                texts_ci.append("")
                pages.append(current_page)
                continue
            kind = str(block.get("kind") or "").lower()
            text = str(block.get("text") or "")
            metadata = block.get("metadata") if isinstance(block.get("metadata"), dict) else {}

            if kind == "page_break":
                page_val = metadata.get("page")
                if isinstance(page_val, int):
                    current_page = page_val
                else:
                    # Best-effort parse from "Page N".
                    try:
                        parts = text.strip().split()
                        if len(parts) >= 2 and parts[0].lower() == "page":
                            current_page = int(parts[1])
                    except Exception:
                        pass

            page = metadata.get("page")
            page_num = page if isinstance(page, int) else current_page
            texts.append(text)
            texts_ci.append(text.casefold())
            pages.append(page_num)

        index = _FileTextIndex(
            updated_at=str(updated_at),
            block_texts=texts,
            block_texts_ci=texts_ci,
            pages=pages,
        )
        with self._lock:
            self._cache[file_id] = index
            self._cache.move_to_end(file_id, last=True)
            while len(self._cache) > self._max_cached_files:
                self._cache.popitem(last=False)
        logger.debug("Built search index for file %s blocks=%d", file_id, len(texts))
        return index
