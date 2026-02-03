from __future__ import annotations

import itertools
import logging
import os
import uuid
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from .models import ChunkPayload, ExtractionResult

logger = logging.getLogger(__name__)

try:  # optional dependency
    from tree_sitter_languages import get_parser as _ts_get_parser
except Exception:  # pragma: no cover - optional
    _ts_get_parser = None


@dataclass
class ChunkerConfig:
    """Configuration for the chunking strategy."""

    chunk_size_tokens: int = 1200
    chunk_overlap_tokens: int = 120


class SimpleChunker:
    """
    Token-approximate chunker that operates on character count approximation.

    This provides a placeholder implementation; a production version should rely on
    the tokenizer associated with the embedding model for accurate budgeting.
    """

    def __init__(self, config: ChunkerConfig | None = None) -> None:
        self._config = config or ChunkerConfig()

    def chunk(self, extraction: ExtractionResult) -> List[ChunkPayload]:
        segments = self._prepare_segments(extraction.text_segments)
        if not segments:
            return [
                ChunkPayload(
                    chunk_id=self._build_chunk_id(),
                    file_id=extraction.file_id,
                    sequence=0,
                    text="",
                    metadata={"source": "empty"},
                    page=None,
                )
            ]

        chunks: List[ChunkPayload] = []
        seq = 0
        cursor = 0
        size_tokens = max(1, int(self._config.chunk_size_tokens))
        overlap_tokens = max(0, int(self._config.chunk_overlap_tokens))
        if overlap_tokens >= size_tokens:
            overlap_tokens = max(0, size_tokens - 1)

        while cursor < len(segments):
            tokens = 0
            end = cursor
            while end < len(segments):
                seg = segments[end]
                seg_tokens = self._approx_token_count(seg)
                # Always include at least one segment.
                if end == cursor or tokens + seg_tokens <= size_tokens:
                    tokens += seg_tokens
                    end += 1
                    continue
                break

            text = "\n\n".join(segments[cursor:end]).strip()
            if text:
                chunks.append(
                    ChunkPayload(
                        chunk_id=self._build_chunk_id(),
                        file_id=extraction.file_id,
                        sequence=seq,
                        text=text,
                        metadata={"source": "chunker"},
                        page=None,
                    )
                )
                seq += 1

            if end >= len(segments):
                break

            if overlap_tokens <= 0:
                cursor = end
                continue

            # Advance cursor, keeping a small overlap on segment boundaries.
            overlap_start = end
            overlap_used = 0
            while overlap_start > cursor and overlap_used < overlap_tokens:
                overlap_start -= 1
                overlap_used += self._approx_token_count(segments[overlap_start])
            next_cursor = overlap_start if overlap_start > cursor else end
            if next_cursor <= cursor:
                next_cursor = end
            cursor = next_cursor
        return chunks

    @staticmethod
    def _iter_segments(segments: Iterable[str]) -> Iterable[str]:
        for segment in segments:
            if segment:
                yield segment

    def _prepare_segments(self, segments: Iterable[str]) -> List[str]:
        """
        Normalize and pre-split very large segments.

        Extraction often returns a small number of large text blocks (sometimes a
        single block for a whole PDF). If we treat `chunk_size_tokens` as "number
        of segments", we end up with 1 chunk per document, which harms retrieval.
        """
        out: List[str] = []
        for raw in self._iter_segments(segments):
            seg = str(raw).strip()
            if not seg:
                continue
            if self._approx_token_count(seg) > int(self._config.chunk_size_tokens):
                out.extend(self._split_text(seg))
            else:
                out.append(seg)
        return out

    def _split_text(self, text: str) -> List[str]:
        """
        Split a long segment into overlapping chunks (character-based approximation).
        """
        if not text:
            return []
        size_chars = max(1, int(self._config.chunk_size_tokens) * 4)
        overlap_chars = max(0, int(self._config.chunk_overlap_tokens) * 4)
        if overlap_chars >= size_chars:
            overlap_chars = max(0, size_chars - 1)
        parts: List[str] = []
        cursor = 0
        while cursor < len(text):
            end = min(len(text), cursor + size_chars)
            piece = text[cursor:end].strip()
            if piece:
                parts.append(piece)
            if end >= len(text):
                break
            cursor = end - overlap_chars if overlap_chars else end
        return parts

    @staticmethod
    def _approx_token_count(text: str) -> int:
        # ~4 characters per token on average; used only for chunk sizing.
        if not text:
            return 0
        return max(1, len(text) // 4)

    @staticmethod
    def _build_chunk_id() -> str:
        return f"chunk_{uuid.uuid4().hex}"


@dataclass(frozen=True)
class _BlockSegment:
    block_index: int
    text: str
    page: Optional[int]
    section_path: List[str]
    has_ocr: bool
    layout_hint: str


class BlockChunker:
    """
    Structure-aware chunker that preserves page + heading context.

    Instead of chunking raw text segments, this chunker walks the extracted block
    stream and builds chunks from *blocks in order*, attaching citeable metadata:
      - page_start/page_end
      - section_path (best-effort heading hierarchy)
      - block_start/block_end (indices into ExtractionResult.blocks / file_text.blocks_json)
      - has_ocr
      - layout_hint

    If no blocks exist, falls back to SimpleChunker behavior.
    """

    def __init__(self, config: ChunkerConfig | None = None) -> None:
        self._config = config or ChunkerConfig()
        self._fallback = SimpleChunker(config=self._config)

    def chunk(self, extraction: ExtractionResult) -> List[ChunkPayload]:
        blocks = getattr(extraction, "blocks", None)
        if not isinstance(blocks, list) or not blocks:
            return self._fallback.chunk(extraction)

        segments = self._segments_from_blocks(blocks)
        if not segments:
            return self._fallback.chunk(extraction)

        chunks: List[ChunkPayload] = []
        seq = 0
        cursor = 0
        size_tokens = max(1, int(self._config.chunk_size_tokens))
        overlap_tokens = max(0, int(self._config.chunk_overlap_tokens))
        if overlap_tokens >= size_tokens:
            overlap_tokens = max(0, size_tokens - 1)

        while cursor < len(segments):
            tokens = 0
            end = cursor
            while end < len(segments):
                seg = segments[end]
                seg_tokens = self._approx_token_count(seg.text)
                if end == cursor or tokens + seg_tokens <= size_tokens:
                    tokens += seg_tokens
                    end += 1
                    continue
                break

            seg_slice = segments[cursor:end]
            text = "\n\n".join(s.text for s in seg_slice if s.text).strip()
            if text:
                pages = [s.page for s in seg_slice if isinstance(s.page, int)]
                page_start = min(pages) if pages else None
                page_end = max(pages) if pages else None
                block_start = seg_slice[0].block_index
                block_end = seg_slice[-1].block_index
                section_path = list(seg_slice[0].section_path)
                has_ocr = any(bool(s.has_ocr) for s in seg_slice)
                layout_hint = self._layout_hint(seg_slice)

                chunks.append(
                    ChunkPayload(
                        chunk_id=self._build_chunk_id(),
                        file_id=extraction.file_id,
                        sequence=seq,
                        text=text,
                        metadata={
                            "source": "block_chunker",
                            "page_start": page_start,
                            "page_end": page_end,
                            "section_path": section_path,
                            "block_start": block_start,
                            "block_end": block_end,
                            "has_ocr": has_ocr,
                            "layout_hint": layout_hint,
                        },
                        # Keep legacy `page` populated for filtering/citations.
                        page=page_start,
                    )
                )
                seq += 1

            if end >= len(segments):
                break

            if overlap_tokens <= 0:
                cursor = end
                continue

            # Advance cursor, keeping a small overlap on segment boundaries.
            overlap_start = end
            overlap_used = 0
            while overlap_start > cursor and overlap_used < overlap_tokens:
                overlap_start -= 1
                overlap_used += self._approx_token_count(segments[overlap_start].text)
            next_cursor = overlap_start if overlap_start > cursor else end
            if next_cursor <= cursor:
                next_cursor = end
            cursor = next_cursor

        return chunks

    def _segments_from_blocks(self, blocks: list) -> List[_BlockSegment]:
        current_page: Optional[int] = None
        heading_stack: List[tuple[int, str]] = []

        def push_heading(level: int, title: str) -> None:
            nonlocal heading_stack
            level = max(1, min(int(level or 2), 6))
            title = (title or "").strip()
            if not title:
                return
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))

        def section_path() -> List[str]:
            return [t for _, t in heading_stack]

        segments: List[_BlockSegment] = []
        for idx, raw in enumerate(blocks):
            if not isinstance(raw, dict):
                continue
            kind = str(raw.get("kind") or "").strip().lower()
            text = raw.get("text")
            meta = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}

            # Track page breaks without chunking them as content.
            if kind == "page_break":
                p = meta.get("page")
                if isinstance(p, int):
                    current_page = p
                continue

            page = meta.get("page")
            if isinstance(page, int):
                current_page = page
            else:
                page = current_page

            has_ocr = bool(meta.get("ocr") is True)

            if not isinstance(text, str) or not text.strip():
                # Some list blocks may be represented in metadata.items only.
                if kind != "list":
                    continue

            rendered = self._render_block(kind, text or "", meta, push_heading)
            if not rendered.strip():
                continue

            # If a single rendered block is huge, split it into smaller pieces but keep metadata.
            if self._approx_token_count(rendered) > int(self._config.chunk_size_tokens):
                for piece in self._split_text(rendered):
                    segments.append(
                        _BlockSegment(
                            block_index=idx,
                            text=piece,
                            page=page if isinstance(page, int) else None,
                            section_path=section_path(),
                            has_ocr=has_ocr,
                            layout_hint=kind or "paragraph",
                        )
                    )
            else:
                segments.append(
                    _BlockSegment(
                        block_index=idx,
                        text=rendered,
                        page=page if isinstance(page, int) else None,
                        section_path=section_path(),
                        has_ocr=has_ocr,
                        layout_hint=kind or "paragraph",
                    )
                )

        return segments

    @staticmethod
    def _render_block(
        kind: str,
        text: str,
        meta: dict,
        push_heading,
    ) -> str:
        kind = (kind or "paragraph").strip().lower()
        raw = (text or "").strip()
        if kind == "heading":
            level = meta.get("level", 2)
            try:
                level_i = int(level)
            except Exception:
                level_i = 2
            push_heading(level_i, raw)
            level_i = max(1, min(level_i, 6))
            return f"{'#' * level_i} {raw}".strip()
        if kind == "list":
            items = meta.get("items") if isinstance(meta.get("items"), list) else None
            lines: List[str] = []
            if items:
                for it in items:
                    s = str(it).strip()
                    if s:
                        lines.append(f"- {s}")
            else:
                for ln in raw.splitlines():
                    s = ln.strip()
                    if s:
                        lines.append(f"- {s}")
            return "\n".join(lines).strip()
        if kind == "code":
            if not raw:
                return ""
            return "```\n" + raw.rstrip() + "\n```"
        # Default: paragraph-like.
        return raw

    @staticmethod
    def _layout_hint(segments: List[_BlockSegment]) -> str:
        kinds = [s.layout_hint for s in segments if s.layout_hint]
        if not kinds:
            return "mixed"
        first = kinds[0]
        if all(k == first for k in kinds):
            return first
        # Prefer "code" if any code is present; otherwise mixed.
        if any(k == "code" for k in kinds):
            return "mixed_code"
        if any(k == "list" for k in kinds):
            return "mixed_list"
        return "mixed"

    def _split_text(self, text: str) -> List[str]:
        if not text:
            return []
        # ~4 chars/token; keep the split behavior aligned with the configured chunk sizes.
        size_chars = max(1, int(self._config.chunk_size_tokens)) * 4
        overlap_chars = max(0, int(self._config.chunk_overlap_tokens)) * 4
        if overlap_chars >= size_chars:
            overlap_chars = max(0, size_chars - 1)
        parts: List[str] = []
        cursor = 0
        while cursor < len(text):
            end = min(len(text), cursor + size_chars)
            piece = text[cursor:end].strip()
            if piece:
                parts.append(piece)
            if end >= len(text):
                break
            cursor = end - overlap_chars if overlap_chars else end
        return parts

    @staticmethod
    def _approx_token_count(text: str) -> int:
        if not text:
            return 0
        return max(1, len(text) // 4)

    @staticmethod
    def _build_chunk_id() -> str:
        return f"chunk_{uuid.uuid4().hex}"


class CodeAwareChunker:
    """
    Code-aware chunker that uses Tree-sitter for code files and falls back to BlockChunker.

    Tree-sitter usage is optional; if the dependency or language grammar is missing,
    this silently falls back to the block/simple chunkers.
    """

    _EXT_TO_LANG = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".rs": "rust",
        ".go": "go",
        ".java": "java",
        ".kt": "kotlin",
        ".c": "c",
        ".h": "c",
        ".hpp": "cpp",
        ".cc": "cpp",
        ".cpp": "cpp",
        ".cs": "c_sharp",
        ".rb": "ruby",
        ".php": "php",
        ".swift": "swift",
    }

    _MIME_TO_LANG = {
        "text/x-python": "python",
        "text/x-python-script": "python",
        "text/x-java-source": "java",
        "text/x-c": "c",
        "text/x-c++": "cpp",
        "text/x-csharp": "c_sharp",
        "text/x-rust": "rust",
        "text/x-go": "go",
        "text/x-javascript": "javascript",
        "application/javascript": "javascript",
        "text/typescript": "typescript",
        "application/typescript": "typescript",
    }

    _NODE_TYPES = {
        "python": {"function_definition", "class_definition"},
        "javascript": {"function_declaration", "class_declaration"},
        "typescript": {"function_declaration", "class_declaration"},
        "tsx": {"function_declaration", "class_declaration"},
        "rust": {"function_item", "impl_item", "struct_item", "enum_item", "trait_item"},
        "go": {"function_declaration", "method_declaration", "type_declaration"},
        "java": {"method_declaration", "class_declaration", "interface_declaration", "enum_declaration"},
        "kotlin": {"function_declaration", "class_declaration", "object_declaration"},
        "c": {"function_definition"},
        "cpp": {"function_definition", "class_specifier", "struct_specifier"},
        "c_sharp": {"method_declaration", "class_declaration", "interface_declaration"},
        "ruby": {"method", "class", "module"},
        "php": {"function_definition", "class_declaration"},
        "swift": {"function_declaration", "class_declaration", "struct_declaration", "enum_declaration"},
    }

    def __init__(self, config: ChunkerConfig | None = None) -> None:
        self._config = config or ChunkerConfig()
        self._fallback = BlockChunker(config=self._config)
        self._parser_cache: dict[str, object] = {}

    def chunk(self, extraction: ExtractionResult) -> List[ChunkPayload]:
        lang = self._detect_language(extraction)
        if not lang or _ts_get_parser is None:
            return self._fallback.chunk(extraction)

        text = extraction.plain_text or ""
        if not isinstance(text, str) or not text.strip():
            return self._fallback.chunk(extraction)

        parser = self._get_parser(lang)
        if parser is None:
            return self._fallback.chunk(extraction)

        segments = self._segments_from_tree(text, parser, lang)
        if not segments:
            return self._fallback.chunk(extraction)

        return self._chunks_from_segments(extraction.file_id, segments)

    def _detect_language(self, extraction: ExtractionResult) -> Optional[str]:
        meta = extraction.metadata or {}
        ext = None
        raw_name = meta.get("source_name") if isinstance(meta.get("source_name"), str) else None
        raw_path = meta.get("source_path") if isinstance(meta.get("source_path"), str) else None
        if raw_name and "." in raw_name:
            ext = os.path.splitext(raw_name)[1].lower()
        if not ext and raw_path and "." in raw_path:
            ext = os.path.splitext(raw_path)[1].lower()
        if ext and ext in self._EXT_TO_LANG:
            return self._EXT_TO_LANG[ext]

        mime = meta.get("content_type") if isinstance(meta.get("content_type"), str) else None
        if mime and mime in self._MIME_TO_LANG:
            return self._MIME_TO_LANG[mime]
        return None

    def _get_parser(self, lang: str):
        if lang in self._parser_cache:
            return self._parser_cache[lang]
        if _ts_get_parser is None:
            return None
        try:
            parser = _ts_get_parser(lang)
        except Exception:
            logger.info("Tree-sitter parser not available for %s; falling back", lang)
            self._parser_cache[lang] = None
            return None
        self._parser_cache[lang] = parser
        return parser

    def _segments_from_tree(self, text: str, parser, lang: str) -> List[Tuple[str, dict]]:
        nodes_of_interest = self._NODE_TYPES.get(lang, set())
        try:
            tree = parser.parse(text.encode("utf-8", errors="ignore"))
        except Exception:
            logger.info("Tree-sitter parse failed; falling back")
            return []

        root = tree.root_node
        if not root:
            return []

        segments: List[Tuple[str, dict]] = []
        byte_text = text.encode("utf-8", errors="ignore")
        node_children = [c for c in root.children if c.type in nodes_of_interest]
        if not node_children:
            return []

        # Header (imports/module docstring/etc.)
        header_end = node_children[0].start_byte
        if header_end > 0:
            header_text = byte_text[:header_end].decode("utf-8", errors="ignore").strip()
            if header_text:
                segments.append((header_text, {"source": "tree_sitter", "node_type": "header"}))

        for node in node_children:
            seg_text = byte_text[node.start_byte:node.end_byte].decode("utf-8", errors="ignore").strip()
            if not seg_text:
                continue
            meta = {
                "source": "tree_sitter",
                "node_type": node.type,
            }
            segments.append((seg_text, meta))

        # Tail content after last node (e.g., trailing constants)
        tail_start = node_children[-1].end_byte
        if tail_start < len(byte_text):
            tail_text = byte_text[tail_start:].decode("utf-8", errors="ignore").strip()
            if tail_text:
                segments.append((tail_text, {"source": "tree_sitter", "node_type": "tail"}))

        return segments

    def _chunks_from_segments(self, file_id: str, segments: List[Tuple[str, dict]]) -> List[ChunkPayload]:
        chunks: List[ChunkPayload] = []
        seq = 0
        size_tokens = max(1, int(self._config.chunk_size_tokens))
        overlap_tokens = max(0, int(self._config.chunk_overlap_tokens))
        if overlap_tokens >= size_tokens:
            overlap_tokens = max(0, size_tokens - 1)

        # Expand overly large segments.
        expanded: List[Tuple[str, dict]] = []
        for text, meta in segments:
            if self._approx_token_count(text) > size_tokens:
                for piece in SimpleChunker(config=self._config)._split_text(text):
                    expanded.append((piece, {**meta, "node_type": f"{meta.get('node_type')}_split"}))
            else:
                expanded.append((text, meta))

        cursor = 0
        while cursor < len(expanded):
            tokens = 0
            end = cursor
            while end < len(expanded):
                seg_text = expanded[end][0]
                seg_tokens = self._approx_token_count(seg_text)
                if end == cursor or tokens + seg_tokens <= size_tokens:
                    tokens += seg_tokens
                    end += 1
                    continue
                break

            seg_slice = expanded[cursor:end]
            text = "\n\n".join(t for t, _ in seg_slice if t).strip()
            if text:
                meta = {"source": "tree_sitter", "node_types": [m.get("node_type") for _, m in seg_slice]}
                chunks.append(
                    ChunkPayload(
                        chunk_id=self._build_chunk_id(),
                        file_id=file_id,
                        sequence=seq,
                        text=text,
                        metadata=meta,
                        page=None,
                    )
                )
                seq += 1

            if end >= len(expanded):
                break

            if overlap_tokens <= 0:
                cursor = end
                continue

            overlap_start = end
            overlap_used = 0
            while overlap_start > cursor and overlap_used < overlap_tokens:
                overlap_start -= 1
                overlap_used += self._approx_token_count(expanded[overlap_start][0])
            next_cursor = overlap_start if overlap_start > cursor else end
            if next_cursor <= cursor:
                next_cursor = end
            cursor = next_cursor

        return chunks

    @staticmethod
    def _approx_token_count(text: str) -> int:
        if not text:
            return 0
        return max(1, len(text) // 4)

    @staticmethod
    def _build_chunk_id() -> str:
        return f"chunk_{uuid.uuid4().hex}"


__all__ = ["SimpleChunker", "BlockChunker", "CodeAwareChunker", "ChunkerConfig"]
