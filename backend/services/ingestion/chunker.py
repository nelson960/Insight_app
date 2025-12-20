from __future__ import annotations

import itertools
import uuid
from dataclasses import dataclass
from typing import Iterable, List

from .models import ChunkPayload, ExtractionResult


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


__all__ = ["SimpleChunker", "ChunkerConfig"]
