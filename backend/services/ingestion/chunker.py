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
    Token-agnostic chunker that operates on character count approximation.

    This provides a placeholder implementation; a production version should rely on
    the tokenizer associated with the embedding model for accurate budgeting.
    """

    def __init__(self, config: ChunkerConfig | None = None) -> None:
        self._config = config or ChunkerConfig()

    def chunk(self, extraction: ExtractionResult) -> List[ChunkPayload]:
        segments = list(self._iter_segments(extraction.text_segments))
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
        while cursor < len(segments):
            window = segments[cursor : cursor + self._config.chunk_size_tokens]
            text = "".join(window).strip()
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
            cursor += self._config.chunk_size_tokens - self._config.chunk_overlap_tokens
            if cursor <= 0:
                # Guard against infinite loops if overlap exceeds size.
                cursor = len(chunks)
        return chunks

    @staticmethod
    def _iter_segments(segments: Iterable[str]) -> Iterable[str]:
        for segment in segments:
            if segment:
                yield segment

    @staticmethod
    def _build_chunk_id() -> str:
        return f"chunk_{uuid.uuid4().hex}"


__all__ = ["SimpleChunker", "ChunkerConfig"]
