from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..base import ExtractedDocument, ExtractionError, SimpleExtractor


class AudioExtractor(SimpleExtractor):
    name = "audio_transcript"
    supported_mime_types = ("audio/wav", "audio/mpeg", "audio/mp4")

    def extract(self, file_path: Path, *, mime_type: Optional[str] = None) -> ExtractedDocument:
        raise ExtractionError(
            "Audio transcription is not configured. Integrate Whisper.cpp or another offline model to enable it."
        )


__all__ = ["AudioExtractor"]
