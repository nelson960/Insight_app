from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .base import ExtractedDocument, ExtractionError
from .detector import detect_mime_type
from .normalizer import normalize_text
from .registry import ExtractorRegistry, build_default_registry

logger = logging.getLogger(__name__)


@dataclass
class FileExtractionResult:
    text: str
    metadata: dict
    mime_type: str


class FileExtractionService:
    """High-level facade that detects MIME types and delegates to extractors."""

    def __init__(self, registry: Optional[ExtractorRegistry] = None) -> None:
        self._registry = registry or build_default_registry()
        extractor_list = getattr(self._registry, "_extractors", [])
        logger.info("FileExtractionService initialized with %d extractors", len(extractor_list))

    def extract(self, file_path: Path, *, mime_type: Optional[str] = None) -> FileExtractionResult:
        resolved_path = Path(file_path)
        if not resolved_path.exists():
            logger.error("Extraction requested for missing file %s", resolved_path)
            raise ExtractionError(f"File not found: {resolved_path}")
        mime = mime_type or detect_mime_type(resolved_path)
        logger.debug("Detected MIME %s for %s", mime, resolved_path)
        extractor = self._registry.resolve(mime)
        logger.info("Starting extraction for %s via %s", resolved_path, extractor.name)
        document = extractor.extract(resolved_path, mime_type=mime)
        text = normalize_text(document.text)
        metadata = {
            "content_type": mime,
            **(document.metadata or {}),
        }
        logger.info("Extraction complete for %s (%d chars)", resolved_path, len(text))
        return FileExtractionResult(text=text, metadata=metadata, mime_type=mime)


def create_extraction_service() -> FileExtractionService:
    service = FileExtractionService()
    return service


__all__ = ["FileExtractionService", "FileExtractionResult", "create_extraction_service"]
