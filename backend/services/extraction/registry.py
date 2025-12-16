from __future__ import annotations

import logging
from typing import Iterable, List, Sequence

from .base import BaseExtractor, ExtractionError
from .extractors import (
    DocxExtractor,
    HtmlExtractor,
    ImageExtractor,
    PDFExtractor,
    PlainTextExtractor,
    PptxExtractor,
)

logger = logging.getLogger(__name__)


class ExtractorRegistry:
    """Registry that selects the appropriate extractor for a MIME type."""

    def __init__(self) -> None:
        self._extractors: List[BaseExtractor] = []

    def register(self, extractor: BaseExtractor) -> None:
        self._extractors.append(extractor)
        logger.debug("Registered extractor %s", extractor.name)

    def extend(self, extractors: Iterable[BaseExtractor]) -> None:
        for extractor in extractors:
            self.register(extractor)

    def resolve(self, mime_type: str) -> BaseExtractor:
        for extractor in self._extractors:
            if extractor.supports(mime_type):
                logger.debug("Resolved extractor %s for mime %s", extractor.name, mime_type)
                return extractor
        # Fallback to first extractor that declared fallback behavior.
        for extractor in self._extractors:
            if getattr(extractor, "fallback", False):
                return extractor
        logger.error("No extractor registered for MIME type %s", mime_type)
        raise ExtractionError(f"No extractor registered for MIME type {mime_type}")


def build_default_registry() -> ExtractorRegistry:
    registry = ExtractorRegistry()
    registry.extend(
        [
            PDFExtractor(),
            DocxExtractor(),
            PptxExtractor(),
            HtmlExtractor(),
            ImageExtractor(),
            PlainTextExtractor(),
        ]
    )
    return registry


__all__ = ["ExtractorRegistry", "build_default_registry"]
