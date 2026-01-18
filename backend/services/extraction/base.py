from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Protocol


class ExtractionError(RuntimeError):
    """Raised when a document cannot be extracted."""


@dataclass
class ExtractedBlock:
    """
    A structured, display-ready block of extracted content.

    `kind` is a small stable enum-ish string so the UI can render it consistently:
      - "heading"
      - "paragraph"
      - "list"
      - "code"
      - "page_break"
    """

    kind: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedDocument:
    """Structured output returned by each extractor."""

    text: str
    metadata: Dict[str, Any]
    blocks: list[ExtractedBlock] = field(default_factory=list)


class BaseExtractor(Protocol):
    """Protocol all extractor implementations must follow."""

    name: str

    def supports(self, mime_type: str) -> bool:
        ...

    def extract(self, file_path: Path, *, mime_type: str | None = None) -> ExtractedDocument:
        ...


class SimpleExtractor:
    """Helper base class for extractors with static MIME declarations."""

    name: str = "simple_extractor"
    supported_mime_types: tuple[str, ...] = ()
    fallback: bool = False

    def supports(self, mime_type: str) -> bool:
        if not self.supported_mime_types:
            return self.fallback
        if mime_type in self.supported_mime_types:
            return True
        # Allow coarse patterns like "text/*".
        primary = mime_type.split("/", 1)[0] if "/" in mime_type else mime_type
        for declared in self.supported_mime_types:
            if declared.endswith("/*") and declared.split("/", 1)[0] == primary:
                return True
        return False

    def extract(self, file_path: Path, *, mime_type: str | None = None) -> ExtractedDocument:
        raise NotImplementedError("SimpleExtractor subclasses must implement extract()")


__all__ = ["BaseExtractor", "ExtractedBlock", "ExtractedDocument", "ExtractionError", "SimpleExtractor"]
