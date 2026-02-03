from __future__ import annotations
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .base import ExtractedBlock, ExtractedDocument, ExtractionError
from .detector import detect_mime_type
from .normalizer import normalize_text
from .registry import ExtractorRegistry, build_default_registry

logger = logging.getLogger(__name__)

PAGE_MARKER_RE = re.compile(r"^---\s*Page\s+(\d+)\s*---$", re.IGNORECASE)
MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
# Note: put '-' at the end (or escape it) to avoid character range parsing.
# Some PDFs/DOCX exports include zero-width spaces after bullets.
BULLET_RE = re.compile(r"^\s*([*•●○▪‣–-]|\d+[.)])[\s\u200b]+")


@dataclass
class FileExtractionResult:
    text: str
    metadata: dict
    mime_type: str
    blocks: list[ExtractedBlock]


def _looks_like_list(lines: list[str]) -> bool:
    content_lines = [ln for ln in lines if ln.strip()]
    if len(content_lines) < 2:
        return False
    hits = sum(1 for ln in content_lines if BULLET_RE.match(ln))
    return hits >= max(2, int(len(content_lines) * 0.6))


def _blocks_from_text(text: str) -> list[ExtractedBlock]:
    blocks: list[ExtractedBlock] = []
    for raw in (text or "").split("\n\n"):
        para = raw.strip("\n")
        if not para.strip():
            continue

        marker = PAGE_MARKER_RE.match(para.strip())
        if marker:
            page = int(marker.group(1))
            blocks.append(ExtractedBlock(kind="page_break", text=f"Page {page}", metadata={"page": page}))
            continue

        lines = para.splitlines()

        # Markdown headings (common for text/markdown inputs).
        if len(lines) == 1:
            heading = MD_HEADING_RE.match(lines[0].strip())
            if heading:
                level = len(heading.group(1))
                blocks.append(
                    ExtractedBlock(
                        kind="heading",
                        text=heading.group(2).strip(),
                        metadata={"level": level},
                    )
                )
                continue

        if _looks_like_list(lines):
            items = []
            for ln in lines:
                s = ln.strip()
                if not s:
                    continue
                items.append(BULLET_RE.sub("", s).strip() or s)
            blocks.append(ExtractedBlock(kind="list", text="\n".join(items), metadata={"items": items}))
            continue

        # Default: unwrap line-wrapped paragraphs into a single readable line.
        joined = " ".join(ln.strip() for ln in lines if ln.strip())
        joined = re.sub(r"\s+", " ", joined).strip()
        blocks.append(ExtractedBlock(kind="paragraph", text=joined, metadata={}))

    return blocks


def blocks_from_text(text: str) -> list[ExtractedBlock]:
    """
    Convert normalized plain text into display-friendly blocks.

    This is intentionally shared between ingestion (authoritative) and any fallback
    reconstruction paths (e.g., rebuilding from stored chunks).
    """
    return _blocks_from_text(text)


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
        if document.blocks:
            blocks: list[ExtractedBlock] = []
            for block in document.blocks:
                if block.kind == "page_break":
                    blocks.append(block)
                    continue
                blocks.append(ExtractedBlock(kind=block.kind, text=normalize_text(block.text), metadata=block.metadata))
        else:
            blocks = blocks_from_text(text)

        metadata = {
            "content_type": mime,
            "source_path": str(resolved_path),
            "source_name": resolved_path.name,
            "source_ext": resolved_path.suffix.lower(),
            **(document.metadata or {}),
        }
        logger.info("Extraction complete for %s (%d chars)", resolved_path, len(text))
        return FileExtractionResult(text=text, metadata=metadata, mime_type=mime, blocks=blocks)


def create_extraction_service() -> FileExtractionService:
    service = FileExtractionService()
    return service


__all__ = ["FileExtractionService", "FileExtractionResult", "blocks_from_text", "create_extraction_service"]
