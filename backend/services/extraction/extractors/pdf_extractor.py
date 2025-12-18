from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from ..base import ExtractedDocument, ExtractionError, SimpleExtractor

try:
    import fitz  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    fitz = None

logger = logging.getLogger(__name__)


class PDFExtractor(SimpleExtractor):
    name = "pdf"
    supported_mime_types = ("application/pdf",)

    def extract(self, file_path: Path, *, mime_type: Optional[str] = None) -> ExtractedDocument:
        if fitz is None:
            raise ExtractionError(
                "PyMuPDF (fitz) is required for PDF extraction. Install with `pip install pymupdf`."
            )
        logger.info("Extracting PDF text from %s", file_path)
        doc = fitz.open(str(file_path))
        try:
            if doc.is_encrypted:
                raise ExtractionError("Encrypted PDF detected; unable to extract without password.")

            page_char_counts: list[int] = []
            pages_with_text = 0
            text_parts: list[str] = []

            for page_index in range(len(doc)):
                page = doc.load_page(page_index)
                page_text = page.get_text("text") or ""
                page_text = page_text.rstrip()
                page_char_counts.append(len(page_text))
                if page_text.strip():
                    pages_with_text += 1
                # Preserve page boundaries for downstream chunking + citations.
                text_parts.append(f"\n\n--- Page {page_index + 1} ---\n\n{page_text}")

            raw_text = "".join(text_parts).lstrip("\n")

            metadata = {
                "page_count": len(doc),
                "pages_with_text": pages_with_text,
                "page_char_counts": page_char_counts,
                "page_markers": True,
                "extraction_method": "pdf_text",
                "content_type": mime_type or "application/pdf",
                "title": doc.metadata.get("title") if doc.metadata else None,
            }
            logger.debug("PDF extraction completed for %s (%d pages)", file_path, metadata["page_count"])
            return ExtractedDocument(text=raw_text, metadata=metadata)
        finally:
            try:
                doc.close()
            except Exception:
                pass


__all__ = ["PDFExtractor"]
