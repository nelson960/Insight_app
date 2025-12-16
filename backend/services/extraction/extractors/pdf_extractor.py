from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from ..base import ExtractedDocument, ExtractionError, SimpleExtractor
from ..normalizer import normalize_text

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
        if doc.is_encrypted:
            raise ExtractionError("Encrypted PDF detected; unable to extract without password.")
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text("text"))
        raw_text = "\n".join(text_parts)
        metadata = {
            "page_count": len(doc),
            "extraction_method": "pdf_text",
            "content_type": mime_type or "application/pdf",
            "title": doc.metadata.get("title") if doc.metadata else None,
        }
        doc.close()
        logger.debug("PDF extraction completed for %s (%d pages)", file_path, metadata["page_count"])
        return ExtractedDocument(text=normalize_text(raw_text), metadata=metadata)


__all__ = ["PDFExtractor"]
