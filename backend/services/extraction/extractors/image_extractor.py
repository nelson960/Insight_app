from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from ..base import ExtractedDocument, ExtractionError, SimpleExtractor
from ..normalizer import normalize_text

try:
    from PIL import Image  # type: ignore
except ImportError:  # pragma: no cover
    Image = None

try:
    import pytesseract  # type: ignore
except ImportError:  # pragma: no cover
    pytesseract = None

logger = logging.getLogger(__name__)


class ImageExtractor(SimpleExtractor):
    name = "image_ocr"
    supported_mime_types = ("image/png", "image/jpeg", "image/jpg")

    def extract(self, file_path: Path, *, mime_type: Optional[str] = None) -> ExtractedDocument:
        if Image is None or pytesseract is None:
            raise ExtractionError(
                "Pillow and pytesseract are required for OCR extraction. Install with `pip install pillow pytesseract`."
            )
        logger.info("Running OCR on %s", file_path)
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
        metadata = {
            "extraction_method": "ocr",
            "content_type": mime_type,
        }
        logger.debug("OCR extraction complete for %s (chars=%d)", file_path, len(text))
        return ExtractedDocument(text=normalize_text(text), metadata=metadata)


__all__ = ["ImageExtractor"]
