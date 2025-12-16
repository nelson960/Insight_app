from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from ..base import ExtractedDocument, ExtractionError, SimpleExtractor
from ..normalizer import normalize_text

try:
    from bs4 import BeautifulSoup  # type: ignore
except ImportError:  # pragma: no cover
    BeautifulSoup = None

logger = logging.getLogger(__name__)


class HtmlExtractor(SimpleExtractor):
    name = "html"
    supported_mime_types = ("text/html",)

    def extract(self, file_path: Path, *, mime_type: Optional[str] = None) -> ExtractedDocument:
        if BeautifulSoup is None:
            raise ExtractionError(
                "BeautifulSoup (bs4) is required for HTML extraction. Install with `pip install beautifulsoup4`."
            )
        logger.info("Extracting HTML text from %s", file_path)
        text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(text, "html.parser")
        extracted = soup.get_text(separator="\n")
        metadata = {
            "title": soup.title.string if soup.title else None,
            "extraction_method": "html_text",
            "content_type": mime_type or "text/html",
        }
        logger.debug("HTML extraction captured title=%s for %s", metadata["title"], file_path)
        return ExtractedDocument(text=normalize_text(extracted), metadata=metadata)


__all__ = ["HtmlExtractor"]
