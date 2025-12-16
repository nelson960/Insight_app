from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Optional

from ..base import ExtractedDocument, SimpleExtractor
from ..normalizer import normalize_text

logger = logging.getLogger(__name__)


class PlainTextExtractor(SimpleExtractor):
    name = "plain_text"
    supported_mime_types = (
        "text/plain",
        "text/markdown",
        "application/json",
        "text/csv",
        "application/xml",
    )
    fallback = True

    def extract(self, file_path: Path, *, mime_type: Optional[str] = None) -> ExtractedDocument:
        logger.info("Extracting text content from %s (mime=%s)", file_path, mime_type)
        text = Path(file_path).read_text(encoding="utf-8", errors="ignore")

        if mime_type == "application/json":
            try:
                obj = json.loads(text)
                text = json.dumps(obj, indent=2)
            except Exception:
                pass
        elif mime_type in {"text/csv"}:
            text = self._extract_csv(file_path)

        metadata = {
            "extraction_method": "plain_text",
            "content_type": mime_type,
        }
        logger.debug("Text extraction complete for %s (length=%d)", file_path, len(text))
        return ExtractedDocument(text=normalize_text(text), metadata=metadata)

    @staticmethod
    def _extract_csv(file_path: Path) -> str:
        rows = []
        with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
            reader = csv.reader(handle)
            for row in reader:
                rows.append(", ".join(cell.strip() for cell in row))
        return "\n".join(rows)


__all__ = ["PlainTextExtractor"]
