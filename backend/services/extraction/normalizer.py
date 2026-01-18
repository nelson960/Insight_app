from __future__ import annotations

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

HEADER_FOOTER_RE = re.compile(r"(^Page \d+\s*of\s*\d+$)|(^Page \d+$)", re.IGNORECASE)
HYPHEN_LINE_BREAK_RE = re.compile(r"(\w+)-\n(\w+)")
BLANK_LINES_RE = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    if not text:
        return ""

    original_len = len(text)

    # Unicode normalization
    text = unicodedata.normalize("NFC", text)

    # Newline normalization
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Fix hyphenated line breaks
    text = HYPHEN_LINE_BREAK_RE.sub(r"\1\2", text)

    # Remove obvious headers/footers but preserve paragraph structure.
    # Important: keep blank lines so downstream segmentation can split on "\n\n".
    lines: list[str] = []
    for raw_line in text.split("\n"):
        candidate = raw_line.strip()
        if candidate and HEADER_FOOTER_RE.match(candidate):
            lines.append("")
            continue
        lines.append(raw_line.rstrip())

    final_text = "\n".join(lines)
    final_text = BLANK_LINES_RE.sub("\n\n", final_text).strip()

    logger.debug(
        "Normalized text reduced from %d → %d characters",
        original_len,
        len(final_text),
    )

    return final_text


__all__ = ["normalize_text"]
