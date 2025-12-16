from __future__ import annotations

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

WHITESPACE_RE = re.compile(r"\s+")
HEADER_FOOTER_RE = re.compile(r"(^Page \d+\s*of\s*\d+$)|(^Page \d+$)", re.IGNORECASE)
BULLET_RE = re.compile(r"^[•\-\*\u2022]\s+")
HYPHEN_LINE_BREAK_RE = re.compile(r"(\w+)-\n(\w+)")


def normalize_text(text: str) -> str:
    if not text:
        return ""

    # Unicode normalization
    text = unicodedata.normalize("NFC", text)

    # Newline normalization
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Fix hyphenated line breaks
    text = HYPHEN_LINE_BREAK_RE.sub(r"\1\2", text)

    # Remove headers/footers
    cleaned_lines = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if HEADER_FOOTER_RE.match(line):
            continue
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)

    # Remove bullets
    cleaned_lines = []
    for line in text.split("\n"):
        line = BULLET_RE.sub("", line).strip()
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)

    # Collapse whitespace
    lines = [WHITESPACE_RE.sub(" ", line).strip() for line in text.split("\n")]
    lines = [line for line in lines if line]

    # Deduplicate lines
    deduped = []
    seen = set()
    for line in lines:
        if line not in seen:
            seen.add(line)
            deduped.append(line)

    final_text = "\n".join(deduped)

    logger.debug(
        "Normalized text reduced from %d → %d characters",
        len(text), len(final_text)
    )

    return final_text


__all__ = ["normalize_text"]
