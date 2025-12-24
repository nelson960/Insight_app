from __future__ import annotations

import logging
import re
import statistics
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Optional

from ..base import ExtractedBlock, ExtractedDocument, ExtractionError, SimpleExtractor

try:
    import fitz  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    fitz = None

try:
    from PIL import Image  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    Image = None

try:
    import pytesseract  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pytesseract = None

logger = logging.getLogger(__name__)

_BULLET_RE = re.compile(r"^\s*([*•●○▪‣–-]|\d+[.)])[\s\u200b]+")
_HEADING_NUM_RE = re.compile(r"^(\d+(\.\d+){0,6}|[IVXLCDM]{1,8})[.)]?\s+", re.IGNORECASE)


def _norm_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").strip().lower())


def _match_toc_heading(first_line: str, toc_entries: list[tuple[int, str]], *, y0: float) -> int | None:
    """
    Best-effort match of a PDF outline/TOC title to a block's first line.

    Only attempts matching near the top of the page to avoid false positives.
    """
    if not toc_entries or y0 > 250:
        return None
    fl_key = _norm_key(first_line)
    if not fl_key:
        return None
    best_level: int | None = None
    best_len = 0
    for level, title in toc_entries:
        title_key = _norm_key(title)
        if not title_key:
            continue
        if fl_key == title_key:
            if len(title_key) > best_len:
                best_len = len(title_key)
                best_level = level
            continue
        if len(title_key) >= 14 and (fl_key.startswith(title_key) or title_key.startswith(fl_key)):
            if len(title_key) > best_len:
                best_len = len(title_key)
                best_level = level
    return best_level


def _looks_like_list(lines: list[str]) -> bool:
    content_lines = [ln for ln in lines if ln.strip()]
    if len(content_lines) < 2:
        return False
    hits = sum(1 for ln in content_lines if _BULLET_RE.match(ln))
    return hits >= max(2, int(len(content_lines) * 0.6))


def _is_preformatted(lines: list[str]) -> bool:
    if len(lines) < 2:
        return False
    markers = sum(1 for ln in lines if ("\t" in ln) or ("|" in ln) or re.search(r"\s{2,}", ln))
    return markers >= max(2, int(len(lines) * 0.5))


def _iter_span_sizes(block: dict[str, Any]) -> Iterable[float]:
    for line in block.get("lines") or []:
        if not isinstance(line, dict):
            continue
        for span in line.get("spans") or []:
            if not isinstance(span, dict):
                continue
            size = span.get("size")
            if isinstance(size, (int, float)):
                yield float(size)


def _render_line_text(line: dict[str, Any]) -> str:
    spans = line.get("spans") or []
    if not isinstance(spans, list):
        return ""
    ordered = []
    for span in spans:
        if isinstance(span, dict):
            ordered.append(span)
    ordered.sort(key=lambda s: (s.get("bbox") or (0, 0, 0, 0))[0])
    parts: list[str] = []
    prev_x1: float | None = None
    for span in ordered:
        bbox = span.get("bbox") if isinstance(span.get("bbox"), (list, tuple)) else None
        if bbox and len(bbox) >= 3 and prev_x1 is not None:
            gap = float(bbox[0]) - prev_x1
            if gap >= 10:
                parts.append(" ")
            if gap >= 20:
                parts.append(" ")
        parts.append(str(span.get("text") or ""))
        if bbox and len(bbox) >= 3:
            prev_x1 = float(bbox[2])
    text = "".join(parts)
    # PDFs often include invisible separators to enforce line breaking; treat as whitespace.
    text = (
        text.replace("\u200b", " ")  # zero-width space
        .replace("\ufeff", " ")  # zero-width no-break space / BOM
        .replace("\u2060", " ")  # word joiner
        .replace("\xa0", " ")  # NBSP
        .replace("\u00ad", "")  # soft hyphen
    )
    return text.strip("\n")


def _sort_text_blocks(blocks: list[dict[str, Any]], *, page_width: float) -> list[dict[str, Any]]:
    """
    Sort blocks into a best-effort reading order.

    PyMuPDF's `sort=True` helps, but two-column PDFs can still interleave. We detect
    a large x0 gap and sort columns left-to-right.
    """
    if not blocks:
        return []

    def key_xy(b: dict[str, Any]) -> tuple[float, float]:
        bbox = b.get("bbox") or (0, 0, 0, 0)
        return (float(bbox[1]), float(bbox[0]))

    xs = sorted(float((b.get("bbox") or (0, 0, 0, 0))[0]) for b in blocks)
    if len(xs) >= 2 and page_width > 0:
        gaps = [(xs[i + 1] - xs[i], i) for i in range(len(xs) - 1)]
        max_gap, max_idx = max(gaps, key=lambda t: t[0])
        if max_gap > page_width * 0.25:
            split_x = (xs[max_idx] + xs[max_idx + 1]) / 2.0
            left = [b for b in blocks if float((b.get("bbox") or (0, 0, 0, 0))[0]) <= split_x]
            right = [b for b in blocks if float((b.get("bbox") or (0, 0, 0, 0))[0]) > split_x]
            return sorted(left, key=key_xy) + sorted(right, key=key_xy)

    return sorted(blocks, key=key_xy)


def _heading_level(max_size: float, median_size: float) -> int:
    delta = max_size - median_size
    if delta >= 8:
        return 1
    if delta >= 5:
        return 2
    if delta >= 3:
        return 3
    return 4


def _maybe_heading(
    lines: list[str],
    *,
    max_size: float,
    median_size: float,
    bold_ratio: float,
    gap_before: float,
    gap_after: float,
    y0: float,
) -> int | None:
    if not lines:
        return None
    if len(lines) > 2:
        return None
    joined = " ".join(ln.strip() for ln in lines if ln.strip()).strip()
    if not joined or len(joined) > 160:
        return None
    words = joined.split()
    if len(words) > 18:
        return None
    if joined[-1] in {".", "?", "!"}:
        # Avoid common false positives ("Heading." in body text).
        return None

    numbered = bool(_HEADING_NUM_RE.match(joined))

    score = 0.0
    delta = max_size - median_size
    if delta >= 6:
        score += 3
    elif delta >= 4:
        score += 2
    elif delta >= 2:
        score += 1
    elif delta >= 1:
        score += 0.5

    if bold_ratio >= 0.8:
        score += 2
    elif bold_ratio >= 0.5:
        score += 1.5
    elif bold_ratio >= 0.3:
        score += 1

    line_h = max(8.0, median_size * 1.2)
    if gap_before >= line_h * 1.2:
        score += 1
    if gap_before >= line_h * 2.0:
        score += 1
    if gap_after >= line_h * 1.2:
        score += 0.5

    if y0 <= 110:
        score += 0.5
    if joined.isupper():
        score += 0.5
    if numbered:
        score += 0.5

    if score < 3.5:
        return None

    if numbered:
        m = _HEADING_NUM_RE.match(joined)
        if m:
            token = m.group(1)
            depth = token.count(".")
            return max(2, min(6, 2 + depth))
        return 3

    return _heading_level(max_size, median_size)
    return None


def _ocr_pdf_page(page: Any) -> str:
    """
    OCR a rendered PDF page image.

    This is a fallback for scanned PDFs (no extractable text blocks).
    """
    if Image is None or pytesseract is None:
        return ""

    try:
        pix = page.get_pixmap(dpi=200)
        png_bytes = pix.tobytes("png")
    except Exception:
        logger.exception("Failed to render PDF page for OCR")
        return ""

    try:
        image = Image.open(BytesIO(png_bytes))
        return str(pytesseract.image_to_string(image) or "")
    except Exception:
        logger.exception("pytesseract OCR failed for PDF page")
        return ""


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

            toc_by_page: dict[int, list[tuple[int, str]]] = {}
            try:
                toc = doc.get_toc(simple=True) or []
                for entry in toc:
                    if not isinstance(entry, (list, tuple)) or len(entry) < 3:
                        continue
                    level, title, page = entry[0], entry[1], entry[2]
                    try:
                        page_num = int(page)
                    except Exception:
                        continue
                    clean_title = str(title or "").strip()
                    if not clean_title:
                        continue
                    try:
                        lvl = int(level)
                    except Exception:
                        lvl = 2
                    toc_by_page.setdefault(page_num, []).append((lvl, clean_title))
                for page_num, entries in list(toc_by_page.items()):
                    entries.sort(key=lambda t: len(_norm_key(t[1])), reverse=True)
                    toc_by_page[page_num] = entries
            except Exception:
                toc_by_page = {}

            page_char_counts: list[int] = []
            pages_with_text = 0
            ocr_pages: list[int] = []
            blocks: list[ExtractedBlock] = []
            text_parts: list[str] = []

            for page_index in range(len(doc)):
                page = doc.load_page(page_index)
                page_num = page_index + 1
                blocks.append(ExtractedBlock(kind="page_break", text=f"Page {page_num}", metadata={"page": page_num}))

                page_dict = page.get_text("dict", sort=True) or {}
                raw_blocks = page_dict.get("blocks")
                if not isinstance(raw_blocks, list):
                    raw_blocks = []

                text_blocks: list[dict[str, Any]] = []
                span_sizes: list[float] = []
                for b in raw_blocks:
                    if not isinstance(b, dict) or b.get("type") != 0:
                        continue
                    if not b.get("lines"):
                        continue
                    text_blocks.append(b)
                    span_sizes.extend(list(_iter_span_sizes(b)))

                median_size = float(statistics.median(span_sizes)) if span_sizes else 11.0
                ordered_blocks = _sort_text_blocks(text_blocks, page_width=float(getattr(page.rect, "width", 0) or 0))
                page_toc = toc_by_page.get(page_num, [])

                page_text_parts: list[str] = []
                list_buffer: list[str] = []

                if not ordered_blocks:
                    ocr_text = _ocr_pdf_page(page).strip()
                    if ocr_text:
                        blocks.append(
                            ExtractedBlock(
                                kind="paragraph",
                                text=ocr_text,
                                metadata={"page": page_num, "ocr": True},
                            )
                        )
                        page_text_parts.append(ocr_text)
                        ocr_pages.append(page_num)

                for idx, b in enumerate(ordered_blocks):
                    bbox = b.get("bbox") or (0, 0, 0, 0)
                    y0 = float(bbox[1]) if len(bbox) >= 2 else 0.0
                    y1 = float(bbox[3]) if len(bbox) >= 4 else y0
                    prev_bbox = ordered_blocks[idx - 1].get("bbox") if idx - 1 >= 0 else None
                    prev_y1 = float(prev_bbox[3]) if isinstance(prev_bbox, (list, tuple)) and len(prev_bbox) >= 4 else y0
                    next_bbox = ordered_blocks[idx + 1].get("bbox") if idx + 1 < len(ordered_blocks) else None
                    next_y0 = (
                        float(next_bbox[1])
                        if isinstance(next_bbox, (list, tuple)) and len(next_bbox) >= 2
                        else y1
                    )
                    gap_before = max(0.0, y0 - prev_y1)
                    gap_after = max(0.0, next_y0 - y1)

                    lines: list[str] = []
                    bold_spans = 0
                    total_spans = 0
                    for line in b.get("lines") or []:
                        if not isinstance(line, dict):
                            continue
                        rendered = _render_line_text(line).strip()
                        if rendered:
                            lines.append(rendered)
                        for span in line.get("spans") or []:
                            if not isinstance(span, dict):
                                continue
                            total_spans += 1
                            font = str(span.get("font") or "")
                            flags = span.get("flags")
                            is_bold = False
                            if isinstance(flags, int) and (flags & 16):
                                is_bold = True
                            if "bold" in font.lower():
                                is_bold = True
                            if is_bold:
                                bold_spans += 1
                    if not lines:
                        continue

                    sizes = list(_iter_span_sizes(b))
                    max_size = max(sizes) if sizes else median_size
                    bold_ratio = (bold_spans / total_spans) if total_spans else 0.0

                    if _looks_like_list(lines):
                        items = []
                        for ln in lines:
                            s = ln.strip()
                            if not s:
                                continue
                            cleaned = _BULLET_RE.sub("", s).strip()
                            items.append(cleaned or s)
                        list_buffer.extend(items)
                        continue

                    if len(lines) == 1 and _BULLET_RE.match(lines[0]):
                        cleaned = _BULLET_RE.sub("", lines[0].strip()).strip()
                        list_buffer.append(cleaned or lines[0].strip())
                        continue

                    if list_buffer:
                        blocks.append(
                            ExtractedBlock(
                                kind="list",
                                text="\n".join(list_buffer),
                                metadata={"items": list_buffer, "page": page_num},
                            )
                        )
                        page_text_parts.append("\n".join(list_buffer))
                        list_buffer = []

                    toc_level = _match_toc_heading(lines[0], page_toc, y0=y0)
                    if toc_level is not None:
                        heading_text = lines[0].strip()
                        level = max(1, min(int(toc_level or 2), 6))
                        blocks.append(
                            ExtractedBlock(
                                kind="heading",
                                text=heading_text,
                                metadata={"level": level, "page": page_num, "source": "toc"},
                            )
                        )
                        page_text_parts.append(heading_text)
                        rest = [ln for ln in lines[1:] if ln.strip()]
                        if not rest:
                            continue
                        if _looks_like_list(rest):
                            items = [_BULLET_RE.sub("", ln.strip()).strip() for ln in rest if ln.strip()]
                            items = [it for it in items if it]
                            if items:
                                blocks.append(
                                    ExtractedBlock(
                                        kind="list",
                                        text="\n".join(items),
                                        metadata={"items": items, "page": page_num},
                                    )
                                )
                                page_text_parts.append("\n".join(items))
                                continue
                        if _is_preformatted(rest):
                            text = "\n".join(rest).rstrip()
                            blocks.append(ExtractedBlock(kind="code", text=text, metadata={"page": page_num}))
                            page_text_parts.append(text)
                            continue
                        joined = " ".join(ln.strip() for ln in rest if ln.strip())
                        joined = re.sub(r"\s+", " ", joined).strip()
                        if joined:
                            blocks.append(ExtractedBlock(kind="paragraph", text=joined, metadata={"page": page_num}))
                            page_text_parts.append(joined)
                        continue

                    heading_level = _maybe_heading(
                        lines,
                        max_size=max_size,
                        median_size=median_size,
                        bold_ratio=bold_ratio,
                        gap_before=gap_before,
                        gap_after=gap_after,
                        y0=y0,
                    )
                    if heading_level is not None:
                        text = " ".join(ln.strip() for ln in lines if ln.strip()).strip()
                        blocks.append(
                            ExtractedBlock(
                                kind="heading",
                                text=text,
                                metadata={"level": heading_level, "page": page_num},
                            )
                        )
                        page_text_parts.append(text)
                        continue

                    if _is_preformatted(lines):
                        text = "\n".join(lines).rstrip()
                        blocks.append(ExtractedBlock(kind="code", text=text, metadata={"page": page_num}))
                        page_text_parts.append(text)
                        continue

                    joined = " ".join(ln.strip() for ln in lines if ln.strip())
                    joined = re.sub(r"\s+", " ", joined).strip()
                    blocks.append(ExtractedBlock(kind="paragraph", text=joined, metadata={"page": page_num}))
                    page_text_parts.append(joined)

                if list_buffer:
                    blocks.append(
                        ExtractedBlock(
                            kind="list",
                            text="\n".join(list_buffer),
                            metadata={"items": list_buffer, "page": page_num},
                        )
                    )
                    page_text_parts.append("\n".join(list_buffer))
                    list_buffer = []

                page_text = "\n\n".join([p for p in page_text_parts if p.strip()]).strip()
                page_char_counts.append(len(page_text))
                if page_text.strip():
                    pages_with_text += 1
                # Preserve page boundaries for downstream chunking + citations.
                text_parts.append(f"\n\n--- Page {page_num} ---\n\n{page_text}")

            raw_text = "".join(text_parts).lstrip("\n")

            metadata = {
                "page_count": len(doc),
                "pages_with_text": pages_with_text,
                "ocr_pages": ocr_pages,
                "page_char_counts": page_char_counts,
                "page_markers": True,
                "extraction_method": "pdf_blocks",
                "content_type": mime_type or "application/pdf",
                "title": doc.metadata.get("title") if doc.metadata else None,
            }
            logger.debug("PDF extraction completed for %s (%d pages)", file_path, metadata["page_count"])
            return ExtractedDocument(text=raw_text, metadata=metadata, blocks=blocks)
        finally:
            try:
                doc.close()
            except Exception:
                pass


__all__ = ["PDFExtractor"]
