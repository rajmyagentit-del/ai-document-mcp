"""Document text extraction: text-native PDFs via PyMuPDF, scanned pages via Claude vision.

Design decision: rather than running a self-hosted OCR model (heavy, memory-hungry,
a poor fit for a free-tier server), scanned/image-only pages are sent to Claude's
vision API for extraction. Text-native pages never touch the API at all -- they're
extracted directly and for free. This keeps cost near-zero for typical PDFs while
still handling scanned documents.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass

import anthropic
import pymupdf

logger = logging.getLogger(__name__)

# Below this many characters of extracted text, a page is treated as "likely scanned"
# and routed to vision extraction instead of trusted as-is.
MIN_TEXT_CHARS_PER_PAGE = 20

# Cheap, fast model -- sufficient for OCR-style transcription, not full reasoning.
VISION_MODEL = "claude-haiku-4-5-20251001"


@dataclass
class ExtractedPage:
    page_number: int  # 1-indexed, matches how humans reference PDF pages
    text: str
    method: str  # "text_layer" or "vision_ocr"


def extract_pdf(pdf_bytes: bytes, anthropic_client: anthropic.Anthropic) -> list[ExtractedPage]:
    """Extract text from every page of a PDF, using vision OCR only where needed."""
    pages: list[ExtractedPage] = []

    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_index, page in enumerate(doc):
            native_text = page.get_text().strip()

            if len(native_text) >= MIN_TEXT_CHARS_PER_PAGE:
                pages.append(
                    ExtractedPage(
                        page_number=page_index + 1,
                        text=native_text,
                        method="text_layer",
                    )
                )
                continue

            logger.info("Page %d has little/no text layer; routing to vision OCR", page_index + 1)
            image_bytes = _render_page_to_png(page)
            vision_text = _extract_via_vision(image_bytes, anthropic_client)
            pages.append(
                ExtractedPage(
                    page_number=page_index + 1,
                    text=vision_text,
                    method="vision_ocr",
                )
            )

    return pages


def _render_page_to_png(page: pymupdf.Page) -> bytes:
    """Rasterize a PDF page to PNG bytes at a resolution readable by vision models."""
    # 2x zoom improves small-text legibility without producing an excessively large image.
    matrix = pymupdf.Matrix(2, 2)
    pixmap = page.get_pixmap(matrix=matrix)
    return pixmap.tobytes("png")


def _extract_via_vision(image_bytes: bytes, client: anthropic.Anthropic) -> str:
    """Ask Claude to transcribe the visible text from a page image."""
    image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

    response = client.messages.create(
        model=VISION_MODEL,
        max_tokens=2048,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "Transcribe all visible text from this document page exactly "
                            "as it appears. Preserve reading order. Output only the "
                            "transcribed text, no commentary."
                        ),
                    },
                ],
            }
        ],
    )
    return response.content[0].text.strip()