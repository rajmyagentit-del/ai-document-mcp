"""Tests for extraction.py. Only the text-native path is covered here --
it requires no API calls, so it's fast and free to run on every commit.

The vision-OCR path is tested separately in test_extraction_vision.py,
marked so it can be skipped in routine CI runs since it costs real API money.
"""

import pymupdf

from ai_document_mcp.extraction import extract_pdf


def _make_pdf_with_text(text: str) -> bytes:
    """Build a minimal in-memory PDF containing the given text."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_extract_pdf_text_native_page_uses_text_layer(monkeypatch):
    """A PDF with a real text layer should be extracted without calling the vision API."""
    pdf_bytes = _make_pdf_with_text("Hello from a real text layer.")

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("Vision API should not be called for text-native pages")

    # No anthropic_client method should be invoked for a text-native page --
    # patch messages.create to fail loudly if it's ever reached.
    class _FakeClient:
        class messages:
            create = staticmethod(_fail_if_called)

    pages = extract_pdf(pdf_bytes, _FakeClient())

    assert len(pages) == 1
    assert pages[0].method == "text_layer"
    assert "Hello from a real text layer." in pages[0].text
    assert pages[0].page_number == 1


def test_extract_pdf_multiple_pages_numbered_correctly():
    """Page numbers should be 1-indexed and match document order."""
    doc = pymupdf.open()
    for i in range(3):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page number {i + 1} content.")
    pdf_bytes = doc.tobytes()
    doc.close()

    class _FakeClient:
        pass

    pages = extract_pdf(pdf_bytes, _FakeClient())

    assert [p.page_number for p in pages] == [1, 2, 3]
    assert "Page number 1" in pages[0].text
    assert "Page number 3" in pages[2].text