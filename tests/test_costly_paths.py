"""Tests that make real Anthropic API and/or Chroma Cloud calls.

These are marked `costs_money` and are excluded from the default `pytest` run
(see addopts in pyproject.toml). Run them explicitly with:

    pytest -m costs_money -v

Requires ANTHROPIC_API_KEY, CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE
to be set in the environment.
"""

import base64

import anthropic
import pymupdf
import pytest

from ai_document_mcp.extraction import extract_pdf
from ai_document_mcp.server import get_document_status, ingest_document, search_documents
from ai_document_mcp.storage import delete_document, get_chroma_client, get_collection


@pytest.mark.costs_money
def test_extract_pdf_scanned_page_uses_vision_ocr():
    """A page rendered as an image (no text layer) should route through vision OCR
    and successfully transcribe the visible text."""
    doc = pymupdf.open()
    page = doc.new_page()
    # Draw the text as vector graphics on a fresh page, then flatten to an image-only
    # page by rasterizing and re-inserting as an image with no text layer, so
    # get_text() returns nothing and the vision path is genuinely exercised.
    page.insert_text((72, 72), "INVOICE TOTAL: 4200 DOLLARS")
    pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))

    image_only_doc = pymupdf.open()
    image_page = image_only_doc.new_page(width=page.rect.width, height=page.rect.height)
    image_page.insert_image(image_page.rect, pixmap=pix)
    pdf_bytes = image_only_doc.tobytes()
    doc.close()
    image_only_doc.close()

    client = anthropic.Anthropic()
    pages = extract_pdf(pdf_bytes, client)

    assert len(pages) == 1
    assert pages[0].method == "vision_ocr"
    assert "4200" in pages[0].text or "4,200" in pages[0].text


@pytest.mark.costs_money
def test_ingest_and_search_round_trip():
    """Full pipeline: ingest a real PDF into Chroma Cloud, then search for it,
    then clean up after itself."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "The refund window is 30 days from purchase date.")
    pdf_bytes = doc.tobytes()
    doc.close()

    document_id = "test_integration_refund_policy"
    pdf_b64 = base64.b64encode(pdf_bytes).decode()

    try:
        ingest_result = ingest_document(document_id=document_id, pdf_base64=pdf_b64)
        assert ingest_result["status"] == "success"
        assert ingest_result["chunks_stored"] >= 1

        status = get_document_status(document_id=document_id)
        assert status["found"] is True

        search_result = search_documents(query="How many days is the refund window?")
        matching_doc_ids = {r["document_id"] for r in search_result["results"]}
        assert document_id in matching_doc_ids
    finally:
        # Always clean up test data, even if an assertion above fails.
        client = get_chroma_client()
        collection = get_collection(client)
        delete_document(collection, document_id)