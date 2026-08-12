"""AI Document Processing MCP Server.

Exposes three tools over MCP:
  - ingest_document: extract, chunk, contextually enrich, and store a PDF
  - search_documents: hybrid + agentic retrieval over stored documents
  - get_document_status: check what's stored for a given document id
"""

from __future__ import annotations

import base64
import binascii
import logging
import os
from functools import lru_cache

import anthropic
from fastmcp import FastMCP

from ai_document_mcp.chunking import chunk_pages, enrich_chunk
from ai_document_mcp.extraction import extract_pdf
from ai_document_mcp.retrieval import agentic_search
from ai_document_mcp.storage import (
    delete_document,
    fetch_all_chunks,
    get_chroma_client,
    get_collection,
    store_chunks,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Guardrail: reject absurdly large uploads before we ever decode/parse them.
MAX_PDF_BYTES = 20 * 1024 * 1024  # 20 MB

mcp = FastMCP("ai-document-mcp")


@lru_cache(maxsize=1)
def _get_anthropic_client() -> anthropic.Anthropic:
    """Lazily create the Anthropic client on first use, not at import time.

    Creating this eagerly at module level would mean simply *importing*
    server.py requires a live ANTHROPIC_API_KEY -- which breaks things like
    CI test collection, where the module gets imported but the key isn't
    (and shouldn't need to be) present.
    """
    return anthropic.Anthropic()


@lru_cache(maxsize=1)
def _get_document_collection():
    """Lazily create the Chroma Cloud connection on first use, same reasoning."""
    client = get_chroma_client()
    return get_collection(client)


@mcp.tool()
def ingest_document(document_id: str, pdf_base64: str) -> dict:
    """Ingest a PDF: extract text (with vision OCR for scanned pages), chunk it,
    contextually enrich each chunk, and store it for retrieval.

    Args:
        document_id: A unique identifier you choose for this document (e.g. a
            filename or UUID). Re-ingesting the same document_id replaces the
            previous version.
        pdf_base64: The PDF file's bytes, base64-encoded.
    """
    try:
        pdf_bytes = base64.b64decode(pdf_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        return {"status": "error", "message": f"Invalid base64 PDF data: {exc}"}

    if len(pdf_bytes) > MAX_PDF_BYTES:
        return {
            "status": "error",
            "message": f"PDF exceeds {MAX_PDF_BYTES // (1024 * 1024)}MB limit.",
        }
    if pdf_bytes[:4] != b"%PDF":
        return {"status": "error", "message": "File does not look like a valid PDF."}

    logger.info("Ingesting document_id=%s (%d bytes)", document_id, len(pdf_bytes))

    # Replace any previous version of this document.
    delete_document(_get_document_collection(), document_id)

    pages = extract_pdf(pdf_bytes, _get_anthropic_client())
    full_document_text = "\n\n".join(p.text for p in pages)

    raw_chunks = chunk_pages(document_id, [(p.page_number, p.text) for p in pages])

    enriched_chunks = [
        enrich_chunk(
            document_id=document_id,
            chunk_index=i,
            page_number=page_number,
            chunk_text=chunk_text,
            full_document_text=full_document_text,
            client=_get_anthropic_client(),
        )
        for i, (page_number, chunk_text) in enumerate(raw_chunks)
    ]

    store_chunks(_get_document_collection(), enriched_chunks)

    vision_pages = sum(1 for p in pages if p.method == "vision_ocr")
    logger.info(
        "Ingested document_id=%s: %d pages (%d via vision OCR), %d chunks",
        document_id,
        len(pages),
        vision_pages,
        len(enriched_chunks),
    )

    return {
        "status": "success",
        "document_id": document_id,
        "pages_extracted": len(pages),
        "pages_via_vision_ocr": vision_pages,
        "chunks_stored": len(enriched_chunks),
    }


@mcp.tool()
def search_documents(query: str, n_results: int = 5) -> dict:
    """Search ingested documents using hybrid (vector + keyword) retrieval with
    agentic self-correction: if initial results look weak, the query is
    automatically reformulated and retried once.

    Args:
        query: The natural-language question or search query.
        n_results: How many top chunks to return (default 5).
    """
    result = agentic_search(_get_document_collection(), _get_anthropic_client(), query, n_results)

    return {
        "query_used": result.query_used,
        "self_corrected": result.self_corrected,
        "confidence": result.confidence,
        "results": [
            {
                "document_id": c.document_id,
                "page_number": c.page_number,
                "text": c.raw_text,
                "context_note": c.context_note,
                "score": round(c.score, 4),
            }
            for c in result.chunks
        ],
    }


@mcp.tool()
def get_document_status(document_id: str) -> dict:
    """Check how many chunks are stored for a given document id."""
    all_chunks = fetch_all_chunks(_get_document_collection())
    all_metadatas = all_chunks["metadatas"] or []
    matching = [
        (chunk_id, meta)
        for chunk_id, meta in zip(all_chunks["ids"], all_metadatas, strict=True)
        if meta["document_id"] == document_id
    ]

    if not matching:
        return {"document_id": document_id, "found": False}

    pages = sorted({int(meta["page_number"]) for _, meta in matching})  # type: ignore[arg-type]
    return {
        "document_id": document_id,
        "found": True,
        "chunks_stored": len(matching),
        "pages": pages,
    }


@mcp.custom_route("/health", methods=["GET"])
async def health(request):
    from starlette.responses import JSONResponse

    return JSONResponse({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    mcp.run(transport="http", host="0.0.0.0", port=port)