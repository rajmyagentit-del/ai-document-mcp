"""AI Document Processing MCP Server.

Exposes three tools over MCP:
  - ingest_document: extract, chunk, contextually enrich, and store a PDF
  - search_documents: hybrid + agentic retrieval over stored documents
  - get_document_status: check what's stored for a given document id

Also exposes a browser-friendly live demo at /demo (HTML page) and
/demo/search (JSON endpoint), on top of the same search_documents logic
used by the MCP tool.
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

MAX_PDF_BYTES = 20 * 1024 * 1024  # 20 MB

mcp = FastMCP("ai-document-mcp")


@lru_cache(maxsize=1)
def _get_anthropic_client() -> anthropic.Anthropic:
    """Lazily create the Anthropic client on first use, not at import time."""
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


DEMO_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>AI Document MCP -- Live Demo</title>
<style>
  :root {
    --bg: #0b0d12;
    --panel: #12151c;
    --border: #232838;
    --text: #e6e8ee;
    --muted: #8b93a7;
    --accent: #7c9cff;
    --good: #4ade80;
    --warn: #fbbf24;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    padding: 40px 20px;
  }
  .wrap { max-width: 720px; margin: 0 auto; }
  h1 { font-size: 22px; margin-bottom: 4px; }
  p.sub { color: var(--muted); margin-top: 0; font-size: 14px; }
  .searchbox { display: flex; gap: 8px; margin: 24px 0; }
  input[type=text] {
    flex: 1;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 14px;
    color: var(--text);
    font-size: 15px;
  }
  button {
    background: var(--accent);
    border: none;
    border-radius: 8px;
    padding: 12px 20px;
    color: #0b0d12;
    font-weight: 600;
    cursor: pointer;
    font-size: 15px;
  }
  button:disabled { opacity: 0.5; cursor: default; }
  .badges { display: flex; gap: 8px; margin-bottom: 16px; }
  .badge {
    font-size: 12px;
    padding: 4px 10px;
    border-radius: 999px;
    border: 1px solid var(--border);
    color: var(--muted);
  }
  .badge.high { color: var(--good); border-color: var(--good); }
  .badge.low { color: var(--warn); border-color: var(--warn); }
  .badge.corrected { color: var(--accent); border-color: var(--accent); }
  .result {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 12px;
  }
  .result .meta { color: var(--muted); font-size: 12px; margin-bottom: 8px; }
  .result .context { color: var(--accent); font-size: 13px; margin-bottom: 8px; font-style: italic; }
  .result .text { font-size: 14px; line-height: 1.5; white-space: pre-wrap; }
  .empty { color: var(--muted); font-size: 14px; padding: 20px 0; }
  .loading { color: var(--muted); font-size: 14px; padding: 20px 0; }
</style>
</head>
<body>
<div class="wrap">
  <h1>AI Document MCP -- Live Demo</h1>
  <p class="sub">Hybrid (vector + BM25) retrieval with contextual enrichment and agentic self-correction, running live against Chroma Cloud.</p>
  <div class="searchbox">
    <input type="text" id="query" placeholder="Ask something about the ingested documents..." />
    <button id="go">Search</button>
  </div>
  <div id="output"></div>
</div>
<script>
const input = document.getElementById('query');
const button = document.getElementById('go');
const output = document.getElementById('output');

async function runSearch() {
  const query = input.value.trim();
  if (!query) return;
  button.disabled = true;
  output.innerHTML = '<div class="loading">Searching...</div>';
  try {
    const res = await fetch('/demo/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, n_results: 5 })
    });
    const data = await res.json();
    render(data);
  } catch (err) {
    output.innerHTML = '<div class="empty">Request failed: ' + err + '</div>';
  } finally {
    button.disabled = false;
  }
}

function render(data) {
  let html = '<div class="badges">';
  html += '<span class="badge ' + (data.confidence === 'high' ? 'high' : 'low') + '">confidence: ' + data.confidence + '</span>';
  if (data.self_corrected) {
    html += '<span class="badge corrected">self-corrected (query rewritten)</span>';
  }
  html += '</div>';
  if (data.query_used) {
    html += '<div class="sub" style="margin-bottom:12px;">query used: "' + escapeHtml(data.query_used) + '"</div>';
  }
  if (!data.results || data.results.length === 0) {
    html += '<div class="empty">No relevant results found.</div>';
  } else {
    for (const r of data.results) {
      html += '<div class="result">';
      html += '<div class="meta">' + escapeHtml(r.document_id) + ' -- page ' + r.page_number + ' -- score ' + r.score + '</div>';
      html += '<div class="context">' + escapeHtml(r.context_note) + '</div>';
      html += '<div class="text">' + escapeHtml(r.text) + '</div>';
      html += '</div>';
    }
  }
  output.innerHTML = html;
}

function escapeHtml(s) {
  const div = document.createElement('div');
  div.textContent = s;
  return div.innerHTML;
}

button.addEventListener('click', runSearch);
input.addEventListener('keydown', (e) => { if (e.key === 'Enter') runSearch(); });
</script>
</body>
</html>
"""


@mcp.custom_route("/demo", methods=["GET"])
async def demo_page(request):
    from starlette.responses import HTMLResponse

    return HTMLResponse(DEMO_HTML)


@mcp.custom_route("/demo/search", methods=["POST"])
async def demo_search(request):
    from starlette.responses import JSONResponse

    body = await request.json()
    query = body.get("query", "")
    n_results = body.get("n_results", 5)

    if not query:
        return JSONResponse({"error": "query is required"}, status_code=400)

    result = search_documents(query=query, n_results=n_results)
    return JSONResponse(result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    mcp.run(transport="http", host="0.0.0.0", port=port)
