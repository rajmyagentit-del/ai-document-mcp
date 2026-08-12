"""Chunking and contextual enrichment.

Naive chunking (split text into fixed-size blocks) loses context: a chunk reading
"the fee is 2%" is nearly useless for retrieval or grounding once separated from
the surrounding document. Contextual enrichment asks a cheap model to generate a
short situating note for each chunk *before* embedding, so the embedded text
carries the context that made the chunk meaningful in the first place.

Reference: Anthropic's "Contextual Retrieval" technique.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import anthropic

logger = logging.getLogger(__name__)

CHUNK_SIZE_CHARS = 1200
CHUNK_OVERLAP_CHARS = 150

ENRICHMENT_MODEL = "claude-haiku-4-5-20251001"

CONTEXT_PROMPT_TEMPLATE = """\
<document>
{document_excerpt}
</document>

Here is a chunk from the document above:
<chunk>
{chunk_text}
</chunk>

Write a short (1-2 sentence) context note situating this chunk within the \
overall document, so it can be understood correctly on its own. Answer only \
with the context note, nothing else."""

# How much surrounding document to show the model when generating a chunk's
# context note. Full-document context would be more accurate but costs more
# tokens per call; this is a deliberate cost/quality tradeoff.
DOCUMENT_EXCERPT_CHARS = 3000


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    page_number: int
    raw_text: str
    context_note: str
    enriched_text: str  # context_note + raw_text, this is what gets embedded


def chunk_pages(document_id: str, pages: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Split (page_number, text) pairs into (page_number, chunk_text) pairs.

    A simple sliding-window splitter. Each chunk stays tagged with its source
    page number so retrieval results can always be traced back to a page.
    """
    raw_chunks: list[tuple[int, str]] = []

    for page_number, text in pages:
        if not text.strip():
            continue

        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE_CHARS
            piece = text[start:end].strip()
            if piece:
                raw_chunks.append((page_number, piece))
            if end >= len(text):
                break
            start = end - CHUNK_OVERLAP_CHARS

    return raw_chunks


def enrich_chunk(
    document_id: str,
    chunk_index: int,
    page_number: int,
    chunk_text: str,
    full_document_text: str,
    client: anthropic.Anthropic,
) -> Chunk:
    """Generate a context note for one chunk and combine it with the raw text."""
    excerpt = full_document_text[:DOCUMENT_EXCERPT_CHARS]

    prompt = CONTEXT_PROMPT_TEMPLATE.format(
        document_excerpt=excerpt,
        chunk_text=chunk_text,
    )

    response = client.messages.create(
        model=ENRICHMENT_MODEL,
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}],
    )
    context_note = response.content[0].text.strip()

    return Chunk(
        chunk_id=f"{document_id}::chunk_{chunk_index}",
        document_id=document_id,
        page_number=page_number,
        raw_text=chunk_text,
        context_note=context_note,
        enriched_text=f"{context_note}\n\n{chunk_text}",
    )