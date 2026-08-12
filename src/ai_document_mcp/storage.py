"""Chroma Cloud storage: wraps collection access and chunk persistence.

Chunks are stored with their *enriched* text (context note + raw text) as the
embedded document, since that's what should be matched against queries. The
raw text and context note are kept separately in metadata so we can show
clean source text back to the user without the context note glued on.
"""

from __future__ import annotations

import os

import chromadb

from ai_document_mcp.chunking import Chunk

COLLECTION_NAME = "document_chunks"


def get_chroma_client() -> chromadb.ClientAPI:
    """Build a Chroma Cloud client from environment-provided credentials."""
    return chromadb.CloudClient(
        api_key=os.environ["CHROMA_API_KEY"],
        tenant=os.environ["CHROMA_TENANT"],
        database=os.environ["CHROMA_DATABASE"],
    )


def get_collection(client: chromadb.ClientAPI) -> chromadb.Collection:
    return client.get_or_create_collection(COLLECTION_NAME)


def store_chunks(collection: chromadb.Collection, chunks: list[Chunk]) -> None:
    """Persist a batch of enriched chunks to the collection."""
    if not chunks:
        return

    collection.add(
        ids=[c.chunk_id for c in chunks],
        documents=[c.enriched_text for c in chunks],
        metadatas=[
            {
                "document_id": c.document_id,
                "page_number": c.page_number,
                "raw_text": c.raw_text,
                "context_note": c.context_note,
            }
            for c in chunks
        ],
    )


def fetch_all_chunks(collection: chromadb.Collection) -> dict:
    """Fetch every chunk in the collection (ids, documents, metadatas).

    Used to build the in-memory BM25 keyword index. Fine at portfolio scale
    (hundreds to low thousands of chunks); a larger deployment would maintain
    a persistent keyword index instead of rebuilding it per query.
    """
    return collection.get(include=["documents", "metadatas"])


def delete_document(collection: chromadb.Collection, document_id: str) -> int:
    """Delete all chunks belonging to one document. Returns count deleted."""
    existing = collection.get(where={"document_id": document_id}, include=[])
    ids = existing["ids"]
    if ids:
        collection.delete(ids=ids)
    return len(ids)