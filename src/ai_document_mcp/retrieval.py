"""Hybrid retrieval (vector + BM25) with agentic self-correction.

Two layers of "beyond naive RAG" here:

1. Hybrid search: vector search alone misses exact-term matches (e.g. a specific
   product code or clause number); BM25 alone misses semantic paraphrases. We run
   both and merge with Reciprocal Rank Fusion (RRF), a standard, well-understood
   fusion technique -- not a hand-rolled scoring heuristic.

2. Agentic self-correction: naive RAG always returns its top-k results, even when
   they're weak. Here, a cheap model judges whether the retrieved chunks actually
   look sufficient to answer the query. If not, the query is reformulated once and
   retried. This is a deliberately small, bounded version of the "reasoning-driven
   retrieval" pattern -- one retry, not an open-ended agent loop -- to keep cost
   and latency predictable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import anthropic
import chromadb
from rank_bm25 import BM25Okapi

from ai_document_mcp._llm import response_text
from ai_document_mcp.storage import fetch_all_chunks

logger = logging.getLogger(__name__)

CONFIDENCE_MODEL = "claude-haiku-4-5-20251001"
RRF_K = 60  # standard RRF damping constant


@dataclass
class RetrievedChunk:
    chunk_id: str
    raw_text: str
    context_note: str
    document_id: str
    page_number: int
    score: float


@dataclass
class SearchResult:
    chunks: list[RetrievedChunk]
    query_used: str
    self_corrected: bool  # True if we reformulated and retried
    confidence: str  # "high" or "low" -- the judgment on the *final* results


def _vector_search(
    collection: chromadb.Collection, query: str, n_results: int
) -> list[tuple[str, int]]:
    """Returns [(chunk_id, rank)] ranked by vector similarity, best first."""
    result = collection.query(query_texts=[query], n_results=n_results)
    ids = result["ids"][0]
    return [(chunk_id, rank) for rank, chunk_id in enumerate(ids)]


def _bm25_search(
    collection: chromadb.Collection, query: str, n_results: int
) -> list[tuple[str, int]]:
    """Returns [(chunk_id, rank)] ranked by BM25 keyword score, best first."""
    all_chunks = fetch_all_chunks(collection)
    ids = all_chunks["ids"]
    documents = all_chunks["documents"] or []

    if not ids:
        return []

    tokenized_corpus = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(zip(ids, scores, strict=True), key=lambda pair: pair[1], reverse=True)
    return [(chunk_id, rank) for rank, (chunk_id, _score) in enumerate(ranked[:n_results])]


def _reciprocal_rank_fusion(
    *rankings: list[tuple[str, int]], k: int = RRF_K
) -> list[tuple[str, float]]:
    """Merge multiple rankings into one score per id via RRF: score = sum(1 / (k + rank))."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for chunk_id, rank in ranking:
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda pair: pair[1], reverse=True)


def hybrid_search(
    collection: chromadb.Collection, query: str, n_results: int = 5
) -> list[RetrievedChunk]:
    """Vector + BM25 search, merged via RRF, hydrated with metadata."""
    candidate_pool = max(n_results * 3, 10)  # widen the pool before fusing/truncating

    vector_ranking = _vector_search(collection, query, candidate_pool)
    bm25_ranking = _bm25_search(collection, query, candidate_pool)
    fused = _reciprocal_rank_fusion(vector_ranking, bm25_ranking)[:n_results]

    if not fused:
        return []

    top_ids = [chunk_id for chunk_id, _ in fused]
    hydrated = collection.get(ids=top_ids, include=["documents", "metadatas"])
    hydrated_metadatas = hydrated["metadatas"] or []
    meta_by_id = dict(zip(hydrated["ids"], hydrated_metadatas, strict=True))

    results = []
    for chunk_id, score in fused:
        meta = meta_by_id.get(chunk_id)
        if meta is None:
            continue
        results.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                raw_text=str(meta["raw_text"]),
                context_note=str(meta["context_note"]),
                document_id=str(meta["document_id"]),
                page_number=int(meta["page_number"]),  # type: ignore[arg-type]
                score=score,
            )
        )
    return results


def _judge_confidence(
    query: str, chunks: list[RetrievedChunk], client: anthropic.Anthropic
) -> bool:
    """Ask a cheap model whether these chunks look sufficient to answer the query."""
    if not chunks:
        return False

    chunks_preview = "\n\n".join(
        f"[{i + 1}] {c.raw_text[:300]}" for i, c in enumerate(chunks)
    )
    prompt = (
        f"Query: {query}\n\nRetrieved passages:\n{chunks_preview}\n\n"
        "Do these passages contain enough information to answer the query well? "
        "Answer with exactly one word: YES or NO."
    )
    response = client.messages.create(
        model=CONFIDENCE_MODEL,
        max_tokens=5,
        messages=[{"role": "user", "content": prompt}],
    )
    verdict = response_text(response).strip().upper()
    return verdict.startswith("YES")


def _reformulate_query(query: str, client: anthropic.Anthropic) -> str:
    """Ask a cheap model to rewrite a weak query into a more retrievable one."""
    prompt = (
        f"This search query returned weak/insufficient results:\n\"{query}\"\n\n"
        "Rewrite it as a single alternative search query that is more likely to "
        "retrieve relevant passages (e.g. by using different terminology, being "
        "more specific, or breaking down what's really being asked). "
        "Answer only with the rewritten query, nothing else."
    )
    response = client.messages.create(
        model=CONFIDENCE_MODEL,
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}],
    )
    return response_text(response).strip()


def agentic_search(
    collection: chromadb.Collection,
    client: anthropic.Anthropic,
    query: str,
    n_results: int = 5,
) -> SearchResult:
    """Hybrid search with one bounded self-correction retry if confidence is low."""
    chunks = hybrid_search(collection, query, n_results)
    confident = _judge_confidence(query, chunks, client)

    if confident:
        return SearchResult(
            chunks=chunks, query_used=query, self_corrected=False, confidence="high"
        )

    logger.info("Low confidence on query %r; reformulating and retrying once", query)
    reformulated = _reformulate_query(query, client)
    retried_chunks = hybrid_search(collection, reformulated, n_results)
    retried_confident = _judge_confidence(reformulated, retried_chunks, client)

    # Keep whichever attempt looks better; never silently discard the first
    # attempt's results if the retry didn't actually improve things.
    if retried_confident or len(retried_chunks) > len(chunks):
        return SearchResult(
            chunks=retried_chunks,
            query_used=reformulated,
            self_corrected=True,
            confidence="high" if retried_confident else "low",
        )

    return SearchResult(chunks=chunks, query_used=query, self_corrected=True, confidence="low")