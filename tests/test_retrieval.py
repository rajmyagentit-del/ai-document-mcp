"""Unit tests for retrieval logic. No API calls, no Chroma Cloud -- pure logic."""

from ai_document_mcp.retrieval import _reciprocal_rank_fusion


def test_rrf_agreement_ranks_higher_than_single_source():
    """A chunk ranked well in BOTH rankings should outscore one ranked well in only one."""
    vector_ranking = [("chunk_a", 0), ("chunk_b", 1)]
    bm25_ranking = [("chunk_a", 0), ("chunk_c", 1)]

    fused = _reciprocal_rank_fusion(vector_ranking, bm25_ranking)
    fused_ids = [chunk_id for chunk_id, _ in fused]

    # chunk_a appears top in both rankings -> should be first after fusion
    assert fused_ids[0] == "chunk_a"


def test_rrf_includes_ids_present_in_only_one_ranking():
    """Fusion shouldn't drop candidates that only one retrieval method found."""
    vector_ranking = [("chunk_a", 0)]
    bm25_ranking = [("chunk_b", 0)]

    fused = _reciprocal_rank_fusion(vector_ranking, bm25_ranking)
    fused_ids = {chunk_id for chunk_id, _ in fused}

    assert fused_ids == {"chunk_a", "chunk_b"}


def test_rrf_empty_rankings_returns_empty():
    """No candidates from either source -> no fused results, no crash."""
    assert _reciprocal_rank_fusion([], []) == []


def test_rrf_better_rank_yields_higher_score():
    """Within a single ranking, an earlier rank should score higher than a later one."""
    ranking = [("chunk_a", 0), ("chunk_b", 5)]

    fused = _reciprocal_rank_fusion(ranking)
    scores = dict(fused)

    assert scores["chunk_a"] > scores["chunk_b"]