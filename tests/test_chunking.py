"""Unit tests for chunking. No API calls -- pure logic."""

from ai_document_mcp.chunking import CHUNK_OVERLAP_CHARS, CHUNK_SIZE_CHARS, chunk_pages


def test_chunk_pages_short_text_single_chunk():
    """A page shorter than CHUNK_SIZE_CHARS should produce exactly one chunk."""
    pages = [(1, "This is a short page of text.")]
    chunks = chunk_pages("doc1", pages)

    assert len(chunks) == 1
    assert chunks[0] == (1, "This is a short page of text.")


def test_chunk_pages_long_text_splits_with_overlap():
    """A page longer than CHUNK_SIZE_CHARS should split into multiple overlapping chunks."""
    long_text = "word " * 500  # comfortably longer than CHUNK_SIZE_CHARS
    pages = [(1, long_text)]
    chunks = chunk_pages("doc1", pages)

    assert len(chunks) > 1
    # every chunk should be tagged with the correct source page
    assert all(page_number == 1 for page_number, _ in chunks)
    # consecutive chunks should overlap: the tail of one appears near the head of the next
    first_chunk_tail = chunks[0][1][-CHUNK_OVERLAP_CHARS:]
    assert first_chunk_tail[:20] in chunks[1][1]


def test_chunk_pages_preserves_page_numbers_across_multiple_pages():
    """Chunks from different pages should keep their correct page_number tag."""
    pages = [(1, "Content from page one."), (2, "Content from page two.")]
    chunks = chunk_pages("doc1", pages)

    page_numbers = [page_number for page_number, _ in chunks]
    assert 1 in page_numbers
    assert 2 in page_numbers


def test_chunk_pages_skips_blank_pages():
    """A page with only whitespace should produce no chunks."""
    pages = [(1, "   \n\t  "), (2, "Real content here.")]
    chunks = chunk_pages("doc1", pages)

    assert len(chunks) == 1
    assert chunks[0][0] == 2


def test_chunk_pages_empty_input():
    """No pages in, no chunks out -- should not raise."""
    assert chunk_pages("doc1", []) == []