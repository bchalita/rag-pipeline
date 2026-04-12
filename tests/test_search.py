"""Tests for BM25 keyword search and hybrid fusion.

Run with: python -m tests.test_search
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models import Chunk
from app.search import tokenize, BM25Index, reciprocal_rank_fusion


def make_chunk(text: str, source: str = "test.pdf", page: int = 1, idx: int = 0) -> Chunk:
    return Chunk(text=text, source_file=source, page_number=page, chunk_index=idx)


def test_tokenize():
    """Tokenizer should lowercase, remove stopwords, and filter short tokens."""
    tokens = tokenize("The quick brown fox jumps over the lazy dog")
    print(f"  Input: 'The quick brown fox jumps over the lazy dog'")
    print(f"  Tokens: {tokens}")
    assert "the" not in tokens, "Stopwords should be removed"
    assert "is" not in tokens, "Stopwords should be removed"
    assert all(t == t.lower() for t in tokens), "Should be lowercase"
    assert all(len(t) > 1 for t in tokens), "Single-char tokens should be filtered"
    print("  PASSED")


def test_bm25_basic():
    """BM25 should rank documents containing query terms higher."""
    index = BM25Index()
    chunks = [
        make_chunk("Python is a programming language used for machine learning", idx=0),
        make_chunk("Java is popular for enterprise backend development", idx=1),
        make_chunk("Python machine learning frameworks include PyTorch and TensorFlow", idx=2),
    ]
    index.add(chunks)

    results = index.search("Python machine learning", top_k=3)
    print(f"  Query: 'Python machine learning'")
    for rank, (chunk, score) in enumerate(results):
        print(f"    Rank {rank+1}: score={score:.3f} — '{chunk.text[:60]}...'")

    # Chunks 0 and 2 mention Python + ML, should rank above chunk 1
    top_texts = [r[0].text for r in results[:2]]
    assert any("Python" in t for t in top_texts), "Python docs should rank high"
    print("  PASSED")


def test_bm25_empty_query():
    """Empty query should return no results."""
    index = BM25Index()
    index.add([make_chunk("Some text here", idx=0)])
    results = index.search("", top_k=5)
    assert len(results) == 0, "Empty query should return empty results"
    print("  PASSED")


def test_bm25_no_match():
    """Query with no matching terms should score zero."""
    index = BM25Index()
    index.add([make_chunk("Python programming language", idx=0)])
    results = index.search("quantum physics experiments", top_k=5)
    # All scores should be 0 (no matching terms)
    for chunk, score in results:
        assert score == 0.0, f"Non-matching term scored {score}"
    print("  PASSED")


def test_bm25_remove_file():
    """Removing a file should drop its chunks from the index."""
    index = BM25Index()
    index.add([
        make_chunk("Alpha text", source="a.pdf", idx=0),
        make_chunk("Beta text", source="b.pdf", idx=1),
    ])
    assert index.n_docs == 2
    index.remove_file("a.pdf")
    assert index.n_docs == 1
    assert index.chunks[0].source_file == "b.pdf"
    print("  PASSED")


def test_rrf_fusion():
    """RRF should merge two ranked lists and boost chunks that appear in both."""
    # Simulate semantic results: chunk A > B > C
    sem = [
        (make_chunk("A", idx=0), 0.9),
        (make_chunk("B", idx=1), 0.7),
        (make_chunk("C", idx=2), 0.5),
    ]
    # Simulate keyword results: chunk B > C > D
    kw = [
        (make_chunk("B", idx=1), 5.0),
        (make_chunk("C", idx=2), 3.0),
        (make_chunk("D", idx=3), 1.0),
    ]

    fused = reciprocal_rank_fusion(sem, kw, top_k=4)
    fused_indices = [chunk.chunk_index for chunk, _ in fused]
    print(f"  Fused ranking (by chunk_index): {fused_indices}")
    for chunk, score in fused:
        print(f"    Chunk {chunk.chunk_index}: RRF score = {score:.6f}")

    # B appears in both lists → should rank first
    assert fused_indices[0] == 1, "Chunk B (idx=1) should rank first (appears in both lists)"
    print("  PASSED")


def test_bm25_idf_weighting():
    """Rare terms should get higher IDF weight than common terms."""
    index = BM25Index()
    index.add([
        make_chunk("machine learning deep neural networks", idx=0),
        make_chunk("machine learning algorithms classification", idx=1),
        make_chunk("quantum computing research experiments", idx=2),
    ])

    # "quantum" appears in 1/3 docs, "machine" in 2/3 — quantum should have higher IDF
    idf_quantum = index._idf("quantum")
    idf_machine = index._idf("machine")
    print(f"  IDF('quantum') = {idf_quantum:.4f} (appears in 1/3 docs)")
    print(f"  IDF('machine') = {idf_machine:.4f} (appears in 2/3 docs)")
    assert idf_quantum > idf_machine, "Rarer terms should have higher IDF"
    print("  PASSED")


if __name__ == "__main__":
    tests = [
        ("tokenize", test_tokenize),
        ("BM25 — basic ranking", test_bm25_basic),
        ("BM25 — empty query", test_bm25_empty_query),
        ("BM25 — no match", test_bm25_no_match),
        ("BM25 — remove file", test_bm25_remove_file),
        ("BM25 — IDF weighting", test_bm25_idf_weighting),
        ("RRF fusion", test_rrf_fusion),
    ]

    print("=" * 60)
    print("SEARCH MODULE TESTS")
    print("=" * 60)

    passed = 0
    failed = 0
    for name, test_fn in tests:
        print(f"\n[TEST] {name}")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
