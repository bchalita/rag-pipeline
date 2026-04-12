"""Tests for the ingestion pipeline (extraction + chunking).

Run with: python -m tests.test_ingestion
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ingestion import chunk_text, extract_text_from_pdf, process_pdf


def test_chunk_text_basic():
    """Chunks should respect size and overlap parameters."""
    text = "A" * 1000
    chunks = chunk_text(text, chunk_size=512, overlap=100)

    print(f"  Input length: {len(text)}")
    print(f"  Number of chunks: {len(chunks)}")
    print(f"  Chunk lengths: {[len(c) for c in chunks]}")

    assert len(chunks) >= 2, "Should produce at least 2 chunks for 1000 chars"
    assert all(len(c) <= 512 for c in chunks), "No chunk should exceed chunk_size"
    print("  PASSED")


def test_chunk_text_short():
    """Short text should return a single chunk."""
    text = "Short text"
    chunks = chunk_text(text, chunk_size=512, overlap=100)

    assert len(chunks) == 1, "Short text should produce exactly 1 chunk"
    assert chunks[0] == text
    print("  PASSED")


def test_chunk_text_overlap():
    """Consecutive chunks should share overlapping characters."""
    text = "ABCDEFGHIJ" * 100  # 1000 chars
    chunks = chunk_text(text, chunk_size=512, overlap=100)

    # The end of chunk[0] should match the start of chunk[1]
    overlap_end = chunks[0][-100:]
    overlap_start = chunks[1][:100]
    assert overlap_end == overlap_start, f"Overlap mismatch: '{overlap_end}' != '{overlap_start}'"
    print(f"  Overlap verified: last 100 chars of chunk 0 == first 100 chars of chunk 1")
    print("  PASSED")


def test_chunk_text_empty():
    """Empty or whitespace-only text should return empty list."""
    chunks = chunk_text("   ", chunk_size=512, overlap=100)
    assert len(chunks) == 0, "Whitespace-only text should produce 0 chunks"
    print("  PASSED")


def test_extract_pdf():
    """Test PDF extraction on a real file (if available)."""
    test_pdf = "tests/test_sample.pdf"
    if not os.path.exists(test_pdf):
        print("  SKIPPED — no test PDF available. Place a PDF at tests/test_sample.pdf to test.")
        return

    pages = extract_text_from_pdf(test_pdf)
    print(f"  Extracted {len(pages)} pages")
    for p in pages[:2]:
        preview = p["text"][:80].replace("\n", " ")
        print(f"    Page {p['page_number']}: '{preview}...'")
    assert len(pages) > 0, "Should extract at least one page"
    print("  PASSED")


if __name__ == "__main__":
    tests = [
        ("chunk_text — basic", test_chunk_text_basic),
        ("chunk_text — short text", test_chunk_text_short),
        ("chunk_text — overlap", test_chunk_text_overlap),
        ("chunk_text — empty", test_chunk_text_empty),
        ("extract_pdf — sample", test_extract_pdf),
    ]

    print("=" * 60)
    print("INGESTION PIPELINE TESTS")
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
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
