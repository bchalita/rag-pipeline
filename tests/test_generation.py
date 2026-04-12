"""Tests for the generation module: citations, answer shaping, hallucination filter.

Run with: python -m tests.test_generation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from app.models import Chunk, Citation
from app.generation import (
    _format_context,
    _detect_query_type,
    _extract_citations,
    generate_answer,
    SIMILARITY_THRESHOLD,
)


def make_chunk(text: str, source: str = "report.pdf", page: int = 1, idx: int = 0) -> Chunk:
    return Chunk(text=text, source_file=source, page_number=page, chunk_index=idx)


def test_format_context():
    """Context formatting should label each chunk with its source."""
    chunks = [
        (make_chunk("Revenue grew 20% year over year.", page=3), 0.85),
        (make_chunk("Operating costs decreased by 5%.", page=7), 0.72),
    ]
    context = _format_context(chunks)
    print(f"  Context preview:\n{context[:200]}")
    assert "[Source 1: report.pdf, p.3]" in context
    assert "[Source 2: report.pdf, p.7]" in context
    assert "Revenue grew" in context
    print("  PASSED")


def test_detect_query_type():
    """Query type detection should categorize by keywords."""
    assert _detect_query_type("List the main benefits") == "list"
    assert _detect_query_type("Compare product A and product B") == "compare"
    assert _detect_query_type("What is the revenue model?") == "explain"
    assert _detect_query_type("What are the key features?") == "list"
    assert _detect_query_type("How do X and Y differ?") == "compare"
    print("  All query types detected correctly")
    print("  PASSED")


def test_extract_citations():
    """Citation extraction should match [Source N] references to chunks."""
    chunks = [
        (make_chunk("Revenue info", page=1, idx=0), 0.9),
        (make_chunk("Cost info", page=2, idx=1), 0.8),
        (make_chunk("Growth info", page=3, idx=2), 0.7),
    ]
    answer = "Revenue grew 20% [Source 1] and costs dropped [Source 2]. Growth was steady."

    citations = _extract_citations(answer, chunks)
    print(f"  Answer: '{answer}'")
    print(f"  Citations found: {len(citations)}")
    for c in citations:
        print(f"    - {c.source_file} p.{c.page_number} (score={c.relevance_score})")

    assert len(citations) == 2, f"Expected 2 citations, got {len(citations)}"
    assert citations[0].page_number == 1
    assert citations[1].page_number == 2
    print("  PASSED")


def test_insufficient_evidence():
    """Low-similarity chunks should trigger insufficient evidence response."""
    chunks = [
        (make_chunk("Unrelated text about cooking recipes"), SIMILARITY_THRESHOLD - 0.1),
    ]
    answer, citations = generate_answer("What is the company's revenue?", chunks)
    print(f"  Answer: '{answer}'")
    assert "insufficient" in answer.lower() or "couldn't find" in answer.lower()
    assert len(citations) == 0
    print("  PASSED")


def test_no_documents():
    """Empty chunk list should inform user to upload documents."""
    answer, citations = generate_answer("What is this about?", [])
    print(f"  Answer: '{answer}'")
    assert "upload" in answer.lower()
    print("  PASSED")


def test_generate_with_real_chunks():
    """End-to-end generation with realistic chunks (requires API)."""
    chunks = [
        (make_chunk(
            "StackAI is a no-code platform for building AI workflows. "
            "It allows users to connect LLMs, knowledge bases, and APIs "
            "through a visual drag-and-drop interface.",
            source="stackai_overview.pdf", page=1, idx=0
        ), 0.88),
        (make_chunk(
            "StackAI supports multiple LLM providers including OpenAI, "
            "Anthropic, and Mistral. Users can switch between models "
            "without changing their workflow configuration.",
            source="stackai_overview.pdf", page=2, idx=1
        ), 0.82),
    ]

    answer, citations = generate_answer("What is StackAI and what LLMs does it support?", chunks)
    print(f"  Answer: '{answer[:200]}...'")
    print(f"  Citations: {len(citations)}")
    for c in citations:
        print(f"    - {c.source_file} p.{c.page_number}")

    assert len(answer) > 50, "Answer should be substantive"
    print("  PASSED")


if __name__ == "__main__":
    tests = [
        ("Format context", test_format_context),
        ("Detect query type", test_detect_query_type),
        ("Extract citations", test_extract_citations),
        ("Insufficient evidence", test_insufficient_evidence),
        ("No documents", test_no_documents),
        ("End-to-end generation", test_generate_with_real_chunks),
    ]

    print("=" * 60)
    print("GENERATION MODULE TESTS")
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
