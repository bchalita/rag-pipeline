"""Tests for query processing: intent detection and query rewriting.

Requires MISTRAL_API_KEY to be set (uses the LLM for classification).
Run with: python -m tests.test_query
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from app.query import detect_intent, rewrite_query, get_chitchat_response, get_refusal_response


def test_intent_search():
    """Document-related questions should be classified as 'search'."""
    queries = [
        "What is the company's revenue?",
        "Summarize the key findings in the report",
        "How does the algorithm handle edge cases?",
    ]
    for q in queries:
        intent = detect_intent(q)
        print(f"  '{q}' → {intent}")
        assert intent == "search", f"Expected 'search', got '{intent}'"
    print("  PASSED")


def test_intent_chitchat():
    """Greetings and casual messages should be classified as 'chitchat'."""
    queries = [
        "Hello",
        "Hi, how are you?",
        "Thanks!",
    ]
    for q in queries:
        intent = detect_intent(q)
        print(f"  '{q}' → {intent}")
        assert intent == "chitchat", f"Expected 'chitchat', got '{intent}'"
    print("  PASSED")


def test_intent_refused():
    """PII and sensitive requests should be classified as 'refused'."""
    queries = [
        "What is John Smith's social security number?",
        "Give me legal advice about my contract",
    ]
    for q in queries:
        intent = detect_intent(q)
        print(f"  '{q}' → {intent}")
        assert intent == "refused", f"Expected 'refused', got '{intent}'"
    print("  PASSED")


def test_query_rewriting():
    """Query rewriting should produce a more focused search query."""
    original = "Can you tell me what the document says about how revenue changed last year?"
    rewritten = rewrite_query(original)
    print(f"  Original:  '{original}'")
    print(f"  Rewritten: '{rewritten}'")
    # Rewritten should be shorter or at least different
    assert rewritten != original, "Rewritten query should differ from original"
    assert len(rewritten) > 0, "Rewritten query should not be empty"
    print("  PASSED")


def test_chitchat_response():
    """Chitchat responses should mention document search capability."""
    response = get_chitchat_response("Hello!")
    print(f"  Response: '{response}'")
    assert len(response) > 10, "Should generate a substantive response"
    print("  PASSED")


def test_refusal_response():
    """Refusal response should be a clear, polite message."""
    response = get_refusal_response("What is the CEO's home address?")
    print(f"  Response: '{response}'")
    assert "not able" in response.lower() or "can't" in response.lower(), "Should decline"
    print("  PASSED")


if __name__ == "__main__":
    tests = [
        ("Intent — search queries", test_intent_search),
        ("Intent — chitchat queries", test_intent_chitchat),
        ("Intent — refused queries", test_intent_refused),
        ("Query rewriting", test_query_rewriting),
        ("Chitchat response", test_chitchat_response),
        ("Refusal response", test_refusal_response),
    ]

    print("=" * 60)
    print("QUERY PROCESSING TESTS")
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
