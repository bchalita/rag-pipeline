"""Integration tests for the FastAPI endpoints.

Tests the full pipeline: ingest → query → delete.
Run with: python -m tests.test_api
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

TEST_PDF = "tests/test_sample.pdf"


def test_root():
    """Root should serve the HTML page."""
    response = client.get("/")
    assert response.status_code == 200
    assert "html" in response.text.lower()
    print("  PASSED")


def test_list_files_empty():
    """Should return empty list initially."""
    response = client.get("/files")
    assert response.status_code == 200
    assert response.json() == []
    print("  PASSED")


def test_ingest_pdf():
    """Should ingest a PDF and return chunk count."""
    if not os.path.exists(TEST_PDF):
        print("  SKIPPED — no test PDF at tests/test_sample.pdf")
        return

    with open(TEST_PDF, "rb") as f:
        response = client.post("/ingest", files={"files": ("test.pdf", f, "application/pdf")})

    print(f"  Status: {response.status_code}")
    data = response.json()
    print(f"  Response: {data}")
    assert response.status_code == 200
    assert data["total_chunks"] > 0
    assert "test.pdf" in data["files_ingested"]
    print("  PASSED")


def test_ingest_invalid_file():
    """Should reject non-PDF files."""
    response = client.post(
        "/ingest",
        files={"files": ("readme.txt", b"some text", "text/plain")},
    )
    assert response.status_code == 400
    assert "not a PDF" in response.json()["detail"]
    print("  PASSED")


def test_list_files_after_ingest():
    """Should show ingested files."""
    response = client.get("/files")
    data = response.json()
    print(f"  Files: {data}")
    if not os.path.exists(TEST_PDF):
        print("  SKIPPED — no test PDF")
        return
    assert len(data) > 0
    print("  PASSED")


def test_query_chitchat():
    """Chitchat should not trigger search."""
    response = client.post("/query", json={"question": "Hello!"})
    data = response.json()
    print(f"  Intent: {data['intent']}")
    print(f"  Answer: '{data['answer'][:100]}'")
    assert data["intent"] == "chitchat"
    assert len(data["citations"]) == 0
    print("  PASSED")


def test_query_search():
    """Search query should return answer with citations."""
    if not os.path.exists(TEST_PDF):
        print("  SKIPPED — no test PDF")
        return

    response = client.post("/query", json={"question": "What is this document about?"})
    data = response.json()
    print(f"  Intent: {data['intent']}")
    print(f"  Rewritten: '{data.get('query_rewritten', 'N/A')}'")
    print(f"  Answer: '{data['answer'][:200]}...'")
    print(f"  Citations: {len(data['citations'])}")
    assert data["intent"] == "search"
    assert len(data["answer"]) > 20
    print("  PASSED")


def test_query_refused():
    """PII queries should be refused."""
    response = client.post(
        "/query",
        json={"question": "What is the CEO's social security number?"},
    )
    data = response.json()
    print(f"  Intent: {data['intent']}")
    print(f"  Answer: '{data['answer'][:100]}'")
    assert data["intent"] == "refused"
    print("  PASSED")


def test_delete_file():
    """Should delete an ingested file."""
    if not os.path.exists(TEST_PDF):
        print("  SKIPPED — no test PDF")
        return

    response = client.delete("/files/test.pdf")
    print(f"  Response: {response.json()}")
    assert response.status_code == 200

    # Verify it's gone
    files = client.get("/files").json()
    assert not any(f["filename"] == "test.pdf" for f in files)
    print("  PASSED")


if __name__ == "__main__":
    tests = [
        ("GET /", test_root),
        ("GET /files — empty", test_list_files_empty),
        ("POST /ingest — valid PDF", test_ingest_pdf),
        ("POST /ingest — invalid file", test_ingest_invalid_file),
        ("GET /files — after ingest", test_list_files_after_ingest),
        ("POST /query — chitchat", test_query_chitchat),
        ("POST /query — search", test_query_search),
        ("POST /query — refused", test_query_refused),
        ("DELETE /files/{name}", test_delete_file),
    ]

    print("=" * 60)
    print("API INTEGRATION TESTS")
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
