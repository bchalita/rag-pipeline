"""PDF text extraction and chunking.

Design decisions:
- PyMuPDF (fitz) for extraction: fast, handles most PDF layouts well
- Character-based chunking with overlap: simple, predictable chunk sizes
- Overlap ensures we don't lose context at chunk boundaries
- Each chunk carries metadata (file, page, index) for citation tracing
"""

import fitz  # PyMuPDF
from app.models import Chunk

# Chunking parameters — tuned for RAG retrieval
CHUNK_SIZE = 512       # chars per chunk (roughly ~100 tokens)
CHUNK_OVERLAP = 100    # chars overlap between consecutive chunks


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """Extract text from a PDF file, page by page.

    Returns a list of dicts with 'page_number' and 'text' keys.
    Pages with no extractable text (e.g., scanned images) are skipped.
    """
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        text = doc[page_num].get_text()
        # Skip empty pages (scanned docs without OCR)
        if text.strip():
            pages.append({"page_number": page_num + 1, "text": text})
    doc.close()
    return pages


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks using a sliding window.

    Why sliding window over sentence/paragraph splitting:
    - Predictable chunk sizes → consistent embedding quality
    - Overlap preserves context across boundaries
    - Simpler and more robust than regex-based sentence splitting
    """
    if not text.strip():
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        # Only add non-trivial chunks
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def process_pdf(pdf_path: str, filename: str) -> list[Chunk]:
    """Full pipeline: extract text from PDF → chunk → return Chunk objects.

    Each chunk carries its source metadata for downstream citation.
    """
    pages = extract_text_from_pdf(pdf_path)
    chunks = []
    chunk_index = 0

    for page in pages:
        page_chunks = chunk_text(page["text"])
        for text in page_chunks:
            chunks.append(Chunk(
                text=text,
                source_file=filename,
                page_number=page["page_number"],
                chunk_index=chunk_index,
            ))
            chunk_index += 1

    return chunks
