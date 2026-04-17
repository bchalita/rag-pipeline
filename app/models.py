"""Pydantic schemas for request/response validation."""

from pydantic import BaseModel


class Chunk(BaseModel):
    """A text chunk extracted from a PDF document."""
    text: str
    source_file: str
    page_number: int
    chunk_index: int


class QueryRequest(BaseModel):
    """User query to the RAG system."""
    question: str
    history: list[dict] | None = None  # [{role: "user"|"assistant", content: "..."}]


class Citation(BaseModel):
    """A source reference supporting part of the answer."""
    source_file: str
    page_number: int
    text_excerpt: str
    relevance_score: float


class QueryResponse(BaseModel):
    """Response from the RAG system."""
    answer: str
    citations: list[Citation]
    intent: str  # "search", "chitchat", or "refused"
    query_rewritten: str | None = None


class IngestResponse(BaseModel):
    """Response after ingesting PDF files."""
    files_ingested: list[str]
    total_chunks: int
    message: str


class FileInfo(BaseModel):
    """Metadata about an ingested file."""
    filename: str
    num_chunks: int
    num_pages: int
