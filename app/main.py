"""FastAPI application — RAG pipeline over PDF documents.

Endpoints:
- POST /ingest    → Upload and process PDF files
- POST /query     → Ask questions over ingested documents
- GET  /files     → List ingested files
- DELETE /files/{filename} → Remove an ingested file
"""

import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.models import IngestResponse, QueryRequest, QueryResponse, FileInfo, Citation
from app.ingestion import process_pdf
from app.embeddings import VectorStore, embed_texts
from app.search import BM25Index, hybrid_search
from app.query import detect_intent, rewrite_query, get_chitchat_response, get_refusal_response
from app.generation import generate_answer

load_dotenv()

app = FastAPI(title="RAG Pipeline", version="0.1.0")

# CORS — allow the frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Global stores — persist in memory for the lifetime of the server
vector_store = VectorStore()
bm25_index = BM25Index()

# Track ingested files and their metadata
ingested_files: dict[str, FileInfo] = {}


@app.post("/ingest", response_model=IngestResponse)
async def ingest_pdfs(files: list[UploadFile] = File(...)):
    """Upload one or more PDF files for ingestion into the knowledge base.

    Pipeline: Upload → Extract text → Chunk → Embed → Store in both vector + BM25 index
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    processed_files = []
    total_chunks = 0

    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"'{file.filename}' is not a PDF file."
            )

        # Save uploaded file to disk
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Extract text and chunk
        chunks = process_pdf(str(file_path), file.filename)
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail=f"No text could be extracted from '{file.filename}'. It may be a scanned PDF."
            )

        # Generate embeddings and store in vector store
        texts = [chunk.text for chunk in chunks]
        embeddings = embed_texts(texts)
        vector_store.add(chunks, embeddings)

        # Also index in BM25 for keyword search
        bm25_index.add(chunks)

        # Track file metadata
        page_numbers = set(c.page_number for c in chunks)
        ingested_files[file.filename] = FileInfo(
            filename=file.filename,
            num_chunks=len(chunks),
            num_pages=len(page_numbers),
        )

        processed_files.append(file.filename)
        total_chunks += len(chunks)

    return IngestResponse(
        files_ingested=processed_files,
        total_chunks=total_chunks,
        message=f"Successfully ingested {len(processed_files)} file(s) with {total_chunks} chunks.",
    )


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the knowledge base with a natural language question.

    Pipeline:
    1. Detect intent (search / chitchat / refused)
    2. If search: rewrite query → hybrid search → generate answer with citations
    3. If chitchat: respond conversationally
    4. If refused: polite refusal
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Step 1: Intent detection
    intent = detect_intent(question)

    # Step 2: Handle non-search intents
    if intent == "chitchat":
        return QueryResponse(
            answer=get_chitchat_response(question),
            citations=[],
            intent=intent,
        )

    if intent == "refused":
        return QueryResponse(
            answer=get_refusal_response(question),
            citations=[],
            intent=intent,
        )

    # Step 3: Search intent — run the full RAG pipeline
    rewritten = rewrite_query(question)

    # Hybrid search (semantic + BM25 keyword)
    results = hybrid_search(rewritten, vector_store, bm25_index, top_k=5)

    # Generate answer with citations
    answer, citations = generate_answer(question, results)

    return QueryResponse(
        answer=answer,
        citations=citations,
        intent=intent,
        query_rewritten=rewritten,
    )


@app.get("/files", response_model=list[FileInfo])
async def list_files():
    """List all ingested files with their metadata."""
    return list(ingested_files.values())


@app.delete("/files/{filename}")
async def delete_file(filename: str):
    """Remove an ingested file and its chunks from the knowledge base."""
    if filename not in ingested_files:
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found.")

    vector_store.remove_file(filename)
    bm25_index.remove_file(filename)
    del ingested_files[filename]

    # Remove from disk
    file_path = UPLOAD_DIR / filename
    if file_path.exists():
        file_path.unlink()

    return {"message": f"'{filename}' removed from knowledge base."}


# Serve the chat UI
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/index.html")
