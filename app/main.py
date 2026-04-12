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
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.models import IngestResponse, FileInfo
from app.ingestion import process_pdf
from app.embeddings import VectorStore, embed_texts

load_dotenv()

app = FastAPI(title="RAG Pipeline", version="0.1.0")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Global vector store — persists in memory for the lifetime of the server
vector_store = VectorStore()

# Track ingested files and their metadata
ingested_files: dict[str, FileInfo] = {}


@app.post("/ingest", response_model=IngestResponse)
async def ingest_pdfs(files: list[UploadFile] = File(...)):
    """Upload one or more PDF files for ingestion into the knowledge base.

    Pipeline: Upload → Extract text → Chunk → Embed → Store
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    processed_files = []
    total_chunks = 0

    for file in files:
        # Validate file type
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

        # Generate embeddings for all chunks
        texts = [chunk.text for chunk in chunks]
        embeddings = embed_texts(texts)

        # Store in vector store
        vector_store.add(chunks, embeddings)

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
