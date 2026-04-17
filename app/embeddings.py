"""Embedding generation and in-memory vector store.

Uses Mistral's embedding API (mistral-embed, 1024 dimensions).
Vector store is a simple numpy array — no external DB needed.
Cosine similarity for retrieval.
"""

import os
import time
import numpy as np
import httpx
from app.models import Chunk

MISTRAL_API_URL = "https://api.mistral.ai/v1/embeddings"
EMBEDDING_MODEL = "mistral-embed"
EMBEDDING_DIM = 1024
BATCH_SIZE = 16            # Mistral API batch limit; balances throughput vs. rate-limit risk
MAX_RETRIES = 5            # retries on 429 / 5xx transient errors
INTER_BATCH_SLEEP = 0.25   # gentle throttle: cap sustained rate at ~4 req/s


def get_api_key() -> str:
    key = os.getenv("MISTRAL_API_KEY")
    if not key:
        raise ValueError("MISTRAL_API_KEY not set. Add it to your .env file.")
    return key


def _embed_batch(batch: list[str], api_key: str) -> list[list[float]]:
    """Embed a single batch with exponential-backoff retry on transient errors."""
    for attempt in range(MAX_RETRIES):
        response = httpx.post(
            MISTRAL_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={"model": EMBEDDING_MODEL, "input": batch},
            timeout=30.0,
        )
        if response.status_code in (429, 500, 502, 503, 504):
            if attempt < MAX_RETRIES - 1:
                # Exponential backoff: 1s, 2s, 4s, 8s, 16s
                delay = 2 ** attempt
                # Honor Retry-After if the server provides it
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    try:
                        delay = max(delay, int(retry_after))
                    except ValueError:
                        pass
                time.sleep(delay)
                continue
        response.raise_for_status()
        data = response.json()
        # Sort by API response index — defensive against potential reordering
        return [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]

    raise RuntimeError(f"Mistral embeddings API exceeded {MAX_RETRIES} retries")


def embed_texts(texts: list[str]) -> np.ndarray:
    """Generate embeddings for a list of texts via Mistral API.

    Handles batching and rate limits automatically.
    Returns an (N, 1024) numpy array of embeddings.
    """
    api_key = get_api_key()
    all_embeddings: list[list[float]] = []
    n_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        batch_idx = i // BATCH_SIZE + 1
        all_embeddings.extend(_embed_batch(batch, api_key))
        # Gentle throttle between batches to stay under rate limits.
        # Skip on the last batch so single-batch requests stay fast.
        if batch_idx < n_batches:
            time.sleep(INTER_BATCH_SLEEP)

    return np.array(all_embeddings, dtype=np.float32)


class VectorStore:
    """In-memory vector store using numpy arrays.

    Why not use FAISS/ChromaDB:
    - Challenge requirement: no third-party vector DB
    - For <100k chunks, brute-force cosine similarity is fast enough
    - NumPy gives us the math we need with zero overhead

    Storage layout:
    - self.embeddings: (N, 1024) float32 array
    - self.chunks: list of Chunk objects (parallel to embeddings)
    """

    def __init__(self):
        self.embeddings: np.ndarray | None = None  # (N, 1024)
        self.chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk], embeddings: np.ndarray):
        """Add chunks and their embeddings to the store."""
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[tuple[Chunk, float]]:
        """Find the top-k most similar chunks to a query embedding.

        Uses cosine similarity: sim(a, b) = (a · b) / (||a|| * ||b||)
        Returns list of (Chunk, score) tuples, sorted by descending score.
        """
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        # Normalize query vector
        query = query_embedding.flatten()
        query_norm = query / (np.linalg.norm(query) + 1e-10)

        # Normalize all stored embeddings
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-10
        normed = self.embeddings / norms

        # Cosine similarity via dot product of normalized vectors
        similarities = normed @ query_norm

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [(self.chunks[i], float(similarities[i])) for i in top_indices]

    def remove_file(self, filename: str):
        """Remove all chunks belonging to a specific file."""
        keep_mask = [c.source_file != filename for c in self.chunks]
        self.chunks = [c for c, keep in zip(self.chunks, keep_mask) if keep]
        if self.embeddings is not None and any(keep_mask):
            self.embeddings = self.embeddings[keep_mask]
        elif not any(keep_mask):
            self.embeddings = None
            self.chunks = []

    def get_all_chunks(self) -> list[Chunk]:
        """Return all stored chunks."""
        return self.chunks

    @property
    def size(self) -> int:
        return len(self.chunks)
