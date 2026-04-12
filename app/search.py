"""Search module: BM25 keyword search, semantic search, hybrid fusion, and re-ranking.

Three retrieval strategies combined:
1. Semantic search — cosine similarity on Mistral embeddings (in embeddings.py)
2. BM25 keyword search — term-frequency based scoring (implemented here from scratch)
3. Hybrid fusion — Reciprocal Rank Fusion (RRF) merges both result lists

No external search libraries used — everything is built from Python + math.
"""

import math
import re
from collections import Counter

from app.models import Chunk
from app.embeddings import VectorStore, embed_texts


# ──────────────────────────────────────────────
# BM25 Implementation (from scratch)
# ──────────────────────────────────────────────

# BM25 tuning parameters
BM25_K1 = 1.5   # Term frequency saturation — higher values give more weight to term frequency
BM25_B = 0.75   # Length normalization — 1.0 = full normalization, 0.0 = no normalization

# Simple stopwords to filter out common English words that don't carry meaning
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "about", "between",
    "through", "during", "before", "after", "and", "but", "or", "not",
    "no", "if", "then", "than", "that", "this", "these", "those", "it",
    "its", "i", "me", "my", "we", "our", "you", "your", "he", "she",
    "they", "them", "what", "which", "who", "whom", "how", "when", "where",
}


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase words, filtering stopwords and short tokens."""
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 1]


class BM25Index:
    """BM25 keyword search index built from scratch.

    BM25 scores how relevant a document is to a query based on:
    - Term Frequency (TF): how often query terms appear in the document
    - Inverse Document Frequency (IDF): how rare the term is across all documents
    - Document length normalization: shorter docs with the same term count score higher

    Formula per term: IDF * (TF * (k1 + 1)) / (TF + k1 * (1 - b + b * (dl / avgdl)))
    """

    def __init__(self):
        self.chunks: list[Chunk] = []
        self.doc_tokens: list[list[str]] = []  # tokenized version of each chunk
        self.doc_freqs: dict[str, int] = {}    # how many docs contain each term
        self.avg_doc_len: float = 0.0
        self.n_docs: int = 0

    def add(self, chunks: list[Chunk]):
        """Index a batch of chunks for keyword search."""
        for chunk in chunks:
            tokens = tokenize(chunk.text)
            self.doc_tokens.append(tokens)
            self.chunks.append(chunk)

            # Count document frequency (how many docs contain each unique term)
            unique_terms = set(tokens)
            for term in unique_terms:
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        self.n_docs = len(self.chunks)
        total_tokens = sum(len(t) for t in self.doc_tokens)
        self.avg_doc_len = total_tokens / self.n_docs if self.n_docs > 0 else 0

    def remove_file(self, filename: str):
        """Remove all chunks belonging to a specific file and rebuild index."""
        keep_indices = [i for i, c in enumerate(self.chunks) if c.source_file != filename]
        self.chunks = [self.chunks[i] for i in keep_indices]
        self.doc_tokens = [self.doc_tokens[i] for i in keep_indices]

        # Rebuild document frequencies from scratch
        self.doc_freqs = {}
        for tokens in self.doc_tokens:
            for term in set(tokens):
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        self.n_docs = len(self.chunks)
        total_tokens = sum(len(t) for t in self.doc_tokens)
        self.avg_doc_len = total_tokens / self.n_docs if self.n_docs > 0 else 0

    def _idf(self, term: str) -> float:
        """Inverse Document Frequency with smoothing.

        IDF = log((N - n + 0.5) / (n + 0.5) + 1)
        where N = total docs, n = docs containing the term
        """
        n = self.doc_freqs.get(term, 0)
        return math.log((self.n_docs - n + 0.5) / (n + 0.5) + 1)

    def search(self, query: str, top_k: int = 10) -> list[tuple[Chunk, float]]:
        """Score all documents against the query using BM25.

        Returns top-k chunks sorted by descending BM25 score.
        """
        if not self.chunks:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = []
        for idx, doc_tokens in enumerate(self.doc_tokens):
            score = 0.0
            doc_len = len(doc_tokens)
            tf_counter = Counter(doc_tokens)

            for term in query_tokens:
                if term not in tf_counter:
                    continue

                tf = tf_counter[term]
                idf = self._idf(term)

                # BM25 formula
                numerator = tf * (BM25_K1 + 1)
                denominator = tf + BM25_K1 * (1 - BM25_B + BM25_B * (doc_len / self.avg_doc_len))
                score += idf * (numerator / denominator)

            scores.append((self.chunks[idx], score))

        # Sort by score descending, return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ──────────────────────────────────────────────
# Hybrid Search: Reciprocal Rank Fusion (RRF)
# ──────────────────────────────────────────────

RRF_K = 60  # RRF smoothing constant (standard value from the original paper)


def reciprocal_rank_fusion(
    semantic_results: list[tuple[Chunk, float]],
    keyword_results: list[tuple[Chunk, float]],
    top_k: int = 10,
) -> list[tuple[Chunk, float]]:
    """Merge semantic and keyword results using Reciprocal Rank Fusion.

    RRF score = sum over lists of 1 / (k + rank)

    Why RRF over simple score averaging:
    - Scores from different systems aren't on the same scale (cosine sim vs BM25)
    - RRF only uses rank positions, making it scale-invariant
    - Well-established in IR literature, simple to implement
    """
    # Build a map of chunk_index -> RRF score
    # Use (source_file, chunk_index) as a unique key for each chunk
    rrf_scores: dict[tuple[str, int], float] = {}
    chunk_map: dict[tuple[str, int], Chunk] = {}

    for rank, (chunk, _score) in enumerate(semantic_results):
        key = (chunk.source_file, chunk.chunk_index)
        rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (RRF_K + rank + 1)
        chunk_map[key] = chunk

    for rank, (chunk, _score) in enumerate(keyword_results):
        key = (chunk.source_file, chunk.chunk_index)
        rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (RRF_K + rank + 1)
        chunk_map[key] = chunk

    # Sort by RRF score and return top-k
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [(chunk_map[key], score) for key, score in sorted_results[:top_k]]


def hybrid_search(
    query: str,
    vector_store: VectorStore,
    bm25_index: BM25Index,
    top_k: int = 5,
) -> list[tuple[Chunk, float]]:
    """Run both semantic and keyword search, fuse with RRF.

    Pipeline:
    1. Embed the query → semantic search (cosine similarity)
    2. Tokenize the query → BM25 keyword search
    3. Merge with Reciprocal Rank Fusion
    """
    # Semantic search
    query_embedding = embed_texts([query])
    semantic_results = vector_store.search(query_embedding[0], top_k=top_k * 2)

    # Keyword search
    keyword_results = bm25_index.search(query, top_k=top_k * 2)

    # Fuse results
    fused = reciprocal_rank_fusion(semantic_results, keyword_results, top_k=top_k)

    return fused
