# RAG Pipeline — Document Q&A System

A from-scratch Retrieval-Augmented Generation pipeline that lets users upload PDF documents and ask questions about them with cited, grounded answers.

Built with **FastAPI**, **Mistral AI**, and **zero external RAG/search libraries**.

---

## Results

Evaluated with a behavioral harness of **23 queries across 9 categories**
(grounded retrieval, out-of-domain refusal, chitchat, adversarial prompt
injection, malformed input, etc.). **23/23 pass.**

The harness deliberately does **not** string-match answer text. Instead it checks
*observable, gameable-resistant* properties: correct intent routing (search vs.
refuse vs. chitchat), citation count on grounded answers, and substring
presence/absence for injection and refusal cases. The goal is to catch the
failures that actually matter in a RAG system — hallucinated answers, leaked
prompts, and answering questions it should refuse — rather than to reward
surface-level text overlap.

| Behavior | Result |
|---|---|
| Grounded queries return cited answers | pass |
| Out-of-domain questions refused | pass |
| Adversarial prompt-injection resisted | pass |
| Malformed requests rejected (HTTP 400) | pass |

**Latency (observed):** chitchat / refusals 0.4–1.5s · OOD refusal ~2s · grounded retrieval 2.5–6s.
**Corpus:** 2 documents → ~1,758 chunks, ~2 min ingestion, ~$0.02 in embeddings.

See [`EVALUATION.md`](EVALUATION.md) for the full methodology and per-query scorecard.

---

## System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                         CHAT UI                                 │
│              (Upload PDFs, Ask Questions)                        │
└────────────────┬──────────────────────┬─────────────────────────┘
                 │                      │
          POST /ingest            POST /query
                 │                      │
                 ▼                      ▼
┌────────────────────────┐   ┌──────────────────────┐
│    INGESTION PIPELINE  │   │   QUERY PROCESSING   │
│                        │   │                      │
│  PDF → Extract → Chunk │   │  Intent Detection    │
│          │             │   │  (search/chitchat/   │
│          ▼             │   │   refused)           │
│  Embed via Mistral API │   │                      │
│          │             │   │  Query Rewriting     │
│          ▼             │   │  (optimize for       │
│  ┌──────────────────┐  │   │   retrieval)         │
│  │  Vector Store     │  │   └──────────┬───────────┘
│  │  (numpy, 1024d)   │  │              │
│  ├──────────────────┤  │              ▼
│  │  BM25 Index       │  │   ┌──────────────────────┐
│  │  (from scratch)   │  │   │   HYBRID SEARCH      │
│  └──────────────────┘  │   │                      │
└────────────────────────┘   │  Semantic: cosine     │
                              │  similarity on        │
                              │  embeddings           │
                              │                      │
                              │  Keyword: BM25 on     │
                              │  tokenized text       │
                              │                      │
                              │  Fusion: Reciprocal   │
                              │  Rank Fusion (RRF)    │
                              └──────────┬───────────┘
                                         │
                                         ▼
                              ┌──────────────────────┐
                              │    GENERATION        │
                              │                      │
                              │  Answer Shaping      │
                              │  (explain/list/      │
                              │   compare)           │
                              │                      │
                              │  Citation Extraction │
                              │  [Source N] → chunk  │
                              │                      │
                              │  Hallucination       │
                              │  Filter (LLM judge)  │
                              └──────────────────────┘
```

## Key Design Decisions

### Chunking Strategy
- **Character-based sliding window** (512 chars, 100 char overlap) rather than sentence splitting
- Predictable chunk sizes yield more consistent embedding quality
- Overlap prevents information loss at boundaries

### Hybrid Search
- **Semantic search** captures meaning ("revenue growth" matches "sales increased")
- **BM25 keyword search** catches exact terms that embeddings might miss
- **Reciprocal Rank Fusion (RRF)** merges both result lists using only rank positions, avoiding the problem of incomparable score scales

### Why Build BM25 From Scratch?
The challenge requires no external search/RAG libraries. The BM25 implementation uses the standard formula with IDF smoothing, TF saturation (k1=1.5), and document length normalization (b=0.75).

### In-Memory Vector Store
- No FAISS, ChromaDB, or Pinecone — just numpy arrays
- Cosine similarity via normalized dot products
- Scales well for the typical knowledge base size (<100k chunks)

### Query Processing
- **Intent detection** prevents unnecessary searches on greetings or sensitive queries
- **Query rewriting** transforms conversational questions into focused search terms
- **Refusal policies** block PII, legal, and medical advice requests

### Guardrails
- **Citation enforcement**: LLM instructed to cite [Source N] for every claim
- **Hallucination filter**: post-generation LLM-as-judge verifies claims against sources
- **Relevance threshold**: returns "insufficient evidence" when no chunk scores high enough

---

## How to Run

### Prerequisites
- Python 3.10+
- A [Mistral AI](https://console.mistral.ai/) API key

### Setup

```bash
# Clone the repo
git clone https://github.com/bchalita/stackAI_challenge.git
cd stackAI_challenge

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set your API key
cp .env.example .env
# Edit .env and add your MISTRAL_API_KEY
```

### Run the Server

```bash
uvicorn app.main:app --reload
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

### Run Tests

```bash
# Unit tests
# Offline tests (no API key needed)
python -m tests.test_ingestion
python -m tests.test_search

# Online tests (requires MISTRAL_API_KEY)
python -m tests.test_query
python -m tests.test_generation
python -m tests.test_api

# End-to-end retrieval eval (23 queries, requires running server + API key)
# See EVALUATION.md for full methodology and findings.
python -m eval.retrieval_eval
```

See [`EVALUATION.md`](EVALUATION.md) for the full evaluation methodology,
latest scorecard, bugs surfaced during testing, and known limitations.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/ingest` | Upload one or more PDF files for ingestion |
| `POST` | `/query` | Query the knowledge base with a question |
| `POST` | `/load_samples` | Ingest the fixture corpus shipped with the repo |
| `GET` | `/suggest_queries` | LLM-generated example questions for the current corpus |
| `GET` | `/files` | List all ingested files |
| `DELETE` | `/files/{filename}` | Remove a file from the knowledge base |
| `GET` | `/` | Serve the chat UI |

### Example: Ingest

```bash
curl -X POST http://localhost:8000/ingest \
  -F "files=@document.pdf"
```

### Example: Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key findings?"}'
```

### Example: Follow-up query (with conversation history)

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How does that break down by region?",
    "history": [
      {"role": "user", "content": "What is total revenue?"},
      {"role": "assistant", "content": "Total revenue was $41,525 million."}
    ]
  }'
```

---

## Libraries Used

| Library | Purpose |
|---------|---------|
| [FastAPI](https://fastapi.tiangolo.com/) | Web framework |
| [Uvicorn](https://www.uvicorn.org/) | ASGI server |
| [PyMuPDF](https://pymupdf.readthedocs.io/) | PDF text extraction |
| [NumPy](https://numpy.org/) | Vector operations, cosine similarity |
| [httpx](https://www.python-httpx.org/) | HTTP client for Mistral API |
| [python-dotenv](https://pypi.org/project/python-dotenv/) | Environment variable management |
| [python-multipart](https://pypi.org/project/python-multipart/) | File upload parsing |

---

## Project Structure

```
├── app/
│   ├── main.py          # FastAPI app, endpoint routing
│   ├── models.py        # Pydantic request/response schemas
│   ├── ingestion.py     # PDF extraction + chunking
│   ├── embeddings.py    # Mistral embeddings + in-memory vector store
│   ├── search.py        # BM25, semantic search, RRF hybrid fusion
│   ├── query.py         # Intent detection, query rewriting, refusals
│   └── generation.py    # LLM generation, citations, hallucination filter
├── static/
│   └── index.html       # Chat UI (vanilla HTML/CSS/JS)
├── tests/
│   ├── test_ingestion.py   # 5 tests: chunking, extraction
│   ├── test_search.py      # 7 tests: BM25, RRF, tokenization
│   ├── test_query.py       # 6 tests: intent, rewriting, refusals
│   ├── test_generation.py  # 6 tests: citations, thresholds, e2e
│   └── test_api.py         # 9 tests: full API integration
├── requirements.txt
├── .env.example
└── README.md
```
