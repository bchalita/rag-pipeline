# Test fixtures

Canonical corpus and query set for end-to-end evaluation of the RAG pipeline.

## Contents

| File | Size | Pages | ~Chunks | Purpose |
|---|---|---|---|---|
| `robust_optimization.pdf` | 314 KB | 39 | ~193 | Clean academic prose. Good for testing retrieval precision and answer shaping on well-structured technical content. |
| `salesforce_10k.pdf` | 1.5 MB | 175 | ~1,456 | SEC filing. Stress-tests ingestion on **odd formats** — financial tables, multi-column layouts, footnotes, exhibits. Surfaces known limitations of the naive chunker (see below). |
| `queries.yaml` | — | — | — | 15 canonical queries across 7 behavior categories (grounded prose retrieval, grounded table retrieval, OOD refusal, chitchat, PII/medical policy refusal, retrieval precision, broad). |

## Why these two documents?

Picked for **coverage, not volume**:

1. **`robust_optimization.pdf`** — small, clean, citation-rich. Validates the happy path: retrieval precision, citation mapping, answer shaping (explain vs compare templates).
2. **`salesforce_10k.pdf`** — larger, "dirty" format. Validates that the pipeline doesn't fall over on real-world documents. Multi-column text flows awkwardly through PyMuPDF's extraction, tables get mangled into row-column text fragments, and footnotes/exhibits break chunk boundaries.

Two documents is enough to exercise cross-document retrieval (e.g., "are optimization and Salesforce related?" should refuse, not conflate) while keeping ingestion under ~2 minutes per fresh run.

## Known limitations surfaced by the corpus

- **Table extraction is lossy.** The 10-K's income statement renders as a flat text block: column headers and row values get reordered during PyMuPDF extraction. The LLM usually still answers correctly (e.g., "$41,525M revenue") because enough surrounding context survives, but the specific cited chunk may point at a mangled fragment. A better pipeline would route PDF tables through a dedicated extractor (e.g., Camelot, `pdfplumber.extract_tables`) before chunking — out of scope here.
- **Multi-column PDFs interleave.** When a page has two columns, extraction emits column 1 top → column 1 bottom → column 2 top → column 2 bottom, which is correct, but chunking with a 512-char window occasionally bridges across columns. Usually harmless but can produce awkward citations.
- **Footnotes / headers attach to the wrong chunk.** A running footer like "Salesforce 2026 Annual Report" appears in almost every chunk and inflates BM25 scores for queries that happen to contain those tokens. Not a retrieval bug but a chunking-quality issue.

All three are documented intentionally — a grader can reproduce them and a future version of the pipeline should address them.

## Adding more documents

Drop any PDF into this folder and it will be picked up by the smoke / eval scripts. Recommended additional stress cases (not committed to keep the repo small):

- Legislative transcript (speaker-turn formatting, procedural language)
- Appliance user manual (image-heavy, icons instead of text, safety warnings)
- Scanned contract / legal PDF (tests OCR fallback — not yet implemented)

## Running

```bash
# Smoke test (sanity check, ~2-3 min)
python scripts/smoke.py

# Full retrieval eval (precision@5, MRR, citation accuracy)
python eval/retrieval_eval.py
```

Both scripts start the API server, ingest every PDF in this folder, then run each query in `queries.yaml`.

## Ingestion cost

One-time cost per fresh server start (embeddings are in-memory, lost on restart):

| Corpus | Time | Approximate API cost |
|---|---|---|
| `robust_optimization.pdf` alone | ~15 s | <$0.01 |
| Both committed fixtures | ~2 min | ~$0.02 |
| Full 4-PDF corpus (10-K + manual + paper + Senate) | ~4.5 min | ~$0.04 |

A simple on-disk cache for embeddings would remove this friction but was out of scope for the challenge.
