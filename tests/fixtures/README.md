# Test fixtures

Canonical corpus and query set for end-to-end evaluation of the RAG pipeline.

## Contents

| File | Size | Pages | ~Chunks | Description |
|---|---|---|---|---|
| `robust_optimization.pdf` | 314 KB | 39 | ~193 | Bertsimas/den Hertog survey on robust & data-driven optimization. Technical, citation-rich, good for precision testing. |
| `queries.yaml` | — | — | — | 11 canonical queries across 6 behavior categories (grounded, OOD refusal, chitchat, policy refusal, precision, broad). |

## Why only one PDF?

Tradeoff. A richer corpus (e.g., Salesforce 10-K + Senate transcript + washer manual) exposes cross-document retrieval, but bloats the repo and slows every test run. A single ~200-chunk paper is enough to validate:

- Retrieval precision (does the right paragraph surface?)
- Citation accuracy (do [Source N] markers map to the actual page?)
- Refusal triggers (OOD, PII, chitchat all exercised without needing multiple docs)
- Answer shaping (explain / compare templates)

If you want to exercise multi-document retrieval locally, drop additional PDFs into this folder and extend `queries.yaml` — both the smoke script and eval harness ingest whatever's present.

## Running

```bash
# Smoke test (sanity check, ~1-2 min)
python scripts/smoke.py

# Full retrieval eval (precision@5, MRR, citation accuracy)
python eval/retrieval_eval.py
```

Both scripts start the API server, ingest every PDF in this folder, then run each query in `queries.yaml`.

## Ingestion cost

One-time cost per fresh server start (embeddings are in-memory, lost on restart):

- robust_optimization.pdf alone: ~15 seconds, <$0.01 in Mistral embedding API calls
- All four of Bernardo's originals (10-K, manual, paper, Senate): ~4.5 minutes, ~$0.04

A simple on-disk cache for embeddings would remove this friction but was out of scope for the challenge.
