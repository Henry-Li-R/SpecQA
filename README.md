# rag-demo

Minimal, production-compatible RAG demo for Acuvim 3 datasheet + manual.

## What this does
- Chunks datasheet/manual PDFs into structured text.
- Indexes chunks into Weaviate (separate or combined collections).
- Retrieves with hybrid search + optional reranking.
- Calls ChatGPT with strict JSON output + quote-backed citations.
- Evaluates retrieval and generation with simple metrics.
- Traces retrieval/LLM spans with Phoenix (Arize) via OpenTelemetry.

## Repo structure
- `docs_raw/` — source PDFs
- `src/ingest/` — chunkers for datasheet/manual
- `src/ingest/data/chunks/` — JSONL chunk outputs
- `scripts/` — Weaviate ingestion scripts
- `src/LLM/` — ChatGPT client + output validation
- `src/pipeline/` — end-to-end pipeline
- `tests/` — retrieval + generation eval
- `weaviate/` — Docker Compose for Weaviate

## Chunking
- Datasheet: `src/ingest/chunk_datasheet_pdf.py`
- Manual: `src/ingest/chunk_manual_pdf.py`
- Chunks are split to ~500 chars with overlap and a minimum size check.

## Weaviate setup
Docker:
```bash
docker compose -f weaviate/docker-compose.yml up -d
```

Collections:
- `DocChunkDatasheet` (datasheet only)
- `DocChunkManual` (manual only)
- `DocChunkCombined` (both docs, with `doc_type`)

### Ingest
```bash
python3 scripts/weaviate_ingest_datasheet.py
python3 scripts/weaviate_ingest_manual.py
python3 scripts/weaviate_ingest_combined.py
```

## Retrieval + LLM pipeline
Entry point:
```bash
python3 src/pipeline/run_pipeline.py "What is the voltage measurement range?"
```

Retrieval:
- Hybrid search (BM25 + vector)
- `bge-small` embeddings
- `bge-reranker-base` reranker (optional)
- Uses `search_text` (doc/page/section prefixed) for better ranking.

LLM:
- `src/LLM/chatgpt_client.py`
- Strict JSON schema enforced with `OpenAI().responses.parse`.
- Validates each citation quote against chunk text (whitespace-normalized).
- Optional abstain on invalid citations.

## Environment variables
Required:
- `OPENAI_API_KEY`

Optional:
- `WEAVIATE_URL` (default `http://localhost:8080`)
- `WEAVIATE_GRPC_PORT` (default `50051`)
- `PHOENIX_COLLECTOR_ENDPOINT` (default `http://localhost:6006/v1/traces`)
- `PHOENIX_PROJECT_NAME` (default `rag-demo`)

Use `.env` for local development (loaded via `python-dotenv`).

## Phoenix tracing (Arize)
Start Phoenix separately:
```bash
python3 - <<'PY'
import phoenix as px
px.launch_app()
print("Phoenix UI at http://localhost:6006")
PY
```

Then run the pipeline with:
```bash
export PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006/v1/traces
python3 src/pipeline/run_pipeline.py "What is the voltage measurement range?"
```

## Evaluation
### Retrieval recall
```bash
python3 tests/retrieval_binary_recall_datasheet.py
python3 tests/retrieval_binary_recall_manual.py
```

### Generation eval
```bash
python3 tests/generation_eval.py --print-details
```

Metrics:
- citation quote valid rate
- citation-in-gold rate
- answer keyword recall
- abstain precision/recall


## Document-Length Bias & Mitigation

### Problem
In a combined index (e.g., short datasheet + long manual), **long documents dominate retrieval** because they generate many more chunks. This can bury concise, high-signal answers from shorter docs.

---

### Mitigation: retrieval-evidence doc gating (no LLM router)

#### 1. Initial retrieval
- Retrieve `K0` candidates (e.g., 50–80) from the combined index.

#### 2. Doc-level relevance scoring
- Group hits by `doc_id`.
- Compute a doc relevance score using only top evidence:

`doc_score = 0.6 * best_hit + 0.4 * mean(top5_hits)`

This avoids domination by many weak hits from long docs.

#### 3. Soft document selection
- Always keep the top document.
- Include a second document only if:
  - `score2 ≥ 0.9 * score1`, or
  - `score2 ≥ absolute_threshold` (e.g., 0.2 if normalized).

This preserves recall for cross-doc queries without adding noise.

#### 4. Rerank within selected docs
- Drop candidates from other docs.
- Rerank remaining chunks only (cleaner results, lower cost).

---

### Defaults
- `K0 = 50`
- Scoring: `0.6 * best + 0.4 * mean(top5)`
- Include doc2 if `score2 ≥ 0.9 * score1` or `≥ 0.2`

**Note:** This addresses document-length bias. Table-fragmentation issues are handled separately.