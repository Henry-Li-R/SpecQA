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

## Notes / decisions
- Diagram pages are skipped to reduce complexity.
- Datasheet, manual, and any other documents should use one combined index.
- Combined index can bias toward longer documents (i.e. manual > datasheet); use doc-type filters or per-doc top‑k if needed.