# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

This is a RAG system for Acuvim 3 product docs (datasheet + manual). It is actively being evolved from a crude MVP into a complete, production-quality system — prefer clean, well-thought-out implementations over quick patches, and flag adjacent issues when relevant.

## Commands

```bash
# Install dependencies (CPU-only torch for dev speed)
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements.txt

# Start Weaviate
docker compose -f weaviate/docker-compose.yml up -d

# Ingest chunks into Weaviate
python3 scripts/weaviate_ingest_combined.py   # both docs → SpecQAChunks
python3 scripts/weaviate_ingest_datasheet.py  # datasheet only → DocChunkDatasheet
python3 scripts/weaviate_ingest_manual.py     # manual only → DocChunkManual

# Run end-to-end pipeline
python3 src/pipeline/run_pipeline.py "What is the voltage measurement range?"

# Start dev server (localhost:8000)
python3 -m src.server.chat_server --development

# Unit tests (no Weaviate or network needed)
python3 -m unittest tests/test_chatgpt_client.py tests/test_pipeline.py

# Run a single test file
python3 -m unittest tests/test_chatgpt_client.py

# Integration tests (requires Weaviate running + models downloaded)
python3 -m unittest tests/test_pipeline_integration.py

# Retrieval + generation evaluation
python3 tests/retrieval_binary_recall_datasheet.py
python3 tests/retrieval_binary_recall_manual.py
python3 tests/generation_eval.py --print-details
```

## Architecture

The pipeline has four stages:

1. **Ingest** (`src/ingest/`) — PDF → JSONL chunks (~500 chars, with overlap). Outputs land in `src/ingest/data/chunks/`. `merged_chunks.jsonl` combines both docs.

2. **Index** (`scripts/`) — Reads JSONL, builds `search_text` (doc/page/section-prefixed text), embeds with `bge-small-en-v1.5` using `"passage: {search_text}"` prefix, and upserts into Weaviate with self-provided vectors. The main collection is `SpecQAChunks` (combined, with `doc_type` field).

3. **Retrieve** (`src/pipeline/run_pipeline.py`) — Hybrid search (BM25 + vector, `alpha=0.6`) on `search_text` using `"query: {query}"` prefix for asymmetric BGE encoding. Optional `bge-reranker-base` reranking of top-20 candidates before returning top-5.

4. **Generate** (`src/LLM/chatgpt_client.py`) — Sends chunks to GPT-4o with structured output (`OpenAI().responses.parse`) enforcing `AnswerOutput` schema. Post-generation citation validation checks each quote is an exact substring (whitespace-normalized) of its cited chunk. Invalid citations either trigger abstain or attach `validation_errors` depending on `abstain_on_invalid`.

The server (`src/server/chat_server.py`) is a plain `http.server` wrapper around `run_pipeline()` — no framework. It serves `static/chat.html` as the UI and exposes `POST /api/chat`.

## Key design details

- **`search_text` field**: the embedding and BM25 target for both ingestion and retrieval. Contains structured prefix (`DOCUMENT / PAGE / CHAPTER / SECTION`) followed by raw text. Do not confuse with `text`, which is the clean chunk used in LLM context and citation validation.
- **Weaviate connection**: `connect_client()` auto-detects localhost vs. cloud URL and uses `connect_to_custom` vs. `connect_to_weaviate_cloud` accordingly.
- **Citation validation**: uses whitespace normalization (`re.sub(r"\s+", " ")`) and trailing-period stripping before substring match.
- **Tracing**: Phoenix (Arize) via OpenTelemetry. Disabled during CI. Set `PHOENIX_COLLECTOR_ENDPOINT` to enable.

## Environment variables

Required:
- `OPENAI_API_KEY`

Optional (defaults shown):
- `WEAVIATE_URL=http://localhost:8080`
- `WEAVIATE_GRPC_PORT=50051`
- `WEAVIATE_API_KEY` (only for Weaviate Cloud)
- `PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006/v1/traces`

Use `.env` for local development — loaded via `python-dotenv`.

## CI

Two jobs in `.github/workflows/ci.yml`:
- **unit-tests**: fully mocked, no Weaviate. Runs `test_chatgpt_client.py` and `test_pipeline.py`.
- **integration-tests**: spins up Weaviate via docker compose, downloads HuggingFace models (cached), runs `test_pipeline_integration.py`. Ingest step is commented out because this job uses S3-based ingestion (`weaviate_ingest_s3.py`), not the local scripts.

Secrets/vars (`OPENAI_API_KEY`, `WEAVIATE_API_KEY`, etc.) must be explicitly set in the GitHub Actions UI under repo settings.
