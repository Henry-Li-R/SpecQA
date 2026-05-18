# Eval Harness

Automated quality baseline for the Acuvim 3 RAG system. Runs on every PR via
`integration-and-eval` CI job; results are uploaded as a job artifact.

## Running locally

```bash
make eval                   # full eval (retrieval + generation)
make eval-retrieval         # retrieval only (no OpenAI calls)
make eval-generation        # generation only
make update-baselines       # re-anchor baselines to current performance
```

Result JSONs land in `eval/results/` (git-ignored). `eval/baselines.json` is
tracked and holds the minimum acceptable values.

## Metrics

### Enforced — drop >0.05 below baseline blocks the PR

| Metric | Suite | What it measures |
|--------|-------|-----------------|
| `recall_at_5` | Retrieval (datasheet + manual) | At least one gold chunk appears in the top-5 hybrid-search results, after reranking |
| `citation_quote_validity` | Generation | For every non-abstain answer: ≥1 citation present AND every cited quote is an exact whitespace-normalised substring of its chunk. Zero citations → 0.0 (not a pass). |
| `answer_keyword_recall` | Generation | Fraction of gold keywords present in the answer text (averaged over queries that have keywords) |
| `abstain_recall` | Generation | Fraction of must-abstain queries where the model correctly abstained |
| `abstain_precision` | Generation | Fraction of all abstain responses that were on a must-abstain query (penalises over-abstention) |

### Informational only — tracked, never block

- `recall_at_1` — top-1 retrieval hit rate; useful for comparing chunking strategies
- `citation_in_gold` — fraction of non-abstain answers where at least one cited chunk is a gold chunk; retrieval–generation alignment diagnostic
- `answer_rate` — fraction of queries answered (not abstained); sanity check

## Tolerance rationale

All thresholds use a universal **0.05 tolerance**. The golden sets are actively
expanding as coverage gaps are identified. Adding new entries changes metric
denominators, producing apparent fluctuations that are measurement artefacts
rather than regressions. A 0.05 tolerance absorbs this noise while still
catching real regressions: on a 95-query manual set, a 5% drop = ~5 new misses.

At n=15 must-abstain queries, 1 failure drops abstain_recall by 0.067 (blocked);
0 failures = 1.0.

## Chunk ID caveat

Gold chunk IDs in the JSONL files are tied to the current chunking strategy
(PyMuPDF plain-text extraction). When table extraction is added, chunk
boundaries and IDs will shift. After re-chunking and re-ingesting, run:

```bash
# re-ingest
python3 scripts/weaviate_ingest_combined.py

# verify golden sets still hit (check for new misses)
make eval-retrieval

# re-anchor baselines
make update-baselines
```

## Known limitation

**Uncited claim faithfulness** is not measured. The system may include accurate
prose that is not quoted in any citation — or inaccurate prose that happens not
to be cited. Catching this requires LLM-as-judge evaluation, which is out of
scope for this harness.

## Bootstrapping

On a fresh clone with `eval/baselines.json` at zeros, `make eval` will run all
suites, print results, and exit 0 (no threshold violations since all metrics
exceed 0.0 − 0.05 = −0.05). Run `make update-baselines` once to anchor real
baselines, then subsequent runs gate against those values.
