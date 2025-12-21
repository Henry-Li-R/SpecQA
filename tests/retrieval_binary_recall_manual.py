import json
import os
from pathlib import Path
from urllib.parse import urlparse

import weaviate
from sentence_transformers import CrossEncoder, SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parents[1]
QUERIES_PATH = Path(
    os.environ.get(
        "RECALL_QUERIES_PATH",
        (REPO_ROOT / "tests/retrieval_binary_recall_manual.jsonl").as_posix(),
    )
)
WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_GRPC_PORT = int(os.environ.get("WEAVIATE_GRPC_PORT", "50051"))
CLASS_NAME = "DocChunkManual"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
RERANK_MODEL_NAME = "BAAI/bge-reranker-base"
HYBRID_LIMIT = 20
RERANK_TOP_K = 20
RERANK_ENABLED = True

ALPHA = 0.6

print(f"Rerank enabled: {RERANK_ENABLED}")

def connect_client() -> weaviate.WeaviateClient:
    parsed = urlparse(WEAVIATE_URL)
    scheme = parsed.scheme or "http"
    host = parsed.hostname or parsed.path
    port = parsed.port or (443 if scheme == "https" else 8080)
    return weaviate.connect_to_custom(
        http_host=host,
        http_port=port,
        http_secure=(scheme == "https"),
        grpc_host=host,
        grpc_port=WEAVIATE_GRPC_PORT,
        grpc_secure=(scheme == "https"),
    )


def load_queries(path: Path):
    queries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        queries.append(json.loads(line))
    return queries


def main() -> None:
    if not QUERIES_PATH.exists():
        raise FileNotFoundError(f"Missing queries file: {QUERIES_PATH}")

    queries = load_queries(QUERIES_PATH)
    if not queries:
        raise ValueError(f"No queries found in: {QUERIES_PATH}")

    client = connect_client()
    collection = client.collections.get(CLASS_NAME)
    embedder = SentenceTransformer(MODEL_NAME)
    embedder.max_seq_length = 512
    reranker = CrossEncoder(RERANK_MODEL_NAME) if RERANK_ENABLED else None

    total = 0
    hit_at_1 = 0
    hit_at_5 = 0
    misses_at_1 = []
    misses_at_5 = []
    miss_details_at_5 = {}

    for item in queries:
        total += 1
        qid = item.get("qid", f"q{total:02d}")
        query = item.get("query", "")
        gold = set(item.get("gold_chunk_ids", []))

        vector = embedder.encode(
            f"query: {query}",
            normalize_embeddings=True,
        ).tolist()

        result = collection.query.hybrid(
            query=query,
            vector=vector,
            alpha=ALPHA,
            limit=HYBRID_LIMIT,
            query_properties=["search_text"],
            return_properties=["chunk_id", "text", "search_text"],
        )

        objects = result.objects
        if RERANK_ENABLED and reranker is not None:
            candidates = objects[:RERANK_TOP_K]
            rerank_pairs = [(query, obj.properties.get("search_text", "")) for obj in candidates]
            scores = reranker.predict(rerank_pairs)
            ranked = [
                obj
                for _, obj in sorted(
                    zip(scores, candidates), key=lambda x: x[0], reverse=True
                )
            ]
        else:
            ranked = objects

        top1 = [obj.properties.get("chunk_id") for obj in ranked[:1]]
        top5 = [obj.properties.get("chunk_id") for obj in ranked[:5]]
        search_text_by_id = {
            obj.properties.get("chunk_id"): obj.properties.get("search_text", "")
            for obj in ranked[:5]
        }

        if gold.intersection(top1):
            hit_at_1 += 1
        else:
            misses_at_1.append(qid)

        if gold.intersection(top5):
            hit_at_5 += 1
        else:
            misses_at_5.append(qid)
            miss_details_at_5[qid] = {
                "query": query,
                "top5": [search_text_by_id.get(cid, "") for cid in top5],
                "gold": sorted(gold),
            }

    client.close()

    recall_1 = hit_at_1 / total if total else 0.0
    recall_5 = hit_at_5 / total if total else 0.0

    print(f"Total queries: {total}")
    print(f"Recall@1: {hit_at_1}/{total} = {recall_1:.3f}")
    print(f"Recall@5: {hit_at_5}/{total} = {recall_5:.3f}")
    if misses_at_1:
        print(f"Misses@1: {', '.join(misses_at_1)}")
    if misses_at_5:
        print(f"Misses@5: {', '.join(misses_at_5)}")
        for qid in misses_at_5:
            details = miss_details_at_5.get(qid, {})
            print(f"Miss@5 {qid}: {details.get('query', '')}")
            print(f"  top5: {details.get('top5', [])}")
            print(f"  gold: {details.get('gold', [])}")


if __name__ == "__main__":
    main()
