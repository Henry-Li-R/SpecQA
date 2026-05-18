import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

import weaviate
from sentence_transformers import CrossEncoder, SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUERIES_PATH = REPO_ROOT / "tests/retrieval_binary_recall_datasheet.jsonl"

WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_GRPC_PORT = int(os.environ.get("WEAVIATE_GRPC_PORT", "50051"))
WEAVIATE_API_KEY = os.environ.get("WEAVIATE_API_KEY", "")
COLLECTION = "SpecQAChunks"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
RERANK_MODEL_NAME = "BAAI/bge-reranker-base"
HYBRID_LIMIT = 20
RERANK_TOP_K = 20
RERANK_ENABLED = True
ALPHA = 0.6


def connect_client() -> weaviate.WeaviateClient:
    parsed = urlparse(WEAVIATE_URL)
    if not parsed.scheme:
        auth = weaviate.auth.AuthApiKey(api_key=WEAVIATE_API_KEY) if WEAVIATE_API_KEY else None
        return weaviate.connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_URL,
            auth_credentials=auth,
        )
    scheme = parsed.scheme
    host = parsed.hostname or parsed.path
    port = parsed.port or (443 if scheme == "https" else 8080)
    auth = weaviate.auth.AuthApiKey(api_key=WEAVIATE_API_KEY) if WEAVIATE_API_KEY else None
    return weaviate.connect_to_custom(
        http_host=host,
        http_port=port,
        http_secure=(scheme == "https"),
        grpc_host=host,
        grpc_port=WEAVIATE_GRPC_PORT,
        grpc_secure=(scheme == "https"),
        auth_credentials=auth,
    )


def load_queries(path: Path) -> List[Dict]:
    queries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        queries.append(json.loads(line))
    return queries


def run_eval(queries_path: Optional[Path] = None) -> Dict:
    path = Path(queries_path) if queries_path else DEFAULT_QUERIES_PATH
    if not path.exists():
        raise FileNotFoundError(f"Missing queries file: {path}")

    queries = load_queries(path)
    if not queries:
        raise ValueError(f"No queries found in: {path}")

    client = connect_client()
    collection = client.collections.get(COLLECTION)
    embedder = SentenceTransformer(MODEL_NAME)
    embedder.max_seq_length = 512
    reranker = CrossEncoder(RERANK_MODEL_NAME) if RERANK_ENABLED else None

    total = 0
    hit_at_1 = 0
    hit_at_5 = 0
    misses_at_1: List[str] = []
    misses_at_5: List[str] = []
    per_query: List[Dict] = []

    for item in queries:
        total += 1
        qid = item.get("qid", f"q{total:02d}")
        query = item.get("query", "")
        gold = set(item.get("gold_chunk_ids", []))

        vector = embedder.encode(f"query: {query}", normalize_embeddings=True).tolist()

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
                obj for _, obj in sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
            ]
        else:
            ranked = objects

        top1 = [obj.properties.get("chunk_id") for obj in ranked[:1]]
        top5 = [obj.properties.get("chunk_id") for obj in ranked[:5]]

        hit1 = bool(gold.intersection(top1))
        hit5 = bool(gold.intersection(top5))

        if hit1:
            hit_at_1 += 1
        else:
            misses_at_1.append(qid)

        if hit5:
            hit_at_5 += 1
        else:
            misses_at_5.append(qid)

        per_query.append({
            "qid": qid,
            "query": query,
            "hit_at_1": hit1,
            "hit_at_5": hit5,
            "top5": top5,
            "gold": sorted(gold),
        })

    client.close()

    recall_1 = hit_at_1 / total if total else 0.0
    recall_5 = hit_at_5 / total if total else 0.0

    misses_detail = [pq for pq in per_query if not pq["hit_at_5"]]

    return {
        "recall_at_1": recall_1,
        "recall_at_5": recall_5,
        "n_queries": total,
        "misses": [{"qid": m["qid"], "query": m["query"], "top5": m["top5"], "gold": m["gold"]}
                   for m in misses_detail],
        "per_query": per_query,
    }


def main() -> None:
    queries_path = Path(
        os.environ.get("RECALL_QUERIES_PATH", DEFAULT_QUERIES_PATH.as_posix())
    )
    result = run_eval(queries_path)
    total = result["n_queries"]
    r1 = result["recall_at_1"]
    r5 = result["recall_at_5"]
    hits1 = round(r1 * total)
    hits5 = round(r5 * total)
    print(f"Collection:  {COLLECTION} (datasheet queries, rerank={RERANK_ENABLED})")
    print(f"Total queries: {total}")
    print(f"Recall@1: {hits1}/{total} = {r1:.3f}")
    print(f"Recall@5: {hits5}/{total} = {r5:.3f}")
    for miss in result["misses"]:
        print(f"Miss@5 {miss['qid']}: {miss['query']}")
        print(f"  top5: {miss['top5']}")
        print(f"  gold: {miss['gold']}")


if __name__ == "__main__":
    main()
