import argparse
import os
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import weaviate
from sentence_transformers import CrossEncoder, SentenceTransformer

from src.LLM.chatgpt_client import answer_with_citations

WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_GRPC_PORT = int(os.environ.get("WEAVIATE_GRPC_PORT", "50051"))
COMBINED_CLASS = "DocChunkCombined"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
RERANK_MODEL_NAME = "BAAI/bge-reranker-base"
ALPHA = 0.6
HYBRID_LIMIT = 20
RERANK_TOP_K = 20
TOP_K = 5
RERANK_ENABLED = True

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


def retrieve_chunks(
    collection: weaviate.collections.collection.Collection,
    query: str,
    embedder: SentenceTransformer,
    reranker: Optional[CrossEncoder],
) -> List[Dict[str, Any]]:
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
        return_properties=[
            "chunk_id",
            "doc_id",
            "chapter",
            "section",
            "page",
            "text",
            "search_text",
        ],
    )

    objects = result.objects
    ranked = objects
    if RERANK_ENABLED and reranker is not None:
        candidates = objects[:RERANK_TOP_K]
        rerank_pairs = [(query, obj.properties.get("search_text", "")) for obj in candidates]
        scores = reranker.predict(rerank_pairs)
        ranked = [
            obj
            for _, obj in sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        ]

    chunks: List[Dict[str, Any]] = []
    for obj in ranked[:TOP_K]:
        chunks.append(obj.properties)
    return chunks


def run_pipeline(
    query: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    client = connect_client()
    try:
        embedder = SentenceTransformer(MODEL_NAME)
        embedder.max_seq_length = 512
        reranker = CrossEncoder(RERANK_MODEL_NAME) if RERANK_ENABLED else None

        collection = client.collections.get(COMBINED_CLASS)
        chunks = retrieve_chunks(collection, query, embedder, reranker)
        return answer_with_citations(
            user_query=query,
            retrieved_chunks=chunks,
            chat_history=chat_history,
            api_key=api_key,
        )
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval + LLM pipeline.")
    parser.add_argument("query", help="User query")
    parser.add_argument("--api-key", dest="api_key", default=None, help="OpenAI API key")
    args = parser.parse_args()

    result = run_pipeline(args.query, api_key=args.api_key)
    print(result)


if __name__ == "__main__":
    main()
