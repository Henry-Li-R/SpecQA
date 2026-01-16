import argparse
import os
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import weaviate
from dotenv import load_dotenv
from weaviate.classes.init import Auth
from sentence_transformers import CrossEncoder, SentenceTransformer
from phoenix.otel import register
from src.LLM.chatgpt_client import answer_with_citations

load_dotenv()

WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_GRPC_PORT = int(os.environ.get("WEAVIATE_GRPC_PORT", "50051"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
COMBINED_CLASS = "DocChunkCombined"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
RERANK_MODEL_NAME = "BAAI/bge-reranker-base"
ALPHA = 0.6 # tune for hybrid search
HYBRID_LIMIT = 20
RERANK_TOP_K = 20
TOP_K = 5
RERANK_ENABLED = True

PHOENIX_COLLECTOR_ENDPOINT = os.environ.get(
    "PHOENIX_COLLECTOR_ENDPOINT",
    "http://localhost:6006",
)
PHOENIX_PROJECT_NAME = os.environ.get("PHOENIX_PROJECT_NAME", "rag-demo")

def setup_tracing():
    if not PHOENIX_COLLECTOR_ENDPOINT:
        return None
    endpoint = PHOENIX_COLLECTOR_ENDPOINT.rstrip("/")
    if "/v1/traces" not in endpoint:
        endpoint = f"{endpoint}/v1/traces"
    provider = register(
        project_name=PHOENIX_PROJECT_NAME,
        endpoint=endpoint,
        batch=True,
        auto_instrument=True,
    )
    return provider.get_tracer(__name__)

def connect_client() -> weaviate.WeaviateClient:
    parsed = urlparse(WEAVIATE_URL)
    host = parsed.hostname or parsed.path
    if host and host not in {"localhost", "127.0.0.1"}:
        return weaviate.connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_URL,
            auth_credentials=Auth.api_key(os.environ.get("WEAVIATE_API_KEY")),
        )
    scheme = parsed.scheme or "http"
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
    tracer=None,
) -> List[Dict[str, Any]]:
    span_ctx = (
        tracer.start_as_current_span("retrieve", openinference_span_kind="retriever")
        if tracer
        else nullcontext()
    )
    with span_ctx as span:
        if span:
            span.set_input(query)
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
        for idx, obj in enumerate(ranked[:TOP_K]):
            chunks.append(obj.properties)
            if span:
                span.set_attribute(
                    f"retrieval.documents.{idx}.document.id",
                    obj.properties.get("chunk_id", ""),
                )
                span.set_attribute(
                    f"retrieval.documents.{idx}.document.metadata",
                    f"doc_id={obj.properties.get('doc_id','')}, "
                    f"page={obj.properties.get('page','')}, "
                    f"section={obj.properties.get('section','')}, "
                    f"chapter={obj.properties.get('chapter','')}",
                )
                span.set_attribute(
                    f"retrieval.documents.{idx}.document.content",
                    obj.properties.get("text", ""),
                )
        return chunks


def run_pipeline(
    query: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    api_key: Optional[str] = None,
    abstain_on_invalid: bool = False,
) -> Dict[str, Any]:
    tracer = setup_tracing()
    client = connect_client()
    try:
        embedder = SentenceTransformer(MODEL_NAME)
        embedder.max_seq_length = 512
        reranker = CrossEncoder(RERANK_MODEL_NAME) if RERANK_ENABLED else None

        collection = client.collections.get(COMBINED_CLASS)
        chunks = retrieve_chunks(collection, query, embedder, reranker, tracer=tracer)
        llm_ctx = (
            tracer.start_as_current_span("llm", openinference_span_kind="llm")
            if tracer
            else nullcontext()
        )
        with llm_ctx as span:
            if span:
                span.set_input(query)
            return answer_with_citations(
                user_query=query,
                retrieved_chunks=chunks,
                chat_history=chat_history,
                api_key=api_key,
                abstain_on_invalid=abstain_on_invalid,
            )
    finally:
        client.close()
