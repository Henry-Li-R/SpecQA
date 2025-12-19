import os
from urllib.parse import urlparse

import weaviate
from sentence_transformers import SentenceTransformer

WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_GRPC_PORT = int(os.environ.get("WEAVIATE_GRPC_PORT", "50051"))
CLASS_NAME = "DocChunk"
MODEL_NAME = "BAAI/bge-small-en-v1.5"


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


def main() -> None:
    client = connect_client()
    collection = client.collections.get(CLASS_NAME)
    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = 512

    query = "what are the compliance standards for acuvim 3?"
    vector = model.encode(
        f"query: {query}",
        normalize_embeddings=True,
    ).tolist()

    result = collection.query.hybrid(
        query=query,
        vector=vector,
        alpha=0.6,
        limit=5,
        return_properties=["chunk_id", "doc_id", "section", "page", "text"],
    )

    for obj in result.objects:
        props = obj.properties
        print(
            f"{props.get('chunk_id')} | "
            f"{props.get('doc_id')} | "
            f"{props.get('section')} | "
            f"p{props.get('page')}"
        )
        print(props.get("text", "").strip())
        print("----")

    client.close()


if __name__ == "__main__":
    main()
