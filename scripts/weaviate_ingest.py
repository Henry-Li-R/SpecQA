import json
import os
import uuid
from pathlib import Path
from urllib.parse import urlparse

import weaviate
from sentence_transformers import SentenceTransformer
from weaviate.classes.config import Configure, DataType, Property

REPO_ROOT = Path(__file__).resolve().parents[1]
WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_GRPC_PORT = int(os.environ.get("WEAVIATE_GRPC_PORT", "50051"))
CHUNKS_PATH = Path(
    os.environ.get(
        "CHUNKS_PATH",
        (REPO_ROOT / "src/ingest/data/chunks/acuvim_3_datasheet_chunks.jsonl").as_posix(),
    )
)
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


def ensure_schema(client: weaviate.WeaviateClient) -> None:
    if client.collections.exists(CLASS_NAME):
        return
    client.collections.create(
        name=CLASS_NAME,
        vector_config=Configure.Vectorizer.none(),
        properties=[
            Property(name="chunk_id", data_type=DataType.TEXT),
            Property(name="doc_id", data_type=DataType.TEXT),
            Property(name="section", data_type=DataType.TEXT),
            Property(name="page", data_type=DataType.INT),
            Property(name="text", data_type=DataType.TEXT),
        ],
    )


def main() -> None:
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Missing chunks file: {CHUNKS_PATH}")

    client = connect_client()
    ensure_schema(client)
    collection = client.collections.get(CLASS_NAME)

    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = 512

    with collection.batch.dynamic() as batch:
        for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines():
            obj = json.loads(line)
            text = obj.get("text", "").strip()
            if not text:
                continue
            chunk_id = obj.get("chunk_id") or str(uuid.uuid4())
            vector = model.encode(
                f"passage: {text}",
                normalize_embeddings=True,
            ).tolist()
            batch.add_object(
                properties={
                    "chunk_id": chunk_id,
                    "doc_id": obj.get("doc_id", ""),
                    "section": obj.get("section", ""),
                    "page": obj.get("page", 0),
                    "text": text,
                },
                uuid=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id)),
                vector=vector,
            )

    client.close()


if __name__ == "__main__":
    main()
