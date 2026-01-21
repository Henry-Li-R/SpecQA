import json
import os
import uuid
from pathlib import Path
from urllib.parse import urlparse

import weaviate
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from weaviate.classes.config import Configure, DataType, Property

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[1]
WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_GRPC_PORT = int(os.environ.get("WEAVIATE_GRPC_PORT", "50051"))
MERGED_PATH = Path(
    os.environ.get(
        "MERGED_CHUNKS_PATH",
        (REPO_ROOT / "src/ingest/data/chunks/merged_chunks.jsonl").as_posix(),
    )
)
CLASS_NAME = "SpecQAChunks"
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
        client.collections.delete(CLASS_NAME)
    client.collections.create(
        name=CLASS_NAME,
        vector_config=Configure.Vectors.self_provided(),
        properties=[
            Property(name="chunk_id", data_type=DataType.TEXT),
            Property(name="doc_id", data_type=DataType.TEXT),
            Property(name="doc_type", data_type=DataType.TEXT),
            Property(name="chapter", data_type=DataType.TEXT),
            Property(name="section", data_type=DataType.TEXT),
            Property(name="page", data_type=DataType.INT),
            Property(name="text", data_type=DataType.TEXT),
            Property(name="search_text", data_type=DataType.TEXT),
        ],
    )


def iter_chunks(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        yield json.loads(line)


def infer_doc_type(obj: dict) -> str:
    doc_type = (obj.get("doc_type") or "").strip().lower()
    if doc_type:
        return doc_type
    doc_id = (obj.get("doc_id") or "").lower()
    chunk_id = (obj.get("chunk_id") or "").lower()
    if "manual" in doc_id or "manual" in chunk_id:
        return "manual"
    if "datasheet" in doc_id or "datasheet" in chunk_id:
        return "datasheet"
    return "unknown"


def build_search_text(
    doc_type: str,
    doc_id: str,
    page: int,
    chapter: str,
    section: str,
    text: str,
) -> str:
    lines = [f"DOCUMENT: {doc_id}", f"PAGE: {page}"]
    if doc_type == "manual" or (doc_type == "unknown" and chapter):
        lines.append(f"CHAPTER: {chapter}")
    lines.append(f"SECTION: {section}")
    return "\n".join(lines) + "\n\n" + text


def main() -> None:
    if not MERGED_PATH.exists():
        raise FileNotFoundError(f"Missing merged chunks file: {MERGED_PATH}")

    client = connect_client()
    try:
        ensure_schema(client)
        collection = client.collections.get(CLASS_NAME)

        model = SentenceTransformer(MODEL_NAME)
        model.max_seq_length = 512

        with collection.batch.dynamic() as batch:
            for obj in iter_chunks(MERGED_PATH):
                text = obj.get("text", "").strip()
                if not text:
                    continue
                chunk_id = obj.get("chunk_id") or str(uuid.uuid4())
                doc_id = obj.get("doc_id", "")
                chapter = obj.get("chapter", "")
                section = obj.get("section", "")
                page = obj.get("page", 0)
                doc_type = infer_doc_type(obj)
                search_text = build_search_text(
                    doc_type=doc_type,
                    doc_id=doc_id,
                    page=page,
                    chapter=chapter,
                    section=section,
                    text=text,
                )
                vector = model.encode(
                    f"passage: {search_text}",
                    normalize_embeddings=True,
                ).tolist()
                batch.add_object(
                    properties={
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "doc_type": doc_type,
                        "chapter": chapter,
                        "section": section,
                        "page": page,
                        "text": text,
                        "search_text": search_text,
                    },
                    uuid=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id)),
                    vector=vector,
                )
    finally:
        client.close()

    print(
        f"Ingested combined chunks into {CLASS_NAME} from {MERGED_PATH}"
    )


if __name__ == "__main__":
    main()
