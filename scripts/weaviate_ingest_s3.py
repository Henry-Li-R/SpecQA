import json
import os
import sys
import uuid
from urllib.parse import urlparse

import boto3
import weaviate
from dotenv import load_dotenv
from weaviate.classes.init import Auth
from weaviate.exceptions import UnexpectedStatusCodeError


def get_required_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise ValueError(f"Missing required env var: {name}")
    return value.strip()


def load_jsonl_from_s3(bucket: str, key: str):
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    for line in obj["Body"].iter_lines():
        if not line:
            continue
        yield json.loads(line.decode("utf-8"))


def connect_client(weaviate_url: str, weaviate_api_key: str | None) -> weaviate.WeaviateClient:
    return weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=Auth.api_key(weaviate_api_key),
    )


def build_properties(record: dict) -> dict:
    if not record.get("chunk_id"):
        raise ValueError("Record missing chunk_id")
    if not record.get("text"):
        raise ValueError(f"Record {record.get('chunk_id')} missing text")
    props: dict = {}
    for key in (
        "chunk_id",
        "doc_id",
        "doc_type",
        "chapter",
        "section",
        "page",
        "source",
        "text",
        "search_text",
    ):
        if key in record and record[key] is not None:
            props[key] = record[key]
    if "search_text" not in props:
        props["search_text"] = record["text"]
    return props


def upsert_object(collection, record: dict) -> None:
    chunk_id = record["chunk_id"]
    obj_id = uuid.uuid5(uuid.NAMESPACE_URL, chunk_id)
    props = build_properties(record)
    if collection.data.exists(obj_id):
        collection.data.replace(uuid=obj_id, properties=props)
    else:
        try:
            collection.data.insert(properties=props, uuid=obj_id)
        except UnexpectedStatusCodeError:
            collection.data.replace(uuid=obj_id, properties=props)


def main() -> int:
    load_dotenv()
    bucket = get_required_env("S3_BUCKET")
    key = get_required_env("S3_CHUNKS_KEY")
    collection_name = get_required_env("WEAVIATE_COLLECTION")
    weaviate_url = get_required_env("WEAVIATE_URL")
    weaviate_key = get_required_env("WEAVIATE_API_KEY")

    client = connect_client(weaviate_url, weaviate_key)
    try:
        if not client.collections.exists(collection_name):
            raise ValueError(f"Weaviate collection not found: {collection_name}")
        collection = client.collections.get(collection_name)
        count = 0
        for record in load_jsonl_from_s3(bucket, key):
            upsert_object(collection, record)
            count += 1
    finally:
        client.close()

    print(f"Ingested {count} chunks into {collection_name}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ingest failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
