"""Process-wide singletons for the RAG pipeline.

`init_resources()` is called once at server startup so models, the Weaviate
client, and the Phoenix tracer are loaded a single time per process. The
legacy code path in `run_pipeline.run_pipeline()` (with `resources=None`)
still works for the CLI and unit tests.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Optional

import weaviate
from sentence_transformers import CrossEncoder, SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class PipelineResources:
    embedder: SentenceTransformer
    reranker: Optional[CrossEncoder]
    weaviate_client: weaviate.WeaviateClient
    tracer: Any  # Phoenix tracer or None


_singleton: Optional[PipelineResources] = None
_lock = threading.Lock()


def init_resources() -> PipelineResources:
    """Eagerly load models, open the Weaviate client, register Phoenix.

    Idempotent: a second call returns the existing singleton without
    re-loading models or re-registering the tracer provider.
    """
    global _singleton
    with _lock:
        if _singleton is not None:
            return _singleton

        # Lazy import to avoid a circular dependency: run_pipeline imports
        # PipelineResources from this module for type hints.
        from src.pipeline.run_pipeline import (
            MODEL_NAME,
            RERANK_ENABLED,
            RERANK_MODEL_NAME,
            connect_client,
            setup_tracing,
        )

        logger.info("loading embedding model %s", MODEL_NAME)
        embedder = SentenceTransformer(MODEL_NAME)
        embedder.max_seq_length = 512

        reranker = None
        if RERANK_ENABLED:
            logger.info("loading reranker %s", RERANK_MODEL_NAME)
            reranker = CrossEncoder(RERANK_MODEL_NAME)

        logger.info("connecting to Weaviate")
        client = connect_client()

        tracer = setup_tracing()

        _singleton = PipelineResources(
            embedder=embedder,
            reranker=reranker,
            weaviate_client=client,
            tracer=tracer,
        )
        logger.info("pipeline resources ready")
        return _singleton


def get_resources() -> PipelineResources:
    if _singleton is None:
        raise RuntimeError(
            "PipelineResources not initialised; call init_resources() first."
        )
    return _singleton


def is_initialised() -> bool:
    return _singleton is not None


def close_resources() -> None:
    """Close the Weaviate client and clear the singleton. Idempotent."""
    global _singleton
    with _lock:
        if _singleton is None:
            return
        try:
            _singleton.weaviate_client.close()
        except Exception:  # noqa: BLE001 - shutdown path, log and move on
            logger.exception("error closing Weaviate client")
        _singleton = None
        logger.info("pipeline resources closed")
