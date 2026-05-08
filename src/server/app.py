"""FastAPI application for the RAG demo chat server."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.status import (
    HTTP_413_REQUEST_ENTITY_TOO_LARGE,
    HTTP_500_INTERNAL_SERVER_ERROR,
    HTTP_503_SERVICE_UNAVAILABLE,
)

from src.LLM.chatgpt_client import InputTooLargeError
from src.pipeline.resources import close_resources, get_resources, init_resources, is_initialised
from src.pipeline.run_pipeline import run_pipeline

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parents[2] / "static"
INDEX_FILE = STATIC_DIR / "chat.html"

MAX_BODY_BYTES = int(os.environ.get("CHAT_MAX_BODY_BYTES", str(8 * 1024)))
RATE_LIMIT_RPM = int(os.environ.get("CHAT_RATE_LIMIT_RPM", "60"))

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO)
    logger.info("starting up — loading pipeline resources")
    await asyncio.to_thread(init_resources)
    logger.info("startup complete")
    yield
    logger.info("shutting down — releasing pipeline resources")
    close_resources()


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[f"{RATE_LIMIT_RPM}/minute"],
    headers_enabled=True,  # adds Retry-After / RateLimit-* headers on 429
)

# ---------------------------------------------------------------------------
# Body-size middleware
# ---------------------------------------------------------------------------

class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests whose body exceeds MAX_BODY_BYTES.

    Fast path: reject on Content-Length header before reading any bytes.
    Fallback: read the actual body (Starlette caches it for the handler)
    to catch chunked requests and clients that lie about Content-Length.
    """

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length is not None and int(content_length) > MAX_BODY_BYTES:
            return JSONResponse(
                status_code=HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={"error": f"Request body exceeds {MAX_BODY_BYTES} byte limit."},
            )
        body = await request.body()
        if len(body) > MAX_BODY_BYTES:
            return JSONResponse(
                status_code=HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={"error": f"Request body exceeds {MAX_BODY_BYTES} byte limit."},
            )
        return await call_next(request)


# ---------------------------------------------------------------------------
# App assembly
# ---------------------------------------------------------------------------

app = FastAPI(lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(BodySizeLimitMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    query: str
    api_key: Optional[str] = None
    abstain_on_invalid: bool = True

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query must not be empty")
        return v


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/healthz")
async def healthz():
    if not is_initialised():
        return JSONResponse(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            content={"ok": False, "reason": "resources not yet initialised"},
        )
    return {"ok": True}


@app.get("/")
@app.get("/chat.html")
async def index():
    if not INDEX_FILE.exists():
        return JSONResponse(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "Missing static/chat.html"},
        )
    return FileResponse(INDEX_FILE, media_type="text/html")


@app.post("/api/chat")
@limiter.limit(f"{RATE_LIMIT_RPM}/minute")
async def chat(request: Request, response: Response, body: ChatRequest):
    resources = get_resources()
    try:
        result = await asyncio.to_thread(
            run_pipeline,
            body.query,
            api_key=body.api_key,
            abstain_on_invalid=body.abstain_on_invalid,
            resources=resources,
        )
    except InputTooLargeError as exc:
        return JSONResponse(
            status_code=HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            content={"error": str(exc)},
        )
    except Exception:
        logger.exception("pipeline error for query %r", body.query)
        return JSONResponse(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "Internal server error"},
        )
    return result
