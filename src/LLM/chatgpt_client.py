import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

class AnswerCitation(BaseModel):
    chunk_id: str
    quote: str


class AnswerOutput(BaseModel):
    answer: str
    citations: List[AnswerCitation]
    abstain: bool


DEFAULT_SYSTEM_PROMPT = (
    "You answer only using the provided chunks.\n"
    "If the answer is not directly supported, abstain (say something like, Cannot answer from provided documents.).\n"
    "Each citation must be an **EXACT SUBSTRING** of a chunk.\n"
    "- Do not reorder sentences, change wording, or change punctuation in citations.\n"
    "- **DO NOT OMIT TEXT IN THE MIDDLE OF A CITATION**.\n"
)


def format_retrieved_chunks(chunks: Iterable[Dict[str, Any]]) -> str:
    parts = []
    for idx, chunk in enumerate(chunks, start=1):
        parts.append(f"--- CHUNK {idx} ---")
        parts.append(f"chunk_id: {chunk.get('chunk_id', '')}")
        parts.append(f"doc_id: {chunk.get('doc_id', '')}")
        parts.append(f"page: {chunk.get('page', '')}")
        if "chapter" in chunk:
            parts.append(f"chapter: {chunk.get('chapter', '')}")
        parts.append(f"section: {chunk.get('section', '')}")
        parts.append("text:")
        parts.append(chunk.get("text", ""))
    return "\n".join(parts)


def build_messages(
    system_prompt: str,
    user_query: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, str]]:
    messages = [{"role": "system", "content": system_prompt}]
    if chat_history:
        messages.extend(chat_history)

    retrieved_text = ""
    if retrieved_chunks:
        retrieved_text = format_retrieved_chunks(retrieved_chunks)

    user_content = f"Query:\n{user_query}"
    if retrieved_text:
        user_content += f"\n\nRetrieved Chunks:\n{retrieved_text}"

    messages.append({"role": "user", "content": user_content})
    return messages


def send_chatgpt_request(
    messages: List[Dict[str, str]],
    model: str = "gpt-4o-2024-08-06",
    api_key: Optional[str] = None,
    temperature: float = 0.0,
) -> AnswerOutput:
    load_dotenv()
    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise ValueError("Missing OPENAI_API_KEY or api_key argument.")

    client = OpenAI(api_key=key)
    resp = client.responses.parse(
        model=model,
        input=messages,
        temperature=temperature,
        text_format=AnswerOutput,
    )
    return resp.output_parsed


def validate_citations(
    answer_json: Dict[str, Any],
    retrieved_chunks: List[Dict[str, Any]],
) -> Tuple[bool, List[str]]:
    errors = []
    citations = answer_json.get("citations", [])
    if not isinstance(citations, list):
        return False, ["citations must be a list"]

    chunk_text_by_id = {c.get("chunk_id"): c.get("text", "") for c in retrieved_chunks}
    normalized_chunk_text_by_id = {
        k: re.sub(r"\s+", " ", v).strip() for k, v in chunk_text_by_id.items()
    }

    for idx, citation in enumerate(citations, start=1):
        chunk_id = citation.get("chunk_id")
        quote = citation.get("quote", "")
        if not chunk_id or chunk_id not in chunk_text_by_id:
            errors.append(f"citation {idx} has unknown chunk_id")
            continue
        if not quote:
            errors.append(f"citation {idx} quote not found in chunk")
            continue
        trimmed_quote = quote.rstrip(".")
        if trimmed_quote in chunk_text_by_id.get(chunk_id, ""):
            continue
        normalized_quote = re.sub(r"\s+", " ", trimmed_quote).strip()
        if normalized_quote not in normalized_chunk_text_by_id.get(chunk_id, ""):
            errors.append(f"citation {idx} quote not found in chunk")

    return len(errors) == 0, errors


def normalize_answer(
    answer_json: Dict[str, Any],
    retrieved_chunks: List[Dict[str, Any]],
    abstain_on_invalid: bool = True,
) -> Dict[str, Any]:
    normalized = {
        "answer": answer_json.get("answer", ""),
        "citations": answer_json.get("citations", []),
        "abstain": bool(answer_json.get("abstain", False)),
    }

    ok, errors = validate_citations(normalized, retrieved_chunks)
    if ok:
        return normalized

    if abstain_on_invalid:
        return {
            "answer": "Error: citations are invalid.",
            "citations": [],
            "abstain": True,
        }

    normalized["validation_errors"] = errors
    return normalized


def answer_with_citations(
    user_query: str,
    retrieved_chunks: List[Dict[str, Any]],
    chat_history: Optional[List[Dict[str, str]]] = None,
    system_prompt: Optional[str] = None,
    model: str = "gpt-4o-2024-08-06",
    temperature: float = 0.0,
    abstain_on_invalid: bool = True,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    load_dotenv()
    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise ValueError("Missing OPENAI_API_KEY or api_key argument.")

    prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    messages = build_messages(
        system_prompt=prompt,
        user_query=user_query,
        chat_history=chat_history,
        retrieved_chunks=retrieved_chunks,
    )
    parsed = send_chatgpt_request(
        messages=messages,
        model=model,
        temperature=temperature,
        api_key=key,
    )
    answer_json = parsed.model_dump()
    return normalize_answer(
        answer_json,
        retrieved_chunks,
        abstain_on_invalid=abstain_on_invalid,
    )
