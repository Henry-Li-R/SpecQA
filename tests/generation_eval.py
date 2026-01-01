import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(REPO_ROOT.as_posix())

from src.pipeline import run_pipeline as rp


DEFAULT_EVAL_PATH = REPO_ROOT / "tests/generation_eval.jsonl"


def load_env_key() -> str:
    load_dotenv()
    return os.environ.get("OPENAI_API_KEY", "")


def load_items(path: Path) -> List[Dict]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        items.append(json.loads(line))
    return items


def keyword_recall(answer: str, keywords: List[str]) -> float:
    if not keywords:
        return 0.0
    answer_lower = answer.lower()
    hits = sum(1 for k in keywords if k.lower() in answer_lower)
    return hits / len(keywords)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LLM generation quality.")
    parser.add_argument("--eval-path", default=str(DEFAULT_EVAL_PATH))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--print-details", action="store_true")
    args = parser.parse_args()

    eval_path = Path(args.eval_path)
    if not eval_path.exists():
        raise FileNotFoundError(f"Missing eval file: {eval_path}")

    api_key = load_env_key()
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY in environment or .env")

    items = load_items(eval_path)
    if args.limit:
        items = items[: args.limit]

    total = len(items)
    answered = 0
    citation_valid = 0
    citation_in_gold = 0
    keyword_recall_sum = 0.0
    keyword_recall_count = 0

    must_abstain_total = 0
    abstained_total = 0
    abstained_on_must = 0

    for item in items:
        query = item.get("query", "")
        gold_citations = set(item.get("gold_citations", []))
        gold_keywords = item.get("gold_answer_keywords", [])
        must_abstain = bool(item.get("must_abstain", False))

        result = rp.run_pipeline(
            query,
            api_key=api_key,
            abstain_on_invalid=False,
        )

        abstained = bool(result.get("abstain", False))
        abstained_total += 1 if abstained else 0
        if must_abstain:
            must_abstain_total += 1
            abstained_on_must += 1 if abstained else 0

        if not abstained:
            answered += 1
            errors = result.get("validation_errors", [])
            if not errors:
                citation_valid += 1
            cited_ids = {c.get("chunk_id") for c in result.get("citations", [])}
            if gold_citations and cited_ids.intersection(gold_citations):
                citation_in_gold += 1
            if gold_keywords:
                keyword_recall_sum += keyword_recall(result.get("answer", ""), gold_keywords)
                keyword_recall_count += 1

        if args.print_details:
            qid = item.get("qid", "")
            cited_ids = [c.get("chunk_id") for c in result.get("citations", [])]
            errors = result.get("validation_errors", [])
            gold_hit = bool(gold_citations and set(cited_ids).intersection(gold_citations))
            print(
                f"{qid}: abstain={abstained} citations={len(cited_ids)} "
                f"gold_hit={gold_hit} quote_valid={not errors}"
            )
            if gold_citations:
                print(f"  gold: {sorted(gold_citations)}")
            if cited_ids:
                print(f"  cited: {cited_ids}")
            if errors:
                print(f"  validation_errors: {errors}")

    citation_quote_valid_rate = citation_valid / answered if answered else 0.0
    citation_in_gold_rate = citation_in_gold / answered if answered else 0.0
    avg_keyword_recall = (
        keyword_recall_sum / keyword_recall_count if keyword_recall_count else 0.0
    )

    abstain_precision = (
        abstained_on_must / abstained_total if abstained_total else 0.0
    )
    abstain_recall = (
        abstained_on_must / must_abstain_total if must_abstain_total else 0.0
    )

    print(f"Total queries: {total}")
    print(f"Answered (non-abstain): {answered}")
    print(f"Citation quote valid rate: {citation_quote_valid_rate:.3f}")
    print(f"Citation-in-gold rate: {citation_in_gold_rate:.3f}")
    print(f"Answer keyword recall: {avg_keyword_recall:.3f}")
    print(f"Abstain precision: {abstain_precision:.3f}")
    print(f"Abstain recall: {abstain_recall:.3f}")


if __name__ == "__main__":
    main()
