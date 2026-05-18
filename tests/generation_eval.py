import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

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


def run_eval(eval_path: Optional[Path] = None, limit: int = 0) -> Dict:
    path = Path(eval_path) if eval_path else DEFAULT_EVAL_PATH
    if not path.exists():
        raise FileNotFoundError(f"Missing eval file: {path}")

    api_key = load_env_key()
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY in environment or .env")

    items = load_items(path)
    if limit:
        items = items[:limit]

    total = len(items)
    answered = 0
    citation_valid_count = 0
    citation_in_gold_count = 0
    keyword_recall_sum = 0.0
    keyword_recall_count = 0
    must_abstain_total = 0
    abstained_total = 0
    abstained_on_must = 0
    per_query: List[Dict] = []

    for item in items:
        qid = item.get("qid", "")
        query = item.get("query", "")
        gold_citations = set(item.get("gold_citations", []))
        gold_keywords = item.get("gold_answer_keywords", [])
        must_abstain = bool(item.get("must_abstain", False))

        result = rp.run_pipeline(query, api_key=api_key, abstain_on_invalid=False)

        abstained = bool(result.get("abstain", False))
        abstained_total += 1 if abstained else 0
        if must_abstain:
            must_abstain_total += 1
            abstained_on_must += 1 if abstained else 0

        citations = result.get("citations", [])
        cited_ids = {c.get("chunk_id") for c in citations}
        errors = result.get("validation_errors", [])

        # citation_quote_validity: non-abstain answer must have ≥1 citation AND no validation errors.
        # An answer with zero citations scores 0.0 (not a vacuous pass).
        query_citation_valid = False
        if not abstained:
            answered += 1
            if citations and not errors:
                citation_valid_count += 1
                query_citation_valid = True
            # citation_in_gold: at least one cited chunk matches gold
            if gold_citations and cited_ids.intersection(gold_citations):
                citation_in_gold_count += 1
            if gold_keywords:
                keyword_recall_sum += keyword_recall(result.get("answer", ""), gold_keywords)
                keyword_recall_count += 1

        per_query.append({
            "qid": qid,
            "query": query,
            "abstained": abstained,
            "must_abstain": must_abstain,
            "citation_valid": query_citation_valid,
            "cited_ids": sorted(cited_ids),
            "gold_citations": sorted(gold_citations),
            "validation_errors": errors,
            "answer_snippet": result.get("answer", "")[:200],
        })

    citation_quote_valid_rate = citation_valid_count / answered if answered else 0.0
    citation_in_gold_rate = citation_in_gold_count / answered if answered else 0.0
    avg_keyword_recall = keyword_recall_sum / keyword_recall_count if keyword_recall_count else 0.0
    answer_rate = answered / total if total else 0.0
    abstain_precision = abstained_on_must / abstained_total if abstained_total else 0.0
    abstain_recall = abstained_on_must / must_abstain_total if must_abstain_total else 0.0

    return {
        "answer_rate": answer_rate,
        "citation_quote_validity": citation_quote_valid_rate,
        "citation_in_gold": citation_in_gold_rate,
        "answer_keyword_recall": avg_keyword_recall,
        "abstain_precision": abstain_precision,
        "abstain_recall": abstain_recall,
        "n_queries": total,
        "per_query": per_query,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LLM generation quality.")
    parser.add_argument("--eval-path", default=str(DEFAULT_EVAL_PATH))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--print-details", action="store_true")
    args = parser.parse_args()

    result = run_eval(eval_path=Path(args.eval_path), limit=args.limit)

    print(f"Total queries: {result['n_queries']}")
    print(f"Answered (non-abstain): {round(result['answer_rate'] * result['n_queries'])}")
    print(f"Citation quote valid rate: {result['citation_quote_validity']:.3f}")
    print(f"Citation-in-gold rate:     {result['citation_in_gold']:.3f}")
    print(f"Answer keyword recall:     {result['answer_keyword_recall']:.3f}")
    print(f"Abstain precision:         {result['abstain_precision']:.3f}")
    print(f"Abstain recall:            {result['abstain_recall']:.3f}")

    if args.print_details:
        for pq in result["per_query"]:
            print(
                f"{pq['qid']}: abstain={pq['abstained']} "
                f"citation_valid={pq['citation_valid']} "
                f"gold_hit={bool(set(pq['cited_ids']).intersection(pq['gold_citations']))}"
            )
            if pq["gold_citations"]:
                print(f"  gold: {pq['gold_citations']}")
            if pq["cited_ids"]:
                print(f"  cited: {pq['cited_ids']}")
            if pq["validation_errors"]:
                print(f"  errors: {pq['validation_errors']}")


if __name__ == "__main__":
    main()
