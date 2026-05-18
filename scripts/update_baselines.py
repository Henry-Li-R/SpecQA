#!/usr/bin/env python3
"""
Advance eval/baselines.json from a fresh full eval run.
Run this after confirming that current system performance is acceptable.

    make update-baselines
"""
import json
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(REPO_ROOT.as_posix())

BASELINES_PATH = REPO_ROOT / "eval" / "baselines.json"


def main():
    from dotenv import load_dotenv
    load_dotenv()
    load_dotenv(REPO_ROOT / ".env.development", override=True)

    print("Running retrieval eval — datasheet...")
    try:
        from tests.retrieval_binary_recall_datasheet import run_eval as ds_eval
        rd = ds_eval()
    except Exception:
        traceback.print_exc()
        sys.exit(1)

    print("Running retrieval eval — manual...")
    try:
        from tests.retrieval_binary_recall_manual import run_eval as man_eval
        rm = man_eval()
    except Exception:
        traceback.print_exc()
        sys.exit(1)

    print("Running generation eval...")
    try:
        from tests.generation_eval import run_eval as gen_eval
        gen = gen_eval()
    except Exception:
        traceback.print_exc()
        sys.exit(1)

    baselines = {
        "_note": "Update via: make update-baselines",
        "tolerance": 0.05,
        "retrieval_datasheet": {
            "recall_at_5": round(rd["recall_at_5"], 4),
        },
        "retrieval_manual": {
            "recall_at_5": round(rm["recall_at_5"], 4),
        },
        "generation": {
            "citation_quote_validity": round(gen["citation_quote_validity"], 4),
            "answer_keyword_recall":   round(gen["answer_keyword_recall"], 4),
            "abstain_recall":          round(gen["abstain_recall"], 4),
            "abstain_precision":       round(gen["abstain_precision"], 4),
        },
    }

    BASELINES_PATH.write_text(json.dumps(baselines, indent=2) + "\n")
    print(f"\nBaselines written to {BASELINES_PATH}:")
    print(json.dumps(baselines, indent=2))


if __name__ == "__main__":
    main()
