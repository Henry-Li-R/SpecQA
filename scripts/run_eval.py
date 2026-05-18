#!/usr/bin/env python3
"""
Eval orchestrator: runs retrieval and generation evals, persists results,
and compares against baselines to gate CI.

Usage:
    python3 scripts/run_eval.py [--only retrieval|generation] [--no-persist]
"""
import argparse
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(REPO_ROOT.as_posix())

RESULTS_DIR = Path(os.environ.get("EVAL_RESULTS_DIR", str(REPO_ROOT / "eval" / "results")))
BASELINES_PATH = Path(os.environ.get("EVAL_BASELINES_PATH", str(REPO_ROOT / "eval" / "baselines.json")))

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"
LLM_MODEL = "gpt-4o-2024-08-06"


def _git_info():
    def run(cmd):
        try:
            return subprocess.check_output(cmd, cwd=REPO_ROOT, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return "unknown"
    return run(["git", "rev-parse", "--short", "HEAD"]), run(["git", "rev-parse", "--abbrev-ref", "HEAD"])


def _check(name, current, baseline, tolerance, violations):
    threshold = baseline - tolerance
    if current < threshold:
        violations.append((name, current, baseline, threshold))


def _print_summary(result, only, baselines=None, tolerance=0.05):
    print("\n=== Eval Summary ===")

    def fmt(val):
        return f"{val:.3f}"

    def line(name, current, baseline=None):
        if baseline is not None:
            threshold = baseline - tolerance
            mark = "PASS" if current >= threshold else "FAIL"
            print(f"  [{mark}] {name}: {fmt(current)}  (baseline={fmt(baseline)}, threshold={fmt(threshold)})")
        else:
            print(f"  [----] {name}: {fmt(current)}  (no baseline)")

    run_retrieval = only is None or only == "retrieval"
    run_gen = only is None or only == "generation"

    if run_retrieval and "retrieval_datasheet" in result:
        rd = result["retrieval_datasheet"]
        brd = (baselines or {}).get("retrieval_datasheet", {})
        print(f"\nRetrieval — Datasheet (n={rd['n_queries']}):")
        line("recall_at_5", rd["recall_at_5"], brd.get("recall_at_5"))
        print(f"         recall_at_1: {fmt(rd['recall_at_1'])}  (informational)")

    if run_retrieval and "retrieval_manual" in result:
        rm = result["retrieval_manual"]
        brm = (baselines or {}).get("retrieval_manual", {})
        print(f"\nRetrieval — Manual (n={rm['n_queries']}):")
        line("recall_at_5", rm["recall_at_5"], brm.get("recall_at_5"))
        print(f"         recall_at_1: {fmt(rm['recall_at_1'])}  (informational)")

    if run_gen and "generation" in result:
        gen = result["generation"]
        bg = (baselines or {}).get("generation", {})
        print(f"\nGeneration (n={gen['n_queries']}):")
        for metric in ("citation_quote_validity", "answer_keyword_recall",
                       "abstain_recall", "abstain_precision"):
            line(metric, gen[metric], bg.get(metric))
        print(f"         answer_rate:      {fmt(gen['answer_rate'])}  (informational)")
        print(f"         citation_in_gold: {fmt(gen['citation_in_gold'])}  (informational)")


def _write_step_summary(result, only, baselines, tolerance, violations):
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return

    lines = [
        "# Eval Results\n\n",
        "| Metric | Value | Baseline | Threshold | Status |\n",
        "|--------|-------|----------|-----------|--------|\n",
    ]

    def row(name, current, baseline, tol):
        threshold = baseline - tol
        status = "✅ PASS" if current >= threshold else "❌ FAIL"
        lines.append(f"| {name} | {current:.3f} | {baseline:.3f} | {threshold:.3f} | {status} |\n")

    run_retrieval = only is None or only == "retrieval"
    run_gen = only is None or only == "generation"

    if run_retrieval and "retrieval_datasheet" in result:
        rd = result["retrieval_datasheet"]
        brd = baselines.get("retrieval_datasheet", {})
        row("recall_at_5 (datasheet)", rd["recall_at_5"], brd.get("recall_at_5", 0.0), tolerance)

    if run_retrieval and "retrieval_manual" in result:
        rm = result["retrieval_manual"]
        brm = baselines.get("retrieval_manual", {})
        row("recall_at_5 (manual)", rm["recall_at_5"], brm.get("recall_at_5", 0.0), tolerance)

    if run_gen and "generation" in result:
        gen = result["generation"]
        bg = baselines.get("generation", {})
        for metric in ("citation_quote_validity", "answer_keyword_recall",
                       "abstain_recall", "abstain_precision"):
            row(metric, gen[metric], bg.get(metric, 0.0), tolerance)

    if violations:
        lines.append(f"\n**{len(violations)} metric(s) below threshold — PR blocked.**\n")

    with open(summary_path, "a", encoding="utf-8") as f:
        f.writelines(lines)


def main():
    parser = argparse.ArgumentParser(description="Run evals and compare against baselines.")
    parser.add_argument("--only", choices=["retrieval", "generation"],
                        help="Run only retrieval or only generation eval.")
    parser.add_argument("--no-persist", action="store_true",
                        help="Skip writing results to eval/results/.")
    args = parser.parse_args()

    load_dotenv()
    load_dotenv(REPO_ROOT / ".env.development", override=True)
    sha, branch = _git_info()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": sha,
        "git_branch": branch,
        "embedding_model": EMBEDDING_MODEL,
        "reranker_model": RERANKER_MODEL,
        "llm_model": LLM_MODEL,
    }

    run_retrieval = args.only is None or args.only == "retrieval"
    run_gen = args.only is None or args.only == "generation"

    if run_retrieval:
        print("Running retrieval eval — datasheet...")
        try:
            from tests.retrieval_binary_recall_datasheet import run_eval as ds_eval
            result["retrieval_datasheet"] = ds_eval()
        except Exception:
            traceback.print_exc()
            sys.exit(1)

        print("Running retrieval eval — manual...")
        try:
            from tests.retrieval_binary_recall_manual import run_eval as man_eval
            result["retrieval_manual"] = man_eval()
        except Exception:
            traceback.print_exc()
            sys.exit(1)

    if run_gen:
        print("Running generation eval...")
        try:
            from tests.generation_eval import run_eval as gen_eval
            result["generation"] = gen_eval()
        except Exception:
            traceback.print_exc()
            sys.exit(1)

    if not args.no_persist:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / f"{timestamp}_{sha}.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(f"Results written to {out_path}")

    if not BASELINES_PATH.exists():
        print("\nNo baselines found — run `make update-baselines` to establish them.")
        _print_summary(result, args.only)
        sys.exit(0)

    baselines = json.loads(BASELINES_PATH.read_text())
    tolerance = baselines.get("tolerance", 0.05)
    violations = []

    if run_retrieval and "retrieval_datasheet" in result:
        rd = result["retrieval_datasheet"]
        brd = baselines.get("retrieval_datasheet", {})
        _check("recall_at_5 (datasheet)", rd.get("recall_at_5", 0.0),
               brd.get("recall_at_5", 0.0), tolerance, violations)

    if run_retrieval and "retrieval_manual" in result:
        rm = result["retrieval_manual"]
        brm = baselines.get("retrieval_manual", {})
        _check("recall_at_5 (manual)", rm.get("recall_at_5", 0.0),
               brm.get("recall_at_5", 0.0), tolerance, violations)

    if run_gen and "generation" in result:
        gen = result["generation"]
        bg = baselines.get("generation", {})
        for metric in ("citation_quote_validity", "answer_keyword_recall",
                       "abstain_recall", "abstain_precision"):
            _check(metric, gen.get(metric, 0.0), bg.get(metric, 0.0), tolerance, violations)

    _print_summary(result, args.only, baselines, tolerance)
    _write_step_summary(result, args.only, baselines, tolerance, violations)

    if violations:
        print(f"\nFAIL: {len(violations)} metric(s) below threshold:")
        for name, current, baseline, threshold in violations:
            print(f"  {name}: {current:.3f} < {threshold:.3f}  (baseline={baseline:.3f})")
        sys.exit(1)

    print("\nPASS: All enforced metrics at or above threshold.")
    sys.exit(0)


if __name__ == "__main__":
    main()
