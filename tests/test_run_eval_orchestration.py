#!/usr/bin/env python3
"""
Unit tests for scripts/run_eval.py orchestration logic.

Approach: shadow the real tests/ package via PYTHONPATH with mock modules
that return canned results (no Weaviate, no OpenAI).  EVAL_BASELINES_PATH
and EVAL_RESULTS_DIR env vars redirect all file I/O to an isolated tmpdir.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_EVAL = str(REPO_ROOT / "scripts" / "run_eval.py")

# ---------------------------------------------------------------------------
# Canned results returned by mock eval modules
# ---------------------------------------------------------------------------

_DS_RESULT = {
    "recall_at_1": 0.90,
    "recall_at_5": 0.95,
    "n_queries": 20,
    "misses": [],
    "per_query": [],
}
_MAN_RESULT = {
    "recall_at_1": 0.85,
    "recall_at_5": 0.95,
    "n_queries": 80,
    "misses": [],
    "per_query": [],
}
_GEN_RESULT = {
    "answer_rate": 0.90,
    "citation_quote_validity": 0.95,
    "citation_in_gold": 0.90,
    "answer_keyword_recall": 0.90,
    "abstain_precision": 1.00,
    "abstain_recall": 0.90,
    "n_queries": 20,
    "per_query": [],
}

# Baselines where every mock result comfortably passes (threshold = baseline - 0.05)
_PASSING_BASELINES = {
    "tolerance": 0.05,
    "retrieval_datasheet": {"recall_at_5": 0.90},  # threshold=0.85, mock=0.95 ✓
    "retrieval_manual":    {"recall_at_5": 0.90},  # threshold=0.85, mock=0.95 ✓
    "generation": {
        "citation_quote_validity": 0.85,           # threshold=0.80, mock=0.95 ✓
        "answer_keyword_recall":   0.85,           # threshold=0.80, mock=0.90 ✓
        "abstain_recall":          0.85,           # threshold=0.80, mock=0.90 ✓
        "abstain_precision":       0.90,           # threshold=0.85, mock=1.00 ✓
    },
}


def _mock_src(result_dict: dict) -> str:
    encoded = repr(json.dumps(result_dict))
    return textwrap.dedent(f"""\
        import json as _j
        _R = _j.loads({encoded})
        def run_eval(**kwargs):
            return _R
    """)


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

class _Harness:
    """Isolated tmpdir with mock tests/ package and empty results/."""

    def __init__(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.baselines_path = self.tmpdir / "baselines.json"
        self.results_dir = self.tmpdir / "results"
        self.results_dir.mkdir()

        tests_pkg = self.tmpdir / "tests"
        tests_pkg.mkdir()
        (tests_pkg / "__init__.py").write_text("")
        (tests_pkg / "retrieval_binary_recall_datasheet.py").write_text(_mock_src(_DS_RESULT))
        (tests_pkg / "retrieval_binary_recall_manual.py").write_text(_mock_src(_MAN_RESULT))
        (tests_pkg / "generation_eval.py").write_text(_mock_src(_GEN_RESULT))

    def run(self, *args, baselines=None, extra_env=None):
        if baselines is not None:
            self.baselines_path.write_text(json.dumps(baselines))
        elif self.baselines_path.exists():
            self.baselines_path.unlink()

        pythonpath = str(self.tmpdir)
        if "PYTHONPATH" in os.environ:
            pythonpath += f":{os.environ['PYTHONPATH']}"

        env = {
            **os.environ,
            "PYTHONPATH": pythonpath,
            "EVAL_BASELINES_PATH": str(self.baselines_path),
            "EVAL_RESULTS_DIR": str(self.results_dir),
            "PHOENIX_COLLECTOR_ENDPOINT": "",
        }
        if extra_env:
            env.update(extra_env)

        return subprocess.run(
            [sys.executable, RUN_EVAL, *args],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(REPO_ROOT),
        )

    def result_files(self):
        return list(self.results_dir.glob("*.json"))

    def cleanup(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------

class TestBootstrap(unittest.TestCase):
    def setUp(self):    self.h = _Harness()
    def tearDown(self): self.h.cleanup()

    def test_no_baselines_exits_0(self):
        """No baselines.json → exits 0 (never blocks, prints guidance)."""
        proc = self.h.run()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("No baselines", proc.stdout)

    def test_no_baselines_still_writes_result(self):
        """Even without baselines the results JSON is persisted."""
        self.h.run()
        self.assertEqual(len(self.h.result_files()), 1)


class TestRegressionDetection(unittest.TestCase):
    def setUp(self):    self.h = _Harness()
    def tearDown(self): self.h.cleanup()

    def test_all_pass_exits_0(self):
        proc = self.h.run(baselines=_PASSING_BASELINES)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("PASS", proc.stdout)
        self.assertNotIn("FAIL", proc.stdout)

    def test_regression_exits_1(self):
        """One metric strictly below threshold → exit 1."""
        # DS baseline=1.05 → threshold=1.00, mock DS recall=0.95 → FAIL
        failing = {**_PASSING_BASELINES, "retrieval_datasheet": {"recall_at_5": 1.05}}
        proc = self.h.run(baselines=failing)
        self.assertEqual(proc.returncode, 1, proc.stderr)
        self.assertIn("FAIL", proc.stdout)

    def test_exactly_at_threshold_passes(self):
        """current == baseline − tolerance is a PASS (>= comparison)."""
        # mock DS=0.95; baseline=1.00 → threshold=0.95 exactly → PASS
        boundary = {**_PASSING_BASELINES, "retrieval_datasheet": {"recall_at_5": 1.00}}
        proc = self.h.run(baselines=boundary)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_one_below_threshold_fails(self):
        """current 0.001 below threshold → exit 1."""
        # mock DS=0.95; baseline=1.001 → threshold=0.951 > 0.95 → FAIL
        over = {**_PASSING_BASELINES, "retrieval_datasheet": {"recall_at_5": 1.001}}
        proc = self.h.run(baselines=over)
        self.assertEqual(proc.returncode, 1, proc.stderr)

    def test_failing_metric_named_in_output(self):
        """The name of the failing metric appears in the FAIL line."""
        gen = {**_PASSING_BASELINES["generation"], "abstain_recall": 1.00}
        failing = {**_PASSING_BASELINES, "generation": gen}
        proc = self.h.run(baselines=failing)
        self.assertEqual(proc.returncode, 1, proc.stderr)
        self.assertIn("abstain_recall", proc.stdout)


class TestOnlyFlag(unittest.TestCase):
    def setUp(self):    self.h = _Harness()
    def tearDown(self): self.h.cleanup()

    def test_only_retrieval_omits_generation_output(self):
        proc = self.h.run("--only", "retrieval", baselines=_PASSING_BASELINES)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("Retrieval", proc.stdout)
        self.assertNotIn("Generation", proc.stdout)

    def test_only_generation_omits_retrieval_output(self):
        proc = self.h.run("--only", "generation", baselines=_PASSING_BASELINES)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("Generation", proc.stdout)
        self.assertNotIn("Datasheet", proc.stdout)
        self.assertNotIn("Manual", proc.stdout)


class TestPersistence(unittest.TestCase):
    def setUp(self):    self.h = _Harness()
    def tearDown(self): self.h.cleanup()

    def test_no_persist_writes_no_file(self):
        self.h.run("--no-persist", baselines=_PASSING_BASELINES)
        self.assertEqual(len(self.h.result_files()), 0)

    def test_default_persist_writes_result_file(self):
        self.h.run(baselines=_PASSING_BASELINES)
        files = self.h.result_files()
        self.assertEqual(len(files), 1)
        data = json.loads(files[0].read_text())
        for key in ("timestamp", "git_sha", "git_branch",
                    "retrieval_datasheet", "retrieval_manual", "generation"):
            self.assertIn(key, data)


class TestCIStepSummary(unittest.TestCase):
    def setUp(self):    self.h = _Harness()
    def tearDown(self): self.h.cleanup()

    def test_step_summary_written_and_contains_metrics(self):
        summary_path = self.h.tmpdir / "step_summary.md"
        proc = self.h.run(
            baselines=_PASSING_BASELINES,
            extra_env={"GITHUB_STEP_SUMMARY": str(summary_path)},
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue(summary_path.exists())
        content = summary_path.read_text()
        self.assertIn("recall_at_5", content)
        self.assertIn("citation_quote_validity", content)


class TestPRCISimulation(unittest.TestCase):
    """Simulate the exact CI checks that gate every PR.

    Uses the committed eval/baselines.json so the test stays in sync with
    whatever baselines are currently in the repo.  Skipped on fresh checkouts
    that have not yet run make update-baselines.
    """

    def setUp(self):
        self.h = _Harness()
        baselines_file = REPO_ROOT / "eval" / "baselines.json"
        self.real_baselines = (
            json.loads(baselines_file.read_text()) if baselines_file.exists() else None
        )

    def tearDown(self):
        self.h.cleanup()

    def _skip_if_no_baselines(self):
        if not self.real_baselines:
            self.skipTest("eval/baselines.json not present — run make update-baselines first")

    def test_pr_passes_when_no_regression(self):
        """A PR whose metrics meet current baselines exits 0."""
        self._skip_if_no_baselines()
        # Mock results (_DS_RESULT, _MAN_RESULT, _GEN_RESULT) are comfortably
        # above the real baselines (verified against actual baseline values).
        proc = self.h.run(baselines=self.real_baselines)
        self.assertEqual(proc.returncode, 0, proc.stderr + "\n" + proc.stdout)

    def test_pr_blocked_on_datasheet_retrieval_regression(self):
        """PR that drops datasheet recall@5 >0.05 is blocked (exit 1)."""
        self._skip_if_no_baselines()
        tol = self.real_baselines.get("tolerance", 0.05)
        # Set DS baseline so threshold is 0.001 above mock result → FAIL
        regressed = {
            **self.real_baselines,
            "retrieval_datasheet": {
                "recall_at_5": _DS_RESULT["recall_at_5"] + tol + 0.001
            },
        }
        proc = self.h.run(baselines=regressed)
        self.assertEqual(proc.returncode, 1, proc.stderr)
        self.assertIn("recall_at_5", proc.stdout)
        self.assertIn("datasheet", proc.stdout.lower())

    def test_pr_blocked_on_generation_regression(self):
        """PR that drops citation_quote_validity >0.05 is blocked (exit 1)."""
        self._skip_if_no_baselines()
        tol = self.real_baselines.get("tolerance", 0.05)
        gen = {
            **self.real_baselines["generation"],
            "citation_quote_validity": _GEN_RESULT["citation_quote_validity"] + tol + 0.001,
        }
        proc = self.h.run(baselines={**self.real_baselines, "generation": gen})
        self.assertEqual(proc.returncode, 1, proc.stderr)
        self.assertIn("citation_quote_validity", proc.stdout)


if __name__ == "__main__":
    unittest.main()
