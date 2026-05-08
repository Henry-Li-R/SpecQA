import unittest
from unittest.mock import MagicMock, patch

# Import the app module explicitly so patch.object can reference it directly
# and so the TestClient can import it without triggering real model loads.
import src.server.app as server_app
from fastapi.testclient import TestClient

from src.LLM.chatgpt_client import InputTooLargeError

_fake_resources = MagicMock()


class TestChatServer(unittest.TestCase):
    def setUp(self):
        # Patches must be active before TestClient is constructed so that the
        # lifespan calls our no-op init_resources instead of the real one.
        self._patches = [
            patch.object(server_app, "init_resources"),
            patch.object(server_app, "close_resources"),
            patch.object(server_app, "is_initialised", return_value=True),
            patch.object(server_app, "get_resources", return_value=_fake_resources),
        ]
        for p in self._patches:
            p.start()
        # Reset per-IP counters so rate-limit state doesn't leak between tests.
        server_app.limiter._storage.reset()
        self.client = TestClient(server_app.app, raise_server_exceptions=False)

    def tearDown(self):
        for p in self._patches:
            p.stop()

    # ------------------------------------------------------------------
    # /healthz
    # ------------------------------------------------------------------

    def test_healthz_ok(self):
        resp = self.client.get("/healthz")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["ok"])

    def test_healthz_not_ready(self):
        with patch.object(server_app, "is_initialised", return_value=False):
            resp = self.client.get("/healthz")
        self.assertEqual(resp.status_code, 503)

    # ------------------------------------------------------------------
    # Body size limit
    # ------------------------------------------------------------------

    def test_body_too_large_via_content_length_returns_413(self):
        large_body = b"x" * (9 * 1024)  # 9 KiB > 8 KiB default
        resp = self.client.post(
            "/api/chat",
            content=large_body,
            headers={"Content-Type": "application/json", "Content-Length": str(len(large_body))},
        )
        self.assertEqual(resp.status_code, 413)

    def test_body_too_large_without_content_length_returns_413(self):
        # Simulate a chunked client that omits Content-Length.
        # httpx sends a generator as chunked transfer, so no Content-Length header
        # reaches the middleware; the actual-bytes check catches it instead.
        large_body = b"x" * (9 * 1024)
        resp = self.client.post(
            "/api/chat",
            content=iter([large_body]),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(resp.status_code, 413)

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def test_rate_limit_returns_429_after_burst(self):
        # Send enough requests to exhaust the 60/minute limit. Storage was
        # reset in setUp so no prior-test budget is consumed.
        fake_answer = {"answer": "x", "citations": [], "abstain": False}
        limit = server_app.RATE_LIMIT_RPM
        with patch.object(server_app, "run_pipeline", return_value=fake_answer):
            responses = [
                self.client.post("/api/chat", json={"query": "x"})
                for _ in range(limit + 10)
            ]
        statuses = [r.status_code for r in responses]
        self.assertIn(429, statuses, "expected at least one 429 after exhausting rate limit")
        last_429 = next(r for r in reversed(responses) if r.status_code == 429)
        self.assertIn("Retry-After", last_429.headers)

    # ------------------------------------------------------------------
    # /api/chat validation
    # ------------------------------------------------------------------

    def test_missing_query_returns_422(self):
        resp = self.client.post("/api/chat", json={})
        self.assertEqual(resp.status_code, 422)

    def test_empty_query_returns_422(self):
        resp = self.client.post("/api/chat", json={"query": "   "})
        self.assertEqual(resp.status_code, 422)

    def test_malformed_json_returns_422(self):
        resp = self.client.post(
            "/api/chat",
            content=b"{bad json",
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(resp.status_code, 422)

    # ------------------------------------------------------------------
    # /api/chat happy path
    # ------------------------------------------------------------------

    def test_happy_path_calls_run_pipeline_with_resources(self):
        fake_answer = {"answer": "42", "citations": [], "abstain": False}
        with patch.object(server_app, "run_pipeline", return_value=fake_answer) as mock_pipeline:
            resp = self.client.post("/api/chat", json={"query": "What is the voltage range?"})

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["answer"], "42")
        mock_pipeline.assert_called_once()
        _, kwargs = mock_pipeline.call_args
        self.assertIs(kwargs["resources"], _fake_resources)

    # ------------------------------------------------------------------
    # Token budget exceeded
    # ------------------------------------------------------------------

    def test_input_too_large_returns_413(self):
        with patch.object(server_app, "run_pipeline", side_effect=InputTooLargeError("too big")):
            resp = self.client.post("/api/chat", json={"query": "x"})
        self.assertEqual(resp.status_code, 413)
        self.assertIn("too big", resp.json()["error"])

    # ------------------------------------------------------------------
    # Unexpected pipeline error
    # ------------------------------------------------------------------

    def test_pipeline_exception_returns_500(self):
        with patch.object(server_app, "run_pipeline", side_effect=RuntimeError("boom")):
            resp = self.client.post("/api/chat", json={"query": "x"})
        self.assertEqual(resp.status_code, 500)


if __name__ == "__main__":
    unittest.main()
