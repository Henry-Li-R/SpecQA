import unittest
from unittest.mock import MagicMock, patch

from src.pipeline import run_pipeline as rp


class TestPipeline(unittest.TestCase):
    def test_run_pipeline_returns_answer(self):
        fake_chunks = [{"chunk_id": "c1", "text": "Alpha"}]
        fake_answer = {"answer": "Alpha", "citations": [], "abstain": False}

        with patch.object(rp, "connect_client") as connect_client, \
            patch.object(rp, "retrieve_chunks", return_value=fake_chunks), \
            patch.object(rp, "answer_with_citations", return_value=fake_answer), \
            patch.object(rp, "SentenceTransformer") as embedder_cls, \
            patch.object(rp, "CrossEncoder") as reranker_cls:
            connect_client.return_value = MagicMock()
            embedder = MagicMock()
            embedder_cls.return_value = embedder
            reranker_cls.return_value = MagicMock()

            result = rp.run_pipeline("test query")

        self.assertEqual(result["answer"], "Alpha")
        self.assertFalse(result["abstain"])

    def test_run_pipeline_passes_chunks_to_llm(self):
        fake_chunks = [{"chunk_id": "c1", "text": "Alpha"}]
        fake_answer = {"answer": "", "citations": [], "abstain": True}

        with patch.object(rp, "connect_client") as connect_client, \
            patch.object(rp, "retrieve_chunks", return_value=fake_chunks), \
            patch.object(rp, "answer_with_citations", return_value=fake_answer) as awc, \
            patch.object(rp, "SentenceTransformer"), \
            patch.object(rp, "CrossEncoder"):
            connect_client.return_value = MagicMock()

            rp.run_pipeline("test query")

        awc.assert_called()
        args, kwargs = awc.call_args
        self.assertEqual(kwargs["retrieved_chunks"], fake_chunks)


if __name__ == "__main__":
    unittest.main()
