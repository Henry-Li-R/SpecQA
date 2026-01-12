import unittest

from src.LLM import chatgpt_client as cc


class TestChatGPTClient(unittest.TestCase):
    def test_format_retrieved_chunks_includes_fields(self):
        chunks = [
            {
                "chunk_id": "doc:p1:s1:0",
                "doc_id": "doc",
                "page": 1,
                "section": "SECTION",
                "text": "Some text.",
            }
        ]
        rendered = cc.format_retrieved_chunks(chunks)
        self.assertIn("chunk_id: doc:p1:s1:0", rendered)
        self.assertIn("doc_id: doc", rendered)
        self.assertIn("page: 1", rendered)
        self.assertIn("section: SECTION", rendered)
        self.assertIn("Some text.", rendered)

    def test_validate_citations_happy_path(self):
        chunks = [
            {"chunk_id": "c1", "text": "Alpha Beta Gamma"},
            {"chunk_id": "c2", "text": "Delta Epsilon"},
        ]
        answer = {
            "answer": "Alpha",
            "citations": [{"chunk_id": "c1", "quote": "Beta"}],
            "abstain": False,
            "abstain_reason": "",
        }
        ok, errors = cc.validate_citations(answer, chunks)
        self.assertTrue(ok)
        self.assertEqual(errors, [])

    def test_validate_citations_missing_quote(self):
        chunks = [{"chunk_id": "c1", "text": "Alpha Beta Gamma"}]
        answer = {
            "answer": "Alpha",
            "citations": [{"chunk_id": "c1", "quote": "Zeta"}],
            "abstain": False,
            "abstain_reason": "",
        }
        ok, errors = cc.validate_citations(answer, chunks)
        self.assertFalse(ok)
        self.assertTrue(errors)

    def test_normalize_answer_abstains_on_invalid(self):
        chunks = [{"chunk_id": "c1", "text": "Alpha Beta Gamma"}]
        answer = {
            "answer": "Alpha",
            "citations": [{"chunk_id": "c1", "quote": "Zeta"}],
            "abstain": False,
        }
        normalized = cc.normalize_answer(answer, chunks, abstain_on_invalid=True)
        self.assertTrue(normalized["abstain"])
        self.assertEqual(normalized["answer"], "Error: citations are invalid.")
        self.assertEqual(normalized["citations"], [])


if __name__ == "__main__":
    unittest.main()
