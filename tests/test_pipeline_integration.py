import os
import unittest
import warnings

from dotenv import load_dotenv

from src.pipeline import run_pipeline as rp


class TestPipelineIntegration(unittest.TestCase):
    def _load_env_key(self) -> str:
        load_dotenv()
        return os.environ.get("OPENAI_API_KEY", "")

    def test_pipeline_end_to_end(self):
        api_key = self._load_env_key()
        result = rp.run_pipeline(
            "What is the voltage measurement range?",
            api_key=api_key,
            abstain_on_invalid=False,
        )

        self.assertIsInstance(result, dict)
        self.assertIn("abstain", result)
        self.assertIn("answer", result)
        self.assertIn("citations", result)
        if result.get("abstain") is False:
            self.assertTrue(result.get("answer"))
            self.assertTrue(result.get("citations"))
        if result.get("abstain_reason") == "invalid_citations":
            print("Validation errors:", result.get("validation_errors"))

        print("Pipeline result:", result)

if __name__ == "__main__":
    unittest.main()
