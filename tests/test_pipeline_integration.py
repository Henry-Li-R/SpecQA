import os
import unittest

from src.pipeline import run_pipeline as rp


class TestPipelineIntegration(unittest.TestCase):
    def _load_env_key(self) -> str:
        key = os.environ.get("OPENAI_API_KEY", "")
        if key:
            return key
        env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
        if not os.path.exists(env_path):
            return ""
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("OPENAI_API_KEY="):
                    return line.split("=", 1)[1].strip()
        return ""

    def test_pipeline_end_to_end(self):
        api_key = self._load_env_key()
        result = rp.run_pipeline(
            "What is the voltage measurement range?",
            api_key=api_key,
        )

        self.assertIsInstance(result, dict)
        self.assertIn("abstain", result)
        self.assertIn("answer", result)
        self.assertIn("citations", result)
        if result.get("abstain") is False:
            self.assertTrue(result.get("answer"))
            self.assertTrue(result.get("citations"))

        print("Pipeline result:", result)

if __name__ == "__main__":
    unittest.main()
