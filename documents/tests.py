from types import SimpleNamespace
from unittest.mock import patch

from django.test import TestCase

from .services import generate_answer


class GenerateAnswerTests(TestCase):
    def test_returns_not_found_when_no_context(self):
        result = generate_answer("What is this?")

        self.assertEqual(
            result["answer"],
            "I could not find relevant information in the uploaded documents.",
        )
        self.assertEqual(result["context"], "")
        self.assertEqual(result["search_method"], "simple")

    @patch("documents.services._get_llm", return_value=None)
    @patch(
        "documents.services.retrieve_relevant_chunks",
        return_value=[SimpleNamespace(content="Python and Django")],
    )
    def test_returns_config_error_when_llm_not_configured(self, _mock_chunks, _mock_llm):
        result = generate_answer("What framework?", search_method="bm25")

        self.assertIn("OPENROUTER_API_KEY", result["answer"])
        self.assertEqual(result["context"], "Python and Django")
        self.assertEqual(result["search_method"], "bm25")