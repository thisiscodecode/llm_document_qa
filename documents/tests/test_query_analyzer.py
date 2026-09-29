from django.test import TestCase

from documents.rag.query_analyzer import detect_language, detect_intent, analyze_query


class QueryAnalyzerTests(TestCase):
    def test_detect_language_english(self):
        self.assertEqual(detect_language("what is python"), "en")

    def test_detect_language_persian(self):
        self.assertEqual(detect_language("پایتون چیست"), "fa")

    def test_detect_intent_question(self):
        self.assertEqual(detect_intent("who is mahyar"), "question")

    def test_detect_intent_explanation(self):
        self.assertEqual(detect_intent("explain how RAG works"), "explanation")

    def test_analyze_query_returns_all_fields(self):
        result = analyze_query("what is python")
        self.assertIn("original", result)
        self.assertIn("rewritten", result)
        self.assertIn("language", result)
        self.assertIn("intent", result)
        self.assertIn("filters", result)