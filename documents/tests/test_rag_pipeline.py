from django.test import TestCase

from documents.rag.context_builder import build_context, build_sources
from documents.rag.prompt_builder import build_grounded_prompt
from documents.rag.answer_verifier import verify_answer
from documents.rag.citation_builder import map_claims_to_sources, format_sources


class MockChunk:
    def __init__(self, content, chunk_index=0, page_number=1):
        self.content = content
        self.chunk_index = chunk_index
        self.page_number = page_number
        self.document = type('obj', (object,), {'title': 'Test Document'})()
        self.id = chunk_index


class ContextBuilderTests(TestCase):
    def test_build_context_with_chunks(self):
        chunks = [
            MockChunk("First chunk content"),
            MockChunk("Second chunk content"),
        ]
        context = build_context(chunks)
        self.assertIn("First chunk content", context)
        self.assertIn("Second chunk content", context)

    def test_build_context_empty_chunks(self):
        context = build_context([])
        self.assertEqual(context, "")

    def test_build_sources(self):
        chunks = [MockChunk("content", chunk_index=0, page_number=1)]
        sources = build_sources(chunks)
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]['document'], 'Test Document')


class PromptBuilderTests(TestCase):
    def test_build_grounded_prompt(self):
        prompt = build_grounded_prompt("test question", "test context", "test sources")
        self.assertIn("test question", prompt)
        self.assertIn("test context", prompt)
        self.assertIn("test sources", prompt)


class AnswerVerifierTests(TestCase):
    def test_verify_answer_with_context(self):
        result = verify_answer("Mahyar is a student", "Mahyar is a computer engineering student")
        self.assertTrue(result['is_valid'])

    def test_verify_answer_no_context(self):
        result = verify_answer("some answer", "")
        self.assertFalse(result['is_valid'])

    def test_verify_answer_refusal(self):
        result = verify_answer("I could not find the answer", "some context")
        self.assertTrue(result['is_valid'])


class CitationBuilderTests(TestCase):
    def test_map_claims_to_sources(self):
        answer = "According to [Source 1] and [Source 2]"
        sources = [
            {'document': 'Doc1', 'chunk_index': 0, 'page': 1},
            {'document': 'Doc2', 'chunk_index': 1, 'page': 2},
        ]
        citations = map_claims_to_sources(answer, sources)
        self.assertEqual(len(citations), 2)

    def test_format_sources(self):
        sources = [
            {'index': 1, 'document': 'Doc', 'chunk_index': 0, 'page': 1, 'preview': 'text'},
        ]
        formatted = format_sources(sources)
        self.assertIn("Source 1", formatted)