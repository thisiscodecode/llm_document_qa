from django.test import TestCase
from documents.models import Document, DocumentChunk


class DocumentModelTests(TestCase):
    def test_document_creation(self):
        doc = Document.objects.create(
            title="Test Document",
            status="uploaded",
        )
        self.assertEqual(doc.title, "Test Document")
        self.assertEqual(doc.status, "uploaded")

    def test_document_chunk_creation(self):
        doc = Document.objects.create(title="Test", status="ready")
        chunk = DocumentChunk.objects.create(
            document=doc,
            content="Test content",
            chunk_index=0,
            page_number=1,
        )
        self.assertEqual(chunk.content, "Test content")