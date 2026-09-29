"""Failure and replacement behavior for the document-to-Chroma lifecycle."""

import os
import uuid
from contextlib import contextmanager
from io import StringIO
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from zipfile import ZipFile

from django.core.files.uploadedfile import SimpleUploadedFile
from django.contrib.admin.sites import AdminSite
from django.core.exceptions import ValidationError
from django.test import SimpleTestCase, TestCase
from django.core.management import call_command
from django.core.management.base import CommandError

from documents.indexing.index_sync import rebuild_global_index
from documents.indexing.vector_index import (
    _get_collection_for_identity,
    embedding_identity,
    upsert_vectors,
)
from documents.ingestion.chunkers import create_chunks
from documents.ingestion.validators import validate_upload
from documents.ingestion.document_lifecycle import delete_document, reprocess_document
from documents.ingestion.document_processor import TaskQueue, TaskStatus, process_document
from documents.models import Document, DocumentChunk
from documents.admin import DocumentAdmin


def make_chunk(chunk_id, document_id=7, content='content'):
    document = SimpleNamespace(id=document_id, title='Guide')
    return SimpleNamespace(
        id=chunk_id,
        document_id=document_id,
        document=document,
        content=content,
        chunk_index=chunk_id,
        page_number=1,
    )


class VectorLifecycleTests(SimpleTestCase):
    @patch('documents.indexing.embedding_client.embed_documents', return_value=[[0.1, 0.2]])
    @patch('documents.indexing.vector_index._batch_size', return_value=2)
    @patch('documents.indexing.vector_index._get_collection')
    def test_incomplete_embeddings_do_not_touch_existing_index(
        self, get_collection, _batch_size, embed_documents
    ):
        self.assertFalse(upsert_vectors([make_chunk(1), make_chunk(2)], document_id=7))
        get_collection.assert_not_called()
        embed_documents.assert_called_once()

    @patch('documents.indexing.embedding_client.embed_documents')
    @patch('documents.indexing.vector_index._batch_size', return_value=2)
    @patch('documents.indexing.vector_index._get_collection')
    def test_partial_chroma_write_is_cleaned_up(
        self, get_collection, _batch_size, embed_documents
    ):
        collection = MagicMock()
        collection.upsert.side_effect = [None, RuntimeError('second batch failed')]
        get_collection.return_value = collection
        embed_documents.side_effect = [
            [[0.1, 0.2], [0.2, 0.3]],
            [[0.3, 0.4]],
        ]

        self.assertFalse(upsert_vectors([make_chunk(1), make_chunk(2), make_chunk(3)], document_id=7))
        self.assertEqual(collection.upsert.call_count, 2)
        self.assertEqual(collection.delete.call_count, 2)
        collection.delete.assert_any_call(where={'document_id': 7})

    def test_embedding_identity_changes_with_model_or_endpoint(self):
        with patch.dict(os.environ, {
            'OPENROUTER_EMBEDDING_MODEL': 'model-a',
            'OPENROUTER_BASE_URL': 'https://one.example/v1',
        }):
            initial = embedding_identity()
        with patch.dict(os.environ, {
            'OPENROUTER_EMBEDDING_MODEL': 'model-b',
            'OPENROUTER_BASE_URL': 'https://one.example/v1',
        }):
            changed_model = embedding_identity()
        with patch.dict(os.environ, {
            'OPENROUTER_EMBEDDING_MODEL': 'model-a',
            'OPENROUTER_BASE_URL': 'https://two.example/v1',
        }):
            changed_endpoint = embedding_identity()
        self.assertNotEqual(initial, changed_model)
        self.assertNotEqual(initial, changed_endpoint)

    def test_real_chroma_replacement_drops_old_chunk_ids(self):
        try:
            import chromadb
        except ImportError:
            self.skipTest('chromadb is not installed')
        collection = chromadb.EphemeralClient().get_or_create_collection(
            name=f'index-test-{uuid.uuid4().hex}',
            embedding_function=None,
        )
        with patch('documents.indexing.vector_index._get_collection', return_value=collection), \
                patch('documents.indexing.vector_index._batch_size', return_value=2), \
                patch('documents.indexing.embedding_client.embed_documents') as embed_documents:
            embed_documents.return_value = [[0.1, 0.2], [0.2, 0.3]]
            self.assertTrue(upsert_vectors([make_chunk(1), make_chunk(2)], document_id=7))
            embed_documents.return_value = [[0.3, 0.4]]
            self.assertTrue(upsert_vectors([make_chunk(3)], document_id=7))

        self.assertEqual(collection.get(where={'document_id': 7}, include=[])['ids'], ['3'])

    def test_existing_chroma_collection_rejects_wrong_embedding_identity(self):
        try:
            import chromadb
        except ImportError:
            self.skipTest('chromadb is not installed')
        client = chromadb.EphemeralClient()
        name = f'identity-test-{uuid.uuid4().hex}'
        collection = client.get_or_create_collection(
            name=name, metadata={'embedding_identity': 'old'}, embedding_function=None
        )
        collection.add(ids=['1'], embeddings=[[0.1, 0.2]])
        _get_collection_for_identity.cache_clear()
        with patch('documents.indexing.vector_index._get_client', return_value=client):
            with self.assertRaisesMessage(RuntimeError, 'different embedding identity'):
                _get_collection_for_identity(name, 'new', 'new-model')
        _get_collection_for_identity.cache_clear()


class ChunkLifecycleTests(TestCase):
    def setUp(self):
        self.document = Document.objects.create(title='Guide', file='documents/guide.pdf')

    def test_short_page_and_final_tail_are_preserved(self):
        pages = [
            {'page': 1, 'text': 'brief'},
            {'page': 2, 'text': 'a' * 100 + 'b' * 15},
        ]
        self.assertEqual(create_chunks(pages, self.document, chunk_size=100, overlap=10), 2)
        chunks = list(self.document.chunks.all())
        self.assertEqual(chunks[0].content, 'brief')
        self.assertTrue(chunks[1].content.endswith('b' * 15))
        self.assertEqual(chunks[1].page_number, 2)

    def test_invalid_overlap_preserves_prior_chunks(self):
        original = DocumentChunk.objects.create(
            document=self.document, content='previous', chunk_index=0
        )
        with self.assertRaises(ValueError):
            create_chunks([{'page': 1, 'text': 'new text'}], self.document,
                          chunk_size=50, overlap=50)
        self.assertEqual(list(self.document.chunks.values_list('id', flat=True)), [original.id])

    def test_explicit_zero_overlap_is_used(self):
        self.assertEqual(
            create_chunks([{'page': 1, 'text': 'x' * 100}], self.document,
                          chunk_size=50, overlap=0),
            2,
        )
        self.assertEqual(list(self.document.chunks.values_list('content', flat=True)),
                         ['x' * 50, 'x' * 50])


class ProcessingLifecycleTests(TestCase):
    def setUp(self):
        self.document = Document.objects.create(
            title='Guide', file='documents/guide.pdf', status='ready'
        )

    @patch('documents.ingestion.document_processor.upsert_document_chunks', return_value=True)
    @patch('documents.ingestion.document_processor.extract_text', return_value=[{'page': 1, 'text': 'new text'}])
    @patch('documents.ingestion.document_processor.delete_document_chunks', return_value=True)
    def test_successful_reprocess_replaces_chunks_and_marks_ready(
        self, delete_chunks, _extract_text, _upsert
    ):
        DocumentChunk.objects.create(
            document=self.document, content='old text', chunk_index=0
        )
        self.assertTrue(process_document(self.document.id))
        self.document.refresh_from_db()
        self.assertEqual(self.document.status, 'ready')
        self.assertIsNotNone(self.document.processed_at)
        self.assertEqual(list(self.document.chunks.values_list('content', flat=True)), ['new text'])
        delete_chunks.assert_called_once_with(self.document.id)

    @patch('documents.ingestion.document_processor.upsert_document_chunks', return_value=False)
    @patch('documents.ingestion.document_processor.extract_text', return_value=[{'page': 1, 'text': 'new text'}])
    @patch('documents.ingestion.document_processor.delete_document_chunks', return_value=True)
    def test_index_failure_marks_document_failed_and_cleans_vectors(
        self, delete_chunks, _extract_text, _upsert
    ):
        self.assertFalse(process_document(self.document.id))
        self.document.refresh_from_db()
        self.assertEqual(self.document.status, 'failed')
        self.assertIn('Chroma indexing failed', self.document.error_message)
        self.assertIsNone(self.document.processed_at)
        self.assertEqual(delete_chunks.call_count, 2)

    @patch('documents.ingestion.document_lifecycle.delete_document_chunks', return_value=False)
    def test_failed_chroma_delete_preserves_document_and_chunks(self, _delete):
        DocumentChunk.objects.create(
            document=self.document, content='existing', chunk_index=0
        )
        result = delete_document(self.document.id)
        self.assertFalse(result['success'])
        self.assertTrue(Document.objects.filter(id=self.document.id).exists())
        self.assertEqual(self.document.chunks.count(), 1)

    @patch('documents.ingestion.document_lifecycle._invalidate_caches')
    @patch('documents.ingestion.document_lifecycle.process_document')
    def test_reprocess_reports_refreshed_status(self, process, _invalidate):
        def mark_ready(document_id):
            Document.objects.filter(id=document_id).update(status='ready')
            return True
        process.side_effect = mark_ready
        Document.objects.filter(id=self.document.id).update(status='failed')

        result = reprocess_document(self.document.id)
        self.assertTrue(result['success'])
        self.assertEqual(result['status'], 'ready')

    @patch('documents.indexing.vector_index.upsert_vectors', return_value=True)
    def test_global_rebuild_indexes_only_ready_documents(self, upsert):
        failed = Document.objects.create(title='Failed', status='failed')
        ready_chunk = DocumentChunk.objects.create(
            document=self.document, content='ready', chunk_index=0
        )
        DocumentChunk.objects.create(document=failed, content='stale', chunk_index=0)

        self.assertTrue(rebuild_global_index())
        indexed = upsert.call_args.args[0]
        self.assertEqual([chunk.id for chunk in indexed], [ready_chunk.id])

    @patch('documents.indexing.vector_index._batch_size', return_value=128)
    @patch('documents.indexing.vector_index._get_collection')
    def test_empty_rebuild_removes_stale_vectors(self, get_collection, _batch_size):
        self.document.status = 'failed'
        self.document.save(update_fields=['status'])
        collection = MagicMock()
        collection.get.return_value = {'ids': ['orphan']}
        get_collection.return_value = collection

        self.assertTrue(rebuild_global_index())
        collection.delete.assert_called_once_with(ids=['orphan'])

    @patch('documents.management.commands.rebuild_chroma_index.upsert_document_chunks')
    def test_command_rejects_non_ready_document(self, upsert):
        self.document.status = 'failed'
        self.document.save(update_fields=['status'])
        with self.assertRaisesMessage(CommandError, 'only ready documents'):
            call_command('rebuild_chroma_index', document_id=self.document.id)
        upsert.assert_not_called()

    @patch('documents.management.commands.rebuild_chroma_index.rebuild_global_index', return_value=True)
    def test_command_reports_empty_successful_rebuild(self, rebuild):
        self.document.status = 'failed'
        self.document.save(update_fields=['status'])
        output = StringIO()
        call_command('rebuild_chroma_index', stdout=output)
        self.assertIn('No ready documents', output.getvalue())
        rebuild.assert_called_once()


class TaskQueueTests(SimpleTestCase):
    def test_unsuccessful_result_is_reported_as_failed(self):
        queue = TaskQueue(max_workers=0, max_retries=0)
        task = queue.enqueue('failed-job', lambda: False, document_id=9)
        queue._execute_task(task.task_id, lambda: False)
        self.assertEqual(task.status, TaskStatus.FAILED)
        self.assertIsNotNone(task.completed_at)
        self.assertIn('unsuccessful', task.error)


class UploadSignatureTests(SimpleTestCase):
    def test_rejects_mislabeled_pdf_and_preserves_file_cursor(self):
        upload = SimpleUploadedFile('fake.pdf', b'not a PDF')
        upload.seek(3)
        errors = validate_upload(upload)
        self.assertTrue(any('signature' in error for error in errors))
        self.assertEqual(upload.tell(), 3)

    def test_rejects_zip_without_docx_parts(self):
        content = BytesIO()
        with ZipFile(content, 'w') as archive:
            archive.writestr('other.txt', 'not a Word document')
        errors = validate_upload(SimpleUploadedFile('fake.docx', content.getvalue()))
        self.assertTrue(any('structure' in error for error in errors))

    def test_accepts_docx_package_with_required_parts(self):
        content = BytesIO()
        with ZipFile(content, 'w') as archive:
            archive.writestr('[Content_Types].xml', '<Types/>')
            archive.writestr('word/document.xml', '<document/>')
        self.assertEqual(
            validate_upload(SimpleUploadedFile('guide.docx', content.getvalue())), []
        )

    @patch('documents.ingestion.validators.MAX_DOCX_UNCOMPRESSED_SIZE', 10)
    def test_rejects_docx_with_excessive_expanded_size(self):
        content = BytesIO()
        with ZipFile(content, 'w') as archive:
            archive.writestr('[Content_Types].xml', '<Types/>')
            archive.writestr('word/document.xml', '<document/>')
        errors = validate_upload(SimpleUploadedFile('large.docx', content.getvalue()))
        self.assertTrue(any('extraction size' in error for error in errors))


class AdminDeletionTests(TestCase):
    @patch('documents.indexing.vector_index.delete_vectors_for_document')
    def test_single_delete_holds_document_lock_through_database_delete(self, delete_vectors):
        document = Document.objects.create(title='Delete me')
        document_id = document.id
        active = set()

        @contextmanager
        def tracked_lock(document_id):
            active.add(document_id)
            try:
                yield
            finally:
                self.assertFalse(Document.objects.filter(id=document_id).exists())
                active.remove(document_id)

        def delete_while_locked(document_id):
            self.assertIn(document_id, active)
            return True

        delete_vectors.side_effect = delete_while_locked
        with patch('documents.ingestion.document_processor.document_lock', tracked_lock):
            DocumentAdmin(Document, AdminSite()).delete_model(None, document)
        self.assertFalse(Document.objects.filter(id=document_id).exists())

    @patch('documents.indexing.vector_index.delete_vectors_for_document')
    def test_bulk_delete_failure_preserves_remaining_document(self, delete_vectors):
        first = Document.objects.create(title='First')
        second = Document.objects.create(title='Second')
        delete_vectors.side_effect = [True, False]
        queryset = Document.objects.filter(id__in=[first.id, second.id]).order_by('id')

        with self.assertRaises(ValidationError):
            DocumentAdmin(Document, AdminSite()).delete_queryset(None, queryset)

        self.assertFalse(Document.objects.filter(id=first.id).exists())
        self.assertTrue(Document.objects.filter(id=second.id).exists())
