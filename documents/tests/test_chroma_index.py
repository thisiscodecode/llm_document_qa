from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase, override_settings

from documents.indexing.vector_index import search_vectors, upsert_vectors


@override_settings(CHROMA_BATCH_SIZE=128)
class ChromaIndexTests(SimpleTestCase):
    @patch('documents.indexing.embedding_client.embed_documents')
    @patch('documents.indexing.vector_index._batch_size', return_value=128)
    @patch('documents.indexing.vector_index._get_collection')
    def test_upsert_replaces_document_records_with_metadata(
        self, get_collection, _batch_size, embed_documents
    ):
        collection = MagicMock()
        get_collection.return_value = collection
        embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]
        document = SimpleNamespace(id=7, title='Guide')
        chunks = [
            SimpleNamespace(
                id=11, document_id=7, document=document,
                content='first', chunk_index=0, page_number=1,
            ),
            SimpleNamespace(
                id=12, document_id=7, document=document,
                content='second', chunk_index=1, page_number=2,
            ),
        ]

        self.assertTrue(upsert_vectors(chunks, document_id=7))

        collection.delete.assert_called_once_with(where={'document_id': 7})
        call = collection.upsert.call_args.kwargs
        self.assertEqual(call['ids'], ['11', '12'])
        self.assertEqual(call['documents'], ['first', 'second'])
        self.assertEqual(call['metadatas'][1]['page_number'], 2)
        self.assertEqual(call['metadatas'][0]['document_title'], 'Guide')

    @patch('documents.indexing.vector_index._get_collection')
    def test_search_applies_document_filter_and_converts_cosine_distance(
        self, get_collection
    ):
        collection = MagicMock()
        collection.count.return_value = 20
        collection.query.return_value = {
            'ids': [['12', '11']],
            'distances': [[0.1, 0.4]],
        }
        get_collection.return_value = collection

        results = search_vectors([0.1, 0.2], limit=5, document_ids=[9, 7, 9])

        self.assertEqual(results, [(12, 0.9), (11, 0.6)])
        self.assertEqual(
            collection.query.call_args.kwargs['where'],
            {'document_id': {'$in': [7, 9]}},
        )
