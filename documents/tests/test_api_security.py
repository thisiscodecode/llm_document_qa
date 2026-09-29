import json
from tempfile import TemporaryDirectory
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client, TestCase, override_settings

from documents.models import ChatSession, Document, QuestionHistory


class APISecurityTests(TestCase):
    def setUp(self):
        user_model = get_user_model()
        self.alice = user_model.objects.create_user(username='alice', password='alice-password')
        self.bob = user_model.objects.create_user(username='bob', password='bob-password')
        self.staff = user_model.objects.create_user(
            username='staff', password='staff-password', is_staff=True
        )
        self.alice_doc = Document.objects.create(
            title='Alice document', file='documents/alice.pdf', owner=self.alice, status='ready'
        )
        self.bob_doc = Document.objects.create(
            title='Bob document', file='documents/bob.pdf', owner=self.bob, status='ready'
        )

    def test_anonymous_requests_cannot_access_corpus_or_chat(self):
        self.assertEqual(self.client.get('/documents/').status_code, 401)
        self.assertEqual(self.client.get('/api/history/').status_code, 401)
        self.assertEqual(self.client.get('/api/sessions/').status_code, 401)
        self.assertEqual(self.client.post('/api/ask/', data='{}', content_type='application/json').status_code, 401)
        self.assertEqual(self.client.get('/api/flags/').status_code, 401)
        self.assertEqual(self.client.get('/').status_code, 302)

    def test_nonstaff_user_can_log_in_to_the_application(self):
        response = self.client.post('/accounts/login/', data={
            'username': 'alice', 'password': 'alice-password',
        })
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response['Location'], '/')
        self.assertEqual(self.client.get('/documents/').status_code, 200)

    def test_document_and_session_lists_are_owned(self):
        alice_session = ChatSession.objects.create(owner=self.alice, title='Alice chat')
        bob_session = ChatSession.objects.create(owner=self.bob, title='Bob chat')
        QuestionHistory.objects.create(
            session=alice_session, owner=self.alice, question='Alice?', answer='Yes',
            source_chunks=[{'document': 'Alice document', 'page': 1}],
        )
        QuestionHistory.objects.create(
            session=bob_session, owner=self.bob, question='Bob?', answer='Yes'
        )
        self.client.force_login(self.alice)

        documents = self.client.get('/documents/').json()
        sessions = self.client.get('/api/sessions/').json()
        history = self.client.get('/api/history/').json()

        self.assertEqual([doc['id'] for doc in documents], [self.alice_doc.id])
        self.assertEqual([item['id'] for item in sessions], [alice_session.id])
        self.assertEqual(sessions[0]['messages'][0]['sources'][0]['page'], 1)
        self.assertEqual([item['id'] for item in history], [alice_session.messages.first().id])
        self.assertEqual(self.client.get(f'/documents/{self.bob_doc.id}/status/').status_code, 404)
        self.assertEqual(self.client.delete(f'/api/sessions/{bob_session.id}/delete/').status_code, 404)

    def test_ask_rejects_cross_user_documents_without_creating_session(self):
        self.client.force_login(self.alice)
        body = json.dumps({'question': 'What is this?', 'document_ids': [self.bob_doc.id]})
        with patch('documents.views.generate_answer') as generate_answer:
            response = self.client.post('/api/ask/', data=body, content_type='application/json')
        self.assertEqual(response.status_code, 400)
        self.assertEqual(ChatSession.objects.count(), 0)
        generate_answer.assert_not_called()

    def test_ask_passes_owned_documents_and_owner_to_pipeline(self):
        self.client.force_login(self.alice)
        with patch('documents.views.generate_answer', return_value={
            'answer': 'Alice answer', 'context': 'Alice context',
            'sources': [{'document': 'Alice document', 'page': 1}],
            'search_method': 'hybrid',
        }) as generate_answer:
            response = self.client.post(
                '/api/ask/',
                data=json.dumps({'question': 'What is this?'}),
                content_type='application/json',
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['answer'], 'Alice answer')
        self.assertEqual(QuestionHistory.objects.get().owner, self.alice)
        self.assertEqual(generate_answer.call_args.kwargs['document_ids'], [self.alice_doc.id])
        self.assertEqual(generate_answer.call_args.kwargs['owner'], self.alice)

    def test_failed_answer_does_not_create_chat_history(self):
        self.client.force_login(self.alice)
        with patch('documents.views.generate_answer', return_value={
            'answer': 'The language model is currently unavailable.',
            'context': '', 'search_method': 'hybrid', 'sources': [],
            'error': 'llm_unavailable', 'failure_mode': 'llm_unavailable',
        }):
            response = self.client.post(
                '/api/ask/', data=json.dumps({'question': 'What is this?'}),
                content_type='application/json',
            )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(ChatSession.objects.count(), 0)
        self.assertEqual(QuestionHistory.objects.count(), 0)

    def test_session_mutation_requires_csrf_token(self):
        session = ChatSession.objects.create(owner=self.alice, title='Alice chat')
        client = Client(enforce_csrf_checks=True)
        client.force_login(self.alice)
        self.assertEqual(client.delete(f'/api/sessions/{session.id}/delete/').status_code, 403)
        self.assertTrue(ChatSession.objects.filter(id=session.id).exists())
        client.get('/')
        token = client.cookies['csrftoken'].value
        self.assertEqual(
            client.delete(f'/api/sessions/{session.id}/delete/', HTTP_X_CSRFTOKEN=token).status_code,
            200,
        )

    def test_ask_requires_csrf_for_browser_session(self):
        client = Client(enforce_csrf_checks=True)
        client.force_login(self.alice)
        response = client.post(
            '/api/ask/', data=json.dumps({'question': 'What is this?'}),
            content_type='application/json',
        )
        self.assertEqual(response.status_code, 403)

    def test_upload_rejects_invalid_chunk_configuration_before_saving(self):
        self.client.force_login(self.alice)
        with patch('documents.views.validate_upload', return_value=[]):
            response = self.client.post('/upload/', data={
                'file': SimpleUploadedFile('sample.pdf', b'%PDF-1.4\n'),
                'document_type': 'academic', 'chunk_size': '100',
                'chunk_overlap': '100',
            })
        self.assertEqual(response.status_code, 400)
        self.assertEqual(Document.objects.count(), 2)

    def test_failed_upload_scheduling_marks_document_failed(self):
        self.client.force_login(self.alice)
        with TemporaryDirectory() as media_root, override_settings(MEDIA_ROOT=media_root), \
                patch('documents.views.validate_upload', return_value=[]), \
                patch('documents.views.process_document_async', side_effect=RuntimeError('queue down')):
            response = self.client.post('/upload/', data={
                'file': SimpleUploadedFile('sample.pdf', b'%PDF-1.4\n'),
            })
            uploaded = Document.objects.get(title='sample.pdf')
        self.assertEqual(response.status_code, 500)
        self.assertEqual(uploaded.status, 'failed')

    def test_staff_only_controls_and_flag_validation(self):
        self.client.force_login(self.alice)
        self.assertEqual(self.client.get('/api/stats/').status_code, 403)
        self.assertEqual(self.client.get('/api/flags/').status_code, 403)
        self.assertEqual(self.client.get('/api/evaluate/').status_code, 403)
        self.client.force_login(self.staff)
        response = self.client.post(
            '/api/flags/', data=json.dumps({'name': 'advanced_caching', 'rollout_pct': 'bad'}),
            content_type='application/json',
        )
        self.assertEqual(response.status_code, 400)

    def test_vector_cleanup_failure_is_not_reported_as_not_found(self):
        self.client.force_login(self.alice)
        with patch('documents.ingestion.document_lifecycle.delete_document', return_value={
            'success': False, 'error': 'Vector index cleanup failed; document was not deleted'
        }):
            response = self.client.delete(f'/documents/{self.alice_doc.id}/delete/')
        self.assertEqual(response.status_code, 503)

    def test_evaluation_reports_missing_dataset(self):
        self.alice.is_staff = True
        self.alice.save(update_fields=['is_staff'])
        self.client.force_login(self.alice)
        with patch('documents.views.evaluate_retrieval', side_effect=ValueError('Set RAG_EVAL_DATASET')):
            response = self.client.get('/api/evaluate/')
        self.assertEqual(response.status_code, 400)
        self.assertIn('RAG_EVAL_DATASET', response.json()['error'])

    def test_health_checks_do_not_require_authentication_or_expose_errors(self):
        self.assertEqual(self.client.get('/health/live/').json(), {'status': 'ok'})
        with patch('documents.indexing.vector_index._get_client') as get_client:
            get_client.return_value.heartbeat.return_value = 1
            self.assertEqual(self.client.get('/health/ready/').status_code, 200)
            get_client.return_value.heartbeat.side_effect = RuntimeError('private host')
            response = self.client.get('/health/ready/')
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {'status': 'unavailable'})
