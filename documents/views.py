import uuid
import logging
from functools import wraps

from django.shortcuts import render
from django.http import JsonResponse
from django.db import connections
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import ensure_csrf_cookie
from rest_framework import generics, status
from rest_framework.response import Response
from rest_framework.renderers import JSONRenderer, BrowsableAPIRenderer
from rest_framework.parsers import JSONParser, FormParser, MultiPartParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.throttling import ScopedRateThrottle

from .models import Document, QuestionHistory, ChatSession
from .serializers import (
    AskQuestionSerializer,
    QuestionHistorySerializer,
)
from .rag.answer_service import generate_answer
from .ingestion.document_processor import process_document_async, get_task_queue
from .ingestion.validators import validate_upload
from .ingestion.chunkers import DOCUMENT_TYPE_CONFIGS
from .evaluation import evaluate_retrieval, compare_search_methods

logger = logging.getLogger(__name__)


def health_live(request):
    if request.method != 'GET':
        return JsonResponse({'error': 'Method not allowed.'}, status=405)
    return JsonResponse({'status': 'ok'})


def health_ready(request):
    if request.method != 'GET':
        return JsonResponse({'error': 'Method not allowed.'}, status=405)
    try:
        with connections['default'].cursor() as cursor:
            cursor.execute('SELECT 1')
        from .indexing.vector_index import _get_client
        _get_client().heartbeat()
    except Exception:
        logger.warning('Readiness check failed', exc_info=True)
        return JsonResponse({'status': 'unavailable'}, status=503)
    return JsonResponse({'status': 'ok'})


def api_login_required(view):
    @wraps(view)
    def wrapped(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return JsonResponse({'error': 'Authentication required.'}, status=401)
        return view(request, *args, **kwargs)
    return wrapped


def api_staff_required(view):
    @wraps(view)
    @api_login_required
    def wrapped(request, *args, **kwargs):
        if not request.user.is_staff:
            return JsonResponse({'error': 'Staff access required.'}, status=403)
        return view(request, *args, **kwargs)
    return wrapped


@login_required
@ensure_csrf_cookie
def index(request):
    return render(request, 'index.html')


@api_login_required
def upload_document(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed.'}, status=405)

    file = request.FILES.get('file')
    if not file:
        return JsonResponse({'error': 'No file provided.'}, status=400)

    errors = validate_upload(file)
    if errors:
        return JsonResponse({'error': ' '.join(errors)}, status=400)

    document_type = request.POST.get('document_type', 'default')
    if document_type not in DOCUMENT_TYPE_CONFIGS:
        return JsonResponse({'error': 'Invalid document_type.'}, status=400)
    config = DOCUMENT_TYPE_CONFIGS[document_type]
    try:
        chunk_size = int(request.POST.get('chunk_size') or config['chunk_size'])
        chunk_overlap = int(request.POST.get('chunk_overlap') or config['overlap'])
    except (TypeError, ValueError):
        return JsonResponse({'error': 'chunk_size and chunk_overlap must be integers.'}, status=400)
    if not 50 <= chunk_size <= 4000:
        return JsonResponse({'error': 'chunk_size must be between 50 and 4000.'}, status=400)
    if not 0 <= chunk_overlap < chunk_size:
        return JsonResponse({'error': 'chunk_overlap must be non-negative and smaller than chunk_size.'}, status=400)

    doc = None
    try:
        doc = Document.objects.create(
            title=file.name,
            file=file,
            owner=request.user,
            status='uploaded',
            document_type=document_type,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        task_id = process_document_async(doc.id)

        return JsonResponse({
            'message': f'Uploaded "{doc.title}". Processing in background.',
            'document_id': doc.id,
            'task_id': task_id,
            'status': 'processing',
            'document_type': document_type,
            'chunk_size': doc.chunk_size,
            'chunk_overlap': doc.chunk_overlap,
        }, status=202)

    except Exception:
        logger.exception('Upload failed')
        if doc is not None:
            doc.status = 'failed'
            doc.error_message = 'Could not schedule document processing.'
            doc.save(update_fields=['status', 'error_message'])
        return JsonResponse({'error': 'Upload failed. Please try again.'}, status=500)


@api_login_required
def document_status(request, doc_id):
    if request.method != 'GET':
        return JsonResponse({'error': 'Method not allowed.'}, status=405)

    try:
        doc = Document.objects.get(id=doc_id, owner=request.user)
    except Document.DoesNotExist:
        return JsonResponse({'error': 'Document not found.'}, status=404)

    task_queue = get_task_queue()
    task_info = None
    for tid, task in task_queue._tasks.items():
        if task.document_id == doc_id:
            task_info = {
                'task_id': tid,
                'status': task.status.value,
                'started_at': str(task.started_at) if task.started_at else None,
                'completed_at': str(task.completed_at) if task.completed_at else None,
                'duration_seconds': task.duration_seconds,
                'retry_count': task.retry_count,
                'error': task.error,
            }
            break

    return JsonResponse({
        'document_id': doc.id,
        'title': doc.title,
        'status': doc.status,
        'document_type': doc.document_type,
        'chunk_size': doc.chunk_size,
        'chunk_overlap': doc.chunk_overlap,
        'chunks': doc.chunks.count(),
        'error_message': doc.error_message,
        'processed_at': doc.processed_at.isoformat() if doc.processed_at else None,
        'created_at': doc.created_at.isoformat() if doc.created_at else None,
        'task': task_info,
        'valid_transitions': Document.VALID_TRANSITIONS.get(doc.status, []),
    })


@api_login_required
def list_documents(request):
    if request.method != 'GET':
        return JsonResponse({'error': 'Method not allowed.'}, status=405)

    docs = Document.objects.filter(owner=request.user).order_by('-created_at')

    data = []
    for doc in docs:
        data.append({
            'id': doc.id,
            'title': doc.title,
            'status': doc.status,
            'document_type': doc.document_type,
            'error_message': doc.error_message,
            'chunks': doc.chunks.count(),
            'processed_at': doc.processed_at.isoformat() if doc.processed_at else None,
            'created_at': doc.created_at.isoformat() if doc.created_at else None,
        })
    return JsonResponse(data, safe=False)


@api_login_required
def delete_document(request, doc_id):
    if request.method != 'DELETE':
        return JsonResponse({'error': 'Method not allowed.'}, status=405)

    from .ingestion.document_lifecycle import delete_document as delete_document_with_index

    result = delete_document_with_index(doc_id, owner=request.user)
    if not result['success']:
        if result.get('error') == 'Document not found':
            return JsonResponse({'error': 'Document not found.'}, status=404)
        logger.error('Could not remove document %s from the vector index', doc_id)
        return JsonResponse({'error': 'Document deletion failed. Please retry.'}, status=503)
    return JsonResponse({'message': f'Deleted "{result["title"]}".'})


class AskQuestionAPIView(generics.GenericAPIView):
    serializer_class = AskQuestionSerializer
    renderer_classes = [JSONRenderer, BrowsableAPIRenderer]
    parser_classes = [JSONParser, FormParser, MultiPartParser]
    permission_classes = [IsAuthenticated]
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = 'ask'

    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        question = serializer.validated_data["question"]
        search_method = serializer.validated_data.get("search_method", "hybrid")
        document_ids = serializer.validated_data.get("document_ids", [])
        session_id = serializer.validated_data.get("session_id")
        request_id = str(uuid.uuid4())[:8]

        owner = request.user

        if session_id:
            try:
                session = ChatSession.objects.get(id=session_id, owner=owner)
            except ChatSession.DoesNotExist:
                return Response({
                    "error": "Session not found."
                }, status=status.HTTP_404_NOT_FOUND)
        else:
            session = None

        ready_documents = Document.objects.filter(owner=owner, status='ready')
        if document_ids:
            valid_ids = set(ready_documents.filter(id__in=document_ids).values_list('id', flat=True))
            if valid_ids != set(document_ids):
                return Response({
                    "error": "One or more documents are unavailable or not ready."
                }, status=status.HTTP_400_BAD_REQUEST)
            document_ids = sorted(valid_ids)
        else:
            document_ids = list(ready_documents.values_list('id', flat=True))

        if not document_ids:
            return Response({
                "answer": "No ready documents found. Please upload and process documents first.",
                "context": "",
                "search_method": search_method,
                "sources": [],
                "history_id": None,
                "session_id": session.id if session else None,
                "request_id": request_id,
            })

        try:
            result = generate_answer(
                question=question,
                search_method=search_method,
                document_ids=document_ids,
                owner=owner,
                request_id=request_id,
            )
        except Exception:
            logger.exception('Answer generation failed for request %s', request_id)
            return Response({
                "error": "Question answering is temporarily unavailable.",
                "request_id": request_id,
            }, status=status.HTTP_503_SERVICE_UNAVAILABLE)

        if not isinstance(result, dict) or not isinstance(result.get('answer'), str):
            logger.error('Invalid answer result for request %s', request_id)
            return Response({
                "error": "Question answering is temporarily unavailable.",
                "request_id": request_id,
            }, status=status.HTTP_503_SERVICE_UNAVAILABLE)

        if result.get('error') or result.get('failure_mode'):
            failure_mode = result.get('failure_mode')
            unsupported = failure_mode in (
                'no_chunks_retrieved', 'no_context', 'verification_failed',
                'verification_low_support', 'verification_no_context',
            )
            message = result['answer'] if unsupported else 'Question answering is temporarily unavailable.'
            return Response({
                'error': message,
                'request_id': result.get('request_id', request_id),
            }, status=422 if unsupported else status.HTTP_503_SERVICE_UNAVAILABLE)

        if session is None:
            session = ChatSession.objects.create(title=question[:50], owner=owner)

        history = QuestionHistory.objects.create(
            session=session,
            question=question,
            answer=result["answer"],
            retrieved_context=result.get("context", ""),
            source_chunks=result.get("sources", []),
            owner=owner,
        )

        session.save(update_fields=['updated_at'])

        return Response({
            "question": question,
            "search_method": result.get("search_method", search_method),
            "answer": result["answer"],
            "context": result.get("context", ""),
            "sources": result.get("sources", []),
            "history_id": history.id,
            "session_id": session.id,
            "request_id": result.get("request_id", request_id),
            "cache_hit": result.get("cache_hit", False),
            "web_search_used": result.get("web_search_used", False),
        }, status=status.HTTP_200_OK)


class QuestionHistoryListAPIView(generics.ListAPIView):
    serializer_class = QuestionHistorySerializer
    renderer_classes = [JSONRenderer, BrowsableAPIRenderer]
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return QuestionHistory.objects.filter(owner=self.request.user).order_by("-created_at")


@api_login_required
def delete_history(request, history_id):
    if request.method != 'DELETE':
        return JsonResponse({'error': 'Method not allowed.'}, status=405)

    try:
        h = QuestionHistory.objects.get(id=history_id, owner=request.user)

        h.delete()
        return JsonResponse({'message': 'Deleted.'})
    except QuestionHistory.DoesNotExist:
        return JsonResponse({'error': 'Not found.'}, status=404)


@api_login_required
def list_sessions(request):
    if request.method != 'GET':
        return JsonResponse({'error': 'Method not allowed.'}, status=405)

    sessions = ChatSession.objects.filter(owner=request.user).prefetch_related('messages')

    data = []
    for session in sessions:
        messages = session.messages.all()
        msg_data = []
        for msg in messages:
            msg_data.append({
                'id': msg.id,
                'question': msg.question,
                'answer': msg.answer,
                'sources': msg.source_chunks,
                'created_at': msg.created_at.isoformat() if msg.created_at else None,
            })

        data.append({
            'id': session.id,
            'title': session.title,
            'message_count': len(msg_data),
            'messages': msg_data,
            'created_at': session.created_at.isoformat() if session.created_at else None,
            'updated_at': session.updated_at.isoformat() if session.updated_at else None,
        })

    return JsonResponse(data, safe=False)


@api_login_required
def delete_session(request, session_id):
    if request.method != 'DELETE':
        return JsonResponse({'error': 'Method not allowed.'}, status=405)

    try:
        session = ChatSession.objects.get(id=session_id, owner=request.user)

        session.delete()
        return JsonResponse({'message': 'Deleted.'})
    except ChatSession.DoesNotExist:
        return JsonResponse({'error': 'Not found.'}, status=404)


@api_staff_required
def system_stats(request):
    if request.method != 'GET':
        return JsonResponse({'error': 'Method not allowed.'}, status=405)

    from .rag.cache import get_all_cache_stats
    from .rag.feature_flags import list_flags
    from .indexing.vector_index import get_mapping_stats
    from .ingestion.document_processor import get_task_queue

    task_queue = get_task_queue()
    tasks = task_queue._tasks

    return JsonResponse({
        'cache': get_all_cache_stats(),
        'feature_flags': list_flags(),
        'vector_store': get_mapping_stats(),
        'task_queue': {
            'pending': sum(1 for t in tasks.values() if t.status.value == 'pending'),
            'processing': sum(1 for t in tasks.values() if t.status.value == 'processing'),
            'completed': sum(1 for t in tasks.values() if t.status.value == 'completed'),
            'failed': sum(1 for t in tasks.values() if t.status.value == 'failed'),
            'total': len(tasks),
        },
        'documents': {
            'total': Document.objects.count(),
            'ready': Document.objects.filter(status='ready').count(),
            'processing': Document.objects.filter(status__in=['processing', 'pending_processing', 'extracting_text', 'chunking', 'indexing']).count(),
            'failed': Document.objects.filter(status='failed').count(),
        },
        'chunk_configs': DOCUMENT_TYPE_CONFIGS,
    })


@api_staff_required
def evaluate(request):
    if request.method != 'GET':
        return JsonResponse({'error': 'Method not allowed.'}, status=405)

    search_method = request.GET.get('method', 'hybrid')
    if search_method not in ('simple', 'bm25', 'vector', 'hybrid'):
        return JsonResponse({'error': 'Invalid search method.'}, status=400)

    document_ids = list(
        Document.objects.filter(owner=request.user, status='ready').values_list('id', flat=True)
    )

    if not document_ids:
        return JsonResponse({'error': 'No ready documents found.'}, status=400)

    try:
        result = evaluate_retrieval(
            search_method=search_method, document_ids=document_ids, owner=request.user
        )
    except ValueError as exc:
        return JsonResponse({'error': str(exc)}, status=400)
    except OSError:
        logger.exception('Evaluation dataset is unavailable')
        return JsonResponse({'error': 'Evaluation dataset is unavailable.'}, status=503)
    return JsonResponse(result)


@api_staff_required
def compare_methods(request):
    if request.method != 'GET':
        return JsonResponse({'error': 'Method not allowed.'}, status=405)

    document_ids = list(
        Document.objects.filter(owner=request.user, status='ready').values_list('id', flat=True)
    )

    if not document_ids:
        return JsonResponse({'error': 'No ready documents found.'}, status=400)

    try:
        result = compare_search_methods(document_ids=document_ids, owner=request.user)
    except ValueError as exc:
        return JsonResponse({'error': str(exc)}, status=400)
    except OSError:
        logger.exception('Evaluation dataset is unavailable')
        return JsonResponse({'error': 'Evaluation dataset is unavailable.'}, status=503)
    return JsonResponse(result)


@api_staff_required
def feature_flags_admin(request):
    from .rag.feature_flags import list_flags, get_flag, update_flag

    if request.method == 'GET':
        return JsonResponse({'flags': list_flags()})

    elif request.method == 'POST':
        import json
        try:
            data = json.loads(request.body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return JsonResponse({'error': 'Invalid JSON'}, status=400)

        if not isinstance(data, dict):
            return JsonResponse({'error': 'Expected a JSON object.'}, status=400)

        name = data.get('name')
        if not isinstance(name, str) or not name:
            return JsonResponse({'error': 'Missing flag name'}, status=400)

        rollout_pct = data.get('rollout_pct')
        variants = data.get('variants')
        if rollout_pct is not None and (
            isinstance(rollout_pct, bool) or not isinstance(rollout_pct, (int, float))
            or not 0 <= rollout_pct <= 100
        ):
            return JsonResponse({'error': 'rollout_pct must be between 0 and 100.'}, status=400)
        if variants is not None and (
            not isinstance(variants, list) or not 1 <= len(variants) <= 10
            or any(not isinstance(v, str) or not v or len(v) > 64 for v in variants)
        ):
            return JsonResponse({'error': 'variants must be a nonempty list of names.'}, status=400)

        if not update_flag(name, rollout_pct=rollout_pct, variants=variants):
            return JsonResponse({'error': f'Flag {name} not found'}, status=404)

        return JsonResponse({'flag': get_flag(name)})

    return JsonResponse({'error': 'Method not allowed'}, status=405)
