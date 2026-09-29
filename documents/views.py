import uuid
import logging

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt, ensure_csrf_cookie
from rest_framework import generics, status
from rest_framework.response import Response
from rest_framework.renderers import JSONRenderer, BrowsableAPIRenderer
from rest_framework.parsers import JSONParser, FormParser, MultiPartParser
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.decorators import api_view, permission_classes

from .models import Document, DocumentChunk, QuestionHistory, ChatSession
from .serializers import (
    DocumentSerializer,
    AskQuestionSerializer,
    QuestionHistorySerializer,
    DocumentChunkSerializer,
    ChatSessionSerializer,
)
from .rag.answer_service import generate_answer
from .ingestion.document_processor import process_document_async, get_task_queue
from .ingestion.validators import validate_upload
from .ingestion.chunkers import DOCUMENT_TYPE_CONFIGS
from .evaluation import evaluate_retrieval, compare_search_methods

logger = logging.getLogger(__name__)


@ensure_csrf_cookie
def index(request):
    return render(request, 'index.html')


@csrf_exempt
def upload_document(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed.'}, status=405)

    file = request.FILES.get('file')
    if not file:
        return JsonResponse({'error': 'No file provided.'}, status=400)

    errors = validate_upload(file)
    if errors:
        return JsonResponse({'error': ' '.join(errors)}, status=400)

    try:
        owner = request.user if request.user.is_authenticated else None
        document_type = request.POST.get('document_type', 'default')
        chunk_size = request.POST.get('chunk_size')
        chunk_overlap = request.POST.get('chunk_overlap')

        config = DOCUMENT_TYPE_CONFIGS.get(document_type, DOCUMENT_TYPE_CONFIGS['default'])

        doc = Document.objects.create(
            title=file.name,
            file=file,
            owner=owner,
            status='uploaded',
            document_type=document_type,
            chunk_size=int(chunk_size) if chunk_size else config['chunk_size'],
            chunk_overlap=int(chunk_overlap) if chunk_overlap else config['overlap'],
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
        })

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return JsonResponse({'error': f'Upload failed: {str(e)}'}, status=500)


@csrf_exempt
def document_status(request, doc_id):
    if request.method != 'GET':
        return JsonResponse({'error': 'Method not allowed.'}, status=405)

    try:
        if request.user.is_authenticated:
            doc = Document.objects.get(id=doc_id, owner=request.user)
        else:
            doc = Document.objects.get(id=doc_id, owner__isnull=True)
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


@csrf_exempt
def list_documents(request):
    if request.method != 'GET':
        return JsonResponse({'error': 'Method not allowed.'}, status=405)

    if request.user.is_authenticated:
        docs = Document.objects.filter(owner=request.user).order_by('-created_at')
    else:
        docs = Document.objects.filter(owner__isnull=True).order_by('-created_at')

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


@csrf_exempt
def delete_document(request, doc_id):
    if request.method != 'DELETE':
        return JsonResponse({'error': 'Method not allowed.'}, status=405)

    from .ingestion.document_lifecycle import delete_document as delete_document_with_index

    owner = request.user if request.user.is_authenticated else None
    result = delete_document_with_index(doc_id, owner=owner)
    if not result['success']:
        return JsonResponse({'error': 'Document not found.'}, status=404)
    return JsonResponse({'message': f'Deleted "{result["title"]}".'})


class AskQuestionAPIView(generics.GenericAPIView):
    serializer_class = AskQuestionSerializer
    renderer_classes = [JSONRenderer, BrowsableAPIRenderer]
    parser_classes = [JSONParser, FormParser, MultiPartParser]
    permission_classes = [AllowAny]

    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        question = serializer.validated_data["question"]
        search_method = serializer.validated_data.get("search_method", "hybrid")
        document_ids = serializer.validated_data.get("document_ids", [])
        session_id = serializer.validated_data.get("session_id")
        request_id = str(uuid.uuid4())[:8]

        owner = request.user if request.user.is_authenticated else None

        if session_id:
            try:
                if owner:
                    session = ChatSession.objects.get(id=session_id, owner=owner)
                else:
                    session = ChatSession.objects.get(id=session_id, owner__isnull=True)
            except ChatSession.DoesNotExist:
                return Response({
                    "error": "Session not found."
                }, status=status.HTTP_404_NOT_FOUND)
        else:
            session = ChatSession.objects.create(
                title=question[:50],
                owner=owner,
            )

        if request.user.is_authenticated:
            if document_ids:
                valid_ids = Document.objects.filter(
                    id__in=document_ids,
                    owner=request.user,
                    status='ready'
                ).values_list('id', flat=True)
                document_ids = list(valid_ids)
            else:
                document_ids = list(
                    Document.objects.filter(
                        owner=request.user,
                        status='ready'
                    ).values_list('id', flat=True)
                )
        else:
            if document_ids:
                valid_ids = Document.objects.filter(
                    id__in=document_ids,
                    owner__isnull=True,
                    status='ready'
                ).values_list('id', flat=True)
                document_ids = list(valid_ids)
            else:
                document_ids = list(
                    Document.objects.filter(
                        owner__isnull=True,
                        status='ready'
                    ).values_list('id', flat=True)
                )

        if not document_ids:
            return Response({
                "answer": "No ready documents found. Please upload and process documents first.",
                "context": "",
                "search_method": search_method,
                "sources": [],
                "history_id": None,
                "session_id": session.id,
                "request_id": request_id,
            })

        result = generate_answer(
            question=question,
            search_method=search_method,
            document_ids=document_ids if document_ids else None,
            owner=owner,
            request_id=request_id,
        )

        if not isinstance(result, dict):
            result = {
                "answer": "LLM service error: Invalid response from generate_answer().",
                "context": "",
                "search_method": search_method,
                "sources": [],
            }

        history = QuestionHistory.objects.create(
            session=session,
            question=question,
            answer=result["answer"],
            retrieved_context=result["context"],
            source_chunks=result.get("sources", []),
            owner=owner,
        )

        session.save(update_fields=['updated_at'])

        return Response({
            "question": question,
            "search_method": result["search_method"],
            "answer": result["answer"],
            "context": result["context"],
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
    permission_classes = [AllowAny]

    def get_queryset(self):
        if self.request.user.is_authenticated:
            return QuestionHistory.objects.filter(owner=self.request.user).order_by("-created_at")
        return QuestionHistory.objects.filter(owner__isnull=True).order_by("-created_at")


@csrf_exempt
def delete_history(request, history_id):
    if request.method != 'DELETE':
        return JsonResponse({'error': 'Method not allowed.'}, status=405)

    try:
        if request.user.is_authenticated:
            h = QuestionHistory.objects.get(id=history_id, owner=request.user)
        else:
            h = QuestionHistory.objects.get(id=history_id, owner__isnull=True)

        h.delete()
        return JsonResponse({'message': 'Deleted.'})
    except QuestionHistory.DoesNotExist:
        return JsonResponse({'error': 'Not found.'}, status=404)


@csrf_exempt
def list_sessions(request):
    if request.method != 'GET':
        return JsonResponse({'error': 'Method not allowed.'}, status=405)

    if request.user.is_authenticated:
        sessions = ChatSession.objects.filter(owner=request.user)
    else:
        sessions = ChatSession.objects.filter(owner__isnull=True)

    data = []
    for session in sessions:
        messages = session.messages.all().order_by('created_at')
        msg_data = []
        for msg in messages:
            msg_data.append({
                'id': msg.id,
                'question': msg.question,
                'answer': msg.answer,
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


@csrf_exempt
def delete_session(request, session_id):
    if request.method != 'DELETE':
        return JsonResponse({'error': 'Method not allowed.'}, status=405)

    try:
        if request.user.is_authenticated:
            session = ChatSession.objects.get(id=session_id, owner=request.user)
        else:
            session = ChatSession.objects.get(id=session_id, owner__isnull=True)

        session.delete()
        return JsonResponse({'message': 'Deleted.'})
    except ChatSession.DoesNotExist:
        return JsonResponse({'error': 'Not found.'}, status=404)


@csrf_exempt
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


@csrf_exempt
def evaluate(request):
    if request.method != 'GET':
        return JsonResponse({'error': 'Method not allowed.'}, status=405)

    search_method = request.GET.get('method', 'hybrid')

    if request.user.is_authenticated:
        document_ids = list(
            Document.objects.filter(owner=request.user, status='ready').values_list('id', flat=True)
        )
    else:
        document_ids = list(
            Document.objects.filter(owner__isnull=True, status='ready').values_list('id', flat=True)
        )

    if not document_ids:
        return JsonResponse({'error': 'No ready documents found.'}, status=400)

    result = evaluate_retrieval(search_method=search_method, document_ids=document_ids)
    return JsonResponse(result)


@csrf_exempt
def compare_methods(request):
    if request.method != 'GET':
        return JsonResponse({'error': 'Method not allowed.'}, status=405)

    if request.user.is_authenticated:
        document_ids = list(
            Document.objects.filter(owner=request.user, status='ready').values_list('id', flat=True)
        )
    else:
        document_ids = list(
            Document.objects.filter(owner__isnull=True, status='ready').values_list('id', flat=True)
        )

    if not document_ids:
        return JsonResponse({'error': 'No ready documents found.'}, status=400)

    result = compare_search_methods(document_ids=document_ids)
    return JsonResponse(result)


@csrf_exempt
def feature_flags_admin(request):
    from .rag.feature_flags import list_flags, get_flag, update_flag

    if request.method == 'GET':
        return JsonResponse({'flags': list_flags()})

    elif request.method == 'POST':
        import json
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)

        name = data.get('name')
        if not name:
            return JsonResponse({'error': 'Missing flag name'}, status=400)

        rollout_pct = data.get('rollout_pct')
        variants = data.get('variants')

        if not update_flag(name, rollout_pct=rollout_pct, variants=variants):
            return JsonResponse({'error': f'Flag {name} not found'}, status=404)

        return JsonResponse({'flag': get_flag(name)})

    return JsonResponse({'error': 'Method not allowed'}, status=405)
