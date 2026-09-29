import logging
import threading

from django.utils import timezone

from documents.models import Document, DocumentChunk
from documents.indexing.index_sync import delete_document_chunks, upsert_document_chunks
from .document_processor import process_document

logger = logging.getLogger(__name__)


def _invalidate_caches(document_id: int):
    try:
        from documents.rag.cache import invalidate_queries_by_document, invalidate_permissions_for_document
        invalidate_queries_by_document(document_id)
        invalidate_permissions_for_document(document_id)
    except Exception as e:
        logger.warning(f"Cache invalidation failed for document {document_id}: {e}")


def delete_document(document_id: int, owner=None) -> dict:
    try:
        if owner:
            doc = Document.objects.get(id=document_id, owner=owner)
        else:
            doc = Document.objects.get(id=document_id, owner__isnull=True)
    except Document.DoesNotExist:
        logger.error(f"Document {document_id} not found")
        return {"success": False, "error": "Document not found"}

    title = doc.title
    
    try:
        delete_document_chunks(document_id)
        logger.info(f"Deleted vector index for document {document_id}")
    except Exception as e:
        logger.warning(f"Failed to delete vector index: {e}")

    _invalidate_caches(document_id)

    doc.delete()
    logger.info(f"Deleted document {document_id}: {title}")

    return {"success": True, "document_id": document_id, "title": title}


def reprocess_document(document_id: int, owner=None) -> dict:
    try:
        if owner:
            doc = Document.objects.get(id=document_id, owner=owner)
        else:
            doc = Document.objects.get(id=document_id, owner__isnull=True)
    except Document.DoesNotExist:
        logger.error(f"Document {document_id} not found")
        return {"success": False, "error": "Document not found"}

    logger.info(f"Reprocessing document {document_id}: {doc.title}")

    _invalidate_caches(document_id)

    try:
        delete_document_chunks(document_id)
        logger.info(f"Cleared old index for document {document_id}")
    except Exception as e:
        logger.warning(f"Failed to clear old index: {e}")

    DocumentChunk.objects.filter(document=doc).delete()

    doc.error_message = None
    doc.processed_at = None
    doc.transition_to('processing')
    doc.save(update_fields=['error_message', 'processed_at'])

    success = process_document(document_id)

    return {
        "success": success,
        "document_id": document_id,
        "title": doc.title,
        "status": doc.status,
    }


def reprocess_document_async(document_id: int, owner=None) -> str:
    task_id = f"doc_reprocess_{document_id}_{int(timezone.now().timestamp())}"
    from documents.ingestion.document_processor import get_task_queue
    queue = get_task_queue()
    queue.enqueue(task_id, reprocess_document, document_id=document_id, owner=owner)
    return task_id