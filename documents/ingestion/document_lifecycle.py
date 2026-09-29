import logging
import uuid

from documents.models import Document
from documents.indexing.index_sync import delete_document_chunks
from .document_processor import document_lock, process_document

logger = logging.getLogger(__name__)


def _invalidate_caches(document_id: int):
    try:
        from documents.rag.cache import invalidate_queries_by_document, invalidate_permissions_for_document
        invalidate_queries_by_document(document_id)
        invalidate_permissions_for_document(document_id)
    except Exception as e:
        logger.warning(f"Cache invalidation failed for document {document_id}: {e}")


def delete_document(document_id: int, owner=None) -> dict:
    with document_lock(document_id):
        try:
            if owner:
                doc = Document.objects.get(id=document_id, owner=owner)
            else:
                doc = Document.objects.get(id=document_id, owner__isnull=True)
        except Document.DoesNotExist:
            logger.error(f"Document {document_id} not found")
            return {"success": False, "error": "Document not found"}

        if not delete_document_chunks(document_id):
            logger.error("Chroma cleanup failed; preserving document %s", document_id)
            return {
                "success": False,
                "error": "Vector index cleanup failed; document was not deleted",
            }

        title = doc.title
        _invalidate_caches(document_id)
        doc.delete()
        logger.info("Deleted document %s: %s", document_id, title)
        return {"success": True, "document_id": document_id, "title": title}


def reprocess_document(document_id: int, owner=None) -> dict:
    with document_lock(document_id):
        try:
            if owner:
                doc = Document.objects.get(id=document_id, owner=owner)
            else:
                doc = Document.objects.get(id=document_id, owner__isnull=True)
        except Document.DoesNotExist:
            logger.error(f"Document {document_id} not found")
            return {"success": False, "error": "Document not found"}

        logger.info("Reprocessing document %s: %s", document_id, doc.title)
        _invalidate_caches(document_id)
        success = process_document(document_id)
        doc.refresh_from_db(fields=['status', 'error_message'])
        return {
            "success": success,
            "document_id": document_id,
            "title": doc.title,
            "status": doc.status,
            "error": doc.error_message if not success else None,
        }


def reprocess_document_async(document_id: int, owner=None) -> str:
    task_id = f"doc_reprocess_{document_id}_{uuid.uuid4().hex}"
    from documents.ingestion.document_processor import get_task_queue
    queue = get_task_queue()
    queue.enqueue(task_id, reprocess_document, document_id=document_id, owner=owner)
    return task_id
