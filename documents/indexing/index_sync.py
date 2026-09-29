import logging

from documents.models import DocumentChunk

logger = logging.getLogger(__name__)


def upsert_document_chunks(document_id):
    chunks = list(
        DocumentChunk.objects.select_related('document').filter(document_id=document_id)
    )
    if not chunks:
        logger.info(f"No chunks to index for document {document_id}")
        delete_document_chunks(document_id)
        return False

    from .vector_index import upsert_vectors
    return upsert_vectors(chunks, document_id=document_id)


def delete_document_chunks(document_id):
    from .vector_index import delete_vectors_for_document
    return delete_vectors_for_document(document_id)


def rebuild_global_index():
    # Failed and in-progress documents must never be reintroduced by a rebuild.
    chunks = list(
        DocumentChunk.objects.select_related('document').filter(document__status='ready')
    )

    from .vector_index import upsert_vectors
    return upsert_vectors(chunks, document_id=None)
