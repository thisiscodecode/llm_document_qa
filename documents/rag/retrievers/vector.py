import logging

from documents.models import DocumentChunk

logger = logging.getLogger(__name__)


def search_vector(query, limit=5, document_ids=None):
    from documents.indexing.embedding_client import embed_query
    from documents.indexing.vector_index import search_vectors

    if document_ids is not None and not document_ids:
        return []

    eligible_chunks = DocumentChunk.objects.filter(document__status='ready')
    if document_ids is not None:
        eligible_chunks = eligible_chunks.filter(document_id__in=document_ids)
    if not eligible_chunks.exists():
        return []

    from ..cache import get_cached_embedding, set_cached_embedding
    from ..feature_flags import is_enabled

    try:
        if is_enabled('advanced_caching'):
            cached_emb = get_cached_embedding(query)
            if cached_emb:
                query_embedding = cached_emb
            else:
                query_embedding = embed_query(query)
                if query_embedding:
                    set_cached_embedding(query, query_embedding)
        else:
            query_embedding = embed_query(query)
    except Exception as exc:
        logger.warning("Query embedding failed: %s", exc)
        return []

    if not query_embedding:
        logger.warning("Query embedding failed")
        return []

    try:
        # Fetch more candidates than the final context needs so hybrid fusion and
        # reranking have enough recall to work with.
        ranked = search_vectors(
            query_embedding,
            limit=max(limit * 2, limit),
            document_ids=document_ids,
        )
        if not ranked:
            return []

        chunk_ids = [chunk_id for chunk_id, _score in ranked]
        chunks = DocumentChunk.objects.select_related("document").filter(
            id__in=chunk_ids, document__status='ready',
        )
        chunk_map = {chunk.id: chunk for chunk in chunks}
        results = []
        for chunk_id, score in ranked:
            chunk = chunk_map.get(chunk_id)
            if chunk is not None:
                chunk.retrieval_score = score
                results.append(chunk)

        return results[:limit]

    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        return []
