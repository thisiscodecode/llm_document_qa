import logging
from typing import Optional

from .filters import apply_permission_filter, apply_metadata_filter
from .retrievers.bm25 import search_bm25
from .retrievers.simple import search_simple
from .retrievers.vector import search_vector
from .retrievers.hybrid import search_hybrid

logger = logging.getLogger(__name__)


def retrieve(query: str, search_method: str = "hybrid", limit: int = 5,
             document_ids: list = None, owner=None, filters: dict = None) -> list:
    
    permitted_ids = apply_permission_filter(document_ids, owner)
    
    if not permitted_ids:
        logger.info("No permitted documents found")
        return []
    
    if search_method == "bm25":
        chunks = search_bm25(query, limit, permitted_ids)
    elif search_method == "vector":
        chunks = search_vector(query, limit, permitted_ids)
    elif search_method == "simple":
        chunks = search_simple(query, limit, permitted_ids)
    else:
        chunks = search_hybrid(query, limit, permitted_ids)
    
    chunks = apply_metadata_filter(chunks, filters)

    # The database is the authorization boundary even if a retriever or
    # vector index has stale or incorrectly scoped records.
    allowed = set(permitted_ids)
    seen = set()
    safe_chunks = []
    for chunk in chunks:
        chunk_id = getattr(chunk, 'id', None)
        if chunk.document_id not in allowed or chunk_id in seen:
            continue
        seen.add(chunk_id)
        safe_chunks.append(chunk)

    logger.info(f"Retrieved {len(safe_chunks)} chunks using {search_method} method")
    return safe_chunks[:limit]
