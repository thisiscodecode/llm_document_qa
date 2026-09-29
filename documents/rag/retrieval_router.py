import logging
from typing import Optional

from .filters import apply_permission_filter, apply_metadata_filter
from .retrievers.bm25 import search_bm25
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
        chunks = search_bm25(query, limit, permitted_ids)
    else:
        chunks = search_hybrid(query, limit, permitted_ids)
    
    chunks = apply_metadata_filter(chunks, filters)
    
    logger.info(f"Retrieved {len(chunks)} chunks using {search_method} method")
    return chunks