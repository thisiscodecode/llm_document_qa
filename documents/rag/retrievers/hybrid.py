import logging

from .bm25 import search_bm25
from .vector import search_vector

logger = logging.getLogger(__name__)


def search_hybrid(query, limit=5, document_ids=None):
    candidate_limit = max(limit * 2, limit)
    bm25_chunks = search_bm25(query, candidate_limit, document_ids)
    vector_chunks = search_vector(query, candidate_limit, document_ids)

    # Reciprocal rank fusion combines lexical and semantic rankings without
    # assuming their raw scores share the same scale.
    rrf_k = 60
    chunk_scores = {}
    for ranking in (vector_chunks, bm25_chunks):
        for rank, chunk in enumerate(ranking, start=1):
            entry = chunk_scores.setdefault(chunk.id, {"chunk": chunk, "score": 0.0})
            entry["score"] += 1.0 / (rrf_k + rank)

    merged = []
    max_rrf_score = 2.0 / (rrf_k + 1)
    for entry in chunk_scores.values():
        entry["chunk"].retrieval_score = entry["score"] / max_rrf_score
        merged.append(entry["chunk"])
    merged.sort(key=lambda c: getattr(c, 'retrieval_score', 0.0), reverse=True)
    return merged[:limit]
