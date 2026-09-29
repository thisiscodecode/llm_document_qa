"""A cheap lexical baseline: rank chunks by distinct query-term coverage."""

import heapq

from documents.models import DocumentChunk

from .bm25 import QUERY_STOP_WORDS, tokenize


def search_simple(query, limit=5, document_ids=None):
    if limit <= 0 or (document_ids is not None and not document_ids):
        return []

    query_terms = set(tokenize(query)) - QUERY_STOP_WORDS
    if not query_terms:
        return []

    queryset = DocumentChunk.objects.select_related('document').filter(
        document__status='ready',
    )
    if document_ids is not None:
        queryset = queryset.filter(document_id__in=document_ids)

    top = []
    for chunk in queryset.iterator(chunk_size=1000):
        coverage = len(query_terms.intersection(tokenize(chunk.content)))
        if not coverage:
            continue
        score = coverage / len(query_terms)
        chunk.retrieval_score = score
        # Shorter chunks break coverage ties; lower IDs make ties stable.
        rank = (score, -len(chunk.content), -chunk.pk)
        entry = (rank, chunk.pk, chunk)
        if len(top) < limit:
            heapq.heappush(top, entry)
        elif rank > top[0][0]:
            heapq.heapreplace(top, entry)

    return [entry[2] for entry in sorted(top, key=lambda item: item[0], reverse=True)]
