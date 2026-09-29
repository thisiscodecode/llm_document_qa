import re
import logging

from documents.models import DocumentChunk

logger = logging.getLogger(__name__)

QUERY_STOP_WORDS = {
    'a', 'an', 'and', 'are', 'for', 'how', 'in', 'is', 'of', 'on',
    'the', 'to', 'what', 'when', 'where', 'which', 'who', 'why',
    'از', 'است', 'به', 'در', 'را', 'که', 'چه', 'کی', 'کجا', 'های',
}


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def search_bm25(query, limit=5, document_ids=None):
    from rank_bm25 import BM25Okapi

    if document_ids is not None and not document_ids:
        return []
    queryset = DocumentChunk.objects.select_related('document').filter(document__status='ready')
    if document_ids is not None:
        queryset = queryset.filter(document_id__in=document_ids)
    chunks = list(queryset)
    if not chunks:
        return []

    tokenized_question = tokenize(query)
    query_terms = set(tokenized_question) - QUERY_STOP_WORDS
    if not query_terms:
        return []

    candidates = [(chunk, tokenize(chunk.content)) for chunk in chunks]
    candidates = [item for item in candidates if query_terms.intersection(item[1])]
    if not candidates:
        return []
    chunks = [chunk for chunk, _tokens in candidates]
    tokenized_chunks = [tokens for _chunk, tokens in candidates]
    bm25 = BM25Okapi(tokenized_chunks)

    scores = bm25.get_scores(tokenized_question)

    scored_chunks = list(zip(scores, chunks))
    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    max_score = max(s for s, _ in scored_chunks) if scored_chunks else 1.0
    for score, chunk in scored_chunks:
        chunk.retrieval_score = float(score / max_score) if max_score > 0 else 0.0

    return [chunk for score, chunk in scored_chunks[:limit]]
