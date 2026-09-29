import re
import logging

from documents.models import DocumentChunk

logger = logging.getLogger(__name__)


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def search_bm25(query, limit=5, document_ids=None):
    from rank_bm25 import BM25Okapi

    chunks = list(DocumentChunk.objects.all())
    if document_ids:
        chunks = list(DocumentChunk.objects.filter(document_id__in=document_ids))
    if not chunks:
        return []

    tokenized_chunks = [tokenize(c.content) for c in chunks]
    bm25 = BM25Okapi(tokenized_chunks)

    tokenized_question = tokenize(query)
    scores = bm25.get_scores(tokenized_question)

    scored_chunks = list(zip(scores, chunks))
    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    max_score = max(s for s, _ in scored_chunks) if scored_chunks else 1.0
    for score, chunk in scored_chunks:
        chunk.retrieval_score = float(score / max_score) if max_score > 0 else 0.0

    return [chunk for score, chunk in scored_chunks[:limit]]