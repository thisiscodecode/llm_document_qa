import logging

logger = logging.getLogger(__name__)

MAX_CONTEXT_TOKENS = 3000


def build_context(chunks, max_tokens=MAX_CONTEXT_TOKENS):
    if not chunks:
        return ""

    context_parts = []
    total_tokens = 0

    for chunk in chunks:
        chunk_tokens = len(chunk.content.split())
        if total_tokens + chunk_tokens > max_tokens:
            break
        context_parts.append(chunk.content)
        total_tokens += chunk_tokens

    return "\n\n".join(context_parts)


def build_sources(chunks):
    sources = []
    for i, chunk in enumerate(chunks):
        doc_title = getattr(chunk, 'document_title', None)
        if not doc_title and hasattr(chunk, 'document'):
            doc_title = chunk.document.title if chunk.document else 'Unknown'
        doc_title = doc_title or 'Unknown'

        rerank_score = getattr(chunk, 'rerank_score', 0)
        retrieval_score = getattr(chunk, 'retrieval_score', 0)

        sources.append({
            'index': i + 1,
            'document': doc_title,
            'chunk_index': chunk.chunk_index,
            'page': getattr(chunk, 'page_number', None) or 1,
            'score': rerank_score or retrieval_score,
            'retrieval_score': retrieval_score,
            'rerank_score': rerank_score,
            'lexical_score': getattr(chunk, 'lexical_score', 0),
            'semantic_score': getattr(chunk, 'semantic_score', 0),
            'preview': chunk.content[:200] + '...' if len(chunk.content) > 200 else chunk.content,
        })
    return sources
