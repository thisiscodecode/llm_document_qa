import logging

logger = logging.getLogger(__name__)

MAX_CONTEXT_TOKENS = 3000


def _estimated_tokens(text):
    # A conservative character bound works for both English and Persian when
    # a tokenizer for the configured model is not available.
    return max(len(text.split()), (len(text) + 1) // 2)


def select_context_chunks(chunks, max_tokens=MAX_CONTEXT_TOKENS):
    selected = []
    remaining = max_tokens
    for chunk in chunks:
        content = (chunk.content or '').strip()
        if not content:
            continue
        cost = _estimated_tokens(content) + 40  # source label and metadata
        if cost <= remaining:
            selected.append(chunk)
            remaining -= cost
        elif not selected:
            # A single long chunk is still useful; build_context trims it.
            selected.append(chunk)
            break
    return selected


def build_context(chunks, max_tokens=MAX_CONTEXT_TOKENS):
    context_parts = []
    remaining_chars = max(0, max_tokens * 2)
    for index, chunk in enumerate(select_context_chunks(chunks, max_tokens), 1):
        title = getattr(chunk, 'document_title', None)
        if not title and getattr(chunk, 'document', None):
            title = chunk.document.title
        title = title or 'Unknown'
        header = (f"[Source {index}] {title}, page "
                  f"{getattr(chunk, 'page_number', None) or 1}, "
                  f"chunk {chunk.chunk_index}")
        available = remaining_chars - len(header) - 2
        if available <= 0:
            break
        content = (chunk.content or '').strip()[:available]
        part = f"{header}\n{content}"
        context_parts.append(part)
        remaining_chars -= len(part) + 2

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
            'document_id': getattr(chunk, 'document_id', None),
            'chunk_id': getattr(chunk, 'id', None),
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
