import re
import logging

from langchain_openai import ChatOpenAI
from rank_bm25 import BM25Okapi

from .models import DocumentChunk

logger = logging.getLogger(__name__)

def _get_llm():
    import os
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    openrouter_model = os.getenv("OPENROUTER_MODEL")

    if not openrouter_model or not openrouter_api_key:
        return None

    try:
        client = ChatOpenAI(
            model=openrouter_model,
            api_key=openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        return client
    except Exception as e:
        logger.error(f"LLM initialization failed: {e}")
        return None


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def retrieve_relevant_chunks_simple(question, limit=5, document_ids=None):
    chunks = DocumentChunk.objects.all()
    if document_ids:
        chunks = chunks.filter(document_id__in=document_ids)

    scored_chunks = []
    question_words = question.lower().split()

    for chunk in chunks:
        score = 0
        content_lower = chunk.content.lower()
        for word in question_words:
            if word in content_lower:
                score += 1
        if score > 0:
            scored_chunks.append((score, chunk))

    scored_chunks.sort(reverse=True, key=lambda x: x[0])
    return [chunk for score, chunk in scored_chunks[:limit]]


def retrieve_relevant_chunks_bm25(question, limit=5, document_ids=None):
    chunks = list(DocumentChunk.objects.all())
    if document_ids:
        chunks = list(DocumentChunk.objects.filter(document_id__in=document_ids))
    if not chunks:
        return []

    tokenized_chunks = [tokenize(c.content) for c in chunks]
    bm25 = BM25Okapi(tokenized_chunks)

    tokenized_question = tokenize(question)
    scores = bm25.get_scores(tokenized_question)

    scored_chunks = list(zip(scores, chunks))
    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    return [chunk for score, chunk in scored_chunks[:limit]]


def retrieve_relevant_chunks_vector(question, limit=5, document_ids=None):
    try:
        from .rag.retrievers.vector import search_vector
        return search_vector(question, limit, document_ids)
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        return retrieve_relevant_chunks_bm25(question, limit, document_ids)


def retrieve_relevant_chunks_hybrid(question, limit=5, document_ids=None):
    bm25_chunks = retrieve_relevant_chunks_bm25(question, limit, document_ids)
    vector_chunks = retrieve_relevant_chunks_vector(question, limit, document_ids)

    seen_ids = set()
    merged = []

    for chunk in vector_chunks + bm25_chunks:
        if chunk.id not in seen_ids:
            seen_ids.add(chunk.id)
            merged.append(chunk)

    return merged[:limit]


def rerank_chunks(question, chunks, limit=3):
    if not chunks:
        return chunks

    question_lower = question.lower()
    question_words = set(tokenize(question))

    scored = []
    for chunk in chunks:
        content_lower = chunk.content.lower()

        exact_match_score = 0
        if question_lower in content_lower:
            exact_match_score = 10

        word_overlap = len(question_words & set(tokenize(chunk.content)))

        position_bonus = 0
        first_occurrence = content_lower.find(list(question_words)[0] if question_words else '')
        if first_occurrence != -1 and first_occurrence < 200:
            position_bonus = 2

        total_score = exact_match_score + word_overlap + position_bonus
        scored.append((total_score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for score, chunk in scored[:limit]]


def retrieve_relevant_chunks(question, search_method="hybrid", limit=5, document_ids=None):
    if search_method == "bm25":
        chunks = retrieve_relevant_chunks_bm25(question, limit, document_ids)
    elif search_method == "vector":
        chunks = retrieve_relevant_chunks_vector(question, limit, document_ids)
    elif search_method == "simple":
        chunks = retrieve_relevant_chunks_simple(question, limit, document_ids)
    else:
        chunks = retrieve_relevant_chunks_hybrid(question, limit, document_ids)

    return rerank_chunks(question, chunks, limit=3)


def rebuild_vector_index(document_id=None):
    if document_id:
        from .indexing.index_sync import upsert_document_chunks
        return upsert_document_chunks(document_id)
    from .indexing.index_sync import rebuild_global_index
    return rebuild_global_index()


def generate_answer(question, search_method="hybrid", document_ids=None):
    relevant_chunks = retrieve_relevant_chunks(
        question=question,
        search_method=search_method,
        limit=5,
        document_ids=document_ids,
    )

    context = "\n\n".join([chunk.content for chunk in relevant_chunks])

    source_chunks = []
    for i, chunk in enumerate(relevant_chunks):
        source_chunks.append({
            'index': i + 1,
            'document': chunk.document.title,
            'chunk_index': chunk.chunk_index,
            'page': getattr(chunk, 'page_number', None) or 1,
            'score': 0,
            'preview': chunk.content[:200] + '...' if len(chunk.content) > 200 else chunk.content,
        })

    if not context:
        return {
            "answer": "I could not find relevant information in the uploaded documents.",
            "context": "",
            "search_method": search_method,
            "sources": source_chunks,
        }

    llm = _get_llm()
    if llm is None:
        return {
            "answer": "LLM service error: OPENROUTER_API_KEY and OPENROUTER_MODEL must be set.",
            "context": context,
            "search_method": search_method,
            "sources": source_chunks,
        }

    sources_text = ""
    for s in source_chunks:
        sources_text += f"\n[Source {s['index']}: {s['document']}, page {s['page']}, chunk {s['chunk_index']}]\n{s['preview']}\n"

    prompt = f"""Answer the question using ONLY the provided context.

Rules:
- ALWAYS respond in the SAME LANGUAGE as the question (e.g., if the question is in Persian/Farsi, respond in Persian)
- Give a SHORT, DIRECT answer for simple questions (who, what, where, when)
- Only provide detailed answers when the question explicitly asks for details
- Do NOT dump the entire document - only extract what's relevant
- Use clean formatting: use bullet points or numbered lists for multiple items
- Reference sources like [Source 1] when citing
- If the answer is not in the context, say: "I could not find the answer in the uploaded documents."
- For Persian responses, use proper Persian punctuation and formatting

Context:
{context}

Sources:
{sources_text}

Question:
{question}
"""

    try:
        response = llm.invoke(prompt)
        answer = response.content
    except Exception as e:
        answer = f"LLM service error: {str(e)}"

    return {
        "answer": answer,
        "context": context,
        "search_method": search_method,
        "sources": source_chunks,
    }
