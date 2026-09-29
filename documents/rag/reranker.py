import re
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

CROSS_ENCODER_MAX_CHUNKS = 20


@dataclass
class RerankConfig:
    lexical_weight: float = 0.3
    semantic_weight: float = 0.5
    position_weight: float = 0.2
    use_semantic: bool = True
    use_cross_encoder: bool = False


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def compute_lexical_score(query: str, content: str) -> float:
    query_lower = query.lower()
    content_lower = content.lower()
    query_words = set(tokenize(query))
    content_words = set(tokenize(content))

    exact_match_score = 0.0
    if query_lower in content_lower:
        exact_match_score = 10.0

    word_overlap = len(query_words & content_words)
    word_overlap_score = word_overlap / max(len(query_words), 1) * 5.0

    return exact_match_score + word_overlap_score


@lru_cache(maxsize=1)
def _cross_encoder_model():
    from sentence_transformers import CrossEncoder
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


def _cross_encoder_scores(query: str, chunks: list) -> Optional[list[float]]:
    try:
        # Reuse the loaded model across requests.
        scores = _cross_encoder_model().predict(
            [(query, chunk.content[:512]) for chunk in chunks]
        )
        return [float(score) * 10.0 for score in scores]
    except ImportError:
        logger.debug("sentence-transformers not installed, cross-encoder unavailable")
        return None
    except Exception as e:
        logger.warning(f"Cross-encoder scoring failed: {e}")
        return None


def compute_cross_encoder_score(query: str, content: str) -> Optional[float]:
    candidate = type('Candidate', (), {'content': content})()
    scores = _cross_encoder_scores(query, [candidate])
    return scores[0] if scores else None


def compute_semantic_score(query: str, content: str) -> float:
    cross_score = compute_cross_encoder_score(query, content)
    if cross_score is not None:
        return cross_score

    try:
        from ..indexing.embedding_client import embed_query
        query_embedding = embed_query(query)
        if not query_embedding:
            return 0.0

        content_embedding = embed_query(content[:1000])
        if not content_embedding:
            return 0.0

        import numpy as np
        query_vec = np.array(query_embedding)
        content_vec = np.array(content_embedding)

        similarity = np.dot(query_vec, content_vec) / (
            np.linalg.norm(query_vec) * np.linalg.norm(content_vec) + 1e-10
        )

        return float(similarity) * 10.0

    except Exception as e:
        logger.warning(f"Semantic scoring failed: {e}")
        return 0.0


def compute_position_score(query: str, content: str) -> float:
    query_words = tokenize(query)
    content_lower = content.lower()

    if not query_words:
        return 0.0

    first_word = query_words[0]
    first_occurrence = content_lower.find(first_word)

    if first_occurrence == -1:
        return 0.0

    if first_occurrence < 50:
        return 5.0
    elif first_occurrence < 100:
        return 3.0
    elif first_occurrence < 200:
        return 1.0

    return 0.0


def compute_chunk_quality_score(content: str) -> float:
    length = len(content)
    
    if length < 50:
        return -2.0
    elif length < 100:
        return -1.0
    elif length > 2000:
        return -0.5
    
    return 0.0


def rerank(query: str, chunks: list, limit: int = 3, 
           config: Optional[RerankConfig] = None) -> list:
    if not chunks:
        return chunks

    if config is None:
        from .feature_flags import get_variant
        use_cross = get_variant('cross_encoder_reranker') == 'cross_encoder'
        config = RerankConfig(use_cross_encoder=use_cross)

    input_chunks = chunks
    if config.use_cross_encoder and len(chunks) > CROSS_ENCODER_MAX_CHUNKS:
        logger.info(f"Capping cross-encoder input from {len(chunks)} to {CROSS_ENCODER_MAX_CHUNKS} chunks")
        input_chunks = chunks[:CROSS_ENCODER_MAX_CHUNKS]

    cross_scores = _cross_encoder_scores(query, input_chunks) if config.use_cross_encoder else None
    scored = []
    for index, chunk in enumerate(input_chunks):
        lexical_score = compute_lexical_score(query, chunk.content)

        semantic_score = 0.0
        if config.use_semantic:
            if config.use_cross_encoder:
                # A failed optional model must not trigger one remote
                # embedding request per candidate chunk.
                semantic_score = (
                    cross_scores[index] if cross_scores is not None
                    else float(getattr(chunk, 'retrieval_score', 0.0)) * 10.0
                )
            else:
                # Chroma already calculated semantic similarity during retrieval.
                # Reusing it avoids one embedding API call per candidate chunk.
                semantic_score = float(getattr(chunk, 'retrieval_score', 0.0)) * 10.0

        position_score = compute_position_score(query, chunk.content)
        quality_score = compute_chunk_quality_score(chunk.content)

        total_score = (
            config.lexical_weight * lexical_score +
            config.semantic_weight * semantic_score +
            config.position_weight * position_score +
            quality_score
        )

        scored.append({
            'chunk': chunk,
            'total_score': total_score,
            'lexical_score': lexical_score,
            'semantic_score': semantic_score,
            'position_score': position_score,
            'quality_score': quality_score,
        })

    scored.sort(key=lambda x: x['total_score'], reverse=True)

    from .contracts import RerankedChunk
    result = []
    for item in scored[:limit]:
        chunk = item['chunk']
        result.append(RerankedChunk(
            id=chunk.id,
            content=chunk.content,
            document_id=chunk.document_id,
            document_title=chunk.document.title if hasattr(chunk, 'document') else '',
            chunk_index=chunk.chunk_index,
            page_number=getattr(chunk, 'page_number', 1),
            rerank_score=item['total_score'],
            lexical_score=item['lexical_score'],
            semantic_score=item['semantic_score'],
            position_score=item['position_score'],
            retrieval_score=getattr(chunk, 'retrieval_score', 0.0),
        ))

    return result
