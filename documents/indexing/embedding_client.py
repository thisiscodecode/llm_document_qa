import os
import logging
from typing import Optional, List

from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)


def _create_embeddings_client() -> Optional[OpenAIEmbeddings]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.warning("OPENROUTER_API_KEY not set")
        return None

    try:
        return OpenAIEmbeddings(
            model=os.getenv("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small"),
            openai_api_key=api_key,
            openai_api_base=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            chunk_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "64")),
            max_retries=int(os.getenv("EMBEDDING_MAX_RETRIES", "3")),
        )
    except Exception as e:
        logger.error(f"Embeddings initialization failed: {e}")
        return None


def embed_query(query: str) -> Optional[List[float]]:
    client = _create_embeddings_client()
    if not client:
        return None

    try:
        embedding = client.embed_query(query)
        logger.debug(f"Embedded query: {len(embedding)} dimensions")
        return embedding
    except Exception as e:
        logger.error(f"Query embedding failed: {e}")
        return None


def embed_documents(texts: List[str]) -> Optional[List[List[float]]]:
    client = _create_embeddings_client()
    if not client:
        return None

    try:
        embeddings = client.embed_documents(texts)
        logger.debug(f"Embedded {len(texts)} documents, {len(embeddings[0])} dimensions each")
        return embeddings
    except Exception as e:
        logger.error(f"Document embedding failed: {e}")
        return None
