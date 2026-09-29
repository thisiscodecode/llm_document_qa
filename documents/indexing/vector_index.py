"""Chroma-backed vector index for document chunks.

The relational database remains the source of truth for documents and chunks.
Chroma stores a derived, rebuildable search index keyed by stable chunk IDs.
"""

import logging
import threading
from functools import lru_cache
from typing import Iterable, Optional

from django.conf import settings

from documents.models import DocumentChunk

logger = logging.getLogger(__name__)

_write_lock = threading.RLock()


@lru_cache(maxsize=1)
def _get_client():
    """Create one Chroma client per process."""
    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError as exc:
        raise RuntimeError(
            "ChromaDB is not installed. Run `pip install -r requirements.txt`."
        ) from exc

    host = getattr(settings, "CHROMA_HOST", "")
    common = {
        "tenant": getattr(settings, "CHROMA_TENANT", "default_tenant"),
        "database": getattr(settings, "CHROMA_DATABASE", "default_database"),
    }
    if host:
        return chromadb.HttpClient(
            host=host,
            port=getattr(settings, "CHROMA_PORT", 8000),
            ssl=getattr(settings, "CHROMA_SSL", False),
            **common,
        )

    return chromadb.PersistentClient(
        path=str(settings.CHROMA_PATH),
        settings=Settings(anonymized_telemetry=False),
        **common,
    )


@lru_cache(maxsize=1)
def _get_collection():
    return _get_client().get_or_create_collection(
        name=settings.CHROMA_COLLECTION,
        metadata={
            "hnsw:space": "cosine",
            "description": "Document chunks for grounded RAG",
        },
        embedding_function=None,
    )


def _chunk_id(chunk) -> str:
    return str(chunk.id)


def _chunk_metadata(chunk) -> dict:
    return {
        "document_id": int(chunk.document_id),
        "chunk_id": int(chunk.id),
        "chunk_index": int(chunk.chunk_index),
        "page_number": int(getattr(chunk, "page_number", 1) or 1),
        "document_title": str(chunk.document.title),
    }


def _batches(items: list, batch_size: int) -> Iterable[list]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def _batch_size() -> int:
    configured = max(1, int(getattr(settings, "CHROMA_BATCH_SIZE", 128)))
    try:
        maximum = int(_get_client().get_max_batch_size())
        return min(configured, maximum)
    except (AttributeError, TypeError, ValueError):
        return configured


def upsert_vectors(chunks, document_id: Optional[int] = None) -> bool:
    """Embed and upsert chunks, replacing stale records for one document."""
    chunks = list(chunks)
    if not chunks:
        logger.info("No chunks supplied for vector indexing")
        return False

    if document_id is not None and any(c.document_id != document_id for c in chunks):
        raise ValueError("All chunks must belong to the requested document")

    from .embedding_client import embed_documents

    embeddings = embed_documents([chunk.content for chunk in chunks])
    if not embeddings or len(embeddings) != len(chunks):
        logger.error("Embedding service returned an incomplete result")
        return False

    records = list(zip(chunks, embeddings))
    collection = _get_collection()

    try:
        with _write_lock:
            if document_id is not None:
                collection.delete(where={"document_id": int(document_id)})

            for batch in _batches(records, _batch_size()):
                batch_chunks = [item[0] for item in batch]
                collection.upsert(
                    ids=[_chunk_id(chunk) for chunk in batch_chunks],
                    embeddings=[item[1] for item in batch],
                    documents=[chunk.content for chunk in batch_chunks],
                    metadatas=[_chunk_metadata(chunk) for chunk in batch_chunks],
                )

            # A global rebuild must also remove records whose source chunks vanished.
            if document_id is None:
                current_ids = {_chunk_id(chunk) for chunk in chunks}
                indexed = collection.get(include=[])
                stale_ids = [item for item in indexed.get("ids", []) if item not in current_ids]
                for batch in _batches(stale_ids, _batch_size()):
                    collection.delete(ids=batch)

        logger.info("Indexed %s chunks in Chroma", len(chunks))
        return True
    except Exception:
        logger.exception("Chroma vector upsert failed")
        return False


def delete_vectors_for_document(document_id: int) -> bool:
    try:
        with _write_lock:
            _get_collection().delete(where={"document_id": int(document_id)})
        logger.info("Deleted Chroma vectors for document %s", document_id)
        return True
    except Exception:
        logger.exception("Failed to delete Chroma vectors for document %s", document_id)
        return False


def rebuild_full_index() -> bool:
    chunks = list(DocumentChunk.objects.select_related("document").all())
    if not chunks:
        logger.info("No chunks to index")
        return False
    return upsert_vectors(chunks)


def get_chunk_ids_for_document(document_id: int) -> list[int]:
    try:
        result = _get_collection().get(
            where={"document_id": int(document_id)},
            include=[],
        )
        return [int(item) for item in result.get("ids", [])]
    except Exception:
        logger.exception("Failed to read Chroma IDs for document %s", document_id)
        return []


def get_mapping_stats() -> dict:
    try:
        collection = _get_collection()
        result = collection.get(include=["metadatas"])
        metadatas = result.get("metadatas", []) or []
        document_ids = {
            metadata.get("document_id")
            for metadata in metadatas
            if metadata and metadata.get("document_id") is not None
        }
        return {
            "backend": "chroma",
            "collection": settings.CHROMA_COLLECTION,
            "total_vectors": collection.count(),
            "documents_indexed": len(document_ids),
        }
    except Exception as exc:
        logger.exception("Failed to read Chroma stats")
        return {
            "backend": "chroma",
            "collection": settings.CHROMA_COLLECTION,
            "total_vectors": 0,
            "documents_indexed": 0,
            "error": str(exc),
        }


def _document_filter(document_ids: Optional[list[int]]) -> Optional[dict]:
    if not document_ids:
        return None
    normalized = sorted({int(item) for item in document_ids})
    if len(normalized) == 1:
        return {"document_id": normalized[0]}
    return {"document_id": {"$in": normalized}}


def search_vectors(
    query_embedding,
    limit: int = 5,
    document_id: Optional[int] = None,
    document_ids: Optional[list[int]] = None,
) -> list[tuple[int, float]]:
    """Return ``(chunk_id, cosine relevance)`` pairs in ranked order."""
    if query_embedding is None or len(query_embedding) == 0 or limit <= 0:
        return []

    if document_id is not None:
        document_ids = [document_id]

    try:
        collection = _get_collection()
        available = collection.count()
        if available == 0:
            return []

        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": min(int(limit), available),
            "include": ["distances"],
        }
        where = _document_filter(document_ids)
        if where:
            query_kwargs["where"] = where

        result = collection.query(**query_kwargs)
        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        ranked = []
        for chunk_id, distance in zip(ids, distances):
            relevance = max(0.0, min(1.0, 1.0 - float(distance)))
            ranked.append((int(chunk_id), relevance))
        return ranked
    except Exception:
        logger.exception("Chroma vector search failed")
        return []
