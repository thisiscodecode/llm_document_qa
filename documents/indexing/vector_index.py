"""Chroma-backed vector index for document chunks.

The relational database remains the source of truth for documents and chunks.
Chroma stores a derived, rebuildable search index keyed by stable chunk IDs.
"""

import logging
import hashlib
import json
import math
import os
import tempfile
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


def embedding_identity() -> str:
    """Stable identifier for the embedding model and its provider endpoint."""
    model = os.getenv("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small")
    endpoint = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
    return hashlib.sha256(f"{endpoint}|{model}".encode("utf-8")).hexdigest()[:12]


def _collection_name() -> str:
    return f"{settings.CHROMA_COLLECTION}-{embedding_identity()}"


@lru_cache(maxsize=8)
def _get_collection_for_identity(name: str, identity: str, model: str):
    collection = _get_client().get_or_create_collection(
        name=name,
        metadata={
            "hnsw:space": "cosine",
            "description": "Document chunks for grounded RAG",
            "embedding_model": model,
            "embedding_identity": identity,
        },
        embedding_function=None,
    )
    if (collection.metadata or {}).get("embedding_identity") != identity:
        if collection.count() == 0:
            collection.modify(metadata={
                "hnsw:space": "cosine",
                "description": "Document chunks for grounded RAG",
                "embedding_model": model,
                "embedding_identity": identity,
            })
        else:
            raise RuntimeError(
                f"Chroma collection {name} has a different embedding identity; "
                "use a new collection name and rebuild the index"
            )
    return collection


def _get_collection():
    return _get_collection_for_identity(
        _collection_name(),
        embedding_identity(),
        os.getenv("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small"),
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
        return min(configured, max(1, maximum))
    except (AttributeError, TypeError, ValueError):
        return configured


def _stage_embeddings(chunks: list, destination) -> None:
    """Validate all embeddings before changing Chroma, spilling large sets to disk."""
    from .embedding_client import embed_documents

    batch_size = min(_batch_size(), max(1, int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))))
    dimension = None
    for batch in _batches(chunks, batch_size):
        embedded = embed_documents([chunk.content for chunk in batch])
        if embedded is None or len(embedded) != len(batch):
            raise ValueError("Embedding service returned an incomplete batch")
        for vector in embedded:
            if not vector or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in vector
            ):
                raise ValueError("Embedding service returned an invalid vector")
            if dimension is None:
                dimension = len(vector)
            elif len(vector) != dimension:
                raise ValueError("Embedding dimensions changed within the document")
            destination.write(json.dumps([float(value) for value in vector]))
            destination.write("\n")
    destination.seek(0)


def upsert_vectors(chunks, document_id: Optional[int] = None) -> bool:
    """Embed and upsert chunks, replacing stale records for one document."""
    chunks = list(chunks)
    if document_id is not None and any(c.document_id != document_id for c in chunks):
        raise ValueError("All chunks must belong to the requested document")

    collection = None
    write_started = False

    try:
        # Embedding failures must never erase a document's previous vectors.
        with tempfile.SpooledTemporaryFile(
            max_size=8 * 1024 * 1024, mode="w+t", encoding="utf-8"
        ) as staged:
            if chunks:
                _stage_embeddings(chunks, staged)
            collection = _get_collection()
            with _write_lock:
                if document_id is not None:
                    write_started = True
                    collection.delete(where={"document_id": int(document_id)})

                for batch in _batches(chunks, _batch_size()):
                    embeddings = [json.loads(staged.readline()) for _ in batch]
                    write_started = True
                    collection.upsert(
                        ids=[_chunk_id(chunk) for chunk in batch],
                        embeddings=embeddings,
                        documents=[chunk.content for chunk in batch],
                        metadatas=[_chunk_metadata(chunk) for chunk in batch],
                    )

                # A global rebuild also removes records whose source chunks vanished.
                if document_id is None:
                    current_ids = {_chunk_id(chunk) for chunk in chunks}
                    indexed = collection.get(include=[])
                    stale_ids = [item for item in indexed.get("ids", []) if item not in current_ids]
                    for batch in _batches(stale_ids, _batch_size()):
                        write_started = True
                        collection.delete(ids=batch)

        logger.info("Indexed %s chunks in Chroma", len(chunks))
        return True
    except Exception:
        logger.exception("Chroma vector upsert failed")
        # A failed document write may have left some new batches in Chroma.
        # Failed documents are not queryable, and removing the partial index
        # keeps subsequent reprocessing from inheriting those records.
        if document_id is not None and write_started and collection is not None:
            try:
                with _write_lock:
                    collection.delete(where={"document_id": int(document_id)})
            except Exception:
                logger.exception("Failed to clean up partial vectors for document %s", document_id)
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
    chunks = list(DocumentChunk.objects.select_related("document").filter(document__status="ready"))
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
            "collection": _collection_name(),
            "total_vectors": collection.count(),
            "documents_indexed": len(document_ids),
        }
    except Exception as exc:
        logger.exception("Failed to read Chroma stats")
        return {
            "backend": "chroma",
            "collection": _collection_name(),
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
