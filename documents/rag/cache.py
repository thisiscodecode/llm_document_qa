import time
import hashlib
import logging
import threading
from typing import Optional, Any

logger = logging.getLogger(__name__)

DEFAULT_TTL = 300
MAX_ENTRIES = 1000


class TTLCache:
    def __init__(self, ttl: int = DEFAULT_TTL, max_entries: int = MAX_ENTRIES):
        self._store: dict[str, tuple[float, Any]] = {}
        self._ttl = ttl
        self._max_entries = max_entries
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._store:
                expiry, value = self._store[key]
                if time.time() < expiry:
                    self._hits += 1
                    return value
                del self._store[key]
            self._misses += 1
            return None

    def set(self, key: str, value: Any, ttl: int = None):
        with self._lock:
            if len(self._store) >= self._max_entries:
                self._evict_expired()
                if len(self._store) >= self._max_entries:
                    self._evict_oldest()
            self._store[key] = (time.time() + (ttl or self._ttl), value)

    def invalidate(self, key: str):
        with self._lock:
            self._store.pop(key, None)

    def clear(self):
        with self._lock:
            self._store.clear()

    def invalidate_by_value(self, predicate):
        with self._lock:
            to_remove = []
            for key, (expiry, value) in self._store.items():
                if time.time() < expiry and predicate(value):
                    to_remove.append(key)
            for key in to_remove:
                del self._store[key]
            return len(to_remove)

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': self._hits / total if total else 0,
            'size': len(self._store),
        }

    def _evict_expired(self):
        now = time.time()
        expired = [k for k, (exp, _) in self._store.items() if now >= exp]
        for k in expired:
            del self._store[k]

    def _evict_oldest(self):
        if not self._store:
            return
        oldest_key = min(self._store, key=lambda k: self._store[k][0])
        del self._store[oldest_key]


def _make_key(*args) -> str:
    return hashlib.sha256(repr(args).encode('utf-8')).hexdigest()


query_cache = TTLCache(ttl=300, max_entries=500)
embedding_cache = TTLCache(ttl=600, max_entries=2000)
metadata_cache = TTLCache(ttl=60, max_entries=500)


def get_cached_query(query: str, method: str, doc_ids: tuple) -> Optional[dict]:
    key = _make_key('query', query, method, doc_ids)
    return query_cache.get(key)


def set_cached_query(query: str, method: str, doc_ids: tuple, result: dict):
    key = _make_key('query', query, method, doc_ids)
    query_cache.set(key, result)


def get_cached_embedding(text: str) -> Optional[list]:
    from documents.indexing.vector_index import embedding_identity
    key = _make_key('emb', embedding_identity(), text)
    return embedding_cache.get(key)


def set_cached_embedding(text: str, embedding: list):
    from documents.indexing.vector_index import embedding_identity
    key = _make_key('emb', embedding_identity(), text)
    embedding_cache.set(key, embedding, ttl=600)


def get_cached_permissions(owner_id: Optional[int], doc_ids: tuple) -> Optional[list]:
    key = _make_key('perm', owner_id, doc_ids)
    return metadata_cache.get(key)


def set_cached_permissions(owner_id: Optional[int], doc_ids: tuple, result: list):
    key = _make_key('perm', owner_id, doc_ids)
    metadata_cache.set(key, result, ttl=60)


def get_all_cache_stats() -> dict:
    return {
        'query_cache': query_cache.stats(),
        'embedding_cache': embedding_cache.stats(),
        'metadata_cache': metadata_cache.stats(),
    }


def clear_all_caches():
    query_cache.clear()
    embedding_cache.clear()
    metadata_cache.clear()


def invalidate_queries_by_document(document_id: int) -> int:
    def contains_doc(value):
        if isinstance(value, dict):
            sources = value.get('sources', [])
            for s in sources:
                if isinstance(s, dict) and s.get('document_id') == document_id:
                    return True
        return False

    count = query_cache.invalidate_by_value(contains_doc)
    if count:
        logger.info(f"Invalidated {count} cached queries referencing document {document_id}")
    return count


def invalidate_permissions_for_document(document_id: int) -> int:
    def references_doc(value):
        if isinstance(value, list):
            return document_id in value
        return False

    count = metadata_cache.invalidate_by_value(references_doc)
    if count:
        logger.info(f"Invalidated {count} cached permissions referencing document {document_id}")
    return count
