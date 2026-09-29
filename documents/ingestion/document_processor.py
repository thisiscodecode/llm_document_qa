import logging
import threading
import time
import uuid
import weakref
from contextlib import contextmanager
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Callable
from datetime import datetime

from django.utils import timezone

from documents.models import Document
from .extractors import extract_text
from .chunkers import create_chunks
from documents.indexing.index_sync import delete_document_chunks, upsert_document_chunks

logger = logging.getLogger(__name__)

_document_locks = weakref.WeakValueDictionary()
_document_locks_guard = threading.Lock()


@contextmanager
def document_lock(document_id: int):
    """Serialize process/reprocess/delete operations for a document in this process."""
    with _document_locks_guard:
        lock = _document_locks.get(document_id)
        if lock is None:
            lock = threading.RLock()
            _document_locks[document_id] = lock
    with lock:
        yield


def _log_with_context(level: str, msg: str, **kwargs):
    from config.middleware import get_request_id
    request_id = get_request_id()
    log_fn = getattr(logger, level)
    if request_id:
        log_fn(f"[{request_id}] {msg}", **kwargs)
    else:
        log_fn(msg, **kwargs)


class TaskStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class TaskResult:
    task_id: str
    document_id: int
    status: TaskStatus
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    retry_count: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


class TaskQueue:
    def __init__(self, max_workers: int = 3, max_retries: int = 2):
        self._tasks: dict[str, TaskResult] = {}
        self._queue: list[tuple[str, Callable]] = []
        self._workers: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._start_lock = threading.Lock()
        self._max_workers = max_workers
        self._max_retries = max_retries
        self._running = False

    def start(self):
        with self._start_lock:
            if self._running:
                return
            self._running = True
            for i in range(self._max_workers):
                worker = threading.Thread(target=self._worker_loop, daemon=True)
                worker.start()
                self._workers.append(worker)
            logger.info(f"Task queue started with {self._max_workers} workers")

    def stop(self):
        self._running = False
        logger.info("Task queue stopped")

    def enqueue(self, task_id: str, func: Callable, *args, **kwargs) -> TaskResult:
        self.start()
        with self._lock:
            task = TaskResult(
                task_id=task_id,
                document_id=kwargs.get('document_id', 0),
                status=TaskStatus.PENDING,
            )
            self._tasks[task_id] = task
            self._queue.append((task_id, lambda: func(*args, **kwargs)))

            logger.info(f"Enqueued task {task_id}")
            return task

    def get_task(self, task_id: str) -> Optional[TaskResult]:
        return self._tasks.get(task_id)

    def _worker_loop(self):
        while self._running:
            task_id = None
            func = None

            with self._lock:
                if self._queue:
                    task_id, func = self._queue.pop(0)

            if task_id and func:
                self._execute_task(task_id, func)
            else:
                time.sleep(0.1)

    def _execute_task(self, task_id: str, func: Callable):
        task = self._tasks[task_id]
        task.status = TaskStatus.PROCESSING
        task.started_at = timezone.now()

        try:
            result = func()
            if result is False or (isinstance(result, dict) and result.get('success') is False):
                raise RuntimeError('Task returned an unsuccessful result')
            task.status = TaskStatus.COMPLETED
            task.completed_at = timezone.now()
            task.metadata['result'] = result
            logger.info(f"Task {task_id} completed in {task.duration_seconds:.2f}s")
        except Exception as e:
            task.error = str(e)
            task.retry_count += 1

            if task.retry_count <= self._max_retries:
                task.status = TaskStatus.RETRYING
                with self._lock:
                    self._queue.append((task_id, func))
                logger.warning(f"Task {task_id} failed, retrying ({task.retry_count}/{self._max_retries})")
            else:
                task.status = TaskStatus.FAILED
                task.completed_at = timezone.now()
                logger.error(f"Task {task_id} failed after {self._max_retries} retries: {e}")


_task_queue = TaskQueue(max_workers=3, max_retries=2)


def get_task_queue() -> TaskQueue:
    return _task_queue


def process_document(document_id: int) -> bool:
    with document_lock(document_id):
        try:
            doc = Document.objects.get(id=document_id)
        except Document.DoesNotExist:
            _log_with_context('error', f"Document {document_id} not found")
            return False

        # A previous worker may have stopped midway through processing. Always
        # restart from a known state, including when the stored status is stale.
        doc.status = 'processing'
        doc.error_message = None
        doc.processed_at = None
        doc.save(update_fields=['status', 'error_message', 'processed_at', 'updated_at'])

        try:
            # Once processing starts, the old chunks must not remain searchable.
            if not delete_document_chunks(document_id):
                raise RuntimeError("Could not clear the previous Chroma index")

            doc.transition_to('extracting_text')
            pages = extract_text(doc.file.path, doc.file.name)
            if not pages:
                raise ValueError("No text could be extracted from the document")

            full_text = '\n\n'.join(p['text'] for p in pages)
            doc.full_text = full_text
            doc.save(update_fields=['full_text'])

            doc.transition_to('chunking')
            chunk_count = create_chunks(pages, doc)
            if not chunk_count:
                raise ValueError("No searchable chunks could be created from the document")

            doc.transition_to('indexing')
            if not upsert_document_chunks(doc.id):
                raise RuntimeError("Chroma indexing failed; the document was not marked ready")

            doc.status = 'ready'
            doc.processed_at = timezone.now()
            doc.save(update_fields=['status', 'processed_at', 'updated_at'])

            _log_with_context('info', f"Document {document_id} processed: {chunk_count} chunks created")
            return True

        except Exception as e:
            error_msg = str(e)
            _log_with_context('error', f"Document {document_id} processing failed: {error_msg}")
            if not delete_document_chunks(document_id):
                error_msg += '; Chroma cleanup failed, rebuild the index before retrying'
            doc.status = 'failed'
            doc.error_message = error_msg
            doc.save(update_fields=['status', 'error_message', 'updated_at'])
            return False


def process_document_async(document_id: int) -> str:
    task_id = f"doc_process_{document_id}_{uuid.uuid4().hex}"
    queue = get_task_queue()
    queue.enqueue(task_id, process_document, document_id=document_id)
    return task_id
