"""Compatibility imports for the canonical ingestion pipeline.

Keep older integrations on the same processing path as uploads and the admin.
"""

from .ingestion.chunkers import (
    DEFAULT_CHUNK_OVERLAP as CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE as CHUNK_SIZE,
    create_chunks,
)
from .ingestion.document_processor import process_document, process_document_async
from .ingestion.extractors import (
    extract_text,
    extract_text_from_docx,
    extract_text_from_pdf,
)
from .ingestion.validators import ALLOWED_EXTENSIONS, MAX_FILE_SIZE, validate_upload

__all__ = [
    "ALLOWED_EXTENSIONS", "CHUNK_OVERLAP", "CHUNK_SIZE", "MAX_FILE_SIZE",
    "create_chunks", "extract_text", "extract_text_from_docx",
    "extract_text_from_pdf", "process_document", "process_document_async",
    "validate_upload",
]
