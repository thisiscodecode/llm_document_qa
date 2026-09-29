import logging
from typing import Optional

from documents.models import DocumentChunk

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150

DOCUMENT_TYPE_CONFIGS = {
    'default': {'chunk_size': 1000, 'overlap': 150, 'min_chunk_size': 50},
    'code': {'chunk_size': 500, 'overlap': 100, 'min_chunk_size': 30},
    'legal': {'chunk_size': 1500, 'overlap': 200, 'min_chunk_size': 100},
    'academic': {'chunk_size': 1200, 'overlap': 180, 'min_chunk_size': 80},
    'presentation': {'chunk_size': 800, 'overlap': 100, 'min_chunk_size': 30},
}


def get_chunk_config(document_type: str = 'default', chunk_size: int = None,
                     overlap: int = None) -> dict:
    base = DOCUMENT_TYPE_CONFIGS.get(document_type, DOCUMENT_TYPE_CONFIGS['default']).copy()
    if chunk_size is not None:
        base['chunk_size'] = chunk_size
    if overlap is not None:
        base['overlap'] = overlap
    return base


def create_chunks(pages, document, chunk_size=None, overlap=None):
    config = get_chunk_config(
        document_type=getattr(document, 'document_type', 'default'),
        chunk_size=chunk_size or getattr(document, 'chunk_size', None),
        overlap=overlap or getattr(document, 'chunk_overlap', None),
    )
    cs = config['chunk_size']
    ov = config['overlap']
    min_size = config['min_chunk_size']

    DocumentChunk.objects.filter(document=document).delete()

    chunks = []
    chunk_index = 0

    for page_data in pages:
        page_num = page_data['page']
        text = page_data['text']

        if not text.strip():
            continue

        start = 0
        while start < len(text):
            end = start + cs
            chunk_text = text[start:end].strip()

            if chunk_text and len(chunk_text) >= min_size:
                chunks.append(DocumentChunk(
                    document=document,
                    content=chunk_text,
                    chunk_index=chunk_index,
                    page_number=page_num,
                ))
                chunk_index += 1

            start += cs - ov

    if chunks:
        DocumentChunk.objects.bulk_create(chunks)

    logger.info(f"Created {len(chunks)} chunks (size={cs}, overlap={ov}, type={document.document_type})")
    return len(chunks)
