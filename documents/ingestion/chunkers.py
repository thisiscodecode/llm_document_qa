import logging

from django.db import transaction

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
        chunk_size=chunk_size if chunk_size is not None else getattr(document, 'chunk_size', None),
        overlap=overlap if overlap is not None else getattr(document, 'chunk_overlap', None),
    )
    cs = int(config['chunk_size'])
    ov = int(config['overlap'])
    min_size = config['min_chunk_size']

    if cs <= 0 or ov < 0 or ov >= cs:
        raise ValueError('chunk_size must be positive and chunk_overlap must be smaller than chunk_size')

    chunks = []
    chunk_index = 0

    for page_data in pages:
        page_num = page_data['page']
        text = page_data['text']

        if not text.strip():
            continue

        spans = []
        start = 0
        while start < len(text):
            end = min(start + cs, len(text))
            if text[start:end].strip():
                spans.append((start, end))
            if end == len(text):
                break
            start += cs - ov

        # Keep short pages and fold a small final tail into the prior chunk
        # instead of silently losing the end of a document.
        if len(spans) > 1 and len(text[spans[-1][0]:spans[-1][1]].strip()) < min_size:
            _, final_end = spans.pop()
            previous_start, _ = spans[-1]
            spans[-1] = (previous_start, final_end)

        for start, end in spans:
            chunks.append(DocumentChunk(
                document=document,
                content=text[start:end].strip(),
                chunk_index=chunk_index,
                page_number=page_num,
            ))
            chunk_index += 1

    with transaction.atomic():
        DocumentChunk.objects.filter(document=document).delete()
        if chunks:
            DocumentChunk.objects.bulk_create(chunks)

    logger.info(f"Created {len(chunks)} chunks (size={cs}, overlap={ov}, type={document.document_type})")
    return len(chunks)
