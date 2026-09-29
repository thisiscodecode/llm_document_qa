import logging
from typing import Optional, List

from ..models import Document

logger = logging.getLogger(__name__)


def apply_permission_filter(document_ids: Optional[List[int]], owner=None) -> List[int]:
    # Authorization is checked against the current database state.
    if document_ids is not None:
        if owner:
            permitted = Document.objects.filter(
                id__in=document_ids,
                owner=owner,
                status='ready'
            ).values_list('id', flat=True)
        else:
            permitted = Document.objects.filter(
                id__in=document_ids,
                owner__isnull=True,
                status='ready'
            ).values_list('id', flat=True)

        result = list(permitted)
        logger.info(f"Permission filter: {len(result)}/{len(document_ids)} documents permitted")
    else:
        queryset = Document.objects.filter(status='ready')
        if owner:
            queryset = queryset.filter(owner=owner)
        else:
            queryset = queryset.filter(owner__isnull=True)

        result = list(queryset.values_list('id', flat=True))
        logger.info(f"Permission filter: {len(result)} documents permitted for owner={owner}")

    return result


def apply_metadata_filter(chunks: list, filters: Optional[dict] = None) -> list:
    if not filters:
        return chunks

    filtered = chunks

    if 'document_ids' in filters:
        allowed_ids = set(filters['document_ids'])
        filtered = [c for c in filtered if c.document_id in allowed_ids]

    if 'page_numbers' in filters:
        allowed_pages = set(filters['page_numbers'])
        filtered = [c for c in filtered if getattr(c, 'page_number', None) in allowed_pages]

    if 'min_length' in filters:
        min_len = filters['min_length']
        filtered = [c for c in filtered if len(c.content) >= min_len]

    logger.info(f"Metadata filter: {len(filtered)}/{len(chunks)} chunks passed")
    return filtered
