import logging
import re

logger = logging.getLogger(__name__)


def map_claims_to_sources(answer, sources):
    citations = []
    seen = set()
    for ref in source_references(answer):
        source_idx = ref - 1
        if ref in seen or not 0 <= source_idx < len(sources):
            continue
        seen.add(ref)
        source = sources[source_idx]
        citation = {
            'source_number': ref,
            'document_id': source.get('document_id'),
            'chunk_id': source.get('chunk_id'),
            'document': source['document'],
            'chunk_index': source['chunk_index'],
            'page': source['page'],
        }
        if source.get('url'):
            citation['url'] = source['url']
        citations.append(citation)

    return citations


def source_references(answer):
    return [int(ref) for ref in re.findall(r'\[Source\s+(\d+)\]', answer, re.I)]


def has_uncited_claims(answer):
    # Models often put citations after sentence punctuation. Move each such
    # citation inside that sentence before checking line and sentence spans.
    normalized = re.sub(
        r'([.!?؟])\s*(\[Source\s+\d+\])', r' \2\1', answer, flags=re.I,
    )
    for line in normalized.splitlines():
        for sentence in re.split(r'(?<=[.!?؟])\s+', line.strip()):
            claim = re.sub(r'\[Source\s+\d+\]', '', sentence, flags=re.I).strip(' -*•')
            if not claim or claim.endswith(':'):
                continue
            if re.search(r'[^\W\d_]', claim) and not source_references(sentence):
                return True
    return False


def format_sources(sources):
    if not sources:
        return ""

    parts = []
    for s in sources:
        parts.append(f"[Source {s['index']}] {s['document']}, page {s['page']}, chunk {s['chunk_index']}")

    return "\n".join(parts)
