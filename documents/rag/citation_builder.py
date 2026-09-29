import logging

logger = logging.getLogger(__name__)


def map_claims_to_sources(answer, sources):
    citations = []

    import re
    source_refs = re.findall(r'\[Source\s+(\d+)\]', answer)

    for ref in source_refs:
        try:
            source_idx = int(ref) - 1
            if 0 <= source_idx < len(sources):
                citations.append({
                    'source_number': int(ref),
                    'document': sources[source_idx]['document'],
                    'chunk_index': sources[source_idx]['chunk_index'],
                    'page': sources[source_idx]['page'],
                })
        except (ValueError, IndexError):
            continue

    return citations


def format_sources(sources):
    if not sources:
        return ""

    parts = []
    for s in sources:
        parts.append(f"\n[Source {s['index']}: {s['document']}, page {s['page']}, chunk {s['chunk_index']}]\n{s['preview']}\n")

    return "".join(parts)