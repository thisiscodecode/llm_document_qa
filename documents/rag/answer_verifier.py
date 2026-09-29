import re
import logging

logger = logging.getLogger(__name__)


def verify_answer(answer, context):
    if not context:
        return {
            'is_valid': False,
            'reason': 'no_context',
            'suggestion': 'No context available to answer the question.',
        }

    refusal_phrases = [
        "i could not find",
        "i cannot find",
        "the context does not contain",
        "not enough information",
        "cannot answer",
    ]

    answer_lower = answer.lower()
    for phrase in refusal_phrases:
        if phrase in answer_lower:
            return {
                'is_valid': True,
                'reason': 'refusal',
                'suggestion': None,
            }

    context_lower = context.lower()
    answer_words = set(re.findall(r'\b\w{3,}\b', answer.lower()))
    context_words = set(re.findall(r'\b\w{3,}\b', context.lower()))
    overlap = len(answer_words & context_words)

    # Short factual answers may contain fewer than three meaningful words. For
    # longer answers, require at least three supported terms as a cheap guardrail.
    required_overlap = min(3, len(answer_words))
    if overlap < required_overlap:
        return {
            'is_valid': False,
            'reason': 'low_support',
            'suggestion': 'The answer may not be well-supported by the context.',
        }

    return {
        'is_valid': True,
        'reason': 'supported',
        'suggestion': None,
    }
