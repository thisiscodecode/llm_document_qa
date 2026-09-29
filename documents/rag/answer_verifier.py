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

    if not answer or not answer.strip():
        return {
            'is_valid': False,
            'reason': 'empty_answer',
            'suggestion': 'The model returned no answer.',
        }

    refusal_phrases = [
        "i could not find",
        "i cannot find",
        "the context does not contain",
        "not enough information",
        "cannot answer",
        "پیدا نکردم",
        "یافت نشد",
        "اطلاعات کافی در اسناد",
    ]

    answer_lower = answer.strip().lower()
    for phrase in refusal_phrases:
        if phrase in answer_lower and len(answer_lower) <= 220 and '\n' not in answer_lower:
            return {
                'is_valid': True,
                'reason': 'refusal',
                'suggestion': None,
            }

    answer_content = re.sub(r'\[Source\s+\d+\]', '', answer, flags=re.I)
    context_content = re.sub(r'(?m)^\[Source\s+\d+\].*\n', '', context)
    answer_words = set(re.findall(r'\b\w{3,}\b', answer_content.lower()))
    context_words = set(re.findall(r'\b\w{3,}\b', context_content.lower()))
    overlap = len(answer_words & context_words)

    # Numbers and dates are frequent high-impact hallucinations. A numeric
    # claim must appear in the source text (source labels do not count).
    claimed_numbers = set(re.findall(r'\d+(?:[.,]\d+)?', answer_content))
    source_numbers = set(re.findall(r'\d+(?:[.,]\d+)?', context_content))
    if not claimed_numbers.issubset(source_numbers):
        return {
            'is_valid': False,
            'reason': 'unsupported_number',
            'suggestion': 'A number in the answer is absent from the source text.',
        }

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
