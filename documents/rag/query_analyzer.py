import re
import logging

logger = logging.getLogger(__name__)


def detect_language(text):
    persian_chars = len(re.findall(r'[\u0600-\u06FF]', text))
    total_chars = len(text)

    if total_chars == 0:
        return 'en'

    persian_ratio = persian_chars / total_chars
    if persian_ratio > 0.3:
        return 'fa'
    return 'en'


def detect_intent(text):
    text_lower = text.lower()

    question_starters = ['who', 'what', 'where', 'when', 'why', 'how', 'which']
    for starter in question_starters:
        if text_lower.startswith(starter):
            return 'question'

    if '?' in text:
        return 'question'

    if any(word in text_lower for word in ['tell me', 'explain', 'describe', 'define']):
        return 'explanation'

    if any(word in text_lower for word in ['list', 'show', 'find', 'search']):
        return 'search'

    return 'question'


def extract_filters(text):
    filters = {}

    date_pattern = r'\b(\d{4})\b'
    years = re.findall(date_pattern, text)
    if years:
        filters['years'] = years

    return filters


def rewrite_query(query):
    query = query.strip()

    query = re.sub(r'\s+', ' ', query)

    if len(query) > 500:
        query = query[:500]

    return query


def analyze_query(query):
    return {
        'original': query,
        'rewritten': rewrite_query(query),
        'language': detect_language(query),
        'intent': detect_intent(query),
        'filters': extract_filters(query),
    }