import os
import logging
import requests
from typing import Optional

logger = logging.getLogger(__name__)

SEARCH_API_URL = os.getenv("WEB_SEARCH_API_URL", "https://api.search.brave.com/res/v1/web/search")
SEARCH_API_KEY = os.getenv("WEB_SEARCH_API_KEY", "")
MAX_RESULTS = 5


def web_search(query: str, max_results: int = MAX_RESULTS) -> Optional[list]:
    if not SEARCH_API_KEY:
        logger.warning("WEB_SEARCH_API_KEY not set, web search unavailable")
        return None

    try:
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": SEARCH_API_KEY,
        }
        params = {
            "q": query,
            "count": max_results,
        }
        resp = requests.get(SEARCH_API_URL, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        results = []
        for item in data.get("web", {}).get("results", [])[:max_results]:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("description", ""),
            })

        return results if results else None

    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return None


def format_web_results(results: list) -> str:
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(f"[Web Result {i}] {r['title']}\n{r['snippet']}\nSource: {r['url']}")
    return "\n\n".join(parts)


def build_web_search_prompt(question: str, web_results: str) -> str:
    return f"""Answer the question using the following web search results.

Rules:
- Respond in the SAME LANGUAGE as the question
- Give a SHORT, DIRECT answer
- Reference sources using [Web Result N] format
- If the results don't contain enough information, say so

Web Search Results:
{web_results}

Question:
{question}
"""
