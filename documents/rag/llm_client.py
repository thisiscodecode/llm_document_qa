import os
import time
import logging
from typing import Any, Optional

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
BASE_DELAY = 1.0

FALLBACK_MODELS = [
    "openai/gpt-3.5-turbo",
    "google/gemini-2.0-flash-001",
]


def _invoke_with_retry(llm, prompt: Any, max_retries: int = MAX_RETRIES) -> Optional[str]:
    last_error = None
    for attempt in range(max_retries):
        try:
            response = llm.invoke(prompt)
            return response.content
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            is_transient = any(k in error_str for k in ['timeout', 'rate', '429', '503', '502', '500'])
            if not is_transient:
                logger.error(f"Non-transient LLM error: {e}")
                raise
            delay = BASE_DELAY * (2 ** attempt)
            logger.warning(f"LLM attempt {attempt + 1}/{max_retries} failed: {e}, retrying in {delay:.1f}s")
            time.sleep(delay)
    raise last_error


def get_llm(model_override: str = None):
    api_key = os.getenv("OPENROUTER_API_KEY")
    model = model_override or os.getenv("OPENROUTER_MODEL", "openrouter/free")

    if not api_key:
        return None

    try:
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
    except Exception as e:
        logger.error(f"LLM initialization failed: {e}")
        return None


def invoke_llm(prompt: Any, model: str = None) -> tuple:
    primary_model = model or os.getenv("OPENROUTER_MODEL", "openrouter/free")
    llm = get_llm(primary_model)
    if llm is None:
        return None, "llm_unavailable"

    try:
        content = _invoke_with_retry(llm, prompt)
        return content, None
    except Exception as e:
        error_str = str(e).lower()
        is_transient = any(k in error_str for k in ['timeout', 'rate', '429', '503'])
        if not is_transient:
            return None, "llm_error"

    from .feature_flags import is_enabled
    if is_enabled('model_fallback'):
        for fallback_model in FALLBACK_MODELS:
            if fallback_model == primary_model:
                continue
            logger.info(f"Trying fallback model: {fallback_model}")
            fallback_llm = get_llm(fallback_model)
            if fallback_llm is None:
                continue
            try:
                content = _invoke_with_retry(fallback_llm, prompt, max_retries=2)
                logger.info(f"Fallback model {fallback_model} succeeded")
                return content, None
            except Exception as fallback_err:
                logger.warning(f"Fallback model {fallback_model} also failed: {fallback_err}")
                continue

    return None, "llm_error"
