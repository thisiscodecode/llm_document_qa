import logging
import hashlib
from typing import Optional

logger = logging.getLogger(__name__)

_flags: dict[str, dict] = {}


def register_flag(name: str, description: str = "", rollout_pct: float = 100.0,
                  variants: list = None):
    _flags[name] = {
        'description': description,
        'rollout_pct': rollout_pct,
        'variants': variants or ['control', 'treatment'],
        'variant_weights': None,
    }


def update_flag(name: str, rollout_pct: float = None, variants: list = None) -> bool:
    if name not in _flags:
        return False
    if rollout_pct is not None:
        _flags[name]['rollout_pct'] = max(0.0, min(100.0, rollout_pct))
    if variants is not None:
        _flags[name]['variants'] = variants
    logger.info(f"Updated feature flag '{name}': {_flags[name]}")
    return True


def get_flag(name: str) -> Optional[dict]:
    flag = _flags.get(name)
    if not flag:
        return None
    return {
        'name': name,
        'description': flag['description'],
        'rollout_pct': flag['rollout_pct'],
        'variants': flag['variants'],
    }


def is_enabled(flag_name: str, user_id: Optional[int] = None) -> bool:
    flag = _flags.get(flag_name)
    if not flag:
        return False
    if flag['rollout_pct'] >= 100:
        return True
    bucket = _get_bucket(flag_name, user_id)
    return bucket < flag['rollout_pct']


def get_variant(flag_name: str, user_id: Optional[int] = None) -> str:
    flag = _flags.get(flag_name)
    if not flag:
        return 'control'
    if not is_enabled(flag_name, user_id):
        return 'control'
    variants = flag['variants']
    if len(variants) == 1:
        return variants[0]
    bucket = _get_bucket(flag_name + '_variant', user_id)
    idx = int(bucket / 100 * len(variants))
    return variants[min(idx, len(variants) - 1)]


def _get_bucket(key: str, user_id: Optional[int]) -> float:
    raw = f"{key}:{user_id or 'anon'}"
    h = hashlib.md5(raw.encode()).hexdigest()
    return (int(h[:8], 16) % 10000) / 100.0


def list_flags() -> dict:
    return {name: {
        'description': f['description'],
        'rollout_pct': f['rollout_pct'],
        'variants': f['variants'],
    } for name, f in _flags.items()}


register_flag(
    'cross_encoder_reranker',
    description='Use cross-encoder model for reranking instead of bi-encoder',
    rollout_pct=0.0,
    variants=['bi_encoder', 'cross_encoder'],
)
register_flag(
    'web_search_fallback',
    description='Fall back to web search when no document chunks are found',
    rollout_pct=0.0,
    variants=['disabled', 'enabled'],
)
register_flag(
    'advanced_caching',
    description='Enable query and embedding caching',
    rollout_pct=100.0,
    variants=['disabled', 'enabled'],
)
register_flag(
    'model_fallback',
    description='Allow fallback to secondary LLM model on primary failure',
    rollout_pct=100.0,
    variants=['disabled', 'enabled'],
)
