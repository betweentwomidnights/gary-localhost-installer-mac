"""Policy helpers for native MLX DiT cross-attention caching."""

import os


_CFG_CACHE_ENV = "ACESTEP_MLX_CFG_KV_CACHE"


def cfg_kv_cache_enabled() -> bool:
    """Return whether experimental CFG cross-attention caching is enabled."""
    return os.getenv(_CFG_CACHE_ENV, "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def should_create_cross_attention_cache(*, compiled: bool, do_cfg: bool) -> bool:
    """Return whether an uncompiled diffusion loop should create a KV cache."""
    if compiled:
        return False
    return not do_cfg or cfg_kv_cache_enabled()
