"""
Central model registry for the localhost backend.

This file exists so:
- /api/models can list models grouped by size and checkpoint
- the MLX backend can infer a finetune's base model when config.json isn't present
- we can keep auto-descriptions in one place
"""

from __future__ import annotations

from typing import Optional


# The JUCE UX expects 30s generations for both "process" and "continue".
OUTPUT_DURATION_S: float = 30.0


# ---------------------------------------------------------------------------
# Model Catalog (used by /api/models and base-model inference)
# ---------------------------------------------------------------------------

MODEL_CATALOG: dict[str, list[str]] = {
    "small": [
        "thepatch/vanya_ai_dnb_0.1",
        "thepatch/gary_orchestra_2",
        "thepatch/keygen-gary-v2-small-8",
        "thepatch/keygen-gary-v2-small-12",
        "thepatch/keygen-gary-small-6",
        "thepatch/keygen-gary-small-12",
        "thepatch/keygen-gary-small-20",
    ],
    "medium": [
        "thepatch/bleeps-medium",
        "thepatch/keygen-gary-medium-12",
    ],
    "large": [
        "thepatch/hoenn_lofi",
        "thepatch/bleeps-large-6",
        "thepatch/bleeps-large-8",
        "thepatch/bleeps-large-10",
        "thepatch/bleeps-large-14",
        "thepatch/bleeps-large-20",
        "thepatch/keygen-gary-large-6",
        "thepatch/keygen-gary-large-12",
        "thepatch/keygen-gary-large-20",
        "thepatch/keygen-gary-v2-large-12",
        "thepatch/keygen-gary-v2-large-16",
        "thepatch/gary-grunge-large-stereo-v2-4",
        "thepatch/gary-grunge-large-stereo-12",
    ],
}


BASE_MODEL_BY_SIZE: dict[str, str] = {
    "small": "facebook/musicgen-small",
    "medium": "facebook/musicgen-medium",
    "large": "facebook/musicgen-large",
}

# Per-model base override for checkpoints that do not match the category default.
BASE_MODEL_BY_MODEL: dict[str, str] = {
    "thepatch/gary-grunge-large-stereo-v2-4": "facebook/musicgen-stereo-large",
    "thepatch/gary-grunge-large-stereo-12": "facebook/musicgen-stereo-large",
}


def find_model_size(model_name: str) -> Optional[str]:
    for size, models in MODEL_CATALOG.items():
        if model_name in models:
            return size
    return None


def get_base_model_for_finetune(model_name: str) -> str:
    """
    Return the base model repo to use as a config fallback for finetunes that
    don't ship config.json.

    We intentionally error for unknown models so we don't silently pick the
    wrong base config (small/medium/large mismatch will break weight loading).
    """
    if model_name in BASE_MODEL_BY_MODEL:
        return BASE_MODEL_BY_MODEL[model_name]

    size = find_model_size(model_name)
    if size is None:
        raise KeyError(
            f"Unknown finetune model '{model_name}'. Add it to MODEL_CATALOG in g4l_models.py "
            "so we can infer the correct base model (small/medium/large)."
        )
    return BASE_MODEL_BY_SIZE[size]


def has_explicit_base_model_override(model_name: str) -> bool:
    return model_name in BASE_MODEL_BY_MODEL


# ---------------------------------------------------------------------------
# Auto-descriptions (optional per-model defaults)
# ---------------------------------------------------------------------------

AUTO_DESCRIPTIONS: dict[str, str] = {
    "thepatch/gary_orchestra": "violin, epic, film, piano, strings, orchestra",
    "thepatch/gary_orchestra_2": "violin, epic, film, piano, strings, orchestra",
}


def get_model_description(model_name: str, custom_description: Optional[str] = None) -> Optional[str]:
    if custom_description is not None and custom_description.strip():
        return custom_description
    return AUTO_DESCRIPTIONS.get(model_name)
