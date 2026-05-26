from __future__ import annotations

from stable_audio_3.mlx.compat import summarize_mlx_compatibility
from stable_audio_3.mlx.pipeline import StableAudioMLXPipeline


def inspect_mlx_pipeline(config_or_path) -> StableAudioMLXPipeline:
    return StableAudioMLXPipeline.from_config(config_or_path)


def inspect_pretrained_mlx_pipeline(
    model_name_or_path: str,
    *,
    search_roots=None,
) -> StableAudioMLXPipeline:
    return StableAudioMLXPipeline.from_pretrained_config(
        model_name_or_path,
        search_roots=search_roots,
    )


def summarize_pretrained_mlx_pipeline(
    model_name_or_path: str,
    *,
    search_roots=None,
) -> str:
    pipeline = inspect_pretrained_mlx_pipeline(
        model_name_or_path,
        search_roots=search_roots,
    )
    return summarize_mlx_compatibility(pipeline.compatibility)
