from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download, try_to_load_from_cache


@dataclass(frozen=True)
class ModelConfig:
    repo_id: str
    config_path: str
    ckpt_path: str

    def resolve_config(self):
        """Return the config path without resolving or downloading model weights."""
        cached_config = try_to_load_from_cache(self.repo_id, self.config_path)
        if isinstance(cached_config, str):
            return cached_config
        return hf_hub_download(repo_id=self.repo_id, filename=self.config_path)

    def resolve(self):
        """Download files from HuggingFace Hub and return local cached paths."""
        try:
            local_config = hf_hub_download(repo_id=self.repo_id, filename=self.config_path)
            local_ckpt = hf_hub_download(repo_id=self.repo_id, filename=self.ckpt_path)
        except Exception:
            cached_pair = _find_cached_snapshot_pair(
                self.repo_id,
                self.config_path,
                self.ckpt_path,
            )
            if cached_pair is None:
                raise
            local_config, local_ckpt = cached_pair
        return local_config, local_ckpt


def _find_cached_snapshot_pair(repo_id: str, *filenames: str) -> tuple[str, ...] | None:
    cache_root = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / f"models--{repo_id.replace('/', '--')}"
        / "snapshots"
    )
    if not cache_root.exists():
        return None

    snapshots = sorted(
        (path for path in cache_root.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for snapshot in snapshots:
        paths = tuple(snapshot / filename for filename in filenames)
        if all(path.exists() for path in paths):
            return tuple(str(path) for path in paths)
    return None


@dataclass(frozen=True)
class AutoencoderModelConfig:
    """Config for a standalone autoencoder HF repo (e.g. stabilityai/SAME-S).

    resolve() first checks whether any full Stable Audio 3 checkpoint that contains the same
    autoencoder is already cached locally.  If one is found it is used as-is
    (load_autoencoder strips the pretransform.* prefix automatically), avoiding a
    redundant download.  Otherwise the lightweight AE-only repo is fetched instead.
    """

    ae_repo_id: str
    ae_config_path: str
    ae_ckpt_path: str
    stable_audio_3: tuple[ModelConfig, ...]

    def resolve(self):
        """Return (config_path, ckpt_path), preferring an already-cached Stable Audio 3 checkpoint."""
        for fallback in self.stable_audio_3:
            cached_config = try_to_load_from_cache(
                fallback.repo_id, fallback.config_path
            )
            cached_ckpt = try_to_load_from_cache(fallback.repo_id, fallback.ckpt_path)
            if isinstance(cached_config, str) and isinstance(cached_ckpt, str):
                return cached_config, cached_ckpt

        # No Stable Audio 3 checkpoint found in local cache — download the AE-only repo.
        local_config = hf_hub_download(
            repo_id=self.ae_repo_id, filename=self.ae_config_path
        )
        local_ckpt = hf_hub_download(
            repo_id=self.ae_repo_id, filename=self.ae_ckpt_path
        )
        return local_config, local_ckpt


models: dict[str, ModelConfig] = {
    "small-music": ModelConfig(
        "stabilityai/stable-audio-3-small-music",
        "model_config.json",
        "model.safetensors",
    ),
    "small-music-base": ModelConfig(
        "stabilityai/stable-audio-3-small-music-base",
        "model_config.json",
        "model.safetensors",
    ),
    "small-sfx": ModelConfig(
        "stabilityai/stable-audio-3-small-sfx",
        "model_config.json",
        "model.safetensors",
    ),
    "small-sfx-base": ModelConfig(
        "stabilityai/stable-audio-3-small-sfx-base",
        "model_config.json",
        "model.safetensors",
    ),
    "medium": ModelConfig(
        "stabilityai/stable-audio-3-medium",
        "model_config.json",
        "model.safetensors",
    ),
    "medium-base": ModelConfig(
        "stabilityai/stable-audio-3-medium-base",
        "model_config.json",
        "model.safetensors",
    ),
}

# Stable Audio 3 full-model configs to probe (in order) before downloading the AE-only repo.
# Both ARC and RF variants share the same autoencoder, so either can supply the weights.
_small_stable_audio_3: tuple[ModelConfig, ...] = (
    models["small-music"],
    models["small-sfx"],
)
_medium_stable_audio_3: tuple[ModelConfig, ...] = (
    models["medium"],
)

ae_models: dict[str, AutoencoderModelConfig] = {
    "same-s": AutoencoderModelConfig(
        ae_repo_id="stabilityai/SAME-S",
        ae_config_path="model_config.json",
        ae_ckpt_path="model.safetensors",
        stable_audio_3=_small_stable_audio_3,
    ),
    "same-l": AutoencoderModelConfig(
        ae_repo_id="stabilityai/SAME-L",
        ae_config_path="model_config.json",
        ae_ckpt_path="model.safetensors",
        stable_audio_3=_medium_stable_audio_3,
    ),
}

base_models: dict[str, ModelConfig] = {
    key: value for key, value in models.items() if key.endswith("-base")
}

all_models: dict[str, ModelConfig | AutoencoderModelConfig] = {
    **models,
    **ae_models,
}
