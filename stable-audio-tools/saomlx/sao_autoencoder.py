"""Stable Audio Open autoencoder helpers (torch reference path)."""

from __future__ import annotations

import json
import sys
import typing as tp
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .hf_download import download_file


def _ensure_sat_import_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sat_dir = repo_root / "third_party" / "stable-audio-tools"
    if not sat_dir.exists():
        raise FileNotFoundError(
            f"Could not find stable-audio-tools at {sat_dir}. "
            "Run ./scripts/pin_deps.sh first."
        )
    sat_path = str(sat_dir)
    if sat_path not in sys.path:
        sys.path.insert(0, sat_path)


@dataclass(frozen=True)
class SAOAutoencoderSpec:
    repo_id: str
    config_filename: str = "vae_model_config.json"
    weights_filename: str = "vae_model.ckpt"


class TorchSAOAutoencoder:
    def __init__(self, model, config: dict[str, tp.Any], device: str, dtype: str):
        self.model = model
        self.config = config
        self.device = device
        self.dtype = dtype

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = "stabilityai/stable-audio-open-1.0",
        *,
        device: str = "cpu",
        dtype: str = "float32",
    ) -> "TorchSAOAutoencoder":
        import torch

        _ensure_sat_import_path()
        from stable_audio_tools.models.factory import create_model_from_config
        from stable_audio_tools.models.utils import load_ckpt_state_dict

        config_path = download_file(repo_id, "vae_model_config.json")
        weights_path = download_file(repo_id, "vae_model.ckpt")

        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)

        model = create_model_from_config(config)
        state_dict = load_ckpt_state_dict(str(weights_path))
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            # These checkpoints are expected to be direct matches; surface mismatches early.
            raise RuntimeError(
                f"VAE checkpoint load mismatch for {repo_id}. "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )

        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        if dtype not in dtype_map:
            raise ValueError(f"Unsupported dtype '{dtype}', expected one of {sorted(dtype_map)}")

        model = model.to(device=device, dtype=dtype_map[dtype]).eval()
        return cls(model=model, config=config, device=device, dtype=dtype)

    @property
    def sample_rate(self) -> int:
        return int(self.config["sample_rate"])

    @property
    def downsampling_ratio(self) -> int:
        return int(self.config["model"]["downsampling_ratio"])

    @property
    def channels(self) -> int:
        return int(self.config["audio_channels"])

    def encode(self, audio, *, sample: bool = True, return_info: bool = False):
        import torch
        import torch.nn.functional as F

        with torch.no_grad():
            if sample:
                return self.model.encode(audio, return_info=return_info)

            # Deterministic VAE path: use posterior mean instead of sampling.
            pre_latents, info = self.model.encode(audio, skip_bottleneck=True, return_info=True)
            mean, scale = pre_latents.chunk(2, dim=1)
            stdev = F.softplus(scale) + 1e-4
            var = stdev * stdev
            logvar = torch.log(var)
            kl = (mean * mean + var - logvar - 1).sum(1).mean()
            info["kl"] = kl
            if return_info:
                return mean, info
            return mean

    def decode(self, latents):
        import torch

        with torch.no_grad():
            return self.model.decode(latents)

    def roundtrip(self, audio, *, sample: bool = True):
        latents = self.encode(audio, sample=sample)
        return self.decode(latents), latents

    def roundtrip_numpy(self, audio: np.ndarray, *, sample: bool = True) -> tuple[np.ndarray, np.ndarray]:
        import torch

        arr = np.asarray(audio, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.ndim != 2:
            raise ValueError(f"Expected (samples, channels) audio, got {arr.shape}")
        if arr.shape[1] != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {arr.shape[1]}")

        x = torch.from_numpy(arr.T[None, ...]).to(device=self.device, dtype=torch.float32)
        with torch.no_grad():
            recon, latents = self.roundtrip(x, sample=sample)

        recon_np = recon.squeeze(0).detach().cpu().float().numpy().T
        latent_np = latents.squeeze(0).detach().cpu().float().numpy().T
        source_len = arr.shape[0]
        if recon_np.shape[0] >= source_len:
            recon_np = recon_np[:source_len]
        else:
            pad = np.zeros((source_len - recon_np.shape[0], recon_np.shape[1]), dtype=recon_np.dtype)
            recon_np = np.concatenate([recon_np, pad], axis=0)
        return recon_np, latent_np
