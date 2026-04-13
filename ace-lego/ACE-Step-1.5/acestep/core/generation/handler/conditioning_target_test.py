"""Unit tests for target-latent preparation helpers."""

import unittest
from contextlib import contextmanager

import torch

from acestep.core.generation.handler.conditioning_target import ConditioningTargetMixin


class _Host(ConditioningTargetMixin):
    """Minimal host implementing ConditioningTargetMixin dependencies."""

    def __init__(self):
        self.device = "meta"
        self.dtype = torch.float32
        self.sample_rate = 48000
        self.use_mlx_vae = True
        self.mlx_vae = object()
        self.silence_latent = torch.zeros(1, 128, 16, dtype=torch.float32)
        self.recorded_audio_devices = []

    def _ensure_silence_latent_on_device(self):
        return None

    @contextmanager
    def _load_model_context(self, _model_name):
        yield

    def is_silence(self, audio: torch.Tensor) -> bool:
        return False

    def _encode_audio_to_latents(self, audio: torch.Tensor) -> torch.Tensor:
        self.recorded_audio_devices.append(audio.device.type)
        return torch.zeros(64, 16, dtype=torch.float32)

    def _decode_audio_codes_to_latents(self, _code_hint):
        return None


class ConditioningTargetMixinTests(unittest.TestCase):
    """Tests for MLX-aware target-audio preparation."""

    def test_prepare_target_latents_keeps_raw_audio_on_cpu_when_mlx_vae_is_active(self):
        """Raw target audio should remain on CPU before MLX VAE encoding."""
        host = _Host()

        target_wavs, target_latents, latent_masks, max_latent_length, silence_latent_tiled = (
            host._prepare_target_latents_and_wavs(
                batch_size=1,
                target_wavs=torch.ones(1, 2, 96000, dtype=torch.float32),
                audio_code_hints=[None],
            )
        )

        self.assertEqual(host.recorded_audio_devices, ["cpu"])
        self.assertEqual(target_wavs.device.type, "cpu")
        self.assertEqual(tuple(target_latents.shape), (1, 128, 16))
        self.assertEqual(latent_masks.device.type, "meta")
        self.assertEqual(max_latent_length, 128)
        self.assertEqual(tuple(silence_latent_tiled.shape), (128, 16))


if __name__ == "__main__":
    unittest.main()
