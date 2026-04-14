"""Unit tests for service-generation execution helper mixin."""

import contextlib
import unittest
from unittest.mock import Mock, patch

import torch

from acestep.core.generation.handler.service_generate_execute import ServiceGenerateExecuteMixin
from acestep.core.generation.handler.service_generate_outputs import ServiceGenerateOutputsMixin


class _Host(ServiceGenerateExecuteMixin, ServiceGenerateOutputsMixin):
    """Test host exposing the minimum attributes used by execute helpers."""

    def __init__(self):
        """Initialize static runtime fields for helper-method tests."""
        self.device = "cpu"
        self.silence_latent = torch.zeros(1, 4, 4, dtype=torch.float32)


class ServiceGenerateExecuteMixinTests(unittest.TestCase):
    """Validate helper behavior for kwargs and output assembly."""

    def test_build_generate_kwargs_adds_timesteps_tensor(self):
        """Timesteps input should be converted to a device tensor in kwargs."""
        host = _Host()
        payload = {
            "text_hidden_states": torch.zeros(1, 2),
            "text_attention_mask": torch.ones(1, 2),
            "lyric_hidden_states": torch.zeros(1, 2),
            "lyric_attention_mask": torch.ones(1, 2),
            "refer_audio_acoustic_hidden_states_packed": torch.zeros(1, 2),
            "refer_audio_order_mask": torch.zeros(1, dtype=torch.long),
            "src_latents": torch.zeros(1, 4, 4),
            "chunk_mask": torch.ones(1, 4, dtype=torch.bool),
            "is_covers": torch.tensor([True]),
            "non_cover_text_hidden_states": None,
            "non_cover_text_attention_masks": None,
            "precomputed_lm_hints_25Hz": None,
        }
        kwargs = host._build_service_generate_kwargs(
            payload=payload,
            seed_param=123,
            infer_steps=16,
            guidance_scale=7.0,
            audio_cover_strength=1.0,
            cover_noise_strength=0.0,
            infer_method="ode",
            use_adg=False,
            cfg_interval_start=0.0,
            cfg_interval_end=1.0,
            shift=1.0,
            timesteps=[1.0, 0.5],
        )

        self.assertIn("timesteps", kwargs)
        self.assertEqual(kwargs["seed"], 123)
        self.assertEqual(kwargs["infer_steps"], 16)
        self.assertEqual(kwargs["timesteps"].dtype, torch.float32)
        self.assertEqual(kwargs["timesteps"].device.type, "cpu")

    def test_attach_service_outputs_persists_required_fields(self):
        """Attached payload fields should be available to downstream handlers."""
        host = _Host()
        payload = {
            "src_latents": torch.zeros(1, 4, 4),
            "target_latents": torch.ones(1, 4, 4),
            "chunk_mask": torch.ones(1, 4, dtype=torch.bool),
            "spans": [("full", 0, 4)],
            "lyric_token_idss": torch.ones(1, 3, dtype=torch.long),
        }
        outputs = host._attach_service_generate_outputs(
            outputs={"target_latents": torch.zeros(1, 4, 4)},
            payload=payload,
            batch={"latent_masks": torch.ones(1, 4, dtype=torch.long)},
            encoder_hidden_states=torch.zeros(1, 2),
            encoder_attention_mask=torch.ones(1, 2),
            context_latents=torch.zeros(1, 2),
        )

        self.assertIn("src_latents", outputs)
        self.assertIn("target_latents_input", outputs)
        self.assertIn("latent_masks", outputs)
        self.assertIn("encoder_hidden_states", outputs)
        self.assertIn("lyric_token_idss", outputs)

    def test_resolve_seed_param_none_uses_random_seed(self):
        """None seed list should produce a random integer seed parameter."""
        host = _Host()
        with patch("acestep.core.generation.handler.service_generate_execute.random.randint", return_value=42):
            seed_param = host._resolve_service_seed_param(None)
        self.assertEqual(seed_param, 42)

    def test_execute_service_generate_diffusion_forwards_guidance_controls_to_mlx(self):
        """MLX execution should receive CFG-related runtime controls unchanged."""
        host = _Host()
        host.use_mlx_dit = True
        host.mlx_decoder = object()
        host._load_model_context = lambda _name: contextlib.nullcontext()

        cond = (
            torch.ones(1, 2, 4, dtype=torch.float32),
            torch.ones(1, 2, dtype=torch.bool),
            torch.ones(1, 4, 4, dtype=torch.float32),
        )
        host.model = type(
            "FakeModel",
            (),
            {
                "prepare_condition": Mock(side_effect=[cond, cond]),
            },
        )()

        captured = {}

        def _fake_mlx_run_diffusion(**kwargs):
            captured.update(kwargs)
            return {
                "target_latents": torch.zeros(1, 4, 4, dtype=torch.float32),
                "time_costs": {"diffusion_time_cost": 0.5},
            }

        host._mlx_run_diffusion = _fake_mlx_run_diffusion

        payload = {
            "spans": [("repainting", 0, 4)],
            "src_latents": torch.zeros(1, 4, 4, dtype=torch.float32),
            "text_hidden_states": torch.zeros(1, 2, 4, dtype=torch.float32),
            "text_attention_mask": torch.ones(1, 2, dtype=torch.bool),
            "lyric_hidden_states": torch.zeros(1, 2, 4, dtype=torch.float32),
            "lyric_attention_mask": torch.ones(1, 2, dtype=torch.bool),
            "refer_audio_acoustic_hidden_states_packed": torch.zeros(1, 2, 4, dtype=torch.float32),
            "refer_audio_order_mask": torch.zeros(1, dtype=torch.long),
            "chunk_mask": torch.ones(1, 4, dtype=torch.bool),
            "is_covers": torch.tensor([True]),
            "precomputed_lm_hints_25Hz": None,
            "non_cover_text_hidden_states": torch.zeros(1, 2, 4, dtype=torch.float32),
            "non_cover_text_attention_masks": torch.ones(1, 2, dtype=torch.bool),
        }
        generate_kwargs = {
            "infer_steps": 12,
            "timesteps": torch.tensor([1.0, 0.6, 0.3], dtype=torch.float32),
            "diffusion_guidance_sale": 8.5,
            "cfg_interval_start": 0.2,
            "cfg_interval_end": 0.85,
            "use_adg": True,
        }

        outputs, encoder_hidden_states, encoder_attention_mask, context_latents = host._execute_service_generate_diffusion(
            payload=payload,
            generate_kwargs=generate_kwargs,
            seed_param=999,
            infer_method="ode",
            shift=1.5,
            audio_cover_strength=0.5,
            cover_noise_strength=0.25,
        )

        self.assertEqual(captured["seed"], 999)
        self.assertEqual(captured["infer_steps"], 12)
        self.assertEqual(captured["guidance_scale"], 8.5)
        self.assertEqual(captured["cfg_interval_start"], 0.2)
        self.assertEqual(captured["cfg_interval_end"], 0.85)
        self.assertTrue(captured["use_adg"])
        self.assertEqual(captured["timesteps"].tolist(), [1.0, 0.6, 0.3])
        self.assertEqual(float(outputs["time_costs"]["diffusion_time_cost"]), 0.5)
        self.assertEqual(tuple(encoder_hidden_states.shape), (1, 2, 4))
        self.assertEqual(tuple(encoder_attention_mask.shape), (1, 2))
        self.assertEqual(tuple(context_latents.shape), (1, 4, 4))


if __name__ == "__main__":
    unittest.main()
