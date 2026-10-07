"""Execute actual API routing/workers with a fake MLX renderer (no GPU imports)."""

import ast
import base64
import io
import os
from pathlib import Path
from types import SimpleNamespace
import time
import unittest
from unittest.mock import Mock

from inference_profiles import resolve_audio2audio_inference_settings, resolve_inference_settings


class Audio:
    shape = (2, 80)

    def clamp(self, *args):
        return self


def api_namespace(data):
    source = Path(__file__).with_name("api_foundation.py")
    names = {"validate_request", "reject", "generate", "audio2audio", "generation_worker", "audio2audio_worker"}
    functions = [node for node in ast.parse(source.read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in functions} == names
    for node in functions:
        node.decorator_list = []
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    namespace = dict(
        resolve_inference_settings=resolve_inference_settings,
        resolve_audio2audio_inference_settings=resolve_audio2audio_inference_settings,
        request=SimpleNamespace(get_json=lambda: data), jsonify=lambda value: value,
        cleanup_old_sessions=Mock(), model_ready=SimpleNamespace(is_set=lambda: True),
        resolve_host_bpm=lambda value, **kwargs: 120, HostBpmError=ValueError,
        SUPPORTED_BARS=[4, 8], VALID_KEY_ROOTS=["C"], VALID_KEY_MODES=["minor"],
        nearest_foundation_bpm=lambda bpm: 120, derive_duration=lambda *args: 4,
        build_prompt=lambda data: "piano", time_stretch_ratio=lambda *args: 1,
        uuid=SimpleNamespace(uuid4=lambda: "test-session"), create_session=Mock(),
        threading=SimpleNamespace(Thread=Mock()),
        generation_semaphore=Mock(acquire=Mock(return_value=True)),
        _HAS_SAOMLX=True, Path=lambda _: SimpleNamespace(exists=lambda: True),
        FOUNDATION_CKPT_PATH="fake.safetensors", FOUNDATION_CONFIG_PATH="fake.json",
        load_foundation_runtime_profile=lambda: dict(diffusion_objective="v", sampler_type="ignored"),
        update_session=Mock(), time=time, gc=SimpleNamespace(collect=Mock()), os=os,
        np=SimpleNamespace(shape=lambda audio: audio.shape),
        coerce_channels_first_audio=lambda *args, **kwargs: Audio(),
        encode_audio_to_wav_bytes=Mock(return_value=(b"wav", "encoded-wav")),
        OUTPUT_DIR="unused-output", base64=base64, io=io,
        torchaudio=SimpleNamespace(load=lambda _: (Audio(), 20)), print=Mock(),
        generate_diffusion_cond_mlx=Mock(return_value=dict(sample_rate=20, audio=Audio())),
    )
    module = ast.fix_missing_locations(ast.Module(body=[future, *functions], type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    return namespace


class FoundationAPIProfilesTest(unittest.TestCase):
    def test_profiles_reach_mlx_and_completed_metadata_on_both_routes(self):
        for route in ("generate", "audio2audio"):
            for profile in ("gary_fallback", "royalcities"):
                with self.subTest(route=route, profile=profile):
                    data = dict(seed=42, bars=4, host_bpm=120, inference_profile=profile,
                                audio_data="YQ==", init_noise_level=0.25, steps=64, guidance_scale=5)
                    ns = api_namespace(data)
                    response = ns[route]()
                    self.assertTrue(response["success"])
                    scheduled = ns["threading"].Thread.call_args.kwargs
                    scheduled["target"](*scheduled["args"], **scheduled.get("kwargs", {}))
                    call = ns["generate_diffusion_cond_mlx"].call_args.kwargs
                    configured = resolve_inference_settings(data)
                    self.assertEqual(call["sampler_type"], configured["sampler_type"])
                    self.assertEqual(call["sigma_min"], configured["sigma_min"])
                    self.assertEqual(call["sigma_max"], 0.25 if route == "audio2audio" else configured["sigma_max"])
                    self.assertEqual(call["rho"], configured["rho"])
                    self.assertEqual(call["steps"], 64)
                    self.assertEqual(call["cfg_scale"], 5)
                    completed = ns["update_session"].call_args.kwargs
                    self.assertEqual(completed["status"], "completed")
                    self.assertEqual(completed["meta"]["inference_profile"], profile)
                    self.assertEqual(completed["meta"]["sigma_max"], call["sigma_max"])
                    ns["generation_semaphore"].release.assert_called_once()

    def test_invalid_and_unsupported_profiles_rejected_before_queueing(self):
        for route in ("generate", "audio2audio"):
            for fields in ({"inference_profile": "unknown"}, {"sampler_type": "k-dpm-fast"},
                           {"sampler_type": "k-dpm-adaptive"}, {"rho": "nan"}):
                with self.subTest(route=route, fields=fields):
                    ns = api_namespace(dict(seed=42, audio_data="YQ==", host_bpm=120, **fields))
                    response, status = ns[route]()
                    self.assertEqual(status, 400)
                    self.assertFalse(response["success"])
                    ns["threading"].Thread.assert_not_called()


if __name__ == "__main__":
    unittest.main()
