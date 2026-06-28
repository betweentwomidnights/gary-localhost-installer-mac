"""Tests for the mac Carey wrapper routing and model-selection helpers."""

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch


def _load_wrapper_module():
    module_path = Path(__file__).with_name("main.py")
    module_dir = str(module_path.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    spec = importlib.util.spec_from_file_location("carey_wrapper_main", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


wrapper = _load_wrapper_module()


class CareyWrapperModelSelectionTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._original_current_model = wrapper._current_model
        self._original_lora_registry = dict(wrapper.LORA_REGISTRY)

    def tearDown(self):
        wrapper._current_model = self._original_current_model
        wrapper.LORA_REGISTRY.clear()
        wrapper.LORA_REGISTRY.update(self._original_lora_registry)

    def test_cover_form_data_uses_turbo_generation_defaults_and_cover_nofsq(self):
        job = wrapper.Job(task_id="job-1", task_type="cover", bpm=120, duration=42.0)
        req = SimpleNamespace(
            audio_data="",
            bpm=120,
            caption="orchestral version",
            lyrics="[Instrumental]",
            language="en",
            key_scale="",
            cover_noise_strength=0.2,
            audio_cover_strength=0.3,
            guidance_scale=7.5,
            inference_steps=64,
            use_src_as_ref=False,
            no_fsq=False,
            time_signature="4",
            batch_size=1,
            audio_format="wav",
        )

        data = wrapper._build_form_data(job, req, "ignored.wav")

        self.assertEqual(data["guidance_scale"], str(wrapper.COVER_GUIDANCE_SCALE))
        self.assertEqual(data["inference_steps"], str(wrapper.COVER_INFERENCE_STEPS))
        self.assertEqual(data["task_type"], "cover-nofsq")

    def test_cover_form_data_no_fsq_flag_is_backcompat_noop(self):
        job = wrapper.Job(task_id="job-1b", task_type="cover", bpm=120, duration=42.0)
        for no_fsq in (False, True):
            with self.subTest(no_fsq=no_fsq):
                req = SimpleNamespace(
                    audio_data="",
                    bpm=120,
                    caption="orchestral version",
                    lyrics="[Instrumental]",
                    language="en",
                    key_scale="",
                    cover_noise_strength=0.2,
                    audio_cover_strength=0.3,
                    guidance_scale=1.0,
                    inference_steps=8,
                    use_src_as_ref=False,
                    no_fsq=no_fsq,
                    time_signature="4",
                    batch_size=1,
                    audio_format="wav",
                )

                data = wrapper._build_form_data(job, req, "ignored.wav")

                self.assertEqual(data["task_type"], "cover-nofsq")

    def test_lego_form_data_uses_simple_fallback_caption_pool_when_empty(self):
        job = wrapper.Job(task_id="job-1c", task_type="lego", bpm=120, duration=8.0)
        req = SimpleNamespace(
            bpm=120,
            caption="  ",
            lyrics="",
            language="en",
            key_scale="",
            guidance_scale=7.0,
            inference_steps=50,
            track_name="brass",
            time_signature="4",
            batch_size=1,
            audio_format="wav",
        )

        with patch.object(wrapper.random, "choice", return_value="jazzy trumpet solo") as choice_mock:
            data = wrapper._build_form_data(job, req, "ignored.wav")

        choice_mock.assert_called_once_with(wrapper.TRACK_CAPTION_POOLS["brass"])
        self.assertEqual(data["caption"], "jazzy trumpet solo")

    def test_generation_request_models_expose_seed(self):
        requests = (
            wrapper.LegoRequest(audio_data="audio", track_name="drums", bpm=120),
            wrapper.CompleteRequest(audio_data="audio", bpm=120, audio_duration=16.0),
            wrapper.CoverRequest(audio_data="audio", bpm=120, caption="dub remix"),
        )

        for request in requests:
            with self.subTest(request_type=type(request).__name__):
                self.assertEqual(request.seed, -1)
                fixed_request = request.model_copy(update={"seed": 42})
                self.assertEqual(fixed_request.seed, 42)

    def test_generation_form_data_forwards_fixed_and_random_seed(self):
        cases = (
            (
                wrapper.Job(task_id="seed-lego", task_type="lego", bpm=120, duration=8.0),
                wrapper.LegoRequest(
                    audio_data="audio", track_name="drums", bpm=120, caption="drum break"
                ),
            ),
            (
                wrapper.Job(task_id="seed-complete", task_type="complete", bpm=120, duration=8.0),
                wrapper.CompleteRequest(
                    audio_data="audio", bpm=120, audio_duration=16.0, caption="synthwave"
                ),
            ),
            (
                wrapper.Job(task_id="seed-cover", task_type="cover", bpm=120, duration=8.0),
                wrapper.CoverRequest(
                    audio_data="audio", bpm=120, caption="dub remix"
                ),
            ),
        )

        for job, request in cases:
            with self.subTest(task_type=job.task_type, seed="fixed"):
                fixed_data = wrapper._build_form_data(
                    job, request.model_copy(update={"seed": 42}), "ignored.wav"
                )
                self.assertEqual(fixed_data["seed"], "42")
                self.assertEqual(fixed_data["use_random_seed"], "false")

            with self.subTest(task_type=job.task_type, seed="random"):
                random_data = wrapper._build_form_data(job, request, "ignored.wav")
                self.assertNotIn("seed", random_data)
                self.assertEqual(random_data["use_random_seed"], "true")

    def test_completed_status_returns_backend_seed(self):
        job = wrapper.Job(
            task_id="seed-status",
            task_type="cover",
            bpm=120,
            status=wrapper.JobStatus.COMPLETED,
            audio_b64="audio",
            seed_used="42,84",
        )

        response = wrapper._build_status_response(job)
        payload = json.loads(response.body)

        self.assertEqual(payload["seed"], "42,84")
        self.assertEqual(payload["last_seed"], "42,84")

    def test_lego_tracks_always_use_active_base_backend(self):
        for track_name in ("vocals", "backing_vocals", "drums"):
            with self.subTest(track_name=track_name):
                self.assertEqual(
                    wrapper._backend_key_for("lego", requested_model="sft", track_name=track_name),
                    "base",
                )
                self.assertEqual(
                    wrapper._required_model_for_task("lego", requested_model="sft", track_name=track_name),
                    wrapper.ACESTEP_LEGO_CONFIG,
                )

    def test_lego_lora_request_validates_base_backend_and_family(self):
        wrapper.LORA_REGISTRY.clear()
        wrapper.LORA_REGISTRY.update(
            {
                "standard-style": {
                    "path": "/tmp/standard-style",
                    "model_family": "standard",
                    "backends": ["base"],
                },
                "turbo-only": {
                    "path": "/tmp/turbo-only",
                    "model_family": "standard",
                    "backends": ["turbo"],
                },
                "xl-style": {
                    "path": "/tmp/xl-style",
                    "model_family": "xl",
                    "backends": ["base"],
                },
            }
        )

        wrapper._validate_lora_request(
            "lego",
            SimpleNamespace(lora="standard-style", model="sft", track_name="vocals"),
        )

        for lora_name in ("unknown-adapter", "turbo-only", "xl-style"):
            with self.subTest(lora_name=lora_name), self.assertRaises(wrapper.HTTPException):
                wrapper._validate_lora_request(
                    "lego",
                    SimpleNamespace(lora=lora_name, model="sft", track_name="vocals"),
                )

    async def test_ensure_required_model_swaps_vocal_lego_from_sft_to_base(self):
        lego_job = wrapper.Job(task_id="job-4b", task_type="lego", bpm=120)
        wrapper._current_model = wrapper.ACESTEP_SFT_CONFIG
        client = object()
        req = SimpleNamespace(model="sft", track_name="vocals")

        with patch.object(wrapper, "_unload_model", AsyncMock()) as unload_mock, patch.object(
            wrapper,
            "_load_model",
            AsyncMock(return_value=wrapper.ACESTEP_LEGO_CONFIG),
        ) as load_mock:
            await wrapper._ensure_required_model(client=client, job=lego_job, req=req)

        unload_mock.assert_awaited_once()
        load_mock.assert_awaited_once_with(client, wrapper.ACESTEP_LEGO_CONFIG)
        self.assertEqual(wrapper._current_model, wrapper.ACESTEP_LEGO_CONFIG)


if __name__ == "__main__":
    unittest.main()
