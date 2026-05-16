"""Tests for the mac Carey wrapper routing and model-selection helpers."""

import importlib.util
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
    spec.loader.exec_module(module)
    return module


wrapper = _load_wrapper_module()


class CareyWrapperModelSelectionTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._original_current_model = wrapper._current_model

    def tearDown(self):
        wrapper._current_model = self._original_current_model

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

    def test_lego_tracks_always_use_acestep_base_backend(self):
        for track_name in ("vocals", "backing_vocals", "drums"):
            with self.subTest(track_name=track_name):
                self.assertEqual(
                    wrapper._backend_key_for("lego", requested_model="sft", track_name=track_name),
                    "regular",
                )
                self.assertEqual(
                    wrapper._required_model_for_task("lego", requested_model="sft", track_name=track_name),
                    wrapper.ACESTEP_LEGO_CONFIG,
                )
                self.assertEqual(wrapper.ACESTEP_LEGO_CONFIG, "acestep-v15-base")

    def test_lego_lora_request_is_ignored(self):
        req = SimpleNamespace(lora="unknown-adapter", model="sft", track_name="vocals")

        wrapper._validate_lora_request("lego", req)

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
