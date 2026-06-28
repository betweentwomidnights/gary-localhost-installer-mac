from __future__ import annotations

import json
import types
import tempfile
import unittest
import wave
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import train_mlx_lora_job as job


def _write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(b"\x00\x00" * 800)


def _base_args(temp: Path, dataset_dir: Path) -> list[str]:
    return [
        "--job-id",
        "job-1",
        "--name",
        "Billie Test",
        "--dataset-dir",
        str(dataset_dir),
        "--checkpoint-dir",
        str(temp / "checkpoints"),
        "--run-dir",
        str(temp / "run"),
        "--status-path",
        str(temp / "run" / "status.json"),
        "--current-job-path",
        str(temp / "current-job.json"),
        "--log-path",
        str(temp / "run" / "job.log"),
    ]


class TrainMLXLoraJobTests(unittest.TestCase):
    def test_command_builders_use_mlx_training_contract(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            temp = Path(raw)
            dataset_dir = temp / "dataset"
            dataset_dir.mkdir()
            args = job.build_parser().parse_args(_base_args(temp, dataset_dir))
            args.cancel_path = temp / "run" / "cancel.requested"

            preprocess = job.build_preprocess_command(
                args,
                temp / "run" / "dataset.json",
                temp / "run" / "output" / "tensors",
                temp / "run" / "output",
            )
            train = job.build_train_command(
                args,
                temp / "run" / "output" / "tensors",
                temp / "run" / "output",
            )

            self.assertIn(str(job.TRAIN_ENTRY), preprocess)
            self.assertIn("--preprocess", preprocess)
            self.assertIn("--dataset-json", preprocess)
            self.assertIn("--device", preprocess)
            self.assertIn("auto", preprocess)
            self.assertIn(str(job.MLX_TRAIN_ENTRY), train)
            self.assertIn("--adapter-type", train)
            self.assertIn("dora", train)
            self.assertIn("--module-profile", train)
            self.assertIn("balanced", train)
            self.assertIn("--weight-decay", train)
            self.assertIn("0.01", train)
            self.assertIn("--loss-weighting", train)
            self.assertIn("min_snr", train)
            self.assertIn("--save-best", train)
            self.assertIn("--save-best-after", train)
            self.assertIn("25", train)
            self.assertIn("--gradient-checkpointing", train)
            self.assertNotIn("--memory-limit-gb", train)
            self.assertEqual(train[train.index("--timestep-mu") + 1], "-0.4")

    def test_xl_command_uses_guarded_training_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            temp = Path(raw)
            dataset_dir = temp / "dataset"
            dataset_dir.mkdir()
            args = job.build_parser().parse_args(
                _base_args(temp, dataset_dir)
                + [
                    "--model",
                    "xl-base",
                    "--allow-unsafe-xl",
                    "--max-steps",
                    "1",
                ]
            )
            args.cancel_path = temp / "run" / "cancel.requested"

            train = job.build_train_command(
                args,
                temp / "run" / "output" / "tensors",
                temp / "run" / "output",
            )

            self.assertIn("--gradient-checkpointing", train)
            self.assertIn("--memory-limit-gb", train)
            self.assertIn(str(job.XL_DEFAULT_MEMORY_LIMIT_GB), train)
            self.assertIn("--allow-unsafe-xl", train)
            self.assertIn("--max-steps", train)
            self.assertIn("1", train)

    def test_timestep_mu_defaults_to_sidestep_schedule_unless_overridden(self) -> None:
        default_args = job.build_parser().parse_args(
            _base_args(Path("/tmp"), Path("/tmp/dataset"))
        )
        instrumental_args = job.build_parser().parse_args(
            _base_args(Path("/tmp"), Path("/tmp/dataset")) + ["--instrumental"]
        )
        override_args = job.build_parser().parse_args(
            _base_args(Path("/tmp"), Path("/tmp/dataset")) + ["--timestep-mu", "0.0"]
        )

        self.assertEqual(job.resolve_timestep_mu(default_args), -0.4)
        self.assertEqual(job.resolve_timestep_mu(instrumental_args), -0.4)
        self.assertEqual(job.resolve_timestep_mu(override_args), 0.0)

    def test_dry_run_prepares_billie_style_dataset_and_plan(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            temp = Path(raw)
            dataset_dir = temp / "billie"
            audio = dataset_dir / "bad guy.wav"
            _write_wav(audio)
            audio.with_suffix(".txt").write_text(
                "\n".join(
                    [
                        "caption: A whispered alt-pop track with sparse bass.",
                        "genre: Alt pop, Electro pop",
                        "bpm: 135",
                        "key: G minor",
                        "signature: 4",
                        "is_instrumental: false",
                        "custom_tag: billie",
                        "lyrics: white shirt now red",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            code = job.main(
                _base_args(temp, dataset_dir)
                + [
                    "--caption",
                    "understand_music",
                    "--trigger",
                    "billie",
                    "--genre-ratio",
                    "30",
                    "--dry-run",
                ]
            )

            self.assertEqual(code, 0)
            dataset = json.loads((temp / "run" / "dataset.json").read_text())
            sample = dataset["samples"][0]
            self.assertEqual(dataset["metadata"]["genre_ratio"], 30)
            self.assertEqual(sample["custom_tag"], "billie")
            self.assertEqual(sample["keyscale"], "G minor")
            self.assertEqual(sample["timesignature"], "4")
            self.assertEqual(sample["bpm"], 135)
            plan = json.loads((temp / "run" / "training_plan.json").read_text())
            self.assertEqual(plan["adapterType"], "dora")
            self.assertEqual(plan["moduleProfile"], "balanced")
            self.assertTrue(plan["gradientCheckpointing"])
            self.assertEqual(plan["memoryLimitGb"], 0.0)
            self.assertIn(str(job.MLX_TRAIN_ENTRY), plan["trainCommand"])
            status = json.loads((temp / "run" / "status.json").read_text())
            self.assertEqual(status["status"], "completed")
            self.assertEqual(status["phase"], "prepared")
            self.assertEqual(status["sample_count"], 1)
            self.assertEqual(status["captioned_count"], 0)
            self.assertIn("already have sidecars", status["message"])
            pointer = json.loads((temp / "current-job.json").read_text())
            self.assertEqual(pointer["statusPath"], str((temp / "run" / "status.json").resolve()))
            self.assertEqual(pointer["logPath"], str((temp / "run" / "job.log").resolve()))
            self.assertEqual(pointer["cancelPath"], str((temp / "run" / "cancel.requested").resolve()))
            self.assertEqual(pointer["runDir"], str((temp / "run").resolve()))

    def test_understand_music_captions_missing_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            temp = Path(raw)
            dataset_dir = temp / "dataset"
            audio = dataset_dir / "bpm_135_missing.wav"
            _write_wav(audio)

            class FakeProcess:
                pid = 12345
                returncode = None

                def poll(self) -> None:
                    return None

            class FakeClient:
                def __init__(self, *args, **kwargs) -> None:
                    pass

                def __enter__(self):
                    return self

                def __exit__(self, *args) -> None:
                    pass

            analysis = {
                "caption": "Sparse whisper pop with sub bass and close vocals.",
                "genre": "Alt pop, Electro pop",
                "lyrics": "white shirt now red",
                "bpm": 120,
                "keyscale": "G minor",
                "language": "en",
            }
            fake_httpx = types.SimpleNamespace(Client=FakeClient, Timeout=lambda value: value)
            with patch.dict("sys.modules", {"httpx": fake_httpx}), \
                 patch.object(job, "start_caption_server", return_value=FakeProcess()), \
                 patch.object(job, "stop_caption_server"), \
                 patch.object(job, "wait_for_carey"), \
                 patch.object(job, "ensure_carey_model_loaded"), \
                 patch.object(job, "require_caption_lm_backend"), \
                 patch.object(job, "request_valid_music_analysis", return_value=analysis):
                code = job.main(
                    _base_args(temp, dataset_dir)
                    + [
                        "--caption",
                        "understand_music",
                        "--trigger",
                        "billie",
                        "--no-bpm-analysis",
                        "--no-key-analysis",
                        "--dry-run",
                    ]
                )

            self.assertEqual(code, 0)
            sidecar = audio.with_suffix(".txt").read_text(encoding="utf-8")
            self.assertIn("caption: Sparse whisper pop", sidecar)
            self.assertIn("bpm: 135", sidecar)
            self.assertIn("bpm_source: filename", sidecar)
            self.assertIn("lm_bpm: 120", sidecar)
            self.assertIn("keyscale: G minor", sidecar)
            self.assertIn("custom_tag: billie", sidecar)
            self.assertIn("lyrics: ", sidecar)
            self.assertNotIn("white shirt now red", sidecar)
            dataset = json.loads((temp / "run" / "dataset.json").read_text())
            self.assertEqual(dataset["samples"][0]["bpm"], 135)
            self.assertEqual(dataset["samples"][0]["caption"], analysis["caption"])
            status = json.loads((temp / "run" / "status.json").read_text())
            self.assertEqual(status["status"], "completed")
            self.assertEqual(status["captioned_count"], 1)
            self.assertEqual(status["caption_lm_model"], "acestep-5Hz-lm-1.7B")

    def test_understand_music_preserves_split_human_lyrics(self) -> None:
        with TemporaryDirectory() as tmpdir:
            temp = Path(tmpdir)
            dataset_dir = temp / "dataset"
            dataset_dir.mkdir()
            audio = dataset_dir / "Billie_135bpm_Gmin.wav"
            _write_wav(audio)
            audio.with_suffix(".lyrics.txt").write_text(
                "[Verse]\nreal human lyrics\n",
                encoding="utf-8",
            )

            class FakeProcess:
                pid = 12345
                returncode = None

                def poll(self) -> None:
                    return None

            class FakeClient:
                def __init__(self, *args, **kwargs) -> None:
                    pass

                def __enter__(self):
                    return self

                def __exit__(self, *args) -> None:
                    pass

            analysis = {
                "caption": "Sparse whisper pop with sub bass and close vocals.",
                "genre": "Alt pop, Electro pop",
                "lyrics": "hallucinated lm lyrics",
                "bpm": 120,
                "keyscale": "G minor",
                "language": "en",
            }
            fake_httpx = types.SimpleNamespace(Client=FakeClient, Timeout=lambda value: value)
            with patch.dict("sys.modules", {"httpx": fake_httpx}), \
                 patch.object(job, "start_caption_server", return_value=FakeProcess()), \
                 patch.object(job, "stop_caption_server"), \
                 patch.object(job, "wait_for_carey"), \
                 patch.object(job, "ensure_carey_model_loaded"), \
                 patch.object(job, "require_caption_lm_backend"), \
                 patch.object(job, "request_valid_music_analysis", return_value=analysis):
                code = job.main(
                    _base_args(temp, dataset_dir)
                    + [
                        "--caption",
                        "understand_music",
                        "--no-bpm-analysis",
                        "--no-key-analysis",
                        "--dry-run",
                    ]
                )

            self.assertEqual(code, 0)
            sidecar = audio.with_suffix(".txt").read_text(encoding="utf-8")
            self.assertIn("caption: Sparse whisper pop", sidecar)
            self.assertIn("lyrics: [Verse]", sidecar)
            self.assertIn("real human lyrics", sidecar)
            self.assertNotIn("hallucinated lm lyrics", sidecar)

    def test_overwrite_existing_sidecar_drops_canonical_hallucinated_lyrics(self) -> None:
        with TemporaryDirectory() as tmpdir:
            temp = Path(tmpdir)
            dataset_dir = temp / "dataset"
            dataset_dir.mkdir()
            audio = dataset_dir / "Billie_135bpm_Gmin.wav"
            _write_wav(audio)
            audio.with_suffix(".txt").write_text(
                "caption: old caption\nlyrics: old hallucinated lyric\n",
                encoding="utf-8",
            )

            class FakeProcess:
                pid = 12345
                returncode = None

                def poll(self) -> None:
                    return None

            class FakeClient:
                def __init__(self, *args, **kwargs) -> None:
                    pass

                def __enter__(self):
                    return self

                def __exit__(self, *args) -> None:
                    pass

            analysis = {
                "caption": "Sparse whisper pop with sub bass and close vocals.",
                "genre": "Alt pop, Electro pop",
                "lyrics": "new hallucinated lm lyrics",
                "bpm": 120,
                "keyscale": "G minor",
                "language": "en",
            }
            fake_httpx = types.SimpleNamespace(Client=FakeClient, Timeout=lambda value: value)
            with patch.dict("sys.modules", {"httpx": fake_httpx}), \
                 patch.object(job, "start_caption_server", return_value=FakeProcess()), \
                 patch.object(job, "stop_caption_server"), \
                 patch.object(job, "wait_for_carey"), \
                 patch.object(job, "ensure_carey_model_loaded"), \
                 patch.object(job, "require_caption_lm_backend"), \
                 patch.object(job, "request_valid_music_analysis", return_value=analysis):
                code = job.main(
                    _base_args(temp, dataset_dir)
                    + [
                        "--caption",
                        "understand_music",
                        "--overwrite-captions",
                        "--no-bpm-analysis",
                        "--no-key-analysis",
                        "--dry-run",
                    ]
                )

            self.assertEqual(code, 0)
            sidecar = audio.with_suffix(".txt").read_text(encoding="utf-8")
            self.assertIn("caption: Sparse whisper pop", sidecar)
            self.assertIn("lyrics: ", sidecar)
            self.assertNotIn("old hallucinated lyric", sidecar)
            self.assertNotIn("new hallucinated lm lyrics", sidecar)

    def test_caption_server_env_uses_mac_ace_root_and_mlx_backend(self) -> None:
        args = job.build_parser().parse_args(
            _base_args(Path("/tmp"), Path("/tmp/dataset"))
            + ["--caption", "understand_music", "--model", "base"]
        )
        env = job.build_caption_server_env(args, {"PYTHONPATH": "/existing"})

        self.assertIn(str(job.ACE_ROOT), env["PYTHONPATH"])
        self.assertIn(str(job.SERVICE_DIR), env["PYTHONPATH"])
        self.assertIn("/existing", env["PYTHONPATH"])
        self.assertEqual(env["ACESTEP_CONFIG_PATH"], "acestep-v15-base")
        self.assertEqual(env["ACESTEP_LM_BACKEND"], "mlx")
        self.assertEqual(env["ACESTEP_REQUIRE_MLX_LM"], "true")
        self.assertEqual(env["ACESTEP_USE_MLX_DIT"], "1")

        fallback_args = job.build_parser().parse_args(
            _base_args(Path("/tmp"), Path("/tmp/dataset"))
            + [
                "--caption",
                "understand_music",
                "--model",
                "base",
                "--caption-lm-backend",
                "pt",
            ]
        )
        fallback_env = job.build_caption_server_env(fallback_args, {})
        self.assertEqual(fallback_env["ACESTEP_LM_BACKEND"], "pt")
        self.assertEqual(fallback_env["ACESTEP_REQUIRE_MLX_LM"], "false")

    def test_caption_server_log_filter_suppresses_audio_code_dump(self) -> None:
        noisy = (
            "2026-06-23 | INFO | generated "
            "<|audio_code_51704|><|audio_code_35256|><|im_end|>\n"
        )

        filtered = job.sanitize_caption_server_log_line(noisy)

        self.assertIn("suppressed raw ACE audio-code token dump", filtered)
        self.assertNotIn("<|audio_code_", filtered)
        self.assertNotIn("51704", filtered)

    def test_registration_writes_carey_registry_and_caption_pool(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            temp = Path(raw)
            dataset_dir = temp / "dataset"
            audio = dataset_dir / "song.wav"
            _write_wav(audio)
            audio.with_suffix(".txt").write_text(
                "caption: neon pop hook\ngenre: Pop, Synth pop\nlyrics: hello\n",
                encoding="utf-8",
            )
            args = job.build_parser().parse_args(
                _base_args(temp, dataset_dir)
                + [
                    "--lora-catalog-path",
                    str(temp / "catalog.json"),
                    "--lora-registry-path",
                    str(temp / "registry.json"),
                    "--captions-json-path",
                    str(temp / "captions.json"),
                    "--genre-ratio",
                    "20",
                ]
            )
            args.name = job.slugify(args.name)

            checkpoint = temp / "adapter"
            checkpoint.mkdir()
            job.register_trained_lora(args, checkpoint)

            catalog = json.loads((temp / "catalog.json").read_text())
            registry = json.loads((temp / "registry.json").read_text())
            captions = json.loads((temp / "captions.json").read_text())
            self.assertEqual(catalog["billie-test"]["modelFamily"], "standard")
            self.assertEqual(registry["billie-test"]["model_family"], "standard")
            self.assertEqual(registry["billie-test"]["adapter_type"], "dora")
            self.assertIn("neon pop hook", captions["billie-test"])
            self.assertIn("Pop, Synth pop", captions["billie-test"])

    def test_registration_omits_genre_prompts_when_genre_ratio_is_zero(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            temp = Path(raw)
            dataset_dir = temp / "dataset"
            audio = dataset_dir / "song.wav"
            _write_wav(audio)
            audio.with_suffix(".txt").write_text(
                "caption: neon pop hook\ngenre: Pop, Synth pop\nlyrics: hello\n",
                encoding="utf-8",
            )
            args = job.build_parser().parse_args(
                _base_args(temp, dataset_dir)
                + [
                    "--captions-json-path",
                    str(temp / "captions.json"),
                    "--genre-ratio",
                    "0",
                ]
            )
            args.name = job.slugify(args.name)

            checkpoint = temp / "adapter"
            checkpoint.mkdir()
            job.register_trained_lora(args, checkpoint)

            captions = json.loads((temp / "captions.json").read_text())
            self.assertEqual(captions["billie-test"], ["neon pop hook"])


if __name__ == "__main__":
    unittest.main()
