import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import sa3_autolabel
from sa3_autolabel import discover_sa3_audio, format_sidecar, usable_genre


class SA3AutolabelTests(unittest.TestCase):
    def test_formats_bare_and_labeled_sidecars(self) -> None:
        self.assertEqual(
            format_sidecar("bare", "electro rock", 120, "C minor"),
            "electro rock, 120 bpm, C minor",
        )
        self.assertEqual(
            format_sidecar("labeled", "electro rock", 120, "C minor"),
            (
                "TrackType: Music, VocalType: Instrumental, "
                "Genre: electro rock, BPM: 120, Key: C minor"
            ),
        )

    def test_discovery_is_recursive_sorted_and_audio_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nested").mkdir()
            (root / "Z.wav").touch()
            (root / "nested" / "a.FLAC").touch()
            (root / "ignore.txt").touch()
            self.assertEqual(
                [path.relative_to(root).as_posix() for path in discover_sa3_audio(root)],
                ["nested/a.FLAC", "Z.wav"],
            )

    def test_genre_normalizes_lists_and_rejects_missing_values(self) -> None:
        audio = Path("track.wav")
        self.assertEqual(
            usable_genre({"genres": ["electronic", "rock"]}, audio),
            "electronic, rock",
        )
        with self.assertRaisesRegex(RuntimeError, "no usable genre"):
            usable_genre({"genre": "unknown"}, audio)

    def test_main_writes_sidecar_and_terminal_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            run = root / "run"
            dataset.mkdir()
            audio = dataset / "track.wav"
            audio.touch()
            status = run / "status.json"
            current = run / "current.json"
            log = run / "autolabel.log"
            cancel = run / "cancel.requested"
            client = mock.MagicMock()
            client.__enter__.return_value = client
            client.__exit__.return_value = False
            fake_httpx = SimpleNamespace(
                Client=mock.MagicMock(return_value=client),
                Timeout=lambda value: value,
            )
            bpm = SimpleNamespace(bpm=128, source="local")
            key = SimpleNamespace(keyscale="D minor", source="local")

            with (
                mock.patch.dict(sys.modules, {"httpx": fake_httpx}),
                mock.patch.object(sa3_autolabel, "require_caption_lm_backend"),
                mock.patch.object(sa3_autolabel, "ensure_carey_stopped"),
                mock.patch.object(
                    sa3_autolabel,
                    "start_caption_server",
                    return_value=mock.sentinel.server,
                ),
                mock.patch.object(sa3_autolabel, "stop_caption_server") as stop,
                mock.patch.object(sa3_autolabel, "wait_for_carey"),
                mock.patch.object(sa3_autolabel, "ensure_carey_model_loaded"),
                mock.patch.object(
                    sa3_autolabel,
                    "request_valid_genre_analysis",
                    return_value=({"genre": "electro rock"}, "electro rock"),
                ),
                mock.patch.object(
                    sa3_autolabel,
                    "decide_sidecar_bpm",
                    return_value=bpm,
                ),
                mock.patch.object(
                    sa3_autolabel,
                    "decide_sidecar_key",
                    return_value=key,
                ),
            ):
                result = sa3_autolabel.main(
                    [
                        "--dataset-dir",
                        str(dataset),
                        "--style",
                        "bare",
                        "--job-id",
                        "test-job",
                        "--run-dir",
                        str(run),
                        "--status-path",
                        str(status),
                        "--current-job-path",
                        str(current),
                        "--log-path",
                        str(log),
                        "--cancel-path",
                        str(cancel),
                    ]
                )

            self.assertEqual(result, 0)
            self.assertEqual(
                audio.with_suffix(".txt").read_text(encoding="utf-8"),
                "electro rock, 128 bpm, D minor\n",
            )
            payload = json.loads(status.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "completed")
            self.assertEqual(payload["done"], 1)
            self.assertEqual(payload["dataset_path"], str(dataset.resolve()))
            stop.assert_called_once_with(mock.ANY, mock.sentinel.server)


if __name__ == "__main__":
    unittest.main()
