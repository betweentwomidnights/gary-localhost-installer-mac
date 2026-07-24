from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


SERVICE_ROOT = Path(__file__).resolve().parents[1]
JOB_RUNNER = SERVICE_ROOT / "scripts" / "train_mlx_lora_job.py"


FAKE_TRAINER = """\
import argparse
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--prompt", default="")
args, _ = parser.parse_known_args()

print("Loading torch checkpoint and converting runtime modules to MLX...", flush=True)
if args.prompt == "cancel":
    time.sleep(30)
    raise SystemExit(0)

print("conversion_seconds=0.01", flush=True)
print("trainable_parameters=42", flush=True)
print("step=1/2 loss=1.25000000 step_seconds=0.01 average_seconds=0.01", flush=True)
args.output_dir.mkdir(parents=True, exist_ok=True)
intermediate = args.output_dir / "gary-mlx-lora-step-000001.safetensors"
intermediate.write_bytes(b"fake-checkpoint-step-1")
print(f"checkpoint={intermediate}", flush=True)
print("step=2/2 loss=0.75000000 step_seconds=0.01 average_seconds=0.01", flush=True)
checkpoint = args.output_dir / "gary-mlx-lora-final.safetensors"
checkpoint.write_bytes(b"fake-checkpoint")
print(f"final_checkpoint={checkpoint}", flush=True)
"""


def _command(
    root: Path,
    *,
    prompt: str,
    job_id: str = "test-job",
    run_name: str = "run",
) -> list[str]:
    audio_path = root / "sample.wav"
    audio_path.write_bytes(b"fake-audio")
    fake_trainer = root / "fake_trainer.py"
    fake_trainer.write_text(FAKE_TRAINER)
    run_dir = root / run_name

    return [
        sys.executable,
        str(JOB_RUNNER),
        "--job-id",
        job_id,
        "--name",
        "Bell Arpeggio",
        "--audio-path",
        str(audio_path),
        "--prompt",
        prompt,
        "--steps",
        "2",
        "--rank",
        "4",
        "--adapter-type",
        "dora",
        "--crop-seconds",
        "1",
        "--learning-rate",
        "0.0001",
        "--save-every",
        "1",
        "--output-dir",
        str(run_dir),
        "--status-path",
        str(run_dir / "status.json"),
        "--log-path",
        str(run_dir / "training.log"),
        "--cancel-path",
        str(run_dir / "cancel.requested"),
        "--lora-dir",
        str(root / "loras"),
        "--registry-path",
        str(root / "lora_registry.json"),
        "--catalog-path",
        str(root / "lora_catalog.json"),
        "--prompts-dir",
        str(root / "prompts"),
        "--trainer-path",
        str(fake_trainer),
    ]


def test_job_runner_installs_completed_lora(tmp_path: Path) -> None:
    result = subprocess.run(
        _command(tmp_path, prompt="bright bell arpeggio"),
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr

    state = json.loads((tmp_path / "run" / "status.json").read_text())
    installed = tmp_path / "loras" / "bell-arpeggio.safetensors"
    assert state["status"] == "completed"
    assert state["current_step"] == 2
    assert state["child_pid"] is None
    assert state["final_checkpoint_path"] == str(installed)
    assert installed.read_bytes() == b"fake-checkpoint"

    registry = json.loads((tmp_path / "lora_registry.json").read_text())
    assert registry["bell-arpeggio"]["path"] == str(installed)
    catalog = json.loads((tmp_path / "lora_catalog.json").read_text())
    assert catalog["bell-arpeggio"]["path"] == str(installed)
    assert catalog["bell-arpeggio"]["trainingBaseModel"] == "medium-base"
    assert catalog["bell-arpeggio"]["inferenceModel"] == "medium"
    assert catalog["bell-arpeggio"]["trainingJobId"] == "test-job"
    assert catalog["bell-arpeggio"]["selectedTrainingStep"] == 2
    assert catalog["bell-arpeggio"]["trainingCheckpoints"] == [
        {
            "step": 1,
            "epoch": 0,
            "path": str(
                (
                    tmp_path
                    / "run"
                    / "gary-mlx-lora-step-000001.safetensors"
                ).resolve()
            ),
        },
        {
            "step": 2,
            "epoch": 0,
            "path": str(
                (
                    tmp_path
                    / "run"
                    / "gary-mlx-lora-final.safetensors"
                ).resolve()
            ),
        },
    ]
    prompts = json.loads((tmp_path / "prompts" / "bell-arpeggio.json").read_text())
    assert prompts["dice"]["instrumental"] == ["bright bell arpeggio"]


def test_job_runner_installs_folder_prompt_pool(tmp_path: Path) -> None:
    command = _command(tmp_path, prompt="unused")
    for option in ("--audio-path", "--prompt"):
        index = command.index(option)
        del command[index : index + 2]

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "one.wav").write_bytes(b"fake-audio")
    (dataset / "one.txt").write_text("bright bells, BPM: 145\n")
    (dataset / "two.flac").write_bytes(b"fake-audio")
    (dataset / "two.txt").write_text("soft glass\n")
    command.extend(["--dataset-dir", str(dataset), "--trigger-text", "garybell"])

    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr

    catalog = json.loads((tmp_path / "lora_catalog.json").read_text())
    assert catalog["bell-arpeggio"]["promptsPath"] == str(dataset)
    prompts = json.loads((tmp_path / "prompts" / "bell-arpeggio.json").read_text())
    assert prompts["dice"]["instrumental"] == [
        "bright bells",
        "soft glass",
    ]


def test_job_runner_forwards_loudness_fix(tmp_path: Path) -> None:
    command = _command(tmp_path, prompt="bright bells")
    command.extend(["--per-track-target-latent-rms", "0.9"])

    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    log_text = (tmp_path / "run" / "training.log").read_text()
    assert "--model-name medium-base" in log_text
    assert "--per-track-target-latent-rms 0.9" in log_text
    state = json.loads((tmp_path / "run" / "status.json").read_text())
    assert state["training_base_model"] == "medium-base"
    assert (
        state["training_base_repo"]
        == "stabilityai/stable-audio-3-medium-base"
    )
    assert state["per_track_target_latent_rms"] == 0.9


def test_job_runner_forwards_full_track_policy(tmp_path: Path) -> None:
    command = _command(tmp_path, prompt="bright bells")
    command.append("--full-tracks")

    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    log_text = (tmp_path / "run" / "training.log").read_text()
    assert "--crop-seconds 285.35" in log_text
    assert "--full-tracks" in log_text
    state = json.loads((tmp_path / "run" / "status.json").read_text())
    assert state["full_tracks"] is True
    assert state["crop_seconds"] == 285.35


def test_job_runner_defaults_to_all_projections_layer_scope(tmp_path: Path) -> None:
    command = _command(tmp_path, prompt="bright bells")

    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    log_text = (tmp_path / "run" / "training.log").read_text()
    assert "--layer-scope all-projections" in log_text
    state = json.loads((tmp_path / "run" / "status.json").read_text())
    assert state["layer_scope"] == "all-projections"


def test_job_runner_forwards_layer_scope(tmp_path: Path) -> None:
    command = _command(tmp_path, prompt="bright bells")
    command.extend(["--layer-scope", "attention-feedforward"])

    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    log_text = (tmp_path / "run" / "training.log").read_text()
    assert "--layer-scope attention-feedforward" in log_text
    state = json.loads((tmp_path / "run" / "status.json").read_text())
    assert state["layer_scope"] == "attention-feedforward"


def test_job_runner_rejects_unknown_layer_scope(tmp_path: Path) -> None:
    command = _command(tmp_path, prompt="bright bells")
    command.extend(["--layer-scope", "attention-only"])

    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode != 0
    assert "attention-only" in result.stderr


def test_job_runner_forwards_advanced_timestep_settings(tmp_path: Path) -> None:
    command = _command(tmp_path, prompt="bright bells")
    command.extend(
        [
            "--timestep-sampler",
            "log_snr",
            "--distribution-shift",
            "full",
            "--shift-base",
            "0.6",
            "--shift-max",
            "1.25",
            "--shift-min-length",
            "192",
            "--shift-max-length",
            "5000",
            "--shift-use-sine",
        ]
    )

    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    log_text = (tmp_path / "run" / "training.log").read_text()
    assert "--timestep-sampler log_snr" in log_text
    assert "--distribution-shift full" in log_text
    assert "--shift-base 0.6" in log_text
    assert "--shift-max 1.25" in log_text
    assert "--shift-min-length 192" in log_text
    assert "--shift-max-length 5000" in log_text
    assert "--shift-use-sine" in log_text

    state = json.loads((tmp_path / "run" / "status.json").read_text())
    assert state["timestep_sampler"] == "log_snr"
    assert state["distribution_shift"] == "full"
    assert state["shift_base"] == 0.6
    assert state["shift_max"] == 1.25
    assert state["shift_min_length"] == 192
    assert state["shift_max_length"] == 5000
    assert state["shift_use_sine"] is True


def test_job_runner_honors_cancel_marker(tmp_path: Path) -> None:
    command = _command(tmp_path, prompt="cancel")
    process = subprocess.Popen(command)
    status_path = tmp_path / "run" / "status.json"

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if status_path.is_file():
            state = json.loads(status_path.read_text())
            if state.get("status") == "running":
                break
        time.sleep(0.05)
    else:
        process.kill()
        raise AssertionError("Job runner did not enter running state.")

    (tmp_path / "run" / "cancel.requested").touch()
    assert process.wait(timeout=10) == 130

    state = json.loads(status_path.read_text())
    assert state["status"] == "cancelled"
    assert state["phase"] == "cancelled"
    assert state["child_pid"] is None


def test_job_runner_rejects_overlapping_training(tmp_path: Path) -> None:
    first = subprocess.Popen(
        _command(
            tmp_path,
            prompt="cancel",
            job_id="first-job",
            run_name="jobs/first",
        )
    )
    first_status_path = tmp_path / "jobs" / "first" / "status.json"

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if first_status_path.is_file():
            first_state = json.loads(first_status_path.read_text())
            if first_state.get("status") == "running":
                break
        time.sleep(0.05)
    else:
        first.kill()
        raise AssertionError("First job runner did not enter running state.")

    second = subprocess.run(
        _command(
            tmp_path,
            prompt="bright bells",
            job_id="second-job",
            run_name="jobs/second",
        ),
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert second.returncode == 1
    second_state = json.loads(
        (tmp_path / "jobs" / "second" / "status.json").read_text()
    )
    assert second_state["status"] == "failed"
    assert "already running" in second_state["error"]

    (tmp_path / "jobs" / "first" / "cancel.requested").touch()
    assert first.wait(timeout=10) == 130
    assert not (tmp_path / "jobs" / "active-job.lock").exists()
