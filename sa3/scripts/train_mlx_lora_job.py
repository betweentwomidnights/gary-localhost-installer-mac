#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import selectors
import signal
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from mlx_lora_dataset import (  # noqa: E402
    LoraDatasetExample,
    compose_trigger_prompt,
    discover_dataset_examples,
    prompt_pool,
)
from mlx_training_assets import (  # noqa: E402
    TRAINING_MODEL_NAME,
    TRAINING_MODEL_REPO,
)

# Mirrored from stable_audio_3.mlx.training so this supervisor does not import
# mlx just to validate a CLI choice. train_mlx_lora.py is the authority and
# re-validates the value against the same list.
LORA_LAYER_SCOPE_CHOICES = ("all-projections", "attention-feedforward")
LORA_LAYER_SCOPE_DEFAULT = "attention-feedforward"

DEFAULT_TRAINER = Path(__file__).with_name("train_mlx_lora.py")
STEP_PATTERN = re.compile(r"^step=(\d+)/(\d+)\s+loss=([^\s]+)")
TRAINING_CHECKPOINT_PATTERN = re.compile(
    r"^gary-mlx-lora-step-(\d+)\.safetensors$",
    re.IGNORECASE,
)
DEFAULT_APP_SUPPORT = (
    Path.home() / "Library" / "Application Support" / "GaryLocalhost" / "sa3"
)
FULL_TRACK_CROP_SECONDS = 285.35


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _acquire_job_lock(path: Path, *, job_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "job_id": job_id,
        "pid": os.getpid(),
        "created_at": _timestamp(),
    }
    encoded = (json.dumps(payload, indent=2) + "\n").encode()

    for _ in range(2):
        try:
            descriptor = os.open(
                path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
        except FileExistsError:
            owner = _read_json(path, {})
            owner_pid = owner.get("pid") if isinstance(owner, dict) else None
            if isinstance(owner_pid, int) and _pid_is_running(owner_pid):
                owner_job = owner.get("job_id", "unknown")
                raise RuntimeError(
                    f"Another MLX LoRA training job is already running: {owner_job}."
                )
            path.unlink(missing_ok=True)
            continue

        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
        return

    raise RuntimeError("Could not acquire the MLX LoRA training lock.")


def _release_job_lock(path: Path, *, job_id: str) -> None:
    owner = _read_json(path, {})
    if not isinstance(owner, dict):
        return
    if owner.get("job_id") == job_id and owner.get("pid") == os.getpid():
        path.unlink(missing_ok=True)


def _slugify(raw: str) -> str:
    value = re.sub(r"[^a-z0-9_-]+", "-", raw.strip().lower())
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value[:64] or "sa3-lora"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an MLX Stable Audio 3 LoRA training job with persistent status."
    )
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--name", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio-path", type=Path)
    source.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--trigger-text", default="")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--adapter-type", required=True)
    parser.add_argument(
        "--dit-engine",
        choices=("gary-generic", "official-specialized"),
        default="gary-generic",
    )
    parser.add_argument(
        "--layer-scope",
        choices=LORA_LAYER_SCOPE_CHOICES,
        default=LORA_LAYER_SCOPE_DEFAULT,
    )
    parser.add_argument("--crop-seconds", type=float, required=True)
    parser.add_argument(
        "--full-tracks",
        action="store_true",
        help=(
            "Train a fixed window from 0:00 instead of choosing a random crop "
            "offset on each step."
        ),
    )
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--save-every", type=int, required=True)
    parser.add_argument("--per-track-target-latent-rms", type=float, default=0.0)
    parser.add_argument("--cfg-dropout-prob", type=float, default=0.1)
    parser.add_argument(
        "--timestep-sampler",
        choices=(
            "uniform",
            "logit_normal",
            "trunc_logit_normal",
            "log_snr",
            "log_snr_uniform",
        ),
        default="trunc_logit_normal",
    )
    parser.add_argument(
        "--distribution-shift",
        choices=("model", "none", "full"),
        default="model",
    )
    parser.add_argument("--shift-base", type=float, default=0.5)
    parser.add_argument("--shift-max", type=float, default=1.15)
    parser.add_argument("--shift-min-length", type=int, default=256)
    parser.add_argument("--shift-max-length", type=int, default=4096)
    parser.add_argument("--shift-use-sine", action="store_true")
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--status-path", type=Path, required=True)
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--cancel-path", type=Path, required=True)
    parser.add_argument("--lock-path", type=Path)
    parser.add_argument(
        "--lora-dir",
        type=Path,
        default=Path(os.environ.get("SA3_LORA_DIR", DEFAULT_APP_SUPPORT / "loras")),
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=Path(
            os.environ.get(
                "SA3_LORA_REGISTRY",
                DEFAULT_APP_SUPPORT / "lora_registry.json",
            )
        ),
    )
    parser.add_argument("--catalog-path", type=Path)
    parser.add_argument(
        "--prompts-dir",
        type=Path,
        default=Path(os.environ.get("SA3_PROMPTS_DIR", DEFAULT_APP_SUPPORT / "prompts")),
    )
    parser.add_argument("--trainer-path", type=Path, default=DEFAULT_TRAINER)
    return parser.parse_args()


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    try:
        process.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait()


def _state_for(args: argparse.Namespace) -> dict[str, Any]:
    now = _timestamp()
    return {
        "job_id": args.job_id,
        "name": args.name,
        "status": "starting",
        "phase": "preparing",
        "message": "Preparing MLX training job.",
        "error": None,
        "pid": os.getpid(),
        "child_pid": None,
        "run_dir": str(args.output_dir),
        "log_path": str(args.log_path),
        "cancel_path": str(args.cancel_path),
        "final_checkpoint_path": None,
        "current_step": 0,
        "max_steps": args.steps,
        "adapter_type": args.adapter_type,
        "dit_engine": args.dit_engine,
        "layer_scope": args.layer_scope,
        "training_base_model": TRAINING_MODEL_NAME,
        "training_base_repo": TRAINING_MODEL_REPO,
        "dataset_path": str(args.dataset_dir or args.audio_path),
        "trigger_text": args.trigger_text,
        "full_tracks": args.full_tracks,
        "crop_seconds": args.crop_seconds,
        "per_track_target_latent_rms": args.per_track_target_latent_rms,
        "timestep_sampler": args.timestep_sampler,
        "distribution_shift": args.distribution_shift,
        "shift_base": args.shift_base,
        "shift_max": args.shift_max,
        "shift_min_length": args.shift_min_length,
        "shift_max_length": args.shift_max_length,
        "shift_use_sine": args.shift_use_sine,
        "example_count": len(args.examples),
        "started_at": now,
        "updated_at": now,
        "finished_at": None,
    }


def _update_state(
    state: dict[str, Any],
    status_path: Path,
    **changes: Any,
) -> None:
    state.update(changes)
    state["updated_at"] = _timestamp()
    _write_json_atomic(status_path, state)


def _update_from_output(
    line: str,
    *,
    state: dict[str, Any],
    status_path: Path,
) -> None:
    if line.startswith("resolving_training_assets="):
        _update_state(
            state,
            status_path,
            phase="downloading",
            message=(
                "Downloading or verifying the Stable Audio 3 medium-base "
                "training model."
            ),
        )
        return

    if line.startswith("Loading torch checkpoint"):
        _update_state(
            state,
            status_path,
            phase="loading",
            message="Loading Stable Audio 3 and converting it to MLX.",
        )
        return

    if line.startswith("conversion_seconds="):
        _update_state(
            state,
            status_path,
            phase="encoding",
            message="Stable Audio 3 is ready. Starting dataset encoding.",
        )
        return

    if line.startswith("encoding_example="):
        progress = line.split(" ", 1)[0].split("=", 1)[1]
        _update_state(
            state,
            status_path,
            phase="encoding",
            message=f"Encoding dataset example {progress}.",
        )
        return

    if line.startswith("encoded_audio="):
        progress = line.split(" ", 1)[0].split("=", 1)[1]
        cache_hit = "cached=true" in line
        verb = "Loaded cached audio latents for" if cache_hit else "Encoded audio for"
        _update_state(
            state,
            status_path,
            phase="conditioning",
            message=f"{verb} dataset example {progress}; preparing its prompt.",
        )
        return

    if line.startswith("conditioned_example="):
        progress = line.split(" ", 1)[0].split("=", 1)[1]
        _update_state(
            state,
            status_path,
            phase="conditioning",
            message=f"Prepared prompt conditioning for dataset example {progress}.",
        )
        return

    if line.startswith("encoded_example="):
        progress = line.split(" ", 1)[0].split("=", 1)[1]
        _update_state(
            state,
            status_path,
            phase="encoding",
            message=f"Encoded dataset example {progress}.",
        )
        return

    if line.startswith("loudness_fix="):
        progress = line.split(" ", 1)[0].split("=", 1)[1]
        _update_state(
            state,
            status_path,
            phase="encoding",
            message=f"Applied latent loudness fix to dataset example {progress}.",
        )
        return

    if line.startswith("trainable_parameters="):
        _update_state(
            state,
            status_path,
            phase="training",
            message="Adapter initialized. Starting training.",
        )
        return

    match = STEP_PATTERN.match(line)
    if match:
        current_step = int(match.group(1))
        max_steps = int(match.group(2))
        loss = match.group(3)
        _update_state(
            state,
            status_path,
            phase="training",
            current_step=current_step,
            max_steps=max_steps,
            message=f"Step {current_step} of {max_steps}, loss {loss}.",
        )
        return

    if line.startswith("checkpoint="):
        checkpoint_path = line.split("=", 1)[1]
        _update_state(
            state,
            status_path,
            phase="training",
            message=f"Saved checkpoint {Path(checkpoint_path).name}.",
        )
        return

    if line.startswith("final_checkpoint="):
        checkpoint_path = line.split("=", 1)[1]
        _update_state(
            state,
            status_path,
            phase="saving",
            final_checkpoint_path=checkpoint_path,
            message="Final checkpoint saved.",
        )


def _install_lora(
    args: argparse.Namespace,
    *,
    source_checkpoint: Path,
    training_checkpoints: list[dict[str, Any]],
) -> Path:
    args.lora_dir.mkdir(parents=True, exist_ok=True)
    installed_path = args.lora_dir / f"{args.name}.safetensors"
    shutil.copy2(source_checkpoint, installed_path)

    registry = _read_json(args.registry_path, {})
    if not isinstance(registry, dict):
        registry = {}
    registry[args.name] = {
        "path": str(installed_path),
        "strength": 1.0,
    }
    _write_json_atomic(args.registry_path, registry)

    catalog = _read_json(args.catalog_path, {})
    if not isinstance(catalog, dict):
        catalog = {}
    catalog[args.name] = {
        "path": str(installed_path),
        "promptsPath": str(args.dataset_dir) if args.dataset_dir is not None else None,
        "strength": 1.0,
        "trainingBaseModel": TRAINING_MODEL_NAME,
        "inferenceModel": "medium",
        "trainingJobId": args.job_id,
        "trainingCheckpoints": training_checkpoints,
        "selectedTrainingStep": args.steps,
    }
    _write_json_atomic(args.catalog_path, catalog)

    args.prompts_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = args.prompts_dir / f"{args.name}.json"
    prompt_payload = {
        "version": 1,
        "source": {
            "lora": args.name,
            "dataset_path": str(args.dataset_dir or args.audio_path),
            "training_job": args.job_id,
            "training_base_model": TRAINING_MODEL_NAME,
            "inference_model": "medium",
            "trigger_text": args.trigger_text,
        },
        "dice": {
            "instrumental": prompt_pool(
                args.examples,
                trigger_text=args.trigger_text,
            )
        },
    }
    _write_json_atomic(prompt_path, prompt_payload)
    return installed_path


def _training_checkpoints(
    output_dir: Path,
    *,
    final_checkpoint: Path,
    final_step: int,
) -> list[dict[str, Any]]:
    by_step: dict[int, Path] = {}
    for checkpoint in output_dir.glob("gary-mlx-lora-step-*.safetensors"):
        match = TRAINING_CHECKPOINT_PATTERN.match(checkpoint.name)
        if match is None or not checkpoint.is_file():
            continue
        by_step[int(match.group(1))] = checkpoint.resolve()

    # The final export is the authoritative representation of the last step.
    # It may duplicate a periodic step checkpoint when max_steps is divisible by
    # save_every, so replace that entry instead of presenting the same step twice.
    by_step[int(final_step)] = final_checkpoint.resolve()
    return [
        {
            "step": step,
            "epoch": 0,
            "path": str(checkpoint),
        }
        for step, checkpoint in sorted(by_step.items())
    ]


def _resolve_examples(args: argparse.Namespace) -> list[LoraDatasetExample]:
    if args.dataset_dir is not None:
        return discover_dataset_examples(
            args.dataset_dir,
            trigger_text=args.trigger_text,
        )

    source_prompt = args.prompt.strip()
    return [
        LoraDatasetExample(
            audio_path=args.audio_path,
            relative_path=args.audio_path.name,
            sidecar_path=None,
            sidecar_kind=None,
            source_prompt=source_prompt,
            prompt=compose_trigger_prompt(args.trigger_text, source_prompt),
        )
    ]


def main() -> int:
    args = _parse_args()
    if args.full_tracks:
        args.crop_seconds = FULL_TRACK_CROP_SECONDS
    args.name = _slugify(args.name)
    args.audio_path = (
        args.audio_path.expanduser().resolve()
        if args.audio_path is not None
        else None
    )
    args.dataset_dir = (
        args.dataset_dir.expanduser().resolve()
        if args.dataset_dir is not None
        else None
    )
    args.output_dir = args.output_dir.expanduser().resolve()
    args.status_path = args.status_path.expanduser().resolve()
    args.log_path = args.log_path.expanduser().resolve()
    args.cancel_path = args.cancel_path.expanduser().resolve()
    args.lock_path = (
        args.lock_path.expanduser().resolve()
        if args.lock_path is not None
        else args.status_path.parent.parent / "active-job.lock"
    )
    args.lora_dir = args.lora_dir.expanduser().resolve()
    args.registry_path = args.registry_path.expanduser().resolve()
    args.catalog_path = (
        args.catalog_path.expanduser().resolve()
        if args.catalog_path is not None
        else args.registry_path.with_name("lora_catalog.json")
    )
    args.prompts_dir = args.prompts_dir.expanduser().resolve()
    args.trainer_path = args.trainer_path.expanduser().resolve()

    if args.audio_path is not None and not args.audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {args.audio_path}")
    if args.dataset_dir is not None and not args.dataset_dir.is_dir():
        raise NotADirectoryError(f"Dataset folder not found: {args.dataset_dir}")
    if not args.trainer_path.is_file():
        raise FileNotFoundError(f"Trainer not found: {args.trainer_path}")
    args.examples = _resolve_examples(args)
    if not args.examples:
        raise FileNotFoundError(
            f"No supported audio files found in {args.dataset_dir or args.audio_path}."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    args.cancel_path.unlink(missing_ok=True)

    state = _state_for(args)
    _write_json_atomic(args.status_path, state)

    command = [
        sys.executable,
        str(args.trainer_path),
        "--output-dir",
        str(args.output_dir),
        "--trigger-text",
        args.trigger_text,
        "--model-name",
        TRAINING_MODEL_NAME,
        "--steps",
        str(args.steps),
        "--rank",
        str(args.rank),
        "--adapter-type",
        args.adapter_type,
        "--dit-engine",
        args.dit_engine,
        "--layer-scope",
        args.layer_scope,
        "--crop-seconds",
        str(args.crop_seconds),
        "--learning-rate",
        str(args.learning_rate),
        "--save-every",
        str(args.save_every),
        "--cfg-dropout-prob",
        str(args.cfg_dropout_prob),
        "--timestep-sampler",
        args.timestep_sampler,
        "--distribution-shift",
        args.distribution_shift,
        "--seed",
        str(args.seed),
    ]
    if args.full_tracks:
        command.append("--full-tracks")
    if args.distribution_shift == "full":
        command.extend(
            [
                "--shift-base",
                str(args.shift_base),
                "--shift-max",
                str(args.shift_max),
                "--shift-min-length",
                str(args.shift_min_length),
                "--shift-max-length",
                str(args.shift_max_length),
            ]
        )
        if args.shift_use_sine:
            command.append("--shift-use-sine")
    if args.per_track_target_latent_rms > 0:
        command.extend(
            [
                "--per-track-target-latent-rms",
                str(args.per_track_target_latent_rms),
            ]
        )
    if args.dataset_dir is not None:
        command.extend(["--dataset-dir", str(args.dataset_dir)])
    else:
        command.extend(
            [
                "--audio-path",
                str(args.audio_path),
                "--prompt",
                args.prompt,
            ]
        )

    child: subprocess.Popen[str] | None = None
    cancellation_requested = False
    lock_acquired = False

    def request_cancellation(_signum: int, _frame: object) -> None:
        nonlocal cancellation_requested
        cancellation_requested = True

    signal.signal(signal.SIGTERM, request_cancellation)
    signal.signal(signal.SIGINT, request_cancellation)

    try:
        _acquire_job_lock(args.lock_path, job_id=args.job_id)
        lock_acquired = True
        with args.log_path.open("a", buffering=1) as log:
            log.write(f"\n---- {_timestamp()} MLX LoRA job {args.job_id} ----\n")
            log.write("$ " + " ".join(command) + "\n")

            child = subprocess.Popen(
                command,
                cwd=SERVICE_ROOT,
                env=os.environ.copy(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            _update_state(
                state,
                args.status_path,
                status="running",
                child_pid=child.pid,
                message="Training process started.",
            )

            assert child.stdout is not None
            selector = selectors.DefaultSelector()
            selector.register(child.stdout, selectors.EVENT_READ)

            while child.poll() is None:
                if cancellation_requested or args.cancel_path.exists():
                    cancellation_requested = True
                    _update_state(
                        state,
                        args.status_path,
                        status="cancelling",
                        message="Cancelling training.",
                    )
                    _terminate_process_group(child)
                    break

                for key, _ in selector.select(timeout=0.25):
                    line = key.fileobj.readline()
                    if not line:
                        continue
                    log.write(line)
                    _update_from_output(
                        line.rstrip("\n"),
                        state=state,
                        status_path=args.status_path,
                    )

            remaining = child.stdout.read()
            if remaining:
                log.write(remaining)
                for line in remaining.splitlines():
                    _update_from_output(
                        line,
                        state=state,
                        status_path=args.status_path,
                    )
            selector.close()

            return_code = child.wait()
            if cancellation_requested:
                _update_state(
                    state,
                    args.status_path,
                    status="cancelled",
                    phase="cancelled",
                    message="Training cancelled.",
                    child_pid=None,
                    finished_at=_timestamp(),
                )
                return 130

            if return_code != 0:
                raise RuntimeError(f"Trainer exited with status {return_code}.")

            source_checkpoint_value = state.get("final_checkpoint_path")
            if not isinstance(source_checkpoint_value, str):
                raise RuntimeError("Trainer completed without reporting a final checkpoint.")
            source_checkpoint = Path(source_checkpoint_value)
            if not source_checkpoint.is_file():
                raise FileNotFoundError(
                    f"Final training checkpoint not found: {source_checkpoint}"
                )

            _update_state(
                state,
                args.status_path,
                phase="installing",
                message="Installing LoRA and updating the SA3 registry.",
            )
            training_checkpoints = _training_checkpoints(
                args.output_dir,
                final_checkpoint=source_checkpoint,
                final_step=args.steps,
            )
            installed_checkpoint = _install_lora(
                args,
                source_checkpoint=source_checkpoint,
                training_checkpoints=training_checkpoints,
            )
            _update_state(
                state,
                args.status_path,
                status="completed",
                phase="completed",
                current_step=args.steps,
                final_checkpoint_path=str(installed_checkpoint),
                message="Training completed.",
                child_pid=None,
                finished_at=_timestamp(),
            )
            return 0
    except Exception as error:
        if child is not None:
            _terminate_process_group(child)
        _update_state(
            state,
            args.status_path,
            status="failed",
            phase="failed",
            error=str(error),
            message="Training failed.",
            child_pid=None,
            finished_at=_timestamp(),
        )
        with args.log_path.open("a") as log:
            log.write(f"[job-error] {error}\n")
        return 1
    finally:
        if lock_acquired:
            _release_job_lock(args.lock_path, job_id=args.job_id)


if __name__ == "__main__":
    raise SystemExit(main())
