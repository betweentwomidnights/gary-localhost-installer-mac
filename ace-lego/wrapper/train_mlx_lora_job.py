#!/usr/bin/env python3
"""Gary-native ACE-Step MLX LoRA/DoRA training job for macOS.

This process owns dataset preparation, optional ``understand_music``
captioning, two-pass ACE preprocessing, native MLX adapter training, status
updates, cancellation, and Carey LoRA registration.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import mimetypes
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ace_training_dataset import (
    audio_duration_seconds,
    build_dataset_json,
    discover_audio_files,
    load_sidecar_metadata,
    load_split_sidecar_metadata,
    write_canonical_sidecar,
)
from bpm_analysis import choose_bpm, estimate_bpm
from key_analysis import choose_key, estimate_key

SERVICE_DIR = Path(__file__).resolve().parent
ACE_ROOT = SERVICE_DIR.parent / "ACE-Step-1.5"
TRAIN_ENTRY = ACE_ROOT / "train.py"
MLX_TRAIN_ENTRY = ACE_ROOT / "scripts" / "train_mlx_lora.py"

MODEL_MAP = {
    "base": {
        "variant": "base",
        "folder": "acestep-v15-base",
        "family": "standard",
    },
    "xl-base": {
        "variant": "acestep-v15-xl-base",
        "folder": "acestep-v15-xl-base",
        "family": "xl",
    },
}
CAPTION_LM_MODELS = (
    "acestep-5Hz-lm-0.6B",
    "acestep-5Hz-lm-1.7B",
    "acestep-5Hz-lm-4B",
)
CAPTION_LM_BACKEND_DEFAULT = "mlx"
XL_DEFAULT_MEMORY_LIMIT_GB = 20.0
STATUS_ALIASES = {
    "job_id": "jobId",
    "run_dir": "runDir",
    "log_path": "logPath",
    "cancel_path": "cancelPath",
    "updated_at": "updatedAt",
    "child_pid": "childPid",
    "dataset_json_path": "datasetJsonPath",
    "training_plan_path": "trainingPlanPath",
    "sample_count": "sampleCount",
    "captioned_count": "captionedCount",
    "caption_lm_model": "captionLmModel",
    "caption_lm_backend": "captionLmBackend",
    "current_file": "currentFile",
    "total_files": "totalFiles",
    "model_family": "modelFamily",
    "adapter_type": "adapterType",
    "module_profile": "moduleProfile",
    "final_checkpoint_path": "finalCheckpointPath",
    "best_checkpoint_path": "bestCheckpointPath",
    "last_epoch_checkpoint_path": "lastEpochCheckpointPath",
    "captions_path": "captionsPath",
    "result_path": "resultPath",
    "registered_lora_name": "registeredLoraName",
    "current_step": "currentStep",
    "max_steps": "maxSteps",
    "current_epoch": "currentEpoch",
    "max_epochs": "maxEpochs",
    "current_loss": "currentLoss",
    "trainable_parameters": "trainableParameters",
}


@dataclass(frozen=True)
class PreparedCaptionAudio:
    path: Path
    cleanup_path: Path | None
    duration: float
    offset: float


class Cancelled(RuntimeError):
    pass


def slugify(raw: str) -> str:
    value = re.sub(r"[^a-z0-9_-]+", "-", raw.strip().lower())
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value[:64] or "ace-lora"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temp_path, path)


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def update_status(args: argparse.Namespace, **updates: Any) -> None:
    payload = read_json(args.status_path, {})
    payload.update(
        {
            "job_id": args.job_id,
            "name": args.name,
            "pid": os.getpid(),
            "run_dir": str(args.run_dir),
            "log_path": str(args.log_path),
            "cancel_path": str(args.cancel_path),
            "updated_at": time.time(),
        }
    )
    payload.update(updates)
    for snake, camel in STATUS_ALIASES.items():
        if snake in payload:
            payload[camel] = payload[snake]
        elif camel in payload:
            payload[snake] = payload[camel]
    write_json(args.status_path, payload)
    write_json(
        args.current_job_path,
        {
            "job_id": args.job_id,
            "jobId": args.job_id,
            "status_path": str(args.status_path),
            "statusPath": str(args.status_path),
            "log_path": str(args.log_path),
            "logPath": str(args.log_path),
            "cancel_path": str(args.cancel_path),
            "cancelPath": str(args.cancel_path),
            "run_dir": str(args.run_dir),
            "runDir": str(args.run_dir),
        },
    )


def cancel_requested(args: argparse.Namespace) -> bool:
    return bool(args.cancel_path and args.cancel_path.exists())


def check_cancel(args: argparse.Namespace) -> None:
    if cancel_requested(args):
        raise Cancelled("Training cancelled.")


def terminate_process_tree(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            return
        except Exception:
            proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            proc.kill()
        else:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                proc.kill()
        proc.wait(timeout=5)


def _training_status_from_line(line: str) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    step_match = re.search(
        r"step=(\d+)/(\d+)\s+loss=([0-9eE+\-.]+)\s+epoch=(\d+)/(\d+)",
        line,
    )
    if step_match:
        updates.update(
            {
                "current_step": int(step_match.group(1)),
                "max_steps": int(step_match.group(2)),
                "current_loss": float(step_match.group(3)),
                "current_epoch": int(step_match.group(4)),
                "max_epochs": int(step_match.group(5)),
                "message": (
                    f"Training step {step_match.group(1)}/{step_match.group(2)} "
                    f"(loss {float(step_match.group(3)):.4f})"
                ),
            }
        )
    if line.startswith("checkpoint="):
        updates["last_epoch_checkpoint_path"] = line.split("=", 1)[1].strip()
    elif line.startswith("best_checkpoint="):
        path = line.split("=", 1)[1].split(" loss=", 1)[0].strip()
        updates["best_checkpoint_path"] = path
    elif line.startswith("final_checkpoint="):
        updates["final_checkpoint_path"] = line.split("=", 1)[1].strip()
    elif line.startswith("trainable_parameters="):
        try:
            updates["trainable_parameters"] = int(line.split("=", 1)[1].strip())
        except ValueError:
            pass
    return updates


def run_step(
    args: argparse.Namespace,
    command: list[str],
    phase: str,
    message: str,
    *,
    cwd: Path | None = None,
) -> None:
    check_cancel(args)
    update_status(args, status="running", phase=phase, message=message)
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[{phase}] {subprocess.list2cmdline(command)}", flush=True)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    pythonpath_parts = [str(SERVICE_DIR), str(ACE_ROOT)]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    popen_kwargs: dict[str, Any] = {
        "cwd": str(cwd or SERVICE_DIR),
        "env": env,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "bufsize": 1,
    }
    if os.name != "nt":
        popen_kwargs["start_new_session"] = True

    proc = subprocess.Popen(command, **popen_kwargs)
    update_status(args, status="running", phase=phase, message=message, child_pid=proc.pid)
    with args.log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{phase}] {subprocess.list2cmdline(command)}\n")
        assert proc.stdout is not None
        try:
            while True:
                line = proc.stdout.readline()
                if line:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                    if phase == "training":
                        updates = _training_status_from_line(line.strip())
                        if updates:
                            update_status(args, status="running", phase=phase, **updates)
                code = proc.poll()
                if code is not None:
                    remainder = proc.stdout.read()
                    if remainder:
                        log.write(remainder)
                        log.flush()
                        print(remainder, end="", flush=True)
                    if code != 0:
                        raise RuntimeError(f"{phase} failed with exit code {code}")
                    return
                if cancel_requested(args):
                    terminate_process_tree(proc)
                    raise Cancelled("Training cancelled.")
                if not line:
                    time.sleep(0.2)
        finally:
            if proc.poll() is None:
                terminate_process_tree(proc)
            update_status(args, child_pid=None)


def require_training_environment(args: argparse.Namespace) -> None:
    check_cancel(args)
    update_status(
        args,
        status="running",
        phase="checking-environment",
        message="Checking ACE MLX training environment",
    )
    missing: list[str] = []
    for module in ("torch", "mlx.core", "safetensors"):
        try:
            importlib.import_module(module)
        except Exception:
            missing.append(module)
    if missing:
        raise RuntimeError(
            "Carey's Python environment is missing required ACE MLX training modules: "
            + ", ".join(missing)
            + ". Rebuild the Carey environment and try again."
        )


def require_model_checkpoint(args: argparse.Namespace) -> None:
    model = MODEL_MAP[args.model]
    model_dir = args.checkpoint_dir / model["folder"]
    if not model_dir.is_dir() or not (model_dir / "config.json").is_file():
        raise RuntimeError(f"ACE-Step model checkpoint is incomplete: {model_dir}")


def build_preprocess_command(
    args: argparse.Namespace,
    dataset_json: Path,
    tensors_dir: Path,
    output_dir: Path,
) -> list[str]:
    model = MODEL_MAP[args.model]
    return [
        sys.executable,
        "-u",
        str(TRAIN_ENTRY),
        "--plain",
        "-y",
        "fixed",
        "--checkpoint-dir",
        str(args.checkpoint_dir),
        "--model-variant",
        model["variant"],
        "--base-model",
        "base",
        "--dataset-dir",
        str(tensors_dir),
        "--output-dir",
        str(output_dir),
        "--preprocess",
        "--dataset-json",
        str(dataset_json),
        "--tensor-output",
        str(tensors_dir),
        "--max-duration",
        str(args.max_duration),
        "--device",
        args.preprocess_device,
        "--precision",
        args.preprocess_precision,
        "--num-workers",
        "0",
        "--prefetch-factor",
        "0",
        "--no-persistent-workers",
    ]


def build_train_command(
    args: argparse.Namespace,
    tensors_dir: Path,
    output_dir: Path,
) -> list[str]:
    model = MODEL_MAP[args.model]
    alpha = resolve_alpha(args)
    memory_limit_gb = resolve_memory_limit_gb(args)
    command = [
        sys.executable,
        "-u",
        str(MLX_TRAIN_ENTRY),
        "--checkpoint-dir",
        str(args.checkpoint_dir),
        "--model-variant",
        model["variant"],
        "--tensor-dir",
        str(tensors_dir),
        "--output-dir",
        str(output_dir),
        "--adapter-type",
        args.adapter_type,
        "--rank",
        str(args.rank),
        "--alpha",
        str(alpha),
        "--module-profile",
        args.module_profile,
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--batch-size",
        str(args.batch_size),
        "--gradient-accumulation",
        str(args.gradient_accumulation),
        "--epochs",
        str(args.epochs),
        "--save-every",
        str(args.save_every),
        "--save-best-after",
        str(args.save_best_after),
        "--cfg-ratio",
        str(args.cfg_ratio),
        "--timestep-mu",
        str(resolve_timestep_mu(args)),
        "--loss-weighting",
        args.loss_weighting,
        "--snr-gamma",
        str(args.snr_gamma),
        "--dtype",
        args.dtype,
        "--seed",
        str(args.seed),
        "--cancel-path",
        str(args.cancel_path),
    ]
    command.append(
        "--gradient-checkpointing"
        if resolve_gradient_checkpointing(args)
        else "--no-gradient-checkpointing"
    )
    command.append("--save-best" if args.save_best else "--no-save-best")
    if args.max_steps > 0:
        command.extend(["--max-steps", str(args.max_steps)])
    if memory_limit_gb > 0:
        command.extend(["--memory-limit-gb", str(memory_limit_gb)])
    if args.allow_unsafe_xl:
        command.append("--allow-unsafe-xl")
    if args.fake_decoder:
        command.append("--fake-decoder")
    return command


def resolve_alpha(args: argparse.Namespace) -> int:
    explicit = getattr(args, "alpha", None)
    return int(explicit) if explicit is not None else int(args.rank) * 2


def resolve_timestep_mu(args: argparse.Namespace) -> float:
    """Resolve the explicit override or the ACE-Step/Side-Step default."""
    explicit = getattr(args, "timestep_mu", None)
    if explicit is not None:
        return float(explicit)
    return -0.4


def is_xl_model(args: argparse.Namespace) -> bool:
    return MODEL_MAP[args.model]["family"] == "xl"


def resolve_gradient_checkpointing(args: argparse.Namespace) -> bool:
    explicit = getattr(args, "gradient_checkpointing", None)
    if explicit is not None:
        return bool(explicit)
    return True


def resolve_memory_limit_gb(args: argparse.Namespace) -> float:
    requested = float(getattr(args, "memory_limit_gb", 0.0) or 0.0)
    if requested > 0:
        return requested
    return XL_DEFAULT_MEMORY_LIMIT_GB if is_xl_model(args) else 0.0


def is_complete_peft_adapter(path: Path) -> bool:
    return (
        (path / "adapter_model.safetensors").is_file()
        and (path / "adapter_config.json").is_file()
    )


def append_log(args: argparse.Namespace, text: str) -> None:
    try:
        args.log_path.parent.mkdir(parents=True, exist_ok=True)
        with args.log_path.open("a", encoding="utf-8") as log:
            log.write(text if text.endswith("\n") else text + "\n")
    except Exception:
        pass


def log_caption(args: argparse.Namespace, message: str) -> None:
    line = f"[captioning] {message}"
    print(line, flush=True)
    append_log(args, line)


def sanitize_caption_server_log_line(line: str) -> str:
    if "<|audio_code_" not in line:
        return line
    prefix = line.split("<|audio_code_", 1)[0].strip()
    if prefix:
        return f"{prefix} [suppressed raw ACE audio-code token dump]\n"
    return "[caption-service] suppressed raw ACE audio-code token dump\n"


def stream_caption_server_logs(args: argparse.Namespace, stream: Any) -> None:
    previous_suppressed = False
    try:
        args.log_path.parent.mkdir(parents=True, exist_ok=True)
        with args.log_path.open("a", encoding="utf-8") as log:
            for raw_line in stream:
                line = sanitize_caption_server_log_line(str(raw_line))
                suppressed = "suppressed raw ACE audio-code token dump" in line
                if suppressed and previous_suppressed:
                    continue
                log.write(line if line.endswith("\n") else line + "\n")
                log.flush()
                previous_suppressed = suppressed
    except Exception as exc:
        append_log(args, f"[caption-service] log stream stopped: {exc}")


def parse_carey_endpoint(url: str) -> tuple[str, int]:
    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return host, port


def build_caption_server_command(args: argparse.Namespace) -> list[str]:
    host, port = parse_carey_endpoint(args.carey_url)
    return [
        sys.executable,
        str(ACE_ROOT / "acestep" / "api_server.py"),
        "--host",
        host,
        "--port",
        str(port),
        "--no-init",
        "--init-llm",
        "--lm-model-path",
        args.caption_lm_model,
    ]


def build_caption_server_env(
    args: argparse.Namespace,
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    env = dict(base_env or os.environ)
    existing_pythonpath = env.get("PYTHONPATH", "")
    pythonpath_parts = [str(SERVICE_DIR), str(ACE_ROOT)]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)

    force_dit_offload = args.caption_lm_model != "acestep-5Hz-lm-0.6B"
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(pythonpath_parts),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1",
            "ACESTEP_CONFIG_PATH": MODEL_MAP[args.model]["folder"],
            "ACESTEP_INIT_LLM": "true",
            "ACESTEP_LM_MODEL_PATH": args.caption_lm_model,
            "ACESTEP_LM_BACKEND": args.caption_lm_backend,
            "ACESTEP_REQUIRE_MLX_LM": (
                "true" if args.caption_lm_backend == "mlx" else "false"
            ),
            "ACESTEP_LM_OFFLOAD_TO_CPU": "true",
            "ACESTEP_NO_INIT": "true",
            "ACESTEP_OFFLOAD_TO_CPU": "true",
            "ACESTEP_OFFLOAD_DIT_TO_CPU": "true" if force_dit_offload else "false",
            "ACESTEP_UNDERSTAND_MAX_NEW_TOKENS": "1024",
            "ACESTEP_UNDERSTAND_TEMPERATURE": "0.3",
            "ACESTEP_USE_FLASH_ATTENTION": "false",
            "ACESTEP_COMPILE_MODEL": "false",
            "ACESTEP_API_WORKERS": "1",
            "ACESTEP_USE_MLX_DIT": "1",
            "ACESTEP_USE_MLX_VAE": "1",
            "ACESTEP_MLX_VAE_FP16": "1",
        }
    )
    for key in ("ACESTEP_CONFIG_PATH2", "ACESTEP_CONFIG_PATH3"):
        env.pop(key, None)
    return env


def start_caption_server(args: argparse.Namespace) -> subprocess.Popen[Any]:
    update_status(
        args,
        status="running",
        phase="starting-caption-service",
        message=(
            f"Starting temporary ACE captioner with {args.caption_lm_model} "
            f"({args.caption_lm_backend.upper()})"
        ),
    )
    command = build_caption_server_command(args)
    append_log(args, f"\n[starting-caption-service] {subprocess.list2cmdline(command)}")
    popen_kwargs: dict[str, Any] = {
        "cwd": str(ACE_ROOT),
        "env": build_caption_server_env(args),
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "bufsize": 1,
    }
    if os.name != "nt":
        popen_kwargs["start_new_session"] = True
    process = subprocess.Popen(command, **popen_kwargs)
    if process.stdout is not None:
        log_thread = threading.Thread(
            target=stream_caption_server_logs,
            args=(args, process.stdout),
            daemon=True,
        )
        log_thread.start()
        setattr(process, "_gary_log_thread", log_thread)
    update_status(args, child_pid=process.pid)
    return process


def stop_caption_server(args: argparse.Namespace, process: subprocess.Popen[Any]) -> None:
    try:
        if process.poll() is None:
            update_status(
                args,
                status="running",
                phase="stopping-caption-service",
                message="Stopping temporary ACE captioner and releasing memory",
            )
            terminate_process_tree(process)
        else:
            process.wait(timeout=1)
    finally:
        log_thread = getattr(process, "_gary_log_thread", None)
        if log_thread is not None:
            try:
                log_thread.join(timeout=2)
            except Exception:
                pass
        update_status(args, child_pid=None)


def require_caption_lm_backend(args: argparse.Namespace) -> None:
    if args.caption_lm_backend != "mlx":
        return
    missing: list[str] = []
    for module_name in ("mlx.core", "mlx_lm"):
        try:
            importlib.import_module(module_name)
        except Exception:
            missing.append(module_name)
    if missing:
        raise RuntimeError(
            "ACE captioning is configured for the MLX LM backend, but this "
            "Python environment is missing "
            f"{', '.join(missing)}. Rebuild the Carey environment so mlx-lm "
            "is installed, or run the wrapper with --caption-lm-backend pt "
            "for a temporary fallback."
        )


def prepare_caption_audio(
    args: argparse.Namespace,
    audio_path: Path,
    *,
    caption_window_seconds: float | None = None,
) -> PreparedCaptionAudio:
    window_seconds = float(
        (
            caption_window_seconds
            if caption_window_seconds is not None
            else getattr(args, "caption_window_seconds", 0.0)
        )
        or 0.0
    )
    try:
        actual_duration = audio_duration_seconds(audio_path)
    except Exception as exc:
        print(
            f"[captioning] Could not read duration for {audio_path.name}; "
            f"using full file for captioning: {exc}",
            flush=True,
        )
        actual_duration = 0.0
    if window_seconds <= 0 or actual_duration <= window_seconds + 1.0:
        return PreparedCaptionAudio(
            path=audio_path,
            cleanup_path=None,
            duration=max(1.0, float(actual_duration or 0.0)),
            offset=0.0,
        )

    try:
        import soundfile as sf

        start_seconds = max(0.0, (actual_duration - window_seconds) / 2.0)
        with sf.SoundFile(str(audio_path)) as source:
            sample_rate = int(source.samplerate)
            start_frame = int(round(start_seconds * sample_rate))
            frames = int(round(window_seconds * sample_rate))
            source.seek(start_frame)
            audio = source.read(frames, always_2d=True, dtype="float32")
        if getattr(audio, "size", 0) == 0:
            raise RuntimeError("excerpt read returned no samples")

        temp = tempfile.NamedTemporaryFile(
            prefix="gary_ace_caption_",
            suffix=".wav",
            delete=False,
        )
        temp_path = Path(temp.name)
        temp.close()
        sf.write(str(temp_path), audio, sample_rate)
        excerpt_duration = float(len(audio) / sample_rate)
        return PreparedCaptionAudio(
            path=temp_path,
            cleanup_path=temp_path,
            duration=max(1.0, excerpt_duration),
            offset=start_seconds,
        )
    except Exception as exc:
        print(
            f"[captioning] Could not create {window_seconds:.0f}s caption excerpt "
            f"for {audio_path.name}; using full file instead: {exc}",
            flush=True,
        )
        return PreparedCaptionAudio(
            path=audio_path,
            cleanup_path=None,
            duration=max(1.0, float(actual_duration or 0.0)),
            offset=0.0,
        )


def wait_for_carey(
    args: argparse.Namespace,
    client: Any,
    process: subprocess.Popen[Any] | None = None,
) -> None:
    deadline = time.monotonic() + args.caption_startup_timeout
    last_error = "not reachable"
    while time.monotonic() < deadline:
        check_cancel(args)
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                "Temporary ACE captioner exited before becoming ready "
                f"(code {process.returncode}). Check the job log for details."
            )
        try:
            response = client.get(f"{args.carey_url}/health", timeout=5)
            if response.is_success:
                return
            last_error = f"HTTP {response.status_code}"
        except Exception as exc:
            last_error = str(exc)
        time.sleep(2)
    raise RuntimeError(f"Carey analysis backend did not become ready: {last_error}")


def ensure_carey_model_loaded(args: argparse.Namespace, client: Any) -> None:
    health = client.get(f"{args.carey_url}/health", timeout=10)
    health.raise_for_status()
    data = health.json().get("data") or {}
    model = MODEL_MAP[args.model]["folder"]
    if data.get("initialized") and data.get("current_model") == model:
        return

    response = client.post(
        f"{args.carey_url}/v1/load",
        params={"config_path": model},
        timeout=args.model_load_timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("code", 200) not in (0, 200):
        raise RuntimeError(f"Carey model load failed: {payload}")


def caption_text_quality_error(value: Any, *, field: str) -> str | None:
    text = str(value or "")
    if not text.strip():
        return f"{field} is empty"
    if "<|audio_code_" in text:
        return f"{field} contains raw ACE audio-code tokens"
    if "\ufffd" in text or "ï¿½" in text:
        return f"{field} contains invalid replacement characters"
    compact = re.sub(r"\s+", "", text)
    if len(compact) >= 24 and len(set(compact)) <= 3:
        return f"{field} is dominated by repeated characters"
    if re.search(r"([!?._=-])\1{20,}", compact):
        return f"{field} is dominated by repeated punctuation"
    if field == "caption":
        alpha_count = sum(1 for char in text if char.isalpha())
        if len(text.strip()) >= 80 and alpha_count / max(1, len(text)) < 0.15:
            return f"{field} has too little word content"
        words = re.findall(r"[A-Za-z][A-Za-z'-]{2,}", text)
        if len(text.strip()) >= 40 and len(words) < 3:
            return f"{field} has too little descriptive text"
    if field == "caption" and re.search(
        r"(?:^|[^a-z])(?:bpm|duration|genres?|keyscale|language|timesignature)\s*:",
        text,
        re.IGNORECASE,
    ):
        return f"{field} appears to contain embedded metadata fields"
    return None


def validate_caption_analysis_result(
    audio_path: Path,
    result: dict[str, Any],
    *,
    require_caption: bool = True,
) -> None:
    checks = (
        ("caption", result.get("prompt") or result.get("caption")),
        ("genre", result.get("genre") or result.get("genres")),
        ("lyrics", result.get("lyrics")),
    )
    for field, value in checks:
        if field == "caption" and not require_caption and not str(value or "").strip():
            continue
        if field != "caption" and not str(value or "").strip():
            continue
        reason = caption_text_quality_error(value, field=field)
        if reason:
            raise RuntimeError(
                f"ACE understand_music returned unusable metadata for "
                f"{audio_path.name}: {reason}. Re-run captioning, overwrite "
                "captions, or edit the sidecar manually."
            )


def validate_dataset_sidecars(dataset_dir: Path) -> None:
    bad: list[str] = []
    for audio_path in discover_audio_files(dataset_dir):
        meta = load_sidecar_metadata(audio_path)
        if not meta:
            continue
        try:
            validate_caption_analysis_result(audio_path, meta, require_caption=False)
        except RuntimeError as exc:
            bad.append(str(exc))
    if bad:
        preview = "\n".join(f"- {item}" for item in bad[:5])
        extra = "\n..." if len(bad) > 5 else ""
        raise RuntimeError(
            "One or more ACE sidecars look corrupted and were not used to "
            f"build dataset.json:\n{preview}{extra}"
        )


def caption_with_understand_music(args: argparse.Namespace) -> int:
    audio_files = discover_audio_files(args.dataset_dir)
    pending = [
        audio
        for audio in audio_files
        if args.overwrite_captions or not audio.with_suffix(".txt").is_file()
    ]
    if not pending:
        message = (
            f"All {len(audio_files)} track"
            f"{'' if len(audio_files) == 1 else 's'} already "
            "have sidecars; skipping understand_music. Enable overwrite "
            "captions to recaption existing .txt files."
        )
        setattr(args, "_caption_skip_message", message)
        log_caption(args, message)
        update_status(
            args,
            status="running",
            phase="captioning-skipped",
            message=message,
            current_file=0,
            total_files=len(audio_files),
            captioned_count=0,
            caption_lm_model=args.caption_lm_model,
            caption_lm_backend=args.caption_lm_backend,
        )
        return 0

    import httpx

    require_caption_lm_backend(args)
    update_status(
        args,
        status="running",
        phase="captioning",
        message=(
            f"Captioning with {args.caption_lm_model} "
            f"({args.caption_lm_backend.upper()})"
        ),
        current_file=0,
        total_files=len(pending),
        caption_lm_model=args.caption_lm_model,
        caption_lm_backend=args.caption_lm_backend,
    )
    caption_server = start_caption_server(args)
    try:
        with httpx.Client(timeout=httpx.Timeout(args.caption_timeout)) as client:
            wait_for_carey(args, client, caption_server)
            update_status(
                args,
                status="running",
                phase="loading-caption-model",
                message=f"Loading {MODEL_MAP[args.model]['folder']} for audio analysis",
            )
            ensure_carey_model_loaded(args, client)
            for index, audio_path in enumerate(pending, 1):
                check_cancel(args)
                update_status(
                    args,
                    status="running",
                    phase="captioning",
                    message=f"Analyzing {audio_path.name}",
                    current_file=index,
                    total_files=len(pending),
                    caption_lm_backend=args.caption_lm_backend,
                )
                result = request_valid_music_analysis(args, client, audio_path)
                existing_meta = (
                    load_split_sidecar_metadata(audio_path)
                    if args.overwrite_captions
                    else load_sidecar_metadata(audio_path)
                )
                bpm_decision = decide_sidecar_bpm(args, audio_path, result)
                key_decision = decide_sidecar_key(args, audio_path, result)
                lyrics = str(existing_meta.get("lyrics") or "").strip()
                is_instrumental = analysis_is_instrumental(
                    result,
                    default=args.instrumental,
                )
                language = str(result.get("language") or "")
                if is_instrumental:
                    language = "unknown"
                write_canonical_sidecar(
                    audio_path.with_suffix(".txt"),
                    caption=str(
                        existing_meta.get("caption")
                        or result.get("prompt")
                        or result.get("caption")
                        or ""
                    ),
                    genre=str(result.get("genre") or result.get("genres") or ""),
                    lyrics=lyrics,
                    bpm=bpm_decision.bpm,
                    bpm_source=bpm_decision.source,
                    lm_bpm=bpm_decision.lm_bpm,
                    local_bpm=bpm_decision.local_bpm,
                    filename_bpm=bpm_decision.filename_bpm,
                    keyscale=key_decision.keyscale,
                    key_source=key_decision.source,
                    lm_keyscale=key_decision.lm_keyscale,
                    local_keyscale=key_decision.local_keyscale,
                    timesignature=auto_timesignature(args, result),
                    language=language,
                    is_instrumental=is_instrumental,
                    custom_tag=args.trigger,
                )
                log_caption(
                    args,
                    f"{index}/{len(pending)} {audio_path.name} "
                    f"bpm={bpm_decision.bpm or 'n/a'} ({bpm_decision.source}) "
                    f"key={key_decision.keyscale or 'n/a'} ({key_decision.source})",
                )
    finally:
        stop_caption_server(args, caption_server)
    return len(pending)


def request_music_analysis(
    args: argparse.Namespace,
    client: Any,
    audio_path: Path,
    *,
    caption_window_seconds: float | None = None,
) -> dict[str, Any]:
    prepared = prepare_caption_audio(
        args,
        audio_path,
        caption_window_seconds=caption_window_seconds,
    )
    upload_name = (
        f"{audio_path.stem}_caption_excerpt.wav"
        if prepared.cleanup_path
        else audio_path.name
    )
    if prepared.cleanup_path:
        log_caption(
            args,
            f"Using {prepared.duration:.1f}s excerpt from "
            f"{prepared.offset:.1f}s for {audio_path.name}",
        )

    try:
        mime = mimetypes.guess_type(upload_name)[0] or "application/octet-stream"
        with prepared.path.open("rb") as audio_file:
            response = client.post(
                f"{args.carey_url}/release_task",
                files={"ctx_audio": (upload_name, audio_file, mime)},
                data={
                    "full_analysis_only": "true",
                    "thinking": "false",
                    "audio_duration": str(
                        resolve_music_analysis_duration(
                            args,
                            audio_path,
                            prepared_duration=prepared.duration,
                        )
                    ),
                    "lm_backend": args.caption_lm_backend,
                    "lm_model_path": args.caption_lm_model,
                },
            )
        response.raise_for_status()
        task_id = response.json().get("data", {}).get("task_id")
        if not task_id:
            raise RuntimeError(f"Carey did not return a task id for {audio_path.name}")

        started_at = time.monotonic()
        last_status_update = 0.0
        deadline = started_at + args.caption_timeout
        while time.monotonic() < deadline:
            check_cancel(args)
            query = client.post(
                f"{args.carey_url}/query_result",
                json={"task_id_list": [task_id]},
            )
            query.raise_for_status()
            records = query.json().get("data") or []
            if records:
                record = records[0]
                status = int(record.get("status", 0))
                if status == 1:
                    return normalize_analysis_result(record.get("result"))
                if status == 2:
                    raise RuntimeError(
                        f"Carey analysis failed for {audio_path.name}: "
                        f"{record.get('error') or record.get('result') or 'unknown error'}"
                    )
            now = time.monotonic()
            if now - last_status_update >= 10:
                update_status(
                    args,
                    status="running",
                    phase="captioning",
                    message=(
                        f"Analyzing {audio_path.name} with {args.caption_lm_model} "
                        f"({args.caption_lm_backend.upper()}, {int(now - started_at)}s)"
                    ),
                    caption_lm_backend=args.caption_lm_backend,
                )
                last_status_update = now
            time.sleep(2)
        raise RuntimeError(f"Carey analysis timed out for {audio_path.name}")
    finally:
        if prepared.cleanup_path:
            try:
                prepared.cleanup_path.unlink()
            except OSError:
                pass


def request_valid_music_analysis(
    args: argparse.Namespace,
    client: Any,
    audio_path: Path,
) -> dict[str, Any]:
    primary_window = float(getattr(args, "caption_window_seconds", 0.0) or 0.0)
    fallback_window = float(
        getattr(args, "caption_fallback_window_seconds", 120.0) or 0.0
    )
    attempts: list[float | None] = [primary_window]
    if primary_window <= 0 and fallback_window > 0:
        attempts.append(fallback_window)

    last_error: RuntimeError | None = None
    for index, window in enumerate(attempts):
        result = request_music_analysis(
            args,
            client,
            audio_path,
            caption_window_seconds=window,
        )
        try:
            validate_caption_analysis_result(audio_path, result)
            return result
        except RuntimeError as exc:
            last_error = exc
            if index + 1 >= len(attempts):
                raise
            log_caption(
                args,
                "Full-track metadata failed quality checks for "
                f"{audio_path.name}; retrying with "
                f"{fallback_window:.0f}s excerpt. Reason: {exc}",
            )

    raise last_error or RuntimeError(f"Carey analysis failed for {audio_path.name}")


def resolve_music_analysis_duration(
    args: argparse.Namespace,
    audio_path: Path,
    *,
    prepared_duration: float | None = None,
) -> float:
    explicit = float(getattr(args, "analysis_duration", 0.0) or 0.0)
    if explicit > 0:
        return max(1.0, explicit)

    if prepared_duration is not None and prepared_duration > 0:
        return max(1.0, float(prepared_duration))

    actual = audio_duration_seconds(audio_path)
    return max(1.0, float(actual or 0.0))


def normalize_analysis_result(value: Any) -> dict[str, Any]:
    while isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            raise RuntimeError(f"Carey returned an invalid analysis result: {value[:200]}")
    if isinstance(value, list):
        value = value[0] if value else {}
    if isinstance(value, dict) and isinstance(value.get("result"), (dict, list, str)):
        return normalize_analysis_result(value["result"])
    if not isinstance(value, dict):
        raise RuntimeError("Carey returned an empty analysis result")
    if value.get("error"):
        raise RuntimeError(str(value["error"]))
    return value


def analysis_is_instrumental(
    result: dict[str, Any],
    *,
    default: bool = False,
) -> bool:
    explicit = result.get("is_instrumental")
    if isinstance(explicit, bool):
        return explicit
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip().lower() in {"1", "true", "yes", "y", "on"}
    lyrics = str(result.get("lyrics") or "").strip()
    return (
        default
        or not lyrics
        or "[instrumental]" in lyrics.lower()
        or lyrics_are_structural_only(lyrics)
    )


def lyrics_are_structural_only(lyrics: str) -> bool:
    lines = [line.strip() for line in lyrics.splitlines() if line.strip()]
    if not lines:
        return True
    return all(re.fullmatch(r"\[[^\]]+\]", line) for line in lines)


def decide_sidecar_bpm(
    args: argparse.Namespace,
    audio_path: Path,
    result: dict[str, Any],
) -> Any:
    filename_bpm = _filename_bpm(audio_path.name)
    local_estimate = None
    if getattr(args, "bpm_analysis", True):
        try:
            local_estimate = estimate_bpm(audio_path)
        except Exception as exc:
            log_caption(args, f"Local BPM analysis failed for {audio_path.name}: {exc}")
    return choose_bpm(
        filename_bpm=filename_bpm,
        lm_bpm=result.get("bpm"),
        local_estimate=local_estimate,
        disagreement_threshold=getattr(args, "bpm_disagreement_threshold", 5.0),
        minimum_local_confidence=getattr(args, "bpm_min_confidence", 1.2),
    )


def decide_sidecar_key(
    args: argparse.Namespace,
    audio_path: Path,
    result: dict[str, Any],
) -> Any:
    local_estimate = None
    if getattr(args, "key_analysis", True):
        try:
            local_estimate = estimate_key(audio_path)
        except Exception as exc:
            log_caption(args, f"Local key analysis failed for {audio_path.name}: {exc}")
    return choose_key(
        lm_keyscale=result.get("keyscale") or result.get("key_scale"),
        local_estimate=local_estimate,
        minimum_local_confidence=getattr(args, "key_min_confidence", 0.15),
    )


def write_plan(
    args: argparse.Namespace,
    dataset_json: Path,
    tensors_dir: Path,
    output_dir: Path,
) -> Path:
    plan_path = args.run_dir / "training_plan.json"
    write_json(
        plan_path,
        {
            "model": args.model,
            "modelFamily": MODEL_MAP[args.model]["family"],
            "datasetJson": str(dataset_json),
            "tensorsDir": str(tensors_dir),
            "outputDir": str(output_dir),
            "adapterType": args.adapter_type,
            "moduleProfile": args.module_profile,
            "rank": args.rank,
            "alpha": resolve_alpha(args),
            "timestepMu": resolve_timestep_mu(args),
            "saveBest": args.save_best,
            "saveBestAfter": args.save_best_after,
            "gradientCheckpointing": resolve_gradient_checkpointing(args),
            "memoryLimitGb": resolve_memory_limit_gb(args),
            "optimizer": "adamw",
            "weightDecay": args.weight_decay,
            "lossWeighting": args.loss_weighting,
            "snrGamma": args.snr_gamma,
            "caption": args.caption,
            "captionLmModel": (
                args.caption_lm_model if args.caption == "understand_music" else None
            ),
            "captionLmBackend": (
                args.caption_lm_backend if args.caption == "understand_music" else None
            ),
            "preprocessCommand": build_preprocess_command(
                args,
                dataset_json,
                tensors_dir,
                output_dir,
            ),
            "trainCommand": build_train_command(args, tensors_dir, output_dir),
        },
    )
    return plan_path


def _genre_variants(genre: str) -> list[str]:
    tokens = [token.strip() for token in genre.split(",") if token.strip()]
    if not tokens:
        return []
    variants = [", ".join(tokens)]
    for size in (2, 3):
        if len(tokens) >= size:
            variants.extend(", ".join(combo) for combo in combinations(tokens, size))
    return variants


def collect_caption_pool(
    dataset_dir: Path,
    *,
    include_genres: bool,
) -> list[str]:
    seen: set[str] = set()
    pool: list[str] = []
    for audio_path in discover_audio_files(dataset_dir):
        meta = load_sidecar_metadata(audio_path)
        entries: list[str] = []
        caption = meta.get("caption", "").strip()
        genre = meta.get("genre", "").strip()
        if caption:
            entries.append(caption)
        if include_genres and genre:
            entries.extend(_genre_variants(genre))
        for entry in entries:
            if entry not in seen:
                seen.add(entry)
                pool.append(entry)
    return pool


def register_trained_lora(args: argparse.Namespace, final_checkpoint: Path) -> None:
    family = MODEL_MAP[args.model]["family"]
    metadata = {
        "path": str(final_checkpoint),
        "captionsPath": str(args.dataset_dir),
        "scale": 1.0,
        "backends": ["base", "turbo"],
        "modelFamily": family,
        "adapterType": args.adapter_type,
        "moduleProfile": args.module_profile,
        "timestepMu": resolve_timestep_mu(args),
    }

    if args.lora_catalog_path:
        catalog = read_json(args.lora_catalog_path, {})
        if not isinstance(catalog, dict):
            catalog = {}
        catalog[args.name] = metadata
        write_json(args.lora_catalog_path, catalog)

    if args.lora_registry_path:
        registry = read_json(args.lora_registry_path, {})
        if not isinstance(registry, dict):
            registry = {}
        registry[args.name] = {
            "path": str(final_checkpoint),
            "scale": 1.0,
            "backends": ["base", "turbo"],
            "model_family": family,
            "adapter_type": args.adapter_type,
            "module_profile": args.module_profile,
            "timestep_mu": resolve_timestep_mu(args),
        }
        write_json(args.lora_registry_path, registry)

    if args.captions_json_path:
        pools = read_json(args.captions_json_path, {})
        if not isinstance(pools, dict):
            pools = {}
        pools[args.name] = collect_caption_pool(
            args.dataset_dir,
            include_genres=args.genre_ratio > 0,
        )
        write_json(args.captions_json_path, pools)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--status-path", type=Path, required=True)
    parser.add_argument("--current-job-path", type=Path, required=True)
    parser.add_argument("--cancel-path", type=Path)
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--model", choices=MODEL_MAP, default="base")
    parser.add_argument("--instrumental", action="store_true")
    parser.add_argument("--trigger", default="")
    parser.add_argument(
        "--tag-position",
        choices=("prepend", "append", "replace"),
        default="prepend",
    )
    parser.add_argument("--genre-ratio", type=int, default=20)
    parser.add_argument("--caption", choices=("understand_music", "skip"), default="skip")
    parser.add_argument(
        "--caption-lm-model",
        choices=CAPTION_LM_MODELS,
        default="acestep-5Hz-lm-1.7B",
    )
    parser.add_argument(
        "--caption-lm-backend",
        choices=("pt", "mlx"),
        default=CAPTION_LM_BACKEND_DEFAULT,
    )
    parser.add_argument("--carey-url", default="http://127.0.0.1:8013")
    parser.add_argument("--inference-carey-url", default="http://127.0.0.1:8003")
    parser.add_argument("--caption-startup-timeout", type=float, default=900.0)
    parser.add_argument("--caption-timeout", type=float, default=900.0)
    parser.add_argument("--caption-window-seconds", type=float, default=0.0)
    parser.add_argument("--caption-fallback-window-seconds", type=float, default=120.0)
    parser.add_argument("--model-load-timeout", type=float, default=900.0)
    parser.add_argument("--carey-stop-timeout", type=float, default=180.0)
    parser.add_argument("--analysis-duration", type=float, default=0.0)
    parser.add_argument("--bpm-analysis", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bpm-disagreement-threshold", type=float, default=5.0)
    parser.add_argument("--bpm-min-confidence", type=float, default=1.2)
    parser.add_argument("--key-analysis", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--key-min-confidence", type=float, default=0.15)
    parser.add_argument("--include-auto-timesignature", action="store_true")
    parser.add_argument("--overwrite-captions", action="store_true")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--alpha", type=int, default=None)
    parser.add_argument(
        "--module-profile",
        choices=("attention", "balanced"),
        default="balanced",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--cfg-ratio", type=float, default=0.15)
    parser.add_argument(
        "--timestep-mu",
        type=float,
        default=None,
        help="Advanced training schedule override. Default: -0.4.",
    )
    parser.add_argument("--loss-weighting", choices=("none", "min_snr"), default="min_snr")
    parser.add_argument("--snr-gamma", type=float, default=5.0)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--save-best", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-best-after", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--max-duration", type=float, default=240.0)
    parser.add_argument("--adapter-type", choices=("lora", "dora"), default="dora")
    parser.add_argument("--preprocess-device", default="auto")
    parser.add_argument(
        "--preprocess-precision",
        choices=("auto", "bf16", "fp16", "fp32"),
        default="fp32",
    )
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--memory-limit-gb", type=float, default=0.0)
    parser.add_argument("--allow-unsafe-xl", action="store_true")
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--fake-decoder", action="store_true")
    parser.add_argument("--lora-catalog-path", type=Path)
    parser.add_argument("--lora-registry-path", type=Path)
    parser.add_argument("--captions-json-path", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.rank <= 0 or (args.alpha is not None and args.alpha <= 0):
        raise ValueError("rank and alpha must be greater than zero")
    if args.batch_size <= 0 or args.gradient_accumulation <= 0:
        raise ValueError("batch size and gradient accumulation must be greater than zero")
    if args.epochs <= 0 or args.save_every <= 0:
        raise ValueError("epochs and save interval must be greater than zero")
    if args.save_best_after <= 0:
        raise ValueError("best-checkpoint start epoch must be greater than zero")
    if not 0 <= args.genre_ratio <= 100:
        raise ValueError("genre ratio must be between 0 and 100")
    if not 0 <= args.cfg_ratio < 1:
        raise ValueError("cfg ratio must be in [0, 1)")
    if args.learning_rate <= 0:
        raise ValueError("learning rate must be greater than zero")
    if args.weight_decay < 0:
        raise ValueError("weight decay must be non-negative")
    if args.snr_gamma <= 0:
        raise ValueError("SNR gamma must be greater than zero")
    if args.max_steps < 0:
        raise ValueError("max steps must be non-negative")
    if args.memory_limit_gb < 0:
        raise ValueError("memory limit must be non-negative")
    for label, value in (
        ("caption startup timeout", args.caption_startup_timeout),
        ("caption timeout", args.caption_timeout),
        ("caption window", args.caption_window_seconds),
        ("caption fallback window", args.caption_fallback_window_seconds),
        ("model load timeout", args.model_load_timeout),
        ("carey stop timeout", args.carey_stop_timeout),
        ("analysis duration", args.analysis_duration),
        ("BPM disagreement threshold", args.bpm_disagreement_threshold),
        ("BPM minimum confidence", args.bpm_min_confidence),
        ("key minimum confidence", args.key_min_confidence),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{label} must be finite")
    if args.caption_startup_timeout <= 0 or args.caption_timeout <= 0:
        raise ValueError("caption timeouts must be greater than zero")
    if args.caption_window_seconds < 0 or args.caption_fallback_window_seconds < 0:
        raise ValueError("caption windows must be non-negative")
    if args.model_load_timeout <= 0 or args.carey_stop_timeout <= 0:
        raise ValueError("service timeouts must be greater than zero")
    if args.analysis_duration < 0:
        raise ValueError("analysis duration must be non-negative")
    if args.bpm_disagreement_threshold < 0:
        raise ValueError("BPM disagreement threshold must be non-negative")
    if args.bpm_min_confidence < 0 or args.key_min_confidence < 0:
        raise ValueError("analysis confidence thresholds must be non-negative")
    if (
        args.model.startswith("xl")
        and _physical_memory_bytes() <= 40 * 1024**3
        and not args.allow_unsafe_xl
    ):
        raise ValueError(
            "XL ACE training is temporarily disabled on systems with 40 GiB or less. "
            "Use the regular base model for now."
        )
    if not math.isfinite(resolve_timestep_mu(args)):
        raise ValueError("timestep mu must be finite")


def _physical_memory_bytes() -> int:
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (OSError, ValueError):
        return 32 * 1024**3


def _filename_bpm(filename: str) -> int | None:
    match = re.search(r"(?:^|[_-])bpm[_-]?(\d{2,3})(?:[_-]|\.|$)", filename, re.I)
    if not match:
        return None
    bpm = int(match.group(1))
    return bpm if 1 <= bpm <= 400 else None


def auto_timesignature(args: argparse.Namespace, result: dict[str, Any]) -> str:
    if not getattr(args, "include_auto_timesignature", False):
        return ""
    return str(result.get("timesignature") or result.get("time_signature") or "")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.name = slugify(args.name)
    args.dataset_dir = args.dataset_dir.expanduser().resolve()
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    args.run_dir = args.run_dir.expanduser().resolve()
    args.status_path = args.status_path.expanduser().resolve()
    args.current_job_path = args.current_job_path.expanduser().resolve()
    args.log_path = args.log_path.expanduser().resolve()
    args.cancel_path = (
        args.cancel_path.expanduser().resolve()
        if args.cancel_path
        else args.run_dir / "cancel.requested"
    )

    dataset_json = args.run_dir / "dataset.json"
    output_dir = args.run_dir / "output"
    tensors_dir = output_dir / "tensors"
    final_checkpoint = output_dir / "final"
    best_checkpoint = output_dir / "best"

    try:
        validate_args(args)
        args.run_dir.mkdir(parents=True, exist_ok=True)
        if args.cancel_path.exists():
            args.cancel_path.unlink()
        update_status(
            args,
            status="running",
            phase="starting",
            message="Starting ACE-Step MLX LoRA training",
            error=None,
            child_pid=None,
            final_checkpoint_path=None,
            result_path=None,
            caption_lm_model=(
                args.caption_lm_model if args.caption == "understand_music" else None
            ),
            caption_lm_backend=(
                args.caption_lm_backend if args.caption == "understand_music" else None
            ),
            adapter_type=args.adapter_type,
            module_profile=args.module_profile,
            model_family=MODEL_MAP[args.model]["family"],
        )

        captioned = 0
        if args.caption == "understand_music":
            captioned = caption_with_understand_music(args)

        update_status(
            args,
            status="running",
            phase="building-dataset",
            message="Building ACE-Step dataset metadata",
        )
        validate_dataset_sidecars(args.dataset_dir)
        dataset_result = build_dataset_json(
            args.dataset_dir,
            dataset_json,
            name=args.name,
            trigger=args.trigger,
            tag_position=args.tag_position,
            genre_ratio=args.genre_ratio,
            instrumental_default=args.instrumental,
        )
        plan_path = write_plan(args, dataset_json, tensors_dir, output_dir)

        if args.prepare_only or args.dry_run:
            prepare_message = "ACE-Step MLX training dataset prepared"
            caption_skip_message = getattr(args, "_caption_skip_message", "")
            if caption_skip_message:
                prepare_message = (
                    "ACE-Step MLX training dataset prepared from existing sidecars. "
                    f"{caption_skip_message}"
                )
            update_status(
                args,
                status="completed",
                phase="prepared",
                message=prepare_message,
                dataset_json_path=str(dataset_json),
                training_plan_path=str(plan_path),
                sample_count=dataset_result["samples"],
                captioned_count=captioned,
                caption_lm_model=(
                    args.caption_lm_model if args.caption == "understand_music" else None
                ),
                caption_lm_backend=(
                    args.caption_lm_backend if args.caption == "understand_music" else None
                ),
                model_family=MODEL_MAP[args.model]["family"],
                adapter_type=args.adapter_type,
                module_profile=args.module_profile,
                error=None,
                child_pid=None,
            )
            return 0

        require_training_environment(args)
        require_model_checkpoint(args)
        tensors_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        run_step(
            args,
            build_preprocess_command(args, dataset_json, tensors_dir, output_dir),
            "preprocessing",
            "Pre-encoding audio with Carey's two-pass pipeline",
        )
        if not any(tensors_dir.glob("*.pt")):
            raise RuntimeError("Preprocessing completed without producing training tensors")

        run_step(
            args,
            build_train_command(args, tensors_dir, output_dir),
            "training",
            "Training ACE-Step LoRA with MLX",
            cwd=output_dir,
        )
        if not is_complete_peft_adapter(final_checkpoint):
            raise RuntimeError(
                f"Training finished without a complete PEFT adapter in {final_checkpoint}"
            )

        selected_checkpoint = (
            best_checkpoint
            if args.save_best and is_complete_peft_adapter(best_checkpoint)
            else final_checkpoint
        )
        result_path = args.run_dir / "result.json"
        family = MODEL_MAP[args.model]["family"]
        checkpoint_selection = (
            "best_loss" if selected_checkpoint == best_checkpoint else "final_epoch"
        )
        result = {
            "jobId": args.job_id,
            "job_id": args.job_id,
            "name": args.name,
            "finalCheckpointPath": str(selected_checkpoint),
            "final_checkpoint_path": str(selected_checkpoint),
            "bestCheckpointPath": (
                str(best_checkpoint) if is_complete_peft_adapter(best_checkpoint) else None
            ),
            "best_checkpoint_path": (
                str(best_checkpoint) if is_complete_peft_adapter(best_checkpoint) else None
            ),
            "lastEpochCheckpointPath": str(final_checkpoint),
            "last_epoch_checkpoint_path": str(final_checkpoint),
            "captionsPath": str(args.dataset_dir),
            "captions_path": str(args.dataset_dir),
            "modelFamily": family,
            "model_family": family,
            "adapterType": args.adapter_type,
            "adapter_type": args.adapter_type,
            "moduleProfile": args.module_profile,
            "module_profile": args.module_profile,
            "timestepMu": resolve_timestep_mu(args),
            "timestep_mu": resolve_timestep_mu(args),
            "checkpointSelection": checkpoint_selection,
            "checkpoint_selection": checkpoint_selection,
            "captionLmModel": (
                args.caption_lm_model if args.caption == "understand_music" else None
            ),
            "caption_lm_model": (
                args.caption_lm_model if args.caption == "understand_music" else None
            ),
            "captionLmBackend": (
                args.caption_lm_backend if args.caption == "understand_music" else None
            ),
            "caption_lm_backend": (
                args.caption_lm_backend if args.caption == "understand_music" else None
            ),
            "backends": ["base", "turbo"],
            "scale": 1.0,
        }
        write_json(selected_checkpoint / "metadata.json", result)
        write_json(result_path, result)
        register_trained_lora(args, selected_checkpoint)
        update_status(
            args,
            status="completed",
            phase="completed",
            message="ACE-Step MLX LoRA training complete",
            final_checkpoint_path=str(selected_checkpoint),
            best_checkpoint_path=(
                str(best_checkpoint) if is_complete_peft_adapter(best_checkpoint) else None
            ),
            last_epoch_checkpoint_path=str(final_checkpoint),
            captions_path=str(args.dataset_dir),
            result_path=str(result_path),
            dataset_json_path=str(dataset_json),
            sample_count=dataset_result["samples"],
            captioned_count=None,
            caption_lm_model=(
                args.caption_lm_model if args.caption == "understand_music" else None
            ),
            caption_lm_backend=(
                args.caption_lm_backend if args.caption == "understand_music" else None
            ),
            model_family=family,
            adapter_type=args.adapter_type,
            module_profile=args.module_profile,
            registered_lora_name=args.name,
            error=None,
            child_pid=None,
        )
        return 0
    except Cancelled:
        update_status(
            args,
            status="cancelled",
            phase="cancelled",
            message="Training cancelled.",
            error=None,
            child_pid=None,
        )
        return 0
    except Exception as exc:
        update_status(
            args,
            status="failed",
            phase="failed",
            message=str(exc),
            error=str(exc),
            child_pid=None,
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
