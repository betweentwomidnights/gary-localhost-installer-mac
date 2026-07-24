#!/usr/bin/env python3
"""Auto-label an SA3 dataset with genre, BPM, and key.

The temporary ACE-Step server and understand_music request path are reused from
the native Carey MLX trainer. Existing text sidecars are replaced one at a time,
and status/cancellation are persisted so the macOS prompt editor can be closed
and reopened while the job continues.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

SERVICE_DIR = Path(__file__).resolve().parent
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from train_mlx_lora_job import (  # noqa: E402
    Cancelled,
    caption_text_quality_error,
    check_cancel,
    decide_sidecar_bpm,
    decide_sidecar_key,
    ensure_carey_model_loaded,
    request_music_analysis,
    require_caption_lm_backend,
    start_caption_server,
    stop_caption_server,
    update_status,
    wait_for_carey,
)

AUDIO_EXTENSIONS = (
    ".wav",
    ".flac",
    ".mp3",
    ".ogg",
    ".opus",
    ".m4a",
    ".aiff",
    ".aif",
)


def discover_sa3_audio(dataset_dir: Path) -> list[Path]:
    files = [
        path
        for path in dataset_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    ]
    files.sort(key=lambda path: path.relative_to(dataset_dir).as_posix().lower())
    return files


def format_sidecar(
    style: str,
    genre: str,
    bpm: int | None,
    keyscale: str,
) -> str:
    genre = (genre or "").strip()
    keyscale = (keyscale or "").strip()
    if style == "labeled":
        parts = ["TrackType: Music", "VocalType: Instrumental"]
        if genre:
            parts.append(f"Genre: {genre}")
        if bpm:
            parts.append(f"BPM: {int(bpm)}")
        if keyscale:
            parts.append(f"Key: {keyscale}")
        return ", ".join(parts)
    return ", ".join(
        part
        for part in (
            genre,
            f"{int(bpm)} bpm" if bpm else "",
            keyscale,
        )
        if part
    )


def usable_genre(result: dict[str, Any], audio_path: Path) -> str:
    value = result.get("genre") or result.get("genres")
    if isinstance(value, (list, tuple)):
        genre = ", ".join(
            str(item).strip() for item in value if str(item).strip()
        )
    elif isinstance(value, str):
        genre = value.strip()
    elif value is None:
        genre = ""
    else:
        raise RuntimeError(
            f"ACE understand_music returned an invalid genre for "
            f"{audio_path.name}: expected text, got {type(value).__name__}"
        )
    if not genre or genre.casefold() in {"n/a", "na", "none", "null", "unknown"}:
        raise RuntimeError(
            f"ACE understand_music returned no usable genre for {audio_path.name}"
        )
    reason = caption_text_quality_error(genre, field="genre")
    if reason:
        raise RuntimeError(
            f"ACE understand_music returned an unusable genre for "
            f"{audio_path.name}: {reason}"
        )
    return genre


def request_valid_genre_analysis(
    args: SimpleNamespace,
    client: Any,
    audio_path: Path,
) -> tuple[dict[str, Any], str]:
    primary_window = float(args.caption_window_seconds or 0.0)
    fallback_window = float(args.caption_fallback_window_seconds or 0.0)
    attempts = [primary_window]
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
            return result, usable_genre(result, audio_path)
        except RuntimeError as exc:
            last_error = exc
            if index + 1 >= len(attempts):
                raise
            print(
                f"[autolabel] Full-track genre failed quality checks for "
                f"{audio_path.name}; retrying with a {fallback_window:.0f}s "
                f"center excerpt. Reason: {exc}",
                flush=True,
            )
    raise last_error or RuntimeError(
        f"Carey genre analysis failed for {audio_path.name}"
    )


def ensure_carey_stopped(args: SimpleNamespace) -> None:
    """Wait until both managed Carey endpoints release the runtime."""
    import httpx

    urls = [args.inference_carey_url]
    if args.carey_url not in urls:
        urls.append(args.carey_url)
    deadline = time.monotonic() + args.carey_stop_timeout
    announced = False
    while time.monotonic() < deadline:
        check_cancel(args)
        running: list[str] = []
        for url in urls:
            try:
                if httpx.get(f"{url}/health", timeout=2).is_success:
                    running.append(url)
            except Exception:
                pass
        if not running:
            return
        if not announced:
            update_status(
                args,
                status="running",
                phase="waiting-for-carey-stop",
                message="Waiting for Carey to stop and release unified memory",
            )
            announced = True
        time.sleep(2)
    raise RuntimeError(
        "Carey is still running. Stop it through gary4local before auto-labeling."
    )


def build_reuse_args(cli: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        job_id=cli.job_id,
        name="sa3-autolabel",
        run_dir=Path(cli.run_dir),
        log_path=Path(cli.log_path),
        cancel_path=Path(cli.cancel_path),
        status_path=Path(cli.status_path),
        current_job_path=Path(cli.current_job_path),
        dataset_dir=Path(cli.dataset_dir),
        carey_url=cli.carey_url,
        inference_carey_url=cli.inference_carey_url,
        caption_lm_model=cli.caption_lm_model,
        caption_lm_backend=cli.caption_lm_backend,
        model=cli.model,
        caption_timeout=cli.caption_timeout,
        caption_startup_timeout=cli.caption_startup_timeout,
        model_load_timeout=cli.model_load_timeout,
        carey_stop_timeout=cli.carey_stop_timeout,
        caption_window_seconds=cli.caption_window_seconds,
        caption_fallback_window_seconds=cli.caption_fallback_window_seconds,
        analysis_duration=cli.analysis_duration,
        overwrite_captions=True,
        bpm_analysis=True,
        key_analysis=True,
        bpm_disagreement_threshold=5.0,
        bpm_min_confidence=1.2,
        key_min_confidence=0.15,
        instrumental=True,
        trigger="",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Auto-label an SA3 dataset.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--style", choices=("bare", "labeled"), default="bare")
    parser.add_argument("--caption-lm-model", default="acestep-5Hz-lm-1.7B")
    parser.add_argument(
        "--caption-lm-backend",
        choices=("pt", "mlx"),
        default="mlx",
    )
    parser.add_argument("--model", default="base")
    parser.add_argument("--carey-url", default="http://127.0.0.1:8013")
    parser.add_argument(
        "--inference-carey-url",
        default="http://127.0.0.1:8003",
    )
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--status-path", type=Path, required=True)
    parser.add_argument("--cancel-path", type=Path, required=True)
    parser.add_argument("--current-job-path", type=Path, required=True)
    parser.add_argument("--caption-timeout", type=float, default=900.0)
    parser.add_argument("--caption-startup-timeout", type=float, default=900.0)
    parser.add_argument("--model-load-timeout", type=float, default=900.0)
    parser.add_argument("--carey-stop-timeout", type=float, default=180.0)
    parser.add_argument("--caption-window-seconds", type=float, default=0.0)
    parser.add_argument(
        "--caption-fallback-window-seconds",
        type=float,
        default=120.0,
    )
    parser.add_argument("--analysis-duration", type=float, default=0.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    cli = build_parser().parse_args(argv)
    cli.dataset_dir = cli.dataset_dir.expanduser().resolve()
    cli.run_dir = cli.run_dir.expanduser().resolve()
    cli.log_path = cli.log_path.expanduser().resolve()
    cli.status_path = cli.status_path.expanduser().resolve()
    cli.cancel_path = cli.cancel_path.expanduser().resolve()
    cli.current_job_path = cli.current_job_path.expanduser().resolve()
    args = build_reuse_args(cli)

    cli.run_dir.mkdir(parents=True, exist_ok=True)
    if cli.cancel_path.exists():
        cli.cancel_path.unlink()
    files = discover_sa3_audio(cli.dataset_dir)
    total = len(files)
    update_status(
        args,
        status="running",
        phase="starting",
        message="Preparing SA3 auto-label",
        error=None,
        child_pid=None,
        total=total,
        done=0,
        dataset_path=str(cli.dataset_dir),
        current_path="",
        style=cli.style,
    )
    if not files:
        update_status(
            args,
            status="completed",
            phase="completed",
            message="No audio files found",
            total=0,
            done=0,
        )
        return 0

    import httpx

    server = None
    terminal_status = ""
    terminal_message = ""
    terminal_error: str | None = None
    try:
        require_caption_lm_backend(args)
        ensure_carey_stopped(args)
        server = start_caption_server(args)
        with httpx.Client(timeout=httpx.Timeout(args.caption_timeout)) as client:
            wait_for_carey(args, client, server)
            update_status(
                args,
                phase="loading-model",
                message="Loading the ACE caption models",
            )
            ensure_carey_model_loaded(args, client)
            done = 0
            for audio_path in files:
                check_cancel(args)
                update_status(
                    args,
                    status="running",
                    phase="analyzing",
                    message=f"Analyzing {audio_path.name}",
                    current_path=str(audio_path),
                    done=done,
                    total=total,
                )
                result, genre = request_valid_genre_analysis(
                    args,
                    client,
                    audio_path,
                )
                bpm = decide_sidecar_bpm(args, audio_path, result)
                key = decide_sidecar_key(args, audio_path, result)
                text = format_sidecar(
                    cli.style,
                    genre,
                    bpm.bpm,
                    key.keyscale,
                )
                audio_path.with_suffix(".txt").write_text(
                    text + "\n",
                    encoding="utf-8",
                )
                done += 1
                update_status(
                    args,
                    done=done,
                    current_path="",
                )
                print(
                    f"[autolabel] {done}/{total} {audio_path.name} -> {text} "
                    f"[bpm={bpm.source}, key={key.source}]",
                    flush=True,
                )
    except Cancelled:
        terminal_status = "cancelled"
        terminal_message = "Auto-label cancelled"
        print("[autolabel] cancelled", flush=True)
    except Exception as exc:
        terminal_status = "failed"
        terminal_message = str(exc)
        terminal_error = str(exc)
        traceback.print_exc()
    finally:
        if server is not None:
            try:
                stop_caption_server(args, server)
            except Exception as exc:
                print(
                    f"[autolabel] caption service cleanup failed: {exc}",
                    flush=True,
                )
                if not terminal_status:
                    terminal_status = "failed"
                    terminal_message = (
                        f"Could not stop the temporary caption service: {exc}"
                    )
                    terminal_error = terminal_message

    if terminal_status:
        update_status(
            args,
            status=terminal_status,
            phase=terminal_status,
            message=terminal_message,
            error=terminal_error,
            current_path="",
            child_pid=None,
        )
        return 0 if terminal_status == "cancelled" else 1
    update_status(
        args,
        status="completed",
        phase="completed",
        message=f"Auto-labeled {total} track{'' if total == 1 else 's'}",
        done=total,
        current_path="",
        error=None,
        child_pid=None,
    )
    print(f"[autolabel] done: {total} track(s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
