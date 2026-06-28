#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time

from api_foundation import (
    FOUNDATION_HF_REPO,
    FOUNDATION_MODEL_DISPLAY_NAME,
    ensure_foundation_model_files,
    foundation_download_queue_status,
    foundation_model_download_status,
)


def emit(payload: dict) -> None:
    print(json.dumps(payload), flush=True)


def emit_status(
    *,
    status: str,
    progress: int,
    message: str,
    stage_name: str,
    stage_index: int,
    stage_total: int,
    download_percent: int,
    error: str | None = None,
    downloaded_bytes: int = 0,
    total_bytes: int = 0,
    speed_bps: float = 0.0,
) -> None:
    payload = {
        "success": error is None,
        "session_id": "local-foundation-predownload",
        "model_name": FOUNDATION_HF_REPO,
        "status": status,
        "progress": max(0, min(100, int(progress))),
        "queue_status": foundation_download_queue_status(
            status=status,
            message=message,
            stage_name=stage_name,
            stage_index=stage_index,
            stage_total=stage_total,
            download_percent=download_percent,
            downloaded_bytes=downloaded_bytes,
            total_bytes=total_bytes,
            speed_bps=speed_bps,
        ),
        "error": error,
        "updated_at": time.time(),
    }
    emit(payload)


def catalog_command() -> int:
    emit(
        {
            "success": True,
            "models": {
                "small": [],
                "medium": [],
                "large": [
                    {
                        "name": FOUNDATION_MODEL_DISPLAY_NAME,
                        "path": FOUNDATION_HF_REPO,
                        "type": "single",
                    }
                ],
            },
            "updated_at": time.time(),
        }
    )
    return 0


def status_command() -> int:
    emit(
        {
            "success": True,
            "models": {
                FOUNDATION_HF_REPO: foundation_model_download_status(),
            },
            "updated_at": time.time(),
        }
    )
    return 0


def download_command(model_name: str) -> int:
    normalized = model_name.strip()
    if normalized not in {FOUNDATION_HF_REPO, FOUNDATION_MODEL_DISPLAY_NAME}:
        emit(
            {
                "success": False,
                "error": f"Unknown model '{model_name}'",
                "updated_at": time.time(),
            }
        )
        return 2

    def progress_callback(
        filename: str,
        stage_index: int,
        stage_total: int,
        download_percent: int,
        message: str,
        downloaded_bytes: int = 0,
        total_bytes: int = 0,
        speed_bps: float = 0.0,
    ) -> None:
        stage_progress = (
            (max(0, stage_index - 1) + (max(0, min(100, int(download_percent))) / 100.0))
            / max(1, stage_total)
        )
        emit_status(
            status="processing",
            progress=max(1, min(99, int(stage_progress * 100))),
            message=message,
            stage_name=filename,
            stage_index=stage_index,
            stage_total=stage_total,
            download_percent=download_percent,
            downloaded_bytes=downloaded_bytes,
            total_bytes=total_bytes,
            speed_bps=speed_bps,
        )

    emit_status(
        status="warming",
        progress=0,
        message=f"preparing download for {FOUNDATION_HF_REPO}",
        stage_name="prepare",
        stage_index=0,
        stage_total=2,
        download_percent=0,
    )

    try:
        ensure_foundation_model_files(progress_callback=progress_callback)
        emit_status(
            status="completed",
            progress=100,
            message="foundation-1 model files downloaded",
            stage_name="completed",
            stage_index=2,
            stage_total=2,
            download_percent=100,
        )
        return 0
    except Exception as exc:
        emit_status(
            status="failed",
            progress=0,
            message=str(exc),
            stage_name="failed",
            stage_index=0,
            stage_total=2,
            download_percent=0,
            error=str(exc),
        )
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local foundation-1 model predownload helper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("catalog")
    subparsers.add_parser("status")

    download_parser = subparsers.add_parser("download")
    download_parser.add_argument("--model-name", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "catalog":
        return catalog_command()
    if args.command == "status":
        return status_command()
    if args.command == "download":
        return download_command(args.model_name)

    parser.error(f"Unsupported command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
