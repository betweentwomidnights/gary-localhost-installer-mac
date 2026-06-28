#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from typing import Any

from g4l_models import MODEL_CATALOG
from g4laudio_mlx import get_model_download_status, predownload_model


def emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload), flush=True)


def all_catalog_models() -> list[str]:
    ordered: list[str] = []
    for size in ("small", "medium", "large"):
        ordered.extend(MODEL_CATALOG.get(size, []))
    return ordered


def format_model_catalog() -> dict[str, list[dict[str, Any]]]:
    def parse_model_info(model_path: str) -> dict[str, Any]:
        name = model_path.split("/")[-1]
        parts = name.rsplit("-", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return {
                "full_path": model_path,
                "display_name": name,
                "base_name": parts[0],
                "checkpoint": int(parts[1]),
                "has_checkpoint": True,
            }
        return {
            "full_path": model_path,
            "display_name": name,
            "base_name": name,
            "checkpoint": None,
            "has_checkpoint": False,
        }

    def group_models(model_list: list[str]) -> list[dict[str, Any]]:
        parsed = [parse_model_info(m) for m in model_list]
        grouped: dict[str, list[dict[str, Any]]] = {}
        for model in parsed:
            grouped.setdefault(str(model["base_name"]), []).append(model)

        result: list[dict[str, Any]] = []
        for base_name, models_group in grouped.items():
            if len(models_group) == 1 and not bool(models_group[0]["has_checkpoint"]):
                result.append(
                    {
                        "name": models_group[0]["display_name"],
                        "path": models_group[0]["full_path"],
                        "type": "single",
                    }
                )
                continue

            checkpoints = sorted(
                [m for m in models_group if bool(m["has_checkpoint"])],
                key=lambda item: int(item["checkpoint"]),
            )
            if len(checkpoints) == 1:
                checkpoint = checkpoints[0]
                result.append(
                    {
                        "name": checkpoint["display_name"],
                        "path": checkpoint["full_path"],
                        "type": "single",
                        "epoch": checkpoint["checkpoint"],
                    }
                )
            else:
                result.append(
                    {
                        "name": base_name,
                        "type": "group",
                        "checkpoints": [
                            {
                                "name": f"{base_name}-{checkpoint['checkpoint']}",
                                "path": checkpoint["full_path"],
                                "epoch": checkpoint["checkpoint"],
                            }
                            for checkpoint in checkpoints
                        ],
                    }
                )
        return result

    return {
        "small": group_models(MODEL_CATALOG.get("small", [])),
        "medium": group_models(MODEL_CATALOG.get("medium", [])),
        "large": group_models(MODEL_CATALOG.get("large", [])),
    }


def format_bytes(value: int) -> str:
    size = float(max(0, int(value)))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            return f"{size:.1f}{unit}" if unit != "B" else f"{int(size)}B"
        size /= 1024.0
    return f"{size:.1f}TB"


def build_status_payload(
    *,
    session_id: str,
    model_name: str,
    status: str,
    progress: int,
    message: str,
    stage_name: str,
    stage_index: int,
    stage_total: int,
    download_percent: int,
    repo_id: str,
    error: str | None = None,
    downloaded_bytes: int = 0,
    total_bytes: int = 0,
    speed_bps: float = 0.0,
    unit: str = "",
    progress_name: str = "",
) -> dict[str, Any]:
    queue_status = {
        "status": status,
        "message": message,
        "position": 0,
        "total_queued": 0,
        "estimated_time": None,
        "estimated_seconds": 0,
        "source": "localhost",
        "phase": "download",
        "repo_id": repo_id,
        "download_percent": max(0, min(100, int(download_percent))),
        "downloaded_bytes": max(0, int(downloaded_bytes)),
        "total_bytes": max(0, int(total_bytes)),
        "speed_bps": max(0.0, float(speed_bps)),
        "stage_name": stage_name,
        "stage_index": max(0, int(stage_index)),
        "stage_total": max(0, int(stage_total)),
        "unit": unit,
        "progress_name": progress_name,
    }
    return {
        "success": error is None,
        "session_id": session_id,
        "model_name": model_name,
        "status": status,
        "progress": max(0, min(100, int(progress))),
        "queue_status": queue_status,
        "error": error,
        "updated_at": time.time(),
    }


def catalog_command() -> int:
    emit(
        {
            "success": True,
            "models": format_model_catalog(),
            "updated_at": time.time(),
        }
    )
    return 0


def status_command() -> int:
    emit(
        {
            "success": True,
            "models": {
                model_name: get_model_download_status(model_name)
                for model_name in all_catalog_models()
            },
            "updated_at": time.time(),
        }
    )
    return 0


def download_command(model_name: str) -> int:
    normalized = model_name.strip()
    if normalized not in set(all_catalog_models()):
        emit(
            {
                "success": False,
                "error": f"Unknown model '{model_name}'",
                "updated_at": time.time(),
            }
        )
        return 2

    session_id = "local-gary-predownload"
    stage_started_at: dict[tuple[int, str, str], float] = {}

    emit(
        build_status_payload(
            session_id=session_id,
            model_name=normalized,
            status="warming",
            progress=0,
            message=f"Preparing download for {normalized}",
            stage_name="prepare",
            stage_index=0,
            stage_total=0,
            download_percent=0,
            repo_id=normalized,
        )
    )

    def progress_callback(event: dict[str, Any]) -> None:
        stage_name = str(event.get("stage_name") or "download")
        repo_id = str(event.get("repo_id") or normalized)
        downloaded = int(event.get("downloaded_bytes") or 0)
        total = int(event.get("total_bytes") or 0)
        stage_percent = max(0, min(100, int(event.get("stage_percent") or 0)))
        stage_total = int(event.get("stage_total") or 0)
        stage_index = int(event.get("stage_index") or 0)
        unit = str(event.get("unit") or "").strip().lower()
        progress_name = str(event.get("progress_name") or "").strip()
        speed_bps = float(event.get("speed_bps") or 0.0)
        progress = max(0, min(99, int(event.get("percent") or 0)))

        stage_prefix = (
            f"Stage {stage_index}/{stage_total} {stage_name}"
            if stage_total > 0
            else stage_name
        )
        speed_suffix = f" • {format_bytes(int(speed_bps))}/s" if speed_bps > 0 else ""
        stage_key = (stage_index, stage_name, repo_id)
        now = time.time()
        started_at = stage_started_at.setdefault(stage_key, now)
        prep_seconds = int(max(0, now - started_at))

        if unit in {"it", "item", "items", "file", "files"} and total > 0:
            message = (
                f"{stage_prefix}: {repo_id} "
                f"({downloaded}/{total} files • {stage_percent}%)"
            )
        elif stage_percent <= 0 and downloaded <= 0 and prep_seconds >= 3:
            message = (
                f"{stage_prefix}: {repo_id} "
                f"(preparing transfer... {prep_seconds}s)"
            )
        elif total >= 4 * 1024 or downloaded >= 4 * 1024:
            message = (
                f"{stage_prefix}: {repo_id} "
                f"({format_bytes(downloaded)}/{format_bytes(total)} • {stage_percent}%{speed_suffix})"
            )
        elif progress_name:
            message = f"{stage_prefix}: {repo_id} ({stage_percent}% • {progress_name})"
        else:
            message = f"{stage_prefix}: {repo_id} ({stage_percent}%)"

        emit(
            build_status_payload(
                session_id=session_id,
                model_name=normalized,
                status="processing",
                progress=progress,
                message=message,
                stage_name=stage_name,
                stage_index=stage_index,
                stage_total=stage_total,
                download_percent=stage_percent,
                repo_id=repo_id,
                downloaded_bytes=downloaded,
                total_bytes=total,
                speed_bps=speed_bps,
                unit=unit,
                progress_name=progress_name,
            )
        )

    try:
        predownload_model(
            model_name=normalized,
            download_progress_callback=progress_callback,
        )
        emit(
            build_status_payload(
                session_id=session_id,
                model_name=normalized,
                status="completed",
                progress=100,
                message=f"{normalized} is ready for offline use.",
                stage_name="completed",
                stage_index=0,
                stage_total=0,
                download_percent=100,
                repo_id=normalized,
            )
        )
        return 0
    except Exception as exc:
        emit(
            build_status_payload(
                session_id=session_id,
                model_name=normalized,
                status="failed",
                progress=0,
                message=str(exc),
                stage_name="failed",
                stage_index=0,
                stage_total=0,
                download_percent=0,
                repo_id=normalized,
                error=str(exc),
            )
        )
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local gary model predownload helper")
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
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
