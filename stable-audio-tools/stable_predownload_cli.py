#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import io
import json
import threading
import time
import warnings

from huggingface_hub import list_repo_files

warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API\..*",
    category=UserWarning,
)

with contextlib.redirect_stdout(io.StringIO()):
    from api import (
        _build_pretrained_inventory_rows,
        _build_stable_predownload_queue_status,
        _is_repo_file_cached,
        _list_cached_repo_checkpoints,
        _read_stable_predownload_session,
        _resolve_hf_hub_cache_dir,
        _run_stable_predownload_task,
        _upsert_stable_predownload_session,
    )


def emit(payload: dict) -> None:
    print(json.dumps(payload), flush=True)


def inventory_command(finetune_repo: str, checkpoints: list[str]) -> int:
    checkpoint_names = [value.strip() for value in checkpoints if value.strip()]
    cached_finetunes: list[str] = []
    finetune_rows: list[dict] = []

    repo = finetune_repo.strip()
    if repo:
        if not checkpoint_names:
            checkpoint_names = _list_cached_repo_checkpoints(repo)
        finetune_rows = [
            {
                "name": checkpoint_name,
                "downloaded": _is_repo_file_cached(repo, checkpoint_name),
            }
            for checkpoint_name in checkpoint_names
        ]
        cached_finetunes = [
            checkpoint_name for checkpoint_name in checkpoint_names
            if _is_repo_file_cached(repo, checkpoint_name)
        ]

    emit(
        {
            "success": True,
            "known_models": _build_pretrained_inventory_rows(),
            "finetune_repo": repo,
            "finetune_checkpoints": finetune_rows,
            "cached_finetunes": cached_finetunes,
            "cache_root": str(_resolve_hf_hub_cache_dir()),
            "updated_at": time.time(),
        }
    )
    return 0


def checkpoints_command(finetune_repo: str) -> int:
    repo = finetune_repo.strip()
    if not repo:
        emit({"success": False, "error": "finetune_repo is required", "updated_at": time.time()})
        return 2

    try:
        all_files = list_repo_files(repo_id=repo, repo_type="model")
    except Exception as exc:
        emit(
            {
                "success": False,
                "repo": repo,
                "error": f"Could not access repository: {exc}",
                "hint": "Check that the repository exists and is public",
                "updated_at": time.time(),
            }
        )
        return 1

    checkpoints = sorted([path for path in all_files if path.endswith(".ckpt")], key=str.lower)
    if not checkpoints:
        emit(
            {
                "success": False,
                "repo": repo,
                "error": "No .ckpt checkpoint files found in repository",
                "updated_at": time.time(),
            }
        )
        return 1

    emit(
        {
            "success": True,
            "repo": repo,
            "checkpoints": checkpoints,
            "count": len(checkpoints),
            "updated_at": time.time(),
        }
    )
    return 0


def session_response(session_id: str) -> dict | None:
    session = _read_stable_predownload_session(session_id)
    if session is None:
        return None

    return {
        "success": True,
        "session_id": session_id,
        "model_name": session.get("target"),
        "status": session.get("status", "unknown"),
        "progress": int(session.get("progress", 0)),
        "queue_status": session.get("queue_status", {}),
        "error": session.get("error"),
        "updated_at": time.time(),
    }


def run_download_command(payload: dict, target_label: str) -> int:
    session_id = "local-stable-audio-predownload"
    _upsert_stable_predownload_session(
        session_id,
        status="warming",
        progress=0,
        queue_status=_build_stable_predownload_queue_status(
            status="warming",
            message=f"preparing download for {target_label}",
            target=target_label,
            stage_name="prepare",
            stage_index=0,
            stage_total=0,
            download_percent=0,
        ),
        error=None,
        target=target_label,
    )

    threading.Thread(
        target=_run_stable_predownload_task,
        args=(session_id, payload),
        daemon=True,
    ).start()

    last_emitted: str | None = None
    missing_session_deadline = time.time() + 10.0

    while True:
        response = session_response(session_id)
        if response is None:
            if time.time() >= missing_session_deadline:
                emit(
                    {
                        "success": False,
                        "session_id": session_id,
                        "model_name": target_label,
                        "status": "failed",
                        "progress": 0,
                        "queue_status": _build_stable_predownload_queue_status(
                            status="failed",
                            message="predownload session not found",
                            target=target_label,
                            stage_name="failed",
                            stage_index=0,
                            stage_total=0,
                            download_percent=0,
                        ),
                        "error": "predownload session not found",
                        "updated_at": time.time(),
                    }
                )
                return 1
            time.sleep(0.25)
            continue

        serialized = json.dumps(response, sort_keys=True)
        if serialized != last_emitted:
            emit(response)
            last_emitted = serialized

        status = str(response.get("status") or "unknown")
        if status == "completed":
            return 0
        if status == "failed":
            return 1

        time.sleep(0.25)


def download_pretrained_command(repo_id: str, require_token: bool) -> int:
    normalized_repo = repo_id.strip() or "stabilityai/stable-audio-open-small"
    return run_download_command(
        {
            "target_type": "pretrained",
            "repo_id": normalized_repo,
            "require_token": require_token,
        },
        normalized_repo,
    )


def download_finetune_command(
    finetune_repo: str,
    finetune_checkpoint: str,
    base_repo: str,
    require_token: bool,
) -> int:
    repo = finetune_repo.strip()
    checkpoint = finetune_checkpoint.strip()
    if not repo or not checkpoint:
        emit(
            {
                "success": False,
                "error": "finetune_repo and finetune_checkpoint are required",
                "updated_at": time.time(),
            }
        )
        return 2

    target_label = f"{repo}/{checkpoint}"
    return run_download_command(
        {
            "target_type": "finetune",
            "finetune_repo": repo,
            "finetune_checkpoint": checkpoint,
            "base_repo": base_repo.strip() or "stabilityai/stable-audio-open-small",
            "require_token": require_token,
        },
        target_label,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local stable-audio model predownload helper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory_parser = subparsers.add_parser("inventory")
    inventory_parser.add_argument("--finetune-repo", default="")
    inventory_parser.add_argument("--checkpoint", action="append", default=[])

    checkpoints_parser = subparsers.add_parser("checkpoints")
    checkpoints_parser.add_argument("--finetune-repo", required=True)

    pretrained_parser = subparsers.add_parser("download-pretrained")
    pretrained_parser.add_argument("--repo-id", default="stabilityai/stable-audio-open-small")
    pretrained_parser.add_argument("--require-token", action="store_true")

    finetune_parser = subparsers.add_parser("download-finetune")
    finetune_parser.add_argument("--finetune-repo", required=True)
    finetune_parser.add_argument("--finetune-checkpoint", required=True)
    finetune_parser.add_argument("--base-repo", default="stabilityai/stable-audio-open-small")
    finetune_parser.add_argument("--require-token", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "inventory":
        return inventory_command(args.finetune_repo, args.checkpoint)
    if args.command == "checkpoints":
        return checkpoints_command(args.finetune_repo)
    if args.command == "download-pretrained":
        return download_pretrained_command(args.repo_id, args.require_token)
    if args.command == "download-finetune":
        return download_finetune_command(
            args.finetune_repo,
            args.finetune_checkpoint,
            args.base_repo,
            args.require_token,
        )

    parser.error(f"Unsupported command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
