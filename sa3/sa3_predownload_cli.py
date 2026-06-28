#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import threading
import time

from api import (
    SA3_MODEL_LINKS,
    build_predownload_queue_status,
    read_predownload_session,
    run_sa3_predownload_task,
    sa3_inventory_rows,
    upsert_predownload_session,
)


def emit(payload: dict) -> None:
    print(json.dumps(payload), flush=True)


def inventory_command() -> int:
    emit(
        {
            "success": True,
            "known_models": sa3_inventory_rows(),
            "gate_links": SA3_MODEL_LINKS,
        }
    )
    return 0


def session_response(session_id: str) -> dict | None:
    session = read_predownload_session(session_id)
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
    }


def download_required_command() -> int:
    session_id = "local-sa3-predownload"
    target_label = "required sa3 models"
    payload = {"target_type": "required"}
    upsert_predownload_session(
        session_id,
        status="warming",
        progress=0,
        queue_status=build_predownload_queue_status(
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
        target=run_sa3_predownload_task,
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
                        "queue_status": build_predownload_queue_status(
                            status="failed",
                            message="predownload session not found",
                            target=target_label,
                            stage_name="failed",
                            stage_index=0,
                            stage_total=0,
                            download_percent=0,
                        ),
                        "error": "predownload session not found",
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local sa3 model predownload helper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("inventory")
    subparsers.add_parser("download-required")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "inventory":
        return inventory_command()
    if args.command == "download-required":
        return download_required_command()

    parser.error(f"Unsupported command {args.command!r}")
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
