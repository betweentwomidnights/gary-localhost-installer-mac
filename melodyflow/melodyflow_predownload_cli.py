#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time

from localhost_melodyflow import (
    MELODYFLOW_MODEL_REPO,
    MELODYFLOW_REQUIRED_FILES,
    _build_model_queue_status,
    _get_model_predownload_session,
    _model_catalog_payload,
    _model_download_status_for,
    _run_model_predownload,
    _upsert_model_predownload_session,
)


def emit(payload: dict) -> None:
    print(json.dumps(payload), flush=True)


def catalog_command() -> int:
    emit(
        {
            "success": True,
            "models": _model_catalog_payload(),
            "updated_at": time.time(),
        }
    )
    return 0


def status_command() -> int:
    emit(
        {
            "success": True,
            "models": {
                MELODYFLOW_MODEL_REPO: _model_download_status_for(MELODYFLOW_MODEL_REPO),
            },
            "updated_at": time.time(),
        }
    )
    return 0


def session_response(session_id: str) -> dict | None:
    session_data = _get_model_predownload_session(session_id)
    if not session_data:
        return None

    response = {
        "success": True,
        "session_id": session_id,
        "model_name": session_data.get("model_name"),
        "status": str(session_data.get("status") or "unknown"),
        "progress": max(0, min(100, int(session_data.get("progress") or 0))),
        "queue_status": session_data.get("queue_status")
        if isinstance(session_data.get("queue_status"), dict)
        else {},
        "updated_at": float(session_data.get("updated_at") or time.time()),
    }
    if response["status"] == "failed":
        response["success"] = False
        response["error"] = str(session_data.get("error") or "Unknown error")
    else:
        response["error"] = None
    return response


def download_command(model_name: str) -> int:
    normalized = model_name.strip()
    if normalized != MELODYFLOW_MODEL_REPO:
        emit(
            {
                "success": False,
                "error": f"Unknown model '{model_name}'",
                "updated_at": time.time(),
            }
        )
        return 2

    session_id = "local-melodyflow-predownload"
    stage_total = len(MELODYFLOW_REQUIRED_FILES)
    _upsert_model_predownload_session(
        session_id,
        model_name=normalized,
        status="queued",
        progress=0,
        queue_status=_build_model_queue_status(
            status="queued",
            message="queued for download",
            model_name=normalized,
            stage_index=0,
            stage_total=stage_total,
            download_percent=0,
        ),
        error=None,
    )
    _run_model_predownload(session_id, normalized)

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
                        "model_name": normalized,
                        "status": "failed",
                        "progress": 0,
                        "queue_status": _build_model_queue_status(
                            status="failed",
                            message="predownload session not found",
                            model_name=normalized,
                            stage_index=0,
                            stage_total=stage_total,
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

        status = response["status"]
        if status == "completed":
            return 0
        if status == "failed":
            return 1

        time.sleep(0.25)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local melodyflow model predownload helper")
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
