#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import time
from contextlib import contextmanager

from huggingface_hub import hf_hub_download


class _NullTqdmStream:
    def write(self, message: str) -> int:
        return len(message) if message is not None else 0

    def flush(self) -> None:
        return None

    def isatty(self) -> bool:
        return False


_NULL_STREAM = _NullTqdmStream()


def _emit(event: dict) -> None:
    print(json.dumps(event, separators=(",", ":")), flush=True)


def _make_tqdm_class():
    from huggingface_hub.utils import tqdm as hf_tqdm

    class _CallbackTqdm(hf_tqdm):  # type: ignore[misc,valid-type]
        def __init__(self, *args, **kwargs):
            self._last_emit_t = 0.0
            self._last_emit_pct = -1
            self._last_emit_bytes = 0
            kwargs.setdefault("file", _NULL_STREAM)
            kwargs.setdefault("leave", False)
            kwargs["disable"] = False
            super().__init__(*args, **kwargs)
            self._emit(force=True)

        def update(self, n=1):
            out = super().update(n)
            self._emit()
            return out

        def refresh(self, *args, **kwargs):
            out = super().refresh(*args, **kwargs)
            self._emit()
            return out

        def set_description(self, desc=None, refresh=True):
            out = super().set_description(desc, refresh=refresh)
            self._emit(force=True)
            return out

        def close(self):
            self._emit(force=True)
            return super().close()

        def display(self, msg=None, pos=None):
            return None

        def _emit(self, force: bool = False) -> None:
            now = time.time()
            total = int(getattr(self, "total", 0) or 0)
            downloaded = int(getattr(self, "n", 0) or 0)
            percent = int((downloaded / total) * 100) if total > 0 else 0

            if not force:
                if (now - self._last_emit_t) < 0.25:
                    return
                pct_delta_ok = self._last_emit_pct < 0 or abs(percent - self._last_emit_pct) >= 1
                bytes_delta_ok = (downloaded - self._last_emit_bytes) >= (1 * 1024 * 1024)
                if not (pct_delta_ok or bytes_delta_ok):
                    return

            self._last_emit_t = now
            self._last_emit_pct = percent
            self._last_emit_bytes = downloaded

            speed_bps = 0.0
            try:
                speed_bps = float((getattr(self, "format_dict", {}) or {}).get("rate") or 0.0)
            except Exception:
                speed_bps = 0.0

            _emit({
                "event": "progress",
                "downloaded_bytes": downloaded,
                "total_bytes": total,
                "percent": max(0, min(100, percent)),
                "speed_bps": max(0.0, speed_bps),
            })

    return _CallbackTqdm


@contextmanager
def _patched_tqdm_if_needed(tqdm_class):
    supports_tqdm_class = "tqdm_class" in inspect.signature(hf_hub_download).parameters
    if supports_tqdm_class:
        yield {"tqdm_class": tqdm_class}
        return

    try:
        import huggingface_hub.file_download as file_download
    except Exception:
        yield {}
        return

    original_tqdm = getattr(file_download, "tqdm", None)
    if original_tqdm is None:
        yield {}
        return

    file_download.tqdm = tqdm_class
    try:
        yield {}
    finally:
        file_download.tqdm = original_tqdm


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--filename", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    _emit({
        "event": "start",
        "repo_id": args.repo_id,
        "filename": args.filename,
    })

    try:
        tqdm_class = _make_tqdm_class()
        with _patched_tqdm_if_needed(tqdm_class) as extra_kwargs:
            file_path = hf_hub_download(
                repo_id=args.repo_id,
                filename=args.filename,
                revision=args.revision,
                force_download=bool(args.force_download),
                **extra_kwargs,
            )
        _emit({
            "event": "complete",
            "repo_id": args.repo_id,
            "filename": args.filename,
            "path": file_path,
        })
        return 0
    except Exception as exc:
        _emit({
            "event": "error",
            "repo_id": args.repo_id,
            "filename": args.filename,
            "error": str(exc),
        })
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
