from __future__ import annotations

import time
from typing import Callable, Optional, TypedDict


class DownloadProgressEvent(TypedDict):
    repo_id: str
    downloaded_bytes: int
    total_bytes: int
    percent: int
    desc: str
    done: bool


class _NullTqdmStream:
    """A write sink that discards tqdm output without buffering in memory."""

    def write(self, message: str) -> int:
        return len(message) if message is not None else 0

    def flush(self) -> None:
        return None

    def isatty(self) -> bool:
        return False


_NULL_TQDM_STREAM = _NullTqdmStream()


def make_hf_tqdm_class(
    *,
    repo_id: str,
    on_progress: Optional[Callable[[DownloadProgressEvent], None]],
    min_interval_s: float = 0.25,
    min_percent_step: int = 1,
    min_bytes_step: int = 1 * 1024 * 1024,
    silence_output: bool = True,
):
    """
    Build a tqdm-compatible class for `huggingface_hub.snapshot_download(tqdm_class=...)`.

    We subclass `huggingface_hub.utils.tqdm.tqdm` to support the extra `name=...`
    kwarg used by the Hub internals (and to keep the global HF progress bar toggles).
    """

    try:
        from huggingface_hub.utils import tqdm as hf_tqdm
    except Exception:  # pragma: no cover
        hf_tqdm = None  # type: ignore[assignment]

    if hf_tqdm is None:
        return None

    class _CallbackTqdm(hf_tqdm):  # type: ignore[misc,valid-type]
        def __init__(self, *args, **kwargs):
            self._repo_id = repo_id
            self._on_progress = on_progress
            self._min_interval_s = float(min_interval_s)
            self._min_percent_step = int(min_percent_step)
            self._min_bytes_step = int(min_bytes_step)
            self._last_emit_t = 0.0
            self._last_emit_pct = -1
            self._last_emit_bytes = 0
            self._progress_name = kwargs.get("name")

            # Keep tqdm enabled so `n`/`total` counters still update, while
            # routing output to a sink to avoid noisy logs and buffer growth.
            if silence_output:
                kwargs.setdefault("file", _NULL_TQDM_STREAM)
                kwargs.setdefault("leave", False)
                # `huggingface_hub` may pass disable=True (often due to NOTSET log level),
                # which prevents tqdm from updating its internal counters. Force-enable
                # so progress callbacks can fire while still silencing output.
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
            # Skip formatting/writes entirely when output is silenced.
            if silence_output:
                return None
            return super().display(msg=msg, pos=pos)

        def _emit(self, force: bool = False) -> None:
            if self._on_progress is None:
                return

            unit = getattr(self, "unit", None)

            now = time.time()
            total = int(getattr(self, "total", 0) or 0)
            downloaded = int(getattr(self, "n", 0) or 0)
            pct = int((downloaded / total) * 100) if total > 0 else 0

            if not force:
                if (now - self._last_emit_t) < self._min_interval_s:
                    return
                pct_delta_ok = (
                    self._last_emit_pct < 0
                    or abs(pct - self._last_emit_pct) >= self._min_percent_step
                )
                bytes_delta_ok = (downloaded - self._last_emit_bytes) >= self._min_bytes_step
                if not (pct_delta_ok or bytes_delta_ok):
                    return

            self._last_emit_t = now
            self._last_emit_pct = pct
            self._last_emit_bytes = downloaded

            desc = getattr(self, "desc", "") or ""
            desc_l = desc.lower().strip()
            done = desc_l.startswith("download complete")
            self._on_progress(
                {
                    "repo_id": self._repo_id,
                    "downloaded_bytes": downloaded,
                    "total_bytes": total,
                    "percent": pct,
                    "desc": desc,
                    "done": done,
                    "unit": unit,
                    "progress_name": self._progress_name,
                }
            )

    return _CallbackTqdm
