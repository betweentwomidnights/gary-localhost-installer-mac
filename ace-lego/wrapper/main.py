"""
ACE-Step wrapper service — lego (stem) + complete (continuation) + cover (remix) modes.

Async submit/poll architecture matching the existing JUCE frontend pattern.

Endpoints:
  POST /lego                     Submit a lego stem job → returns task_id
  GET  /lego/status/{task_id}    Poll lego progress/completion

  POST /complete                 Submit a continuation job → returns task_id
  GET  /complete/status/{task_id} Poll continuation progress/completion

  POST /cover                    Submit a cover/remix job → returns task_id
  GET  /cover/status/{task_id}   Poll cover progress/completion

  GET  /health                   Wrapper + backend health

Environment variables:
  ACESTEP_BACKEND           "local" or "spark"              (default "local")
  ACESTEP_URL               URL of ACE-Step api_server       (auto-set by backend mode)
  ACESTEP_MANAGE_LIFECYCLE  "true"/"false" load/unload mgmt  (default: true if local)
  QUEUE_URL                 URL of gpu-queue-service          (default http://gpu-queue:8085)
  WRAPPER_PORT              Port this service listens on      (default 8003)
  ACESTEP_MAX_CONCURRENT    Max simultaneous generations      (default 1)
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import subprocess
import tempfile
import time
import wave
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional
from uuid import uuid4

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ACESTEP_BACKEND = os.getenv("ACESTEP_BACKEND", "local").lower()

if ACESTEP_BACKEND == "spark":
    ACESTEP_URL = os.getenv("ACESTEP_URL", "http://100.120.181.124:8001").rstrip("/")
    _default_lifecycle = os.getenv("ACESTEP_MANAGE_LIFECYCLE", "false")
else:
    ACESTEP_URL = os.getenv("ACESTEP_URL", "http://localhost:8001").rstrip("/")
    _default_lifecycle = os.getenv("ACESTEP_MANAGE_LIFECYCLE", "true")

MANAGE_MODEL_LIFECYCLE = _default_lifecycle.lower() == "true"

# Model variant configs for turbo/base switching
DEFAULT_STARTUP_CONFIG = (os.getenv("ACESTEP_CONFIG_PATH") or "acestep-v15-base").strip()
ACESTEP_BASE_CONFIG = (os.getenv("ACESTEP_BASE_CONFIG") or DEFAULT_STARTUP_CONFIG).strip()
ACESTEP_TURBO_CONFIG = (os.getenv("ACESTEP_TURBO_CONFIG") or "acestep-v15-turbo").strip()

# Track whether the backend starts with a model already loaded.
# When ACESTEP_INIT_LLM is "false" or ACESTEP_NO_INIT is set, the backend
# starts empty and the wrapper must explicitly /v1/load before the first job.
_backend_starts_loaded = os.getenv("ACESTEP_INIT_LLM", "true").strip().lower() not in {"false", "0", "no"}
_current_model: Optional[str] = DEFAULT_STARTUP_CONFIG if _backend_starts_loaded else None

QUEUE_URL = os.getenv("QUEUE_URL", "http://gpu-queue:8085").rstrip("/")
WRAPPER_PORT = int(os.getenv("WRAPPER_PORT", "8003"))
MAX_CONCURRENT = int(os.getenv("ACESTEP_MAX_CONCURRENT", "1"))
EFFECTIVE_MAX_CONCURRENT = 1 if MANAGE_MODEL_LIFECYCLE else MAX_CONCURRENT

# Generation constants
INFERENCE_STEPS = 50
POLL_INTERVAL = 1.5
GENERATION_TIMEOUT = int(os.getenv("CAREY_GENERATION_TIMEOUT", "600"))
JOB_TTL = 3600

# Cover mode uses turbo model with locked inference params
COVER_INFERENCE_STEPS = 8
COVER_GUIDANCE_SCALE = 1.0

# Default captions per track type (lego mode only)
TRACK_CAPTIONS = {
    "vocals":         "soulful indie vocalist, warm, wordless melody, expressive, intimate",
    "backing_vocals": "background vocals, close harmony, wordless, warm, following the lead vocal",
    "drums":          "live acoustic drum kit, tight kick and snare, brushed hi-hats, warm",
    "bass":           "electric bass, warm fingerstyle, rhythmic, supportive",
    "guitar":         "acoustic guitar, fingerpicked, warm, rhythmic",
    "piano":          "piano, expressive, warm, melodic",
    "strings":        "string ensemble, lush, warm, cinematic",
    "synth":          "analog synth pad, warm, atmospheric",
    "keyboard":       "electric piano, warm, smooth",
    "percussion":     "percussion, shaker, tambourine, tight groove",
    "brass":          "brass section, warm, expressive",
    "woodwinds":      "woodwind ensemble, warm, airy, melodic",
}

ALLOWED_TRACKS = set(TRACK_CAPTIONS.keys())


# ---------------------------------------------------------------------------
# Job tracking
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    QUEUED = "queued"
    LOADING = "loading"
    COMPRESSING = "compressing"
    SUBMITTING = "submitting"
    GENERATING = "generating"
    DOWNLOADING = "downloading"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    task_id: str
    task_type: str               # "lego" or "complete"
    bpm: int
    created_at: float = field(default_factory=time.time)
    status: JobStatus = JobStatus.QUEUED
    progress: int = 0
    progress_text: str = "queued"
    ace_task_id: Optional[str] = None
    audio_b64: Optional[str] = None
    audio_format: str = "wav"
    duration: Optional[float] = None       # actual source duration (probed)
    target_duration: Optional[float] = None # user-requested output duration (complete)
    track_name: Optional[str] = None       # lego only
    error: Optional[str] = None


_jobs: dict[str, Job] = {}
_generation_semaphore: asyncio.Semaphore | None = None


def _cleanup_old_jobs():
    now = time.time()
    expired = [
        tid for tid, job in _jobs.items()
        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED)
        and (now - job.created_at) > JOB_TTL
    ]
    for tid in expired:
        del _jobs[tid]


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class LegoRequest(BaseModel):
    """Lego mode: generate a single stem over existing audio."""
    audio_data: str = Field(..., description="Base64-encoded audio")
    track_name: str = Field(..., description="Track type: vocals, drums, bass, etc.")
    bpm: int = Field(..., description="BPM of the source audio")
    caption: str = Field("", description="Override default caption for track type")
    lyrics: str = Field("", description="Optional lyrics with structure tags like [Verse 1]")
    language: str = Field("en", description="Language code for lyrics vocalization (e.g. en, ja, zh)")
    guidance_scale: float = Field(7.0, description="CFG scale. 7-9 recommended")
    inference_steps: int = Field(50, description="Diffusion steps. 50 default")
    time_signature: str = Field("4", description="Time signature numerator")
    batch_size: int = Field(1, description="Number of candidates")
    audio_format: str = Field("wav", description="Output format: wav, mp3, flac")


class CompleteRequest(BaseModel):
    """Complete mode: continue/extend audio with full arrangement."""
    audio_data: str = Field(..., description="Base64-encoded source audio")
    bpm: int = Field(..., description="BPM of the source audio")
    audio_duration: float = Field(..., description="Target output duration in seconds")
    caption: str = Field("", description="Style caption — longer = stronger steer")
    lyrics: str = Field("", description="Optional lyrics with structure tags")
    language: str = Field("en", description="Language code for lyrics vocalization (e.g. en, ja, zh)")
    key_scale: str = Field("", description="Optional key/scale e.g. 'F minor', 'C major'")
    guidance_scale: float = Field(7.0, description="CFG scale. 7-9 recommended")
    inference_steps: int = Field(50, description="Diffusion steps. 50 default")
    use_src_as_ref: bool = Field(False, description="Pass source as ref_audio for timbre anchoring")
    time_signature: str = Field("4", description="Time signature numerator")
    batch_size: int = Field(1, description="Number of candidates")
    audio_format: str = Field("wav", description="Output format: wav, mp3, flac")


class CoverRequest(BaseModel):
    """Cover/remix mode: restyle audio guided by caption while preserving structure."""
    audio_data: str = Field(..., description="Base64-encoded source audio")
    bpm: int = Field(..., description="BPM of the source audio")
    caption: str = Field(..., description="Style caption driving the remix")
    lyrics: str = Field("", description="Optional lyrics with structure tags")
    language: str = Field("en", description="Language code for lyrics vocalization (e.g. en, ja, zh)")
    key_scale: str = Field("", description="Optional key/scale e.g. 'F minor', 'C major'")
    cover_noise_strength: float = Field(0.2, description="0=pure noise, 1=closest to source. Recommended 0.2")
    audio_cover_strength: float = Field(0.3, description="Fraction of steps using semantic codes. 0.3 instrumental, 0.5-0.7 vocals")
    guidance_scale: float = Field(1.0, description="Cover mode is locked to CFG 1.0 for turbo generation")
    inference_steps: int = Field(8, description="Cover mode is locked to 8 diffusion steps for turbo generation")
    use_src_as_ref: bool = Field(False, description="Pass source as ref_audio for subtler transformation")
    time_signature: str = Field("4", description="Time signature numerator")
    batch_size: int = Field(1, description="Number of candidates")
    audio_format: str = Field("wav", description="Output format: wav, mp3, flac")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="ACE-Step Wrapper", version="0.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _init():
    global _generation_semaphore
    _generation_semaphore = asyncio.Semaphore(EFFECTIVE_MAX_CONCURRENT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _probe_duration(path: str) -> Optional[float]:
    # First try ffprobe for broad format support.
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=10,
        )
        out = result.stdout.strip()
        if result.returncode == 0 and out:
            value = float(out)
            if value > 0:
                return value
    except Exception:
        pass

    # Fallback for local mac installs where ffprobe is unavailable in PATH.
    # JUCE sends WAV buffers, so this still covers normal localhost flows.
    try:
        with wave.open(path, "rb") as wf:
            frame_rate = wf.getframerate()
            frame_count = wf.getnframes()
            if frame_rate > 0 and frame_count >= 0:
                return frame_count / float(frame_rate)
    except Exception:
        pass

    return None


def _compress_for_proxy(input_path: str) -> str:
    """Skip compression — send raw audio to backend."""
    return input_path


_ANSI_RE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences and control chars from text."""
    stripped = _ANSI_RE.sub('', text)
    return stripped.strip()


def _parse_progress_from_text(text: str, expected_total: Optional[int] = None) -> Optional[int]:
    """Parse step-style progress like ``2/50`` from text payloads.

    We intentionally ignore plain percentages (e.g. ``99%``) because upstream
    logs can briefly emit stale/noisy percent lines that do not map to this job.
    """
    if not text:
        return None
    best_pct: Optional[int] = None
    for match in re.finditer(r'(\d+)\s*/\s*(\d+)', text):
        current = int(match.group(1))
        total = int(match.group(2))
        if total <= 0 or total > 200 or current < 0 or current > total:
            continue
        if expected_total is not None and expected_total > 0 and abs(total - expected_total) > 2:
            continue
        pct = min(int((current / total) * 100), 99)
        best_pct = pct if best_pct is None else max(best_pct, pct)
    return best_pct


def _coerce_progress_percent(value) -> Optional[int]:
    """Normalize API progress values to integer percent (0..99)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        value_f = float(value)
    except Exception:
        return None
    if value_f <= 1.0:
        value_f *= 100.0
    return max(0, min(int(value_f), 99))


def _extract_query_result_payload(raw_payload) -> list[dict]:
    """Decode /query_result 'result' payload into a list of dicts."""
    payload = raw_payload
    if isinstance(payload, str):
        text = payload.strip()
        if not text:
            return []
        try:
            payload = json.loads(text)
        except Exception:
            return []
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def _remap_structured_progress(
    stage_text: str,
    percent: Optional[int],
    *,
    saw_step_progress: bool = False,
) -> Optional[int]:
    """Map ACE-Step internal phase progress to user-facing generation percent."""
    if percent is None:
        return None
    stage = (stage_text or "").strip().lower()
    if stage in {"running", "generating..."}:
        return 1
    if "preparing inputs" in stage:
        return 1
    if "generating music" in stage:
        if not saw_step_progress:
            return 1
        # Internal estimator spans ~52..79 during diffusion startup/progress.
        if percent >= 52:
            remapped = 1 + int((percent - 52) * 88 / 27)
            return max(1, min(remapped, 89))
        return max(1, min(percent, 89))
    if "decoding audio" in stage:
        return 95
    if "preparing audio data" in stage:
        return 98
    return percent


async def _acquire_gpu_token(client: httpx.AsyncClient, session_id: str) -> bool:
    try:
        resp = await client.post(
            f"{QUEUE_URL}/tasks",
            json={"session_id": session_id, "tokens": 1000},
            timeout=10,
        )
        return resp.status_code in (200, 201)
    except Exception:
        return False


async def _release_gpu_token(client: httpx.AsyncClient, session_id: str) -> None:
    try:
        await client.post(
            f"{QUEUE_URL}/task/status",
            json={"session_id": session_id, "status": "completed"},
            timeout=10,
        )
    except Exception:
        pass


async def _load_model(client: httpx.AsyncClient, config_path: str) -> str:
    global _current_model
    resp = await client.post(
        f"{ACESTEP_URL}/v1/load",
        params={"config_path": config_path},
        timeout=180,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"ACE-Step /v1/load failed for {config_path}: {resp.text}")
    body = resp.json()
    loaded = ((body.get("data") or {}).get("model") or config_path).strip()
    _current_model = loaded
    return loaded


async def _unload_model(client: httpx.AsyncClient) -> None:
    global _current_model
    resp = await client.post(f"{ACESTEP_URL}/v1/unload", timeout=60)
    _current_model = None
    if resp.status_code != 200:
        raise RuntimeError(f"ACE-Step /v1/unload failed: {resp.text}")


def _required_config_for_task(task_type: str) -> str:
    """Return the model config required for a given task type.
    Cover mode uses turbo; lego and complete use base."""
    if task_type == "cover":
        return ACESTEP_TURBO_CONFIG
    return ACESTEP_BASE_CONFIG


def _effective_guidance_scale(task_type: str, requested: float) -> float:
    if task_type == "cover":
        return COVER_GUIDANCE_SCALE
    return requested


def _effective_inference_steps(task_type: str, requested: int) -> int:
    if task_type == "cover":
        return COVER_INFERENCE_STEPS
    return requested


async def _ensure_required_model(client: httpx.AsyncClient, job: Job) -> None:
    """Switch between turbo and base models if needed."""
    global _current_model

    if not MANAGE_MODEL_LIFECYCLE:
        return

    required = _required_config_for_task(job.task_type)
    if _current_model == required:
        return

    job.status = JobStatus.LOADING
    job.progress = max(job.progress, 3)
    job.progress_text = f"loading {required}..."

    if _current_model is not None:
        await _unload_model(client)
        _current_model = None

    _current_model = await _load_model(client, required)


# ---------------------------------------------------------------------------
# Generalized background generation
# ---------------------------------------------------------------------------

def _build_form_data(job: Job, req, send_path: str) -> dict:
    """Build the multipart form data dict for /release_task.

    Works for lego, complete, and cover task types.
    """
    if job.task_type == "lego":
        effective_caption = req.caption.strip() or TRACK_CAPTIONS.get(req.track_name, "")
        audio_duration = str(job.duration)
    elif job.task_type == "cover":
        effective_caption = req.caption.strip()
        audio_duration = str(job.duration)  # cover outputs same length as source
    else:  # complete
        effective_caption = req.caption.strip()
        audio_duration = str(req.audio_duration)

    effective_guidance = _effective_guidance_scale(job.task_type, req.guidance_scale)
    effective_steps = _effective_inference_steps(job.task_type, req.inference_steps)

    data = {
        "task_type":        job.task_type,
        "caption":          effective_caption,
        "lyrics":           req.lyrics,
        "language":         req.language,
        "bpm":              str(req.bpm),
        "time_signature":   req.time_signature,
        "guidance_scale":   str(effective_guidance),
        "thinking":         "false",
        "use_cot_caption":  "false",
        "use_cot_language": "false",
        "batch_size":       str(req.batch_size),
        "audio_duration":   audio_duration,
        "audio_format":     req.audio_format,
    }

    if job.task_type == "lego":
        data["track_name"] = req.track_name
        data["repainting_start"] = "0.0"
        data["repainting_end"] = "-1"
        data["inference_steps"] = str(effective_steps)

    elif job.task_type == "cover":
        data["cover_noise_strength"] = str(req.cover_noise_strength)
        data["audio_cover_strength"] = str(req.audio_cover_strength)
        data["inference_steps"] = str(effective_steps)

    else:  # complete
        data["inference_steps"] = str(effective_steps)

    # key_scale for cover and complete (optional, user-provided)
    if hasattr(req, 'key_scale') and req.key_scale.strip():
        data["key_scale"] = req.key_scale.strip()

    return data


async def _run_generation(job: Job, req):
    """Background task: handles the full generation lifecycle for any task type."""
    raw_audio_path = None
    compressed_path = None
    session_id = f"acestep-{int(time.time() * 1000)}"

    async with _generation_semaphore:
        try:
            # --- Decode audio ---
            job.status = JobStatus.COMPRESSING
            job.progress_text = "preparing audio..."
            try:
                audio_bytes = base64.b64decode(req.audio_data)
            except Exception:
                raise RuntimeError("Invalid base64 audio data")

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                raw_audio_path = tmp.name

            job.duration = _probe_duration(raw_audio_path)
            if job.duration is None:
                if job.task_type == "complete":
                    # Complete mode already carries explicit target duration.
                    # Source probing is best-effort only.
                    job.duration = float(req.audio_duration)
                else:
                    raise RuntimeError("Could not determine audio duration")

            # For complete mode, store the user's target duration
            if job.task_type == "complete":
                job.target_duration = req.audio_duration

            # Compress for Spark proxy
            send_path = _compress_for_proxy(raw_audio_path)
            if send_path != raw_audio_path:
                compressed_path = send_path

            async with httpx.AsyncClient() as client:
                # --- GPU queue + model load (local mode) ---
                use_gpu_queue = MANAGE_MODEL_LIFECYCLE and ACESTEP_BACKEND == "spark"
                if MANAGE_MODEL_LIFECYCLE:
                    if use_gpu_queue:
                        job.status = JobStatus.LOADING
                        job.progress_text = "acquiring gpu..."
                        ok = await _acquire_gpu_token(client, session_id)
                        if not ok:
                            raise RuntimeError("GPU queue unavailable")

                try:
                    if MANAGE_MODEL_LIFECYCLE:
                        await _ensure_required_model(client, job)

                    # --- Submit to ace-step ---
                    job.status = JobStatus.SUBMITTING
                    job.progress_text = "submitting to ace-step..."

                    filename = Path(send_path).name
                    mime = "audio/flac" if filename.endswith(".flac") else \
                            "audio/ogg" if filename.endswith(".ogg") else "audio/wav"
                    form_data = _build_form_data(job, req, send_path)

                    # Build file uploads — ctx_audio always, ref_audio when use_src_as_ref
                    use_ref = getattr(req, 'use_src_as_ref', False)

                    with open(send_path, "rb") as fh:
                        files = [("ctx_audio", (filename, fh, mime))]

                        ref_fh = None
                        if use_ref:
                            ref_fh = open(send_path, "rb")
                            files.append(("ref_audio", (filename, ref_fh, mime)))

                        try:
                            resp = await client.post(
                                f"{ACESTEP_URL}/release_task",
                                data=form_data, files=files, timeout=120,
                            )
                        finally:
                            if ref_fh:
                                ref_fh.close()

                    if resp.status_code != 200:
                        raise RuntimeError(f"/release_task failed: {resp.text}")

                    body = resp.json()
                    job.ace_task_id = body["data"]["task_id"]

                    # --- Poll ace-step for progress ---
                    job.status = JobStatus.GENERATING
                    job.progress_text = "generating..."

                    # For time-based progress estimation when ace-step
                    # doesn't report steps (e.g. cover mode)
                    gen_start_time = time.time()
                    inference_steps = _effective_inference_steps(
                        job.task_type,
                        getattr(req, 'inference_steps', INFERENCE_STEPS),
                    )
                    est_seconds_per_step = 0.35  # rough baseline

                    deadline = time.time() + GENERATION_TIMEOUT
                    saw_step_progress = False
                    while time.time() < deadline:
                        resp = await client.post(
                            f"{ACESTEP_URL}/query_result",
                            json={"task_id_list": [job.ace_task_id]},
                            timeout=15,
                        )
                        result = resp.json()["data"][0]
                        ace_status = result["status"]

                        got_real_progress = False
                        has_structured_status = False
                        parsed_text_progress = None
                        stage_text = ""

                        # Prefer structured per-job payload from /query_result `result`.
                        # Top-level `progress_text` can be noisy/stale in some deployments.
                        payload_items = _extract_query_result_payload(result.get("result"))
                        if payload_items:
                            payload0 = payload_items[0]

                            stage_text = _strip_ansi(str(payload0.get("stage") or ""))
                            if stage_text:
                                parsed_stage_progress = _parse_progress_from_text(
                                    stage_text,
                                    expected_total=inference_steps,
                                )
                                if parsed_stage_progress is not None:
                                    parsed_text_progress = parsed_stage_progress
                                    saw_step_progress = True
                                    job.progress = parsed_stage_progress
                                    got_real_progress = True
                                    job.progress_text = stage_text
                                    has_structured_status = True

                            nested_progress = _coerce_progress_percent(payload0.get("progress"))
                            nested_progress = _remap_structured_progress(
                                stage_text,
                                nested_progress,
                                saw_step_progress=saw_step_progress,
                            )
                            if nested_progress is not None and parsed_text_progress is None:
                                job.progress = nested_progress
                                got_real_progress = True

                            if stage_text:
                                if stage_text == "running":
                                    stage_text = "generating..."
                                elif stage_text == "succeeded":
                                    stage_text = "downloading audio..."
                                elif stage_text == "failed":
                                    stage_text = "generation failed"
                                if parsed_text_progress is None:
                                    job.progress_text = stage_text
                                    has_structured_status = True

                        # Parse step-style text progress from top-level progress_text as an
                        # override source, even when structured payload exists. In practice,
                        # DiT step counters are emitted here on some backends.
                        progress_text = _strip_ansi(result.get("progress_text") or "")
                        if progress_text:
                            relaxed_step_progress = _parse_progress_from_text(progress_text)
                            top_level_step_progress = _parse_progress_from_text(
                                progress_text,
                                expected_total=inference_steps,
                            )
                            # First textual progress sample should not instantly jump to the end.
                            if not saw_step_progress and top_level_step_progress is not None and top_level_step_progress >= 90:
                                top_level_step_progress = None
                            if top_level_step_progress is not None:
                                # Keep step progress monotonic once started.
                                if (not saw_step_progress) or (top_level_step_progress >= job.progress):
                                    parsed_text_progress = top_level_step_progress
                                    saw_step_progress = True
                                    job.progress = top_level_step_progress
                                    got_real_progress = True
                                    job.progress_text = progress_text
                                    has_structured_status = True
                            elif relaxed_step_progress is not None:
                                # Even when step total does not match expected_total exactly,
                                # surface the text so the UI can display "x/y steps" updates.
                                job.progress_text = progress_text
                                has_structured_status = True

                        raw_progress = _coerce_progress_percent(result.get("progress"))
                        raw_progress = _remap_structured_progress(
                            stage_text,
                            raw_progress,
                            saw_step_progress=saw_step_progress,
                        )
                        if raw_progress is not None and parsed_text_progress is None and not got_real_progress:
                            job.progress = raw_progress
                            got_real_progress = True

                        # Hard guard: before real step stream starts, keep generation phase at 1%.
                        # This prevents stale/noisy upstream percent values from spiking to 99%.
                        if not saw_step_progress:
                            stage_norm = (stage_text or "").strip().lower()
                            if stage_norm in {"", "running", "generating..."} or "generating music" in stage_norm:
                                if job.progress > 1:
                                    job.progress = 1

                        # Estimate progress from elapsed time ONLY for cover mode,
                        # which sends only ANSI cursor codes instead of real step counts.
                        # Lego and complete reliably report step progress, so the fallback
                        # just causes ugly jumps when it fights with real updates.
                        if not got_real_progress and job.status == JobStatus.GENERATING \
                                and job.task_type == "cover":
                            elapsed = time.time() - gen_start_time
                            est_total = inference_steps * est_seconds_per_step
                            if est_total > 0:
                                est_pct = min(int((elapsed / est_total) * 90), 90)
                                if est_pct > 0:
                                    job.progress = est_pct
                                    job.progress_text = f"~{est_pct}% ({inference_steps} steps)"

                        if ace_status == 1:
                            break
                        if ace_status == 2:
                            error_msg = result.get("error") or "generation failed"
                            raise RuntimeError(error_msg)

                        await asyncio.sleep(POLL_INTERVAL)
                    else:
                        raise RuntimeError("Generation timed out")

                    # --- Download audio ---
                    job.status = JobStatus.DOWNLOADING
                    job.progress = 95
                    job.progress_text = "downloading audio..."

                    files_list = json.loads(result["result"])
                    if not files_list:
                        raise RuntimeError("No audio files in result")

                    file_path = files_list[0]["file"]
                    resp = await client.get(
                        f"{ACESTEP_URL}{file_path}", timeout=60,
                    )
                    if resp.status_code != 200:
                        raise RuntimeError(f"Failed to download audio: {resp.status_code}")

                    job.audio_b64 = base64.b64encode(resp.content).decode("utf-8")
                    job.status = JobStatus.COMPLETED
                    job.progress = 100
                    job.progress_text = "complete"

                finally:
                    # On localhost with lifecycle management, keep the model
                    # loaded between jobs — _ensure_required_model handles
                    # switching when the next task needs a different variant.
                    # Only release GPU queue tokens (Spark mode).
                    if MANAGE_MODEL_LIFECYCLE and use_gpu_queue:
                        await _unload_model(client)
                        await _release_gpu_token(client, session_id)

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.progress_text = f"failed: {e}"

        finally:
            for p in [raw_audio_path, compressed_path]:
                if p:
                    try:
                        os.unlink(p)
                    except Exception:
                        pass


# ---------------------------------------------------------------------------
# Shared status response builder
# ---------------------------------------------------------------------------

def _build_status_response(job: Job) -> JSONResponse:
    """Build JUCE-compatible status response for any task type."""

    # --- In progress ---
    if job.status in (
        JobStatus.QUEUED, JobStatus.LOADING, JobStatus.COMPRESSING,
        JobStatus.SUBMITTING, JobStatus.GENERATING, JobStatus.DOWNLOADING,
    ):
        status_messages = {
            JobStatus.QUEUED: "queued",
            JobStatus.LOADING: "loading model...",
            JobStatus.COMPRESSING: "preparing audio...",
            JobStatus.SUBMITTING: "submitting...",
            JobStatus.GENERATING: job.progress_text or "generating...",
            JobStatus.DOWNLOADING: "downloading result...",
        }

        return JSONResponse({
            "success": True,
            "generation_in_progress": True,
            "transform_in_progress": False,
            "progress": job.progress,
            "status": "processing",
            "queue_status": {
                "status": "queued" if job.status == JobStatus.QUEUED else "ready",
                "message": status_messages.get(job.status, "processing..."),
                "position": 0,
                "estimated_seconds": 0,
                "estimated_time": "",
            },
        })

    # --- Completed ---
    if job.status == JobStatus.COMPLETED:
        resp = {
            "success": True,
            "generation_in_progress": False,
            "transform_in_progress": False,
            "status": "completed",
            "audio_data": job.audio_b64,
            "progress": 100,
            "bpm": job.bpm,
            "duration": job.target_duration or job.duration,
            "audio_format": job.audio_format,
            "task_type": job.task_type,
        }
        if job.track_name:
            resp["track_name"] = job.track_name
        return JSONResponse(resp)

    # --- Failed ---
    return JSONResponse({
        "success": False,
        "generation_in_progress": False,
        "transform_in_progress": False,
        "status": "failed",
        "error": job.error or "Unknown error",
        "progress": 0,
    })


# ---------------------------------------------------------------------------
# Lego endpoints
# ---------------------------------------------------------------------------

@app.post("/lego")
async def lego_submit(req: LegoRequest):
    """Submit a lego stem generation. Returns task_id immediately."""
    if req.track_name not in ALLOWED_TRACKS:
        raise HTTPException(400, f"track_name must be one of {sorted(ALLOWED_TRACKS)}")

    _cleanup_old_jobs()
    task_id = str(uuid4())
    job = Job(
        task_id=task_id,
        task_type="lego",
        bpm=req.bpm,
        track_name=req.track_name,
        audio_format=req.audio_format,
    )
    _jobs[task_id] = job
    asyncio.create_task(_run_generation(job, req))

    return JSONResponse({
        "success": True,
        "task_id": task_id,
        "status": "queued",
    })


@app.get("/lego/status/{task_id}")
async def lego_status(task_id: str):
    """Poll lego generation progress."""
    job = _jobs.get(task_id)
    if not job:
        return JSONResponse({
            "success": False, "status": "failed", "error": "Unknown task_id",
        }, status_code=404)
    return _build_status_response(job)


# ---------------------------------------------------------------------------
# Complete endpoints
# ---------------------------------------------------------------------------

@app.post("/complete")
async def complete_submit(req: CompleteRequest):
    """Submit a continuation/completion generation. Returns task_id immediately."""
    if req.audio_duration < 5:
        raise HTTPException(400, "audio_duration must be at least 5 seconds")
    if req.audio_duration > 300:
        raise HTTPException(400, "audio_duration must be at most 300 seconds (5 min)")

    _cleanup_old_jobs()
    task_id = str(uuid4())
    job = Job(
        task_id=task_id,
        task_type="complete",
        bpm=req.bpm,
        target_duration=req.audio_duration,
        audio_format=req.audio_format,
    )
    _jobs[task_id] = job
    asyncio.create_task(_run_generation(job, req))

    return JSONResponse({
        "success": True,
        "task_id": task_id,
        "status": "queued",
    })


@app.get("/complete/status/{task_id}")
async def complete_status(task_id: str):
    """Poll continuation generation progress."""
    job = _jobs.get(task_id)
    if not job:
        return JSONResponse({
            "success": False, "status": "failed", "error": "Unknown task_id",
        }, status_code=404)
    return _build_status_response(job)


# ---------------------------------------------------------------------------
# Cover endpoints
# ---------------------------------------------------------------------------

@app.post("/cover")
async def cover_submit(req: CoverRequest):
    """Submit a cover/remix generation. Returns task_id immediately."""
    if req.cover_noise_strength < 0 or req.cover_noise_strength > 1:
        raise HTTPException(400, "cover_noise_strength must be 0.0-1.0")
    if req.audio_cover_strength < 0 or req.audio_cover_strength > 1:
        raise HTTPException(400, "audio_cover_strength must be 0.0-1.0")

    _cleanup_old_jobs()
    task_id = str(uuid4())
    job = Job(
        task_id=task_id,
        task_type="cover",
        bpm=req.bpm,
        audio_format=req.audio_format,
    )
    _jobs[task_id] = job
    asyncio.create_task(_run_generation(job, req))

    return JSONResponse({
        "success": True,
        "task_id": task_id,
        "status": "queued",
    })


@app.get("/cover/status/{task_id}")
async def cover_status(task_id: str):
    """Poll cover/remix generation progress."""
    job = _jobs.get(task_id)
    if not job:
        return JSONResponse({
            "success": False, "status": "failed", "error": "Unknown task_id",
        }, status_code=404)
    return _build_status_response(job)


# ---------------------------------------------------------------------------
# Carey lifecycle passthrough
# ---------------------------------------------------------------------------

async def _proxy_lifecycle_post(path: str, *, timeout: float, params: Optional[dict] = None) -> JSONResponse:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(f"{ACESTEP_URL}{path}", params=params)
    except httpx.RequestError as e:
        return JSONResponse({
            "success": False,
            "error": f"Upstream request failed: {type(e).__name__}",
        }, status_code=502)

    try:
        payload = resp.json()
    except ValueError:
        payload = {
            "success": resp.status_code < 400,
            "status_code": resp.status_code,
            "raw": resp.text,
        }

    return JSONResponse(payload, status_code=resp.status_code)


@app.post("/v1/load")
async def lifecycle_load():
    """Proxy model load requests to ACE-Step."""
    return await _proxy_lifecycle_post("/v1/load", timeout=120)


@app.post("/v1/unload")
async def lifecycle_unload(request: Request):
    """Proxy model unload requests to ACE-Step."""
    mode = request.query_params.get("mode")
    params = {"mode": mode} if mode else None
    return await _proxy_lifecycle_post("/v1/unload", timeout=60, params=params)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    ace_status = "unknown"
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{ACESTEP_URL}/health")
            ace_status = "ok" if r.status_code == 200 else f"http_{r.status_code}"
    except httpx.ConnectError:
        ace_status = "unreachable"
    except Exception as e:
        ace_status = f"error: {type(e).__name__}"

    active_jobs = sum(
        1 for j in _jobs.values()
        if j.status not in (JobStatus.COMPLETED, JobStatus.FAILED)
    )

    return {
        "status": "ok" if ace_status == "ok" else "error",
        "backend": ACESTEP_BACKEND,
        "acestep_url": ACESTEP_URL,
        "acestep_status": ace_status,
        "manage_model_lifecycle": MANAGE_MODEL_LIFECYCLE,
        "current_model": _current_model,
        "base_model": ACESTEP_BASE_CONFIG,
        "turbo_model": ACESTEP_TURBO_CONFIG,
        "active_jobs": active_jobs,
        "max_concurrent": EFFECTIVE_MAX_CONCURRENT,
        "configured_max_concurrent": MAX_CONCURRENT,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=WRAPPER_PORT, reload=False)
