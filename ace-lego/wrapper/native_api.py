"""ace-server's job API for the Mac Carey wrapper: POST /synth, GET/POST /job, GET /props.

gary4juce v5.0.2 and later speak this contract to every Carey backend (acestep.cpp's ace-server on
Windows, the remote wrapper, and this one). The legacy routes in main.py stay as they are for
plugin builds that still call them; the route names do not collide.

This is a translation layer onto the one ACE-Step api_server subprocess the wrapper already
manages. What the legacy routes do server-side and these must not: loop assist, trimming to the
input, default captions, a server-chosen seed, turbo step/CFG overrides and a default model. The
request names the exact DiT and every setting, and gets the full generated length back.

api_server has no cooperative cancel. A job cancelled while it waits never starts; one cancelled
while it runs reports "cancelled" at once, and is then left to finish with its result discarded
before the next job may start (releasing earlier would race the inference still running).
"""
from __future__ import annotations

import ast
import asyncio
import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, ValidationError
from starlette.datastructures import UploadFile

log = logging.getLogger("carey.native")

# gary4juce allows 380 s on localhost (240 s remotely).
MAX_SECONDS = 380.0
# Source plus an identical reference, 380 s of 48 kHz stereo float each.
MAX_UPLOAD = 512 * 1024 * 1024
MAX_REQUEST_JSON = 1024 * 1024
SAMPLE_RATE = 48000
TASKS = {"text2music", "lego", "complete", "repaint", "cover-nofsq", "extract"}
BASE_ONLY_TASKS = {"lego", "complete", "extract"}
TRACKS = {"vocals", "backing_vocals", "drums", "bass", "guitar", "keyboard",
          "percussion", "strings", "synth", "fx", "brass", "woodwinds"}
DIT_NAMES = (
    "acestep-v15-base", "acestep-v15-sft", "acestep-v15-turbo",
    "acestep-v15-xl-base", "acestep-v15-xl-sft", "acestep-v15-xl-turbo",
)
ARTIFACT_REPO = "thepatch/acestep-1.5-mlx"
# The plugin polls every 500 ms; the legacy routes' 1.5 s would skip most turbo steps.
POLL_SECONDS = 0.5


class AceRequest(BaseModel):
    model_config = {"allow_inf_nan": False}
    task_type: str
    caption: str = ""
    lyrics: str = ""
    bpm: int = Field(ge=0, le=400)
    keyscale: str = ""
    timesignature: str = ""
    vocal_language: str = "en"
    inference_steps: int = Field(ge=1, le=200)
    guidance_scale: float = Field(ge=0, le=30)
    seed: int = Field(ge=0, le=2**32 - 1)
    output_format: str = "wav16"
    synth_model: str
    adapter: str = ""
    adapter_scale: float = Field(default=1, ge=0, le=1)
    duration: float = Field(gt=0, le=MAX_SECONDS)
    track: str = ""
    repainting_start: float = Field(default=0, ge=0, le=MAX_SECONDS)
    repainting_end: float = Field(default=-1, ge=-1, le=MAX_SECONDS)
    audio_cover_strength: float = Field(default=1, ge=0, le=1)
    cover_noise_strength: float = Field(default=0, ge=0, le=1)


class JobError(Exception):
    """A failure whose text is fit for the plugin's status bar."""


def model_variant(name: str) -> str:
    if name.endswith("-turbo"):
        return "turbo"
    if name.endswith("-sft"):
        return "sft"
    return "base"


def model_family(name: str) -> str:
    return "xl" if name.startswith("acestep-v15-xl-") else "standard"


def backend_payload(req: AceRequest) -> dict:
    """The api_server /release_task fields for one request, with nothing added or overridden."""
    data = {
        "task_type": req.task_type, "prompt": req.caption, "lyrics": req.lyrics,
        "key_scale": req.keyscale, "time_signature": req.timesignature,
        "vocal_language": req.vocal_language,
        "inference_steps": req.inference_steps, "guidance_scale": req.guidance_scale,
        "seed": req.seed, "use_random_seed": False, "audio_duration": req.duration,
        "audio_format": "wav", "batch_size": 1, "thinking": False,
        "use_cot_caption": False, "use_cot_language": False,
        "audio_cover_strength": req.audio_cover_strength,
        "cover_noise_strength": req.cover_noise_strength,
        # api_server defaults to shift 3 (turbo); base and sft were trained with shift 1.
        "shift": 3.0 if model_variant(req.synth_model) == "turbo" else 1.0,
    }
    if req.bpm > 0:
        data["bpm"] = req.bpm
    if req.task_type in {"lego", "extract"}:
        data["track_name"] = req.track.strip()
    if req.task_type == "complete":
        data["track_classes"] = [t.strip() for t in req.track.split("|")]
    if req.task_type in {"lego", "repaint"}:
        data.update(repainting_start=req.repainting_start,
                    repainting_end=req.repainting_end, chunk_mask_mode="explicit")
    return data


def form_fields(payload: dict) -> dict:
    """Multipart text fields. A list stays a list, which httpx sends as one field per item:
    api_server reads repeated fields back as a list, and would take a JSON string as one name."""
    fields = {}
    for key, value in payload.items():
        if isinstance(value, list):
            fields[key] = [str(item) for item in value]
        elif isinstance(value, bool):
            fields[key] = "true" if value else "false"
        else:
            fields[key] = str(value)
    return fields


# --------------------------------------------------------------------------- installed models


def _hub_cache() -> Path:
    for name in ("HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE"):
        value = (os.getenv(name) or "").strip()
        if value:
            return Path(value).expanduser()
    home = (os.getenv("HF_HOME") or "").strip()
    return (Path(home).expanduser() if home else Path.home() / ".cache" / "huggingface") / "hub"


def _pinned_artifact_revision() -> str | None:
    """The revision api_server loads optimized weights from, read from its own catalog."""
    root = (os.getenv("CAREY_API_ROOT") or "").strip()
    source = (Path(root) if root else Path(__file__).resolve().parent.parent / "ACE-Step-1.5")
    source = source / "acestep" / "models" / "mlx" / "artifact_runtime.py"
    try:
        for node in ast.parse(source.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "ARTIFACT_REVISION" for t in node.targets
            ):
                return str(ast.literal_eval(node.value))
    except Exception:
        pass
    return None


def _optimized_artifact_present(name: str) -> bool:
    override = (os.getenv("ACESTEP_MLX_ARTIFACT_DIR") or "").strip()
    if override:
        roots = [Path(override).expanduser()]
    else:
        snapshots = _hub_cache() / ("models--" + ARTIFACT_REPO.replace("/", "--")) / "snapshots"
        revision = _pinned_artifact_revision()
        if revision:
            roots = [snapshots / revision / "MLX" / f"{name}-f16"]
        else:
            roots = list(snapshots.glob(f"*/MLX/{name}-f16"))
    needed = ("provenance.json", "model.safetensors.index.json", "config.json",
              "conditioning/provenance.json", "conditioning/silence_latent.pt")
    return any(all((root / part).is_file() for part in needed) for root in roots)


def _legacy_checkpoint_present(name: str) -> bool:
    directory = (os.getenv("ACESTEP_CHECKPOINT_DIR") or "").strip()
    if not directory:
        return False
    root = Path(directory).expanduser() / name
    if not (root / "config.json").is_file():
        return False
    return any(p.is_file() and p.stat().st_size > 0 for p in root.glob("*.safetensors"))


def installed_dits() -> list[str]:
    """Every DiT api_server can load right now, 2B and XL. Only looks at the disk."""
    uses_mlx = (os.getenv("ACESTEP_USE_MLX_DIT") or "1").strip() != "0"
    policy = (os.getenv("ACESTEP_MLX_WEIGHT_SOURCE") or "").strip().lower() or "legacy"
    found = []
    for name in DIT_NAMES:
        optimized = uses_mlx and policy in {"auto", "optimized"} and _optimized_artifact_present(name)
        legacy = not (uses_mlx and policy == "optimized") and _legacy_checkpoint_present(name)
        if optimized or legacy:
            found.append(name)
    return found


# --------------------------------------------------------------------------- audio


def audio_seconds(path: Path) -> float | None:
    try:
        import soundfile as sf

        info = sf.info(str(path))
        if info.samplerate > 0 and info.frames > 0:
            return info.frames / float(info.samplerate)
    except Exception:
        pass
    return None


def convert_audio(source: Path, target: Path) -> float:
    """Write `source` as 48 kHz stereo 16-bit WAV, whole length. Returns its seconds.

    Done in Python so it works on a Mac with no ffmpeg; ffmpeg is only the fallback for a file
    libsndfile cannot read.
    """
    import numpy as np
    import soundfile as sf

    try:
        data, rate = sf.read(str(source), dtype="float32", always_2d=True)
    except Exception:
        return _convert_with_ffmpeg(source, target)
    if data.shape[0] == 0 or rate <= 0:
        raise ValueError("Could not decode audio")
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    elif data.shape[1] > 2:
        data = data[:, :2]
    if rate != SAMPLE_RATE:
        from scipy.signal import resample_poly

        divisor = math.gcd(SAMPLE_RATE, int(rate))
        data = resample_poly(data, SAMPLE_RATE // divisor, int(rate) // divisor, axis=0)
    data = np.clip(np.nan_to_num(data), -1.0, 1.0).astype(np.float32, copy=False)
    sf.write(str(target), data, SAMPLE_RATE, subtype="PCM_16", format="WAV")
    return data.shape[0] / float(SAMPLE_RATE)


def _convert_with_ffmpeg(source: Path, target: Path) -> float:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise ValueError("Could not decode audio")
    done = subprocess.run(
        [ffmpeg, "-nostdin", "-v", "error", "-y", "-i", str(source), "-map", "0:a:0",
         "-ar", str(SAMPLE_RATE), "-ac", "2", "-c:a", "pcm_s16le", str(target)],
        capture_output=True, timeout=120,
    )
    seconds = audio_seconds(target) if done.returncode == 0 else None
    if seconds is None:
        log.warning("Audio conversion failed: %s", done.stderr.decode(errors="replace")[-2000:])
        raise ValueError("Could not decode audio")
    return seconds


# --------------------------------------------------------------------------- jobs


@dataclass
class NativeJob:
    id: str
    temp: tempfile.TemporaryDirectory
    status: str = "running"
    stage: str = "queued"
    step: int = 0
    total: int = 0
    error: str | None = None
    finished: float | None = None
    fetched: float | None = None
    worker: asyncio.Task | None = None
    cancel: asyncio.Event = field(default_factory=asyncio.Event)
    active: bool = True


def error(message: str, code: int = 400) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=code)


_STEP_RE = re.compile(r"(\d+)\s*/\s*(\d+)")


def update_progress(job: NativeJob, result: dict, req: AceRequest, strip_ansi) -> None:
    """Stage and diffusion step from what api_server reports. Never a guess from elapsed time."""
    text = strip_ansi(str(result.get("progress_text") or ""))
    lowered = text.lower()
    match = _STEP_RE.search(text)
    if match and int(match[2]) > 0 and (int(match[2]) == req.inference_steps or "diffusion" in lowered):
        total = int(match[2])
        job.stage, job.step, job.total = "dit", min(int(match[1]), total), total
    elif "decod" in lowered:
        job.stage, job.step, job.total = "decode", 0, 0
    elif "encod" in lowered and job.stage != "dit":
        job.stage, job.step, job.total = "encode", 0, 0


def result_file_reference(result: dict) -> str:
    items = json.loads(result["result"])
    if not items or not items[0].get("file"):
        raise JobError("Carey finished without audio")
    return str(items[0]["file"])


def install_native_routes(app, legacy) -> None:
    jobs: dict[str, NativeJob] = {}
    app.state.native_jobs = jobs
    fallback_lock = asyncio.Lock()

    def lock():
        # The legacy routes' semaphore, so one job runs at a time whichever contract asked.
        return legacy._generation_semaphore or fallback_lock

    def cleanup() -> None:
        now = time.time()
        for job_id, job in list(jobs.items()):
            if not job.active and job.finished and now - max(job.finished, job.fetched or 0) > legacy.JOB_TTL:
                job.temp.cleanup()
                del jobs[job_id]

    def adapter_problem(req: AceRequest) -> str | None:
        if not req.adapter:
            return None
        entry = legacy.LORA_REGISTRY.get(req.adapter)
        backend = "turbo" if model_variant(req.synth_model) == "turbo" else "base"
        if (
            not entry
            or str(entry.get("model_family", "standard")) != model_family(req.synth_model)
            or backend not in list(entry.get("backends", ["base", "turbo"]))
        ):
            return "Requested lora is not available for this model"
        return None

    @app.get("/props")
    async def props():
        cleanup()
        return {
            "models": {"dit": installed_dits()},
            "adapters": list(legacy.LORA_REGISTRY),
            "adapter_info": {
                name: {
                    "model_family": str(entry.get("model_family", "standard")),
                    "captions": legacy._caption_pools.get(name, []),
                }
                for name, entry in legacy.LORA_REGISTRY.items()
            },
        }

    async def strict_lora(client: httpx.AsyncClient, req: AceRequest, entry: dict) -> None:
        # The legacy loader logs and carries on, which would quietly generate without the LoRA.
        try:
            for path, body, timeout in (
                ("/v1/lora/load", {"lora_path": str(entry["path"])}, legacy.LORA_LOAD_TIMEOUT),
                ("/v1/lora/scale", {"scale": req.adapter_scale}, legacy.LORA_SCALE_TIMEOUT),
            ):
                resp = await client.post(f"{legacy.ACESTEP_URL}{path}", json=body, timeout=timeout)
                reply = resp.json() if resp.status_code == 200 else {}
                if resp.status_code != 200 or reply.get("error") or reply.get("code", 200) != 200:
                    log.warning("Carey %s failed for %s: %s", path, req.adapter, resp.text[:2000])
                    raise JobError(f"lora {req.adapter} failed to load")
        except (httpx.HTTPError, ValueError) as exc:
            raise JobError(f"lora {req.adapter} failed to load") from exc

    async def fetch_result(client: httpx.AsyncClient, file_ref: str, raw: Path) -> None:
        # The result is on this disk; only fall back to the server's own relative URL.
        candidates = parse_qs(urlsplit(file_ref).query).get("path", [])
        local = Path(candidates[0]) if candidates else None
        if local is not None and local.is_absolute() and local.is_file():
            await asyncio.to_thread(shutil.copyfile, local, raw)
            return
        if not file_ref.startswith("/") or file_ref.startswith("//"):
            raise JobError("Carey finished without audio")
        async with client.stream("GET", legacy.ACESTEP_URL + file_ref, timeout=120) as download:
            download.raise_for_status()
            with raw.open("wb") as out:
                async for chunk in download.aiter_bytes():
                    out.write(chunk)

    async def wait_for_backend(client: httpx.AsyncClient) -> None:
        """After a submission whose fate is unknown: hold on until api_server answers again."""
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            try:
                if (await client.get(f"{legacy.ACESTEP_URL}/health", timeout=5)).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(POLL_SECONDS)

    async def run(job: NativeJob, req: AceRequest, paths: dict[str, Path]) -> None:
        adapter = legacy.LORA_REGISTRY.get(req.adapter) if req.adapter else None
        try:
            async with lock():
                if job.cancel.is_set():
                    return
                adapter_touched = submitted = confirmed_terminal = False
                async with httpx.AsyncClient() as client:
                    try:
                        job.stage = "load"
                        if legacy.MANAGE_MODEL_LIFECYCLE and legacy._current_model != req.synth_model:
                            try:
                                if legacy._current_model is not None:
                                    await legacy._unload_model(client)
                                await legacy._load_model(client, req.synth_model)
                            except Exception as exc:
                                raise JobError(f"could not load {req.synth_model}") from exc
                        # Start from no adapter, as the legacy routes do.
                        await legacy._unload_lora(client)
                        if adapter:
                            adapter_touched = True
                            await strict_lora(client, req, adapter)
                        if job.cancel.is_set():
                            return

                        payload = backend_payload(req)
                        if paths:
                            job.stage = "encode"
                        with ExitStack() as stack:
                            files = [
                                ("ctx_audio" if name == "audio" else name,
                                 (name + ".wav", stack.enter_context(path.open("rb")), "audio/wav"))
                                for name, path in paths.items()
                            ]
                            submitted = True
                            if files:
                                resp = await client.post(f"{legacy.ACESTEP_URL}/release_task",
                                                         data=form_fields(payload), files=files, timeout=120)
                            else:
                                resp = await client.post(f"{legacy.ACESTEP_URL}/release_task",
                                                         json=payload, timeout=120)
                        if resp.status_code != 200:
                            confirmed_terminal = True  # refused outright: nothing is running
                            log.warning("Carey /release_task refused: %s", resp.text[:2000])
                            raise JobError("Carey could not start this generation")
                        ace_id = resp.json()["data"]["task_id"]

                        deadline = time.monotonic() + legacy.GENERATION_TIMEOUT
                        while True:
                            try:
                                resp = await client.post(f"{legacy.ACESTEP_URL}/query_result",
                                                         json={"task_id_list": [ace_id]}, timeout=15)
                                resp.raise_for_status()
                                result = resp.json()["data"][0]
                                status = int(result["status"])
                            except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError):
                                if time.monotonic() > deadline + 120:
                                    raise JobError("Carey stopped responding")
                                await asyncio.sleep(POLL_SECONDS)
                                continue
                            if status in (1, 2):
                                confirmed_terminal = True
                                break
                            update_progress(job, result, req, legacy._strip_ansi)
                            if time.monotonic() > deadline and not job.cancel.is_set():
                                job.status, job.error = "failed", "Carey generation timed out"
                                job.finished = time.time()
                                job.cancel.set()  # keep draining; never publish a late result
                            await asyncio.sleep(POLL_SECONDS)
                        if job.cancel.is_set():
                            return
                        if status == 2:
                            log.warning("Carey generation failed: %s", str(result)[:2000])
                            raise JobError("Carey generation failed; please retry")

                        job.stage, job.step, job.total = "decode", 0, 0
                        raw = Path(job.temp.name) / "result-raw.wav"
                        await fetch_result(client, result_file_reference(result), raw)
                        await asyncio.to_thread(convert_audio, raw, Path(job.temp.name) / "result.wav")
                        raw.unlink(missing_ok=True)
                        if not job.cancel.is_set():
                            job.status = "done"
                    finally:
                        if submitted and not confirmed_terminal:
                            await wait_for_backend(client)
                        try:
                            if adapter_touched:
                                await legacy._unload_lora(client)
                            if legacy.MANAGE_MODEL_LIFECYCLE and not legacy.KEEP_MODEL_LOADED:
                                await legacy._unload_model(client)
                        except Exception:
                            log.exception("Carey cleanup after job %s failed", job.id)
        except JobError as exc:
            log.warning("Carey job %s failed: %s", job.id, exc, exc_info=exc.__cause__ is not None)
            if job.status == "running":
                job.status, job.error = "failed", str(exc)
        except Exception:
            log.exception("Carey job %s failed", job.id)
            if job.status == "running":
                job.status, job.error = "failed", "Carey generation failed; please retry"
        finally:
            job.active = False
            job.finished = time.time()
            for path in paths.values():
                path.unlink(missing_ok=True)

    async def read_submission(request: Request, temp: Path) -> tuple[object, dict[str, Path]] | JSONResponse:
        paths: dict[str, Path] = {}
        if request.headers.get("content-type", "").startswith("application/json"):
            raw = await request.body()
            if len(raw) > MAX_REQUEST_JSON:
                return error("Request is too large", 413)
            return json.loads(raw), paths

        # Starlette spools file parts to disk. Cap the whole streamed body as well, since
        # Content-Length can be missing or wrong.
        received = 0
        receive = request._receive

        async def bounded_receive():
            nonlocal received
            message = await receive()
            received += len(message.get("body", b""))
            if received > MAX_UPLOAD:
                raise ValueError("Upload is too large")
            return message

        request._receive = bounded_receive
        async with request.form(max_files=3, max_fields=4, max_part_size=MAX_REQUEST_JSON) as form:
            part = form.get("request")
            if part is None:
                return error("Missing request part")
            raw = await part.read(MAX_REQUEST_JSON + 1) if isinstance(part, UploadFile) else part
            if len(raw) > MAX_REQUEST_JSON:
                return error("Request is too large", 413)
            data = json.loads(raw)
            for name in ("audio", "ref_audio"):
                upload = form.get(name)
                if upload is None:
                    continue
                if not isinstance(upload, UploadFile):
                    return error(f"{name} must be a WAV file")
                path = temp / (name + "-upload.wav")
                with path.open("wb") as out:
                    while chunk := await upload.read(1024 * 1024):
                        out.write(chunk)
                paths[name] = path
        return data, paths

    @app.post("/synth")
    async def synth(request: Request):
        cleanup()
        temp = tempfile.TemporaryDirectory(prefix="carey-native-")
        accepted = False
        try:
            submission = await read_submission(request, Path(temp.name))
            if isinstance(submission, JSONResponse):
                return submission
            data, paths = submission
            if not isinstance(data, dict):
                return error("Request must be one JSON object")
            req = AceRequest.model_validate(data)
            if req.task_type not in TASKS:
                return error("Unknown task type")
            if req.output_format != "wav16":
                return error("Only wav16 output is supported")
            if req.task_type in {"text2music", "cover-nofsq"} and not req.caption.strip():
                return error("Please enter a caption")
            if req.synth_model not in installed_dits():
                return error("Requested model is not available")
            if req.task_type in BASE_ONLY_TASKS and model_variant(req.synth_model) != "base":
                return error("This task needs a base model")
            if problem := adapter_problem(req):
                return error(problem)
            if req.task_type != "text2music" and "audio" not in paths:
                return error("Please upload source audio")
            if req.task_type == "text2music":
                paths = {}
            if req.task_type in BASE_ONLY_TASKS:
                tracks = [name.strip() for name in req.track.split("|")]
                if any(t not in TRACKS for t in tracks) or (req.task_type != "complete" and len(tracks) != 1):
                    return error("Please choose valid tracks")
            source_seconds = None
            for name, path in list(paths.items()):
                seconds = await asyncio.to_thread(audio_seconds, path)
                if seconds is not None and seconds > MAX_SECONDS + 0.01:
                    return error(f"Source audio must be at most {int(MAX_SECONDS)} seconds")
                converted = Path(temp.name) / (name + ".wav")
                seconds = await asyncio.to_thread(convert_audio, path, converted)
                if seconds > MAX_SECONDS + 0.01:
                    return error(f"Source audio must be at most {int(MAX_SECONDS)} seconds")
                path.unlink()
                paths[name] = converted
                if name == "audio":
                    source_seconds = seconds
            if req.task_type == "repaint" and (source_seconds is None or req.duration <= source_seconds + 1):
                return error("Target must be longer than the source")
            job = NativeJob(uuid4().hex, temp)
            jobs[job.id] = job
            job.worker = asyncio.create_task(run(job, req, paths))
            accepted = True
            return {"id": job.id}
        except ValidationError as exc:
            return error(f"Invalid or missing {exc.errors()[0]['loc'][0]}")
        except (ValueError, TypeError):
            return error("Invalid request or audio")
        except Exception:
            log.exception("Carey submission failed")
            return error("Could not read request")
        finally:
            if not accepted:
                temp.cleanup()

    @app.get("/job")
    async def get_job(id: str = "", result: str = ""):
        cleanup()
        job = jobs.get(id)
        if not job:
            return error("Job not found", 404)
        if result:
            if job.status != "done":
                return error("Result not ready", 404)
            job.fetched = time.time()
            path = Path(job.temp.name) / "result.wav"

            async def stream():
                yield b"--ace-batch-boundary\r\nContent-Type: audio/wav\r\n\r\n"
                with path.open("rb") as audio:
                    while chunk := await asyncio.to_thread(audio.read, 1024 * 1024):
                        yield chunk
                yield b"\r\n--ace-batch-boundary--\r\n"

            return StreamingResponse(stream(), media_type="multipart/mixed; boundary=ace-batch-boundary")
        response = {"status": job.status}
        if job.status == "running":
            response["progress"] = {"stage": job.stage, "step": job.step, "total": job.total}
        elif job.status == "failed":
            response["error"] = job.error
        return response

    @app.post("/job")
    async def cancel_job(id: str = "", cancel: str = ""):
        job = jobs.get(id)
        if not job:
            return error("Job not found", 404)
        if cancel != "1":
            return error("Expected cancel=1")
        if job.status == "running":
            job.cancel.set()
            job.status = "cancelled"
        return {"status": job.status}
