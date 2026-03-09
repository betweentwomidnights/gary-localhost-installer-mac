# ACE-Step T4 Deployment Plan

## Hardware Target

**GPU:** NVIDIA T4 — 16 GB VRAM
**Role:** Shared GPU host running multiple inference models (gary-backend-combined stack)
**Constraint:** ACE-Step must not hold GPU memory when idle — other models need the device.

---

## VRAM Profile (confirmed on GB10 Spark, base model, no LM)

| State | VRAM |
|-------|------|
| Model loaded + warm (idle) | ~7.7 GB |
| Mid-generation (50 steps, batch=1) | ~11 GB (confirmed: 11141 MiB / 10957 MiB) |
| After unload / empty_cache | ~0 GB |

**T4 headroom:** 16 GB − 11 GB peak = ~5 GB free during generation. With the
gpu-queue-service holding an exclusive token during that window, no other model
will be competing for that headroom.

**Conclusion:** T4 fits the base model comfortably. The LM (even 0.6B) pushes total
usage too high and adds CUDA graph complications — **no LM on T4**.

---

## Model Configuration for T4

```
ACESTEP_CONFIG_PATH=acestep-v15-base
ACESTEP_INIT_LLM=false   # always — LM won't fit alongside other models
```

Parameters locked for all T4 requests:
- `thinking=false`
- `use_cot_caption=false`
- `inference_steps=50`
- `task_type=lego`
- `repainting_start=0.0`
- `repainting_end=-1`

---

## Exposed Track Types

Only three track names are exposed in the VST UI and API wrapper:

| Track | Caption hint |
|-------|-------------|
| `vocals` | soulful indie vocalist, warm, wordless melody, expressive |
| `backing_vocals` | background vocals, close harmony, wordless, warm |
| `drums` | live acoustic drum kit, tight kick and snare, brushed hi-hats |

The user can override the caption for advanced use, but these are the defaults.

---

## Load / Unload Strategy

The T4 runs a shared GPU host. ACE-Step must release VRAM between requests.
Usage is very intermittent (single user, building toward iOS + JUCE VST), so
reload latency is acceptable and simplicity beats cleverness here.

**Recommended approach: load on demand, hard unload after generation**

On each request the wrapper:
1. Sends the generation request to ACE-Step api_server
2. After the audio is returned, calls a `/unload` endpoint (see below) that does:
   ```python
   del self.model        # drop reference
   gc.collect()
   torch.cuda.empty_cache()
   ```
3. Releases the gpu-queue-service token

This guarantees a full return to ~0 GB VRAM between requests. The models are
downloaded once and cached on disk — subsequent loads from disk are fast enough
that this is not a UX problem.

**Why not CPU offload?**
CPU offload keeps weights in system RAM and streams to GPU per-layer. It works,
but doesn't fully vacate VRAM and adds complexity around cache coherence.
For intermittent single-user usage, load/unload is simpler and cleaner.

**Adding `/unload` to api_server**
ACE-Step's api_server doesn't have an unload endpoint today. Options:
- Add a `POST /unload` route that drops `self.pipeline` / `self.model` and calls
  `torch.cuda.empty_cache()`. Minimal change.
- Alternatively: just restart the container after each generation (heavy-handed
  but bulletproof — Docker start from cached image is fast).

Start command for T4 container:
```bash
docker run --gpus all -p 8001:8001 \
  -v /path/to/checkpoints:/app/checkpoints \
  -e ACESTEP_CONFIG_PATH=acestep-v15-base \
  -e ACESTEP_INIT_LLM=false \
  ace-step-t4:latest
```

---

## gpu-queue-service Integration

**Service:** Go-based, port 8085, Redis backend
**Repo:** https://github.com/betweentwomidnights/gpu-queue-service
**Token pool:** 13,000 tokens shared across all GPU services

### How ACE-Step slots in

ACE-Step on T4 does not call gpu-queue-service directly — the **wrapper service** (see below)
handles token acquisition before forwarding the request to ACE-Step's api_server.

Flow:
```
VST / client
    │
    ▼
ace-step-wrapper (FastAPI, port 8002)
    │  1. POST /tasks → gpu-queue-service:8085  (acquire tokens)
    │  2. Wait for token
    │  3. POST /release_task → ace-step api_server:8001
    │  4. Poll /query_result until done
    │  5. Stream audio back to client
    │  6. POST /task/status → gpu-queue-service:8085 (release tokens)
    ▼
ace-step api_server (port 8001, cpu-offload mode)
```

The wrapper keeps the VST interface simple (single POST, returns audio) while the
gpu-queue-service ensures ACE-Step doesn't fight other models for VRAM.

---

## Wrapper API Design

Simple FastAPI service wrapping the ACE-Step lego workflow.

### POST /lego

**Request (multipart form):**
```
audio_file    — WAV or MP3, the source stem to layer over
track_type    — "vocals" | "backing_vocals" | "drums"
bpm           — integer (required)
key_scale     — e.g. "F# minor" (optional, model handles ambiguous keys well)
audio_duration — float seconds (optional, auto-detected if omitted)
batch_size    — 1 or 2 (default 1 for T4)
caption       — optional override caption
```

**Response:** audio/mpeg stream (first candidate), or JSON with file URL.

### GET /health

Returns wrapper + ace-step api_server health status.

---

## Docker Container Plan (T4)

Build from the same base as ace-step-spark with these differences:

| | Spark (dev) | T4 (prod) |
|---|---|---|
| LM | Optional (4B) | Never |
| CUDA graphs | Graceful fail | Skipped (enforce_eager) |
| CPU offload | Off | On |
| Batch size default | 2–4 | 1 |
| Wrapper service | No | Yes (port 8002) |
| gpu-queue-service | No | Yes (network: gary-backend) |

### Patches included in T4 image

All three fixes from the Spark dev session must be in the T4 build:

1. **`acestep/core/generation/handler/padding_utils.py`** — lego fix: returns `None`
   for `repainting_end_batch` when `is_lego_task=True`, routing through the full-mask
   branch in `conditioning_masks.py` so `src_latents` = full source audio.

2. **`acestep/third_parts/nano-vllm/nanovllm/engine/model_runner.py`** — graceful CUDA
   graph capture failure with RNG state restore (moot with no LM, but harmless).

3. **`acestep/llm_inference.py`** — forces `enforce_eager=True` on CUDA to skip graph
   capture entirely, preventing RNG contamination of the sampler (moot with no LM).

---

## gary4juce UI Tab

New tab: **"Stems"** (or "ACE-Step" / "AI Stems")

### Controls
- **Source audio** — drag-and-drop or file picker (WAV/MP3)
- **Track type** — dropdown: Vocals / Backing Vocals / Drums
- **BPM** — numeric field (can auto-detect from DAW)
- **Key** — text field, e.g. "F# minor" (optional)
- **Generate** button
- **Progress indicator** — polls wrapper /lego status
- **Result** — waveform preview + "Add to DAW" button

### Backend call
```
POST http://t4-host:8002/lego
  audio_file = <buffer>
  track_type = vocals
  bpm = 133
  key_scale = F# minor
```

---

## Open Questions / Next Steps

1. **Confirm peak VRAM on T4** — expected ~11 GB based on GB10 Spark measurements
   (11141 MiB / 10957 MiB confirmed). Verify this holds on T4 hardware and that the
   5 GB headroom is real (T4 = 16 GB).

2. **Verify `--cpu-offload` behavior** — does the model fully vacate VRAM after generation,
   or does it leave a residual footprint? Test with `nvidia-smi` before/after.

3. **Token cost** — how many of the 13,000 GPU tokens should ACE-Step consume per request?
   Needs to be calibrated against other services in the gary-backend-combined stack.

4. **Audio duration auto-detect** — the wrapper should call `ffprobe` on the uploaded file
   and pass `audio_duration` automatically so clients don't need to know it.

5. **T4 inference time** — 50 steps on T4 vs GB10 Spark will be slower. Benchmark and
   decide if 32 steps is acceptable quality for the VST use case.

6. **Partial lego** — `repainting_start` / `repainting_end` for selective stem replacement
   (e.g. "re-do just the chorus"). Needs further work in `conditioning_masks.py` but
   worth exposing in the UI eventually.

7. **Vocals with lyrics** — architecturally blocked for now (LM codes and guitar src_latents
   are mutually exclusive conditioning in the current model). Revisit if ACE-Step releases
   a model update or if we build a blend pathway in `prepare_condition`.
