# AGENTS.md — ACE-Step 1.5 (DGX Spark exploratory bring-up)

## Goal
Stand up ACE-Step 1.5 on DGX Spark (Linux aarch64, CUDA 13.0) in an isolated Docker image, then:
1) Run BPM + key-aligned text2audio (no LM / no "thinking" at first).
2) Run a small matrix of audio-in tasks via REST API (repaint, cover, lego).
3) Decide which subset is worth a JUCE tab (repaint + lego are likely the DAW-first wins).

We explicitly avoid touching the production T4 VM during this phase.

---

## Ground truth requirements (from repo)
ACE-Step pins the following for DGX Spark class machines:
- torch==2.10.0+cu130
- torchvision==0.25.0+cu130
- torchaudio==2.10.0+cu130

These pins are in the upstream `requirements.txt`. Do not "uv sync" in a way that overwrites our chosen torch build. Keep torch consistent with cu130. (We can still build FA2/xformers against this torch.)

Refs:
- docs/en/INSTALL.md describes `uv run acestep` and `uv run acestep-api`.
- docs/en/API.md documents bpm + key_scale + time_signature fields and multipart audio upload.

---

## Repo actions checklist
### A) Clone on Spark
- git clone https://github.com/ace-step/ACE-Step-1.5
- cd ACE-Step-1.5

### B) Identify API fields we care about (for DAW integration)
From docs/en/API.md:
- bpm: int
- key_scale: string (e.g. "C Major", "Am")
- time_signature: string ("4" for 4/4 etc.)
- audio_duration: seconds
- task_type: text2music | cover | repaint | lego | extract | complete
- multipart upload fields: src_audio=@file, reference_audio=@file
We will use bpm+key_scale as first-class DAW parameters.

---

## Docker plan (Spark-first)
We will produce a Spark image that:
1) Installs torch/vision/audio for cu130 (either wheels or our own build if needed).
2) Installs optional accelerators (FA2 and/or xformers) if feasible.
3) Installs ACE-Step deps WITHOUT replacing torch.
4) Runs the REST API server on :8001.

### Why we start from scratch (recommended)
Even though we already built torch 2.6 + FA2 before, upstream ACE-Step expects torch 2.10.0; keeping close to their pinned versions reduces time lost to incompatibilities.

---

## Implementation steps (inside Docker build)
### 1) Base image
Pick a CUDA 13 compatible base for aarch64.
- Must match the cu130 toolchain expectations for torch 2.10.0+cu130.

### 2) Install torch cu130 for aarch64
Preferred: official wheel install via PyTorch extra index.
- torch==2.10.0+cu130
- torchvision==0.25.0+cu130
- torchaudio==2.10.0+cu130

Verify:
- python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

### 3) Build/install FA2 (optional but recommended)
- Build flash-attn / FA2 only if it supports aarch64 + cu130 in our environment.
- If FA2 build is painful, skip for bring-up and revisit after functionality is proven.

### 4) Install ACE-Step dependencies without overriding torch
Important: `requirements.txt` includes torch pins. Do not pip/uv install that file verbatim.

Strategy:
- Create a filtered requirements file that removes torch/vision/audio pin lines.
- Install remaining deps, then install the local package editable.

Suggested approach:
- Extract "Core dependencies" and everything below, excluding torch/vision/audio lines.
- `pip install -r requirements-no-torch.txt`
- `pip install -e .`

### 5) Start REST API server
- Expose 8001
- Command:
  - python acestep/api_server.py
or
  - uv run acestep-api
(Prefer the direct python entry first so we see stack traces clearly.)

Health check:
- GET http://localhost:8001/health
Models list:
- GET http://localhost:8001/v1/models

---

## First functional tests (no LM / deterministic)
We start with:
- thinking=false
- explicit bpm, key_scale, time_signature, audio_duration
- model set to a turbo DiT for speed if available, but any default is fine initially.

### 1) Text2music (BPM + key)
curl -X POST http://localhost:8001/release_task -H "Content-Type: application/json" -d '{"prompt":"tight electro groove, punchy kick, clean bass","bpm":120,"key_scale":"A minor","time_signature":"4","audio_duration":20,"thinking":false,"inference_steps":8,"batch_size":1}'

Poll:
curl -X POST http://localhost:8001/query_result -H "Content-Type: application/json" -d '{"task_id_list":["<TASK_ID>"]}'

Download:
curl -L "http://localhost:8001/v1/audio?path=<URLENCODED_PATH>" -o out.mp3

Notes:
- key_scale is supported and should be treated as a DAW-global parameter candidate.
- bpm/time_signature/audio_duration are explicitly supported.

---

## Audio-in tests (DAW workflows)
All of these can be done with multipart upload.

### 2) Repaint (regenerate a region of the clip)
curl -X POST http://localhost:8001/release_task -F "task_type=repaint" -F "prompt=regenerate this region with tighter drums and less busy hats" -F "repainting_start=8.0" -F "repainting_end=16.0" -F "src_audio=@/data/in.wav"

### 3) Cover (style transfer-ish)
curl -X POST http://localhost:8001/release_task -F "task_type=cover" -F "prompt=keep structure but make it more lo-fi and warm" -F "audio_cover_strength=0.3" -F "src_audio=@/data/in.wav" -F "reference_audio=@/data/style.wav"

### 4) Lego (add a layer)
curl -X POST http://localhost:8001/release_task -F "task_type=lego" -F "instruction=Generate the bass track based on the audio context:" -F "prompt=warm round bass, minimal fills" -F "src_audio=@/data/in.wav"

---

## Exploration matrix (Spark)
We will keep clips short at first (10–20s) and explicitly set BPM+key.
Test grid:
- bpm: 90 / 120 / 140
- key_scale: "C Major" / "A minor"
- tasks: text2music, repaint (8s region), lego (bass, guitar, vocals)

Record:
- runtime
- VRAM peak
- audible alignment to BPM grid
- artifacts at repaint boundaries

---

## When to enable the 5Hz LM ("thinking")
Only after baseline works, try:
- thinking=true with the 0.6B LM first.
We expect LM to fill missing metadata if we omit bpm/key/time_sig/duration, but for DAW UX we likely want to keep BPM+key user-controlled.

---

## Deliverables to commit back to our Gary ecosystem
1) A minimal "ace-step-api" Docker image for Spark.
2) A minimal curl test script folder (text2music.json, repaint.sh, lego.sh).
3) Notes: which tasks are viable on T4 vs Spark, recommended UI params.

---

## T4 follow-up (later)
After Spark validation:
- Decide whether T4 runs turbo-only or can handle base tasks.
- Build a second image targeting Linux x86_64 cu128 (torch 2.10.0+cu128).
- Keep API surface identical so JUCE can select backend capabilities dynamically.
