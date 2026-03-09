# MLX Fix For Reliable LEGO Inference Steps

## Context

This repo is being tuned for production use of **ACE-Step LEGO mode** (layer/stem generation from context audio), not general text2music workflows.

On Apple Silicon (MPS + MLX path), we observed that `inference_steps=50` requests behaved like ~8-step runs. The output quality/timing pattern confirmed the requested step count was not being honored in MLX diffusion.

## Root Cause

The MLX diffusion path used a turbo-style fixed timestep schedule and did not receive the requested `infer_steps` from service generation.

Specifically:

1. `service_generate_execute.py` called `_mlx_run_diffusion(...)` without passing `infer_steps`.
2. `diffusion.py` did not forward `infer_steps` (or model turbo/base mode) into `mlx_generate_diffusion(...)`.
3. `models/mlx/dit_generate.py` always generated a turbo schedule when custom `timesteps` were absent.

Result: base model requests in LEGO mode defaulted to turbo-like scheduling behavior.

## Files Changed

Only these files were modified:

1. `ACE-Step-1.5/acestep/core/generation/handler/service_generate_execute.py`
2. `ACE-Step-1.5/acestep/core/generation/handler/diffusion.py`
3. `ACE-Step-1.5/acestep/models/mlx/dit_generate.py`

## What Changed

### 1) Forward requested inference steps into MLX execution

`service_generate_execute.py`

- Passes `infer_steps=generate_kwargs.get("infer_steps", 8)` into `_mlx_run_diffusion(...)`.

### 2) Propagate infer steps + model mode into MLX diffusion core

`diffusion.py`

- `_mlx_run_diffusion(...)` now accepts `infer_steps`.
- Forwards both:
  - `infer_steps`
  - `is_turbo=bool(self.config.is_turbo)`
  into `mlx_generate_diffusion(...)`.

### 3) Make timestep scheduling base-aware (not always turbo)

`models/mlx/dit_generate.py`

- `get_timestep_schedule(...)` now accepts:
  - `infer_steps`
  - `is_turbo`
- Behavior:
  - **Turbo**: keeps existing fixed mapped schedule behavior.
  - **Base**: uses PyTorch-parity schedule:
    - `linspace(1.0 -> 0.0, infer_steps + 1)`
    - optional shift transform
    - uses `t[:-1]` for diffusion steps.
- `mlx_generate_diffusion(...)` now accepts and uses `infer_steps` and `is_turbo`.

## Validation (Apple Silicon)

Validated with LEGO smoke runs (`task_type=lego`, `track_name=vocals`, context audio = `smoke.wav`):

- Before fix: runtime profile matched ~8-step behavior even when requesting 50.
- After fix:
  - MLX progress showed `50/50` steps.
  - Time-cost logs matched 50-step diffusion duration.
  - Output quality aligned with expected 50-step result.

## Scope Notes

- This fix intentionally minimizes framework changes and only addresses step-schedule correctness in MLX diffusion.
- No broad architectural refactors were introduced.
- This keeps the repo suitable as a clean reference for LEGO-focused reliability fixes across environments (Apple Silicon now, T4 next).

#these might break everything we don't know yet

## 2026-02-24 Experimental M4 Conditioning Stability Changes

These changes are being tracked as high-risk experiments to resolve M4-specific LEGO conditioning mismatch and mangled source-audio behavior.
Status: rolled back on 2026-02-24 after regressions (poorer source lock / degraded outputs).

### A) MPS VAE dtype raised to float32

File:
- `ACE-Step-1.5/acestep/core/generation/handler/memory_utils.py`

Change:
- `_get_vae_dtype()` now returns `torch.float32` for `target_device == "mps"` (previously `torch.float16`).

Rationale:
- Improve numerical stability in VAE encode/decode on Apple MPS during conditioning.
Current state:
- Rolled back (MPS VAE dtype restored to `torch.float16`).

### B) Conditioning latent extraction switched from stochastic sample to deterministic mode

Files:
- `ACE-Step-1.5/acestep/core/generation/handler/vae_encode.py`
- `ACE-Step-1.5/acestep/core/generation/handler/vae_encode_chunks.py`

Changes:
- Replaced all inference-time conditioning calls:
  - `.latent_dist.sample()`
  - with `.latent_dist.mode()`

Rationale:
- Remove tile-to-tile random posterior noise from conditioning latents, which can produce boundary discontinuities and weak source lock in long/tiled inputs.
Current state:
- Rolled back (`.latent_dist.sample()` restored).

### C) Unload cleanup hardening (already present in this branch)

File:
- `ACE-Step-1.5/acestep/api_server.py`

Notes:
- Hard unload already clears MLX objects and calls MPS cache cleanup.
- Added reset of `use_mlx_dit` and `use_mlx_vae` flags to `False` after hard unload to avoid stale backend state.

### D) MLX VAE conditioning path switched to posterior mean

Files:
- `ACE-Step-1.5/acestep/core/generation/handler/mlx_vae_encode_native.py`
- `ACE-Step-1.5/acestep/core/generation/handler/mlx_vae_init.py`

Changes:
- MLX encode function resolution now prefers `encode_mean` (deterministic) when available, instead of always using `encode_and_sample` (stochastic).
- MLX compiled/uncompiled encode callables now use `encode_mean` instead of `encode_and_sample`.

Rationale:
- Keeps MLX conditioning behavior aligned with deterministic `.mode()`/mean-based conditioning and avoids random tile-to-tile latent drift.
Current state:
- Rolled back (`encode_and_sample` restored for compiled/uncompiled MLX encode path).

### Validation status

- Python syntax check passed for modified files.
- Audio quality impact still requires repeated ear tests in managed-process and standalone runs.

## 2026-02-24 Follow-up: Deterministic Source-Latent Encoding (ACTIVE TEST)

After reviewing current runtime code, the source-audio conditioning path was still stochastic in active code:

- PyTorch VAE encode used `latent_dist.sample()`.
- MLX VAE encode used `encode_and_sample`.

This can destabilize repeated LEGO runs on identical input audio (especially on Apple Silicon paths) because source conditioning latents vary each request.

### Files updated

- `ACE-Step-1.5/acestep/core/generation/handler/vae_encode.py`
- `ACE-Step-1.5/acestep/core/generation/handler/vae_encode_chunks.py`
- `ACE-Step-1.5/acestep/core/generation/handler/mlx_vae_init.py`
- `ACE-Step-1.5/acestep/core/generation/handler/mlx_vae_encode_native.py`
- `ACE-Step-1.5/acestep/api_server.py` (hard-unload cleanup for new MLX callable)
- `ACE-Step-1.5/acestep/handler.py` (state slot for compiled MLX mean encoder)

### What changed

- PyTorch conditioning encode now defaults to deterministic latent extraction (`mode`/`mean`) with fallback behavior preserved.
- MLX conditioning encode now prefers deterministic `encode_mean` (compiled when available).
- Added opt-in sampling flags for debugging/back-compat:
  - `ACESTEP_VAE_ENCODE_SAMPLE=1`
  - `ACESTEP_MLX_VAE_ENCODE_SAMPLE=1` (overrides MLX path specifically)
- Hard unload now also clears `_mlx_compiled_encode_mean`.

### Local verification

- Targeted unit tests pass:
  - `acestep.core.generation.handler.vae_encode_test`
  - `acestep.core.generation.handler.mlx_vae_init_test`
  - `acestep.core.generation.handler.mlx_vae_native_test`

## 2026-02-24 Follow-up: MPS Decoder KV Cache Causing Cross-Run Drift (ACTIVE)

Symptom:
- On Apple MPS, repeated LEGO runs on identical audio often produced one good output followed by degraded/jumbled outputs.

Controlled finding:
- With identical payload + fixed seed (`seed=12345`, `use_random_seed=false`) on MPS:
  - **Before fix**: three WAV outputs had different SHA-256 hashes.
  - **After fix**: three WAV outputs were bit-identical.

Root cause:
- DiT decoder KV cache (`use_cache=True`) in `generate_audio()` was unstable on MPS across iterative diffusion calls.

Change:
- Files:
  - `ACE-Step-1.5/acestep/models/base/modeling_acestep_v15_base.py`
  - `ACE-Step-1.5/acestep/models/sft/modeling_acestep_v15_base.py`
- In `generate_audio()`:
  - Detect MPS device.
  - Force `use_cache=False` for decoder on MPS (`use_decoder_cache = use_cache and not is_mps_device`).
  - Keep KV cache enabled on non-MPS backends.
  - Reinitialize/consume `past_key_values` only when cache is enabled.

Validation artifact:
- Deterministic test outputs:
  - `/Users/kevingriffing/gary/carey-repeat-debug/20260223_231941_wav` (pre-fix mismatch)
  - `/Users/kevingriffing/gary/carey-repeat-debug/20260223_232705_wav_cacheoff` (post-fix identical hashes)
