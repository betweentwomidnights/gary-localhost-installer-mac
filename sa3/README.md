# sa3 localhost backend

Local Stable Audio 3 service for gary4local.

Backend behavior, performance notes, and known quirks for the Apple Silicon
 MLX path now live in:

- [docs/sa3/README.md](/Users/klgriffing/Documents/gary-localhost-installer-mac/docs/sa3/README.md)

The Mac implementation uses an MLX-backed inference path built from the
validated `sa3-mlx` conversion layer while preserving the same HTTP contract
used by the Windows reference service.

The first terminal-first MLX LoRA training smoke is documented in
[`docs/sa3/MLX_LORA_TRAINING.md`](../docs/sa3/MLX_LORA_TRAINING.md).

The Mac control center exposes the MLX trainer directly from the `sa3` service
row. It recursively trains audio folders, supports per-track `.txt` prompts and
a shared trigger, persists jobs under Application Support, supports
cancellation and progress polling, and installs successful checkpoints into
the SA3 LoRA registry automatically.

The trainer also mirrors the Windows experimental loudness fix. When enabled,
each source track is iteratively re-encoded while its input gain is adjusted
toward a target latent RMS. It uses the same four correction rounds and 3%
tolerance as the Windows implementation. The default `0.90` target matches the
base model; the source audio remains untouched and the measured RMS, gain, pass
count, and convergence result are stored with the cached latents.

LoRA registration uses a Carey-style flow in the control center: the user points
at an SA3 `.ckpt` or `.safetensors` checkpoint and, optionally, a dataset folder
containing training `.txt` sidecars. In the Mac app wiring, the registry lives
at `~/Library/Application Support/GaryLocalhost/sa3/lora_registry.json`, and
prompt dice pools live under
`~/Library/Application Support/GaryLocalhost/sa3/prompts`.

## Hugging Face access

SA3 uses the same saved Hugging Face token as the existing stable-audio service.
The token alone is not enough: the same Hugging Face account must also accept
the model terms for the current SA3 repo, which now bundles the T5 assets under
the `t5gemma-b-b-ul2/` subfolder:

- https://huggingface.co/stabilityai/stable-audio-3-medium
- https://huggingface.co/stabilityai/stable-audio-3-medium/tree/main/t5gemma-b-b-ul2

The service health endpoint does not load or download the model. First load
happens when `/load` or a generation endpoint is called.

## LoRA endpoints

- `GET /loras` returns configured LoRAs.
- `POST /reload` rebuilds the loaded model and preloads the current registry.
- `GET /prompts?lora=name` returns default prompt pools merged with any selected
  LoRA prompt JSONs.

## LoRA automation lanes

`/generate`, `/generate/loop`, `/transform` and `/continue` accept sa3.cpp's
automation fields (`lora_lanes`, `lora_envelope`, `lane_isolation`,
`lane_isolation_until`, `lane_average_until`), so gary4juce sends one request
shape to sa3.cpp on Windows, to the remote service and to this one. `/health`
reports `capabilities.lora_lanes` and `lora_lanes_forwarded`, which is what the
plugin checks before it shows the lanes editor. A completed poll carries
`meta.lora_lanes`.

```json
"loras":      [{"name": "kev", "strength": 0.5}],
"lora_lanes": [
  {"loras": [{"name": "keygen", "strength": 0.8}],
   "envelope": [[0, 0], [4, 0], [6, 1.25], [10, 1]],
   "prompt": "demoscene chiptune, fast arpeggios"},
  {"loras": [{"name": "ratatat"}], "envelope": [[0, 1], [10, 0.3]]}
],
"lane_average_until": 0.9, "lane_isolation": 1.0, "lane_isolation_until": 0.0
```

Each lane is one more DiT pass per step. Per latent frame the velocity is
`v_base + sum(e_k(t) * (v_k - v_base))`. The MLX path merges adapters into the
DiT weights, so each pass keeps its own merged weight set and swaps it in
before it runs (`stable_audio_3/mlx/lora_lanes.py`). The sets are kept for the
length of one generation, up to `SA3_LANE_WEIGHT_CACHE_GB` (default: a quarter
of physical memory, between 2 and 12 GB); a pass past the budget is merged
again on every step, which is slower and sounds the same.

Refused with a 400, as on the remote service: `lora_envelope_mode: "layer"`,
one adapter both shared and in a lane (or twice in a lane), and a shared LoRA
with an interval or layer filter while lanes are on.

## Continuation modes

`POST /continue` accepts `continuation_mode` values `inpaint` and
`latent_prefix`. `inpaint` is the default path. `latent_prefix` is
experimental, forces the `pingpong` sampler, and pins the encoded source audio
as a fixed latent prefix before generating the continuation.

## Output shaping

SA3 applies local loudness defaults from the service environment and echoes the
applied values in `meta.loudness`. Gary4local exposes the main knobs as an
advanced "sa3 output shaping" panel:

- `peak_normalize_db` / `SA3_PEAK_NORMALIZE_DB`, default `2.0`
- `limiter_ceiling_db` / `SA3_LIMITER_CEILING_DB`, default `-0.3`
- `latent_rescale` / `SA3_LATENT_RESCALE`, default `1.0`
- `latent_shift` / `SA3_LATENT_SHIFT`, default `0.0`
- `latent_target_std` / `SA3_LATENT_TARGET_STD`, default off
- `continuation_tail_pad` / `SA3_CONTINUE_TAIL_PAD`, default `6`

Gary4local also exposes an experimental `fp32` / `fp16` DiT toggle. Current
testing suggests the most reliable loudness fix is still peak normalization
plus a gentle limiter; the precision toggle is better treated as a fallback or
comparison tool than as the primary solution.

Use `off` for dB fields to disable that stage. A positive peak-normalize target
is intended to be paired with the limiter.

### gary4juce v5 ending choices and local duration

`POST /generate` accepts `tail_pad`; `POST /continue` accepts
`continuation_tail_pad`. Both also accept the aliases `tail_pad_seconds`,
`tail_pad`, and `continuation_tail_pad`, in that precedence order. Explicit
zero is an override, not a missing value. gary4juce's **ends here** sends `0`;
**keeps going** sends `6`, generating six extra seconds internally and cutting
them off so the returned audio stops inside the music. The returned generate
duration, or continuation's source + added duration, remains unchanged.

Requests without an ending choice use the service defaults. Generate uses
`SA3_TAIL_PAD_SECONDS`, falling back to the saved `SA3_CONTINUE_TAIL_PAD` (`6`);
continue uses `SA3_CONTINUE_TAIL_PAD`. The app's advanced continuation setting
does not override an explicit plugin choice. Loop mode keeps its separate
bar-based padding, and transform keeps its existing length behavior.

The default local output ceiling is **380 seconds** (`SA3_MAX_DURATION` can
still override it). Continuation counts the source toward that ceiling; tail
padding is internal and may extend generation beyond it. Submit responses and
completed `meta.generate` / `meta.continue` report the effective tail pad and
internal generation duration. Cached model resolution no longer performs
online login validation; missing gated assets still require valid access.
