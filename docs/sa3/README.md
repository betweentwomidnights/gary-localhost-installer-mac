# SA3 Mac Backend Notes

This note tracks the current Stable Audio 3 backend behavior in `gary4local`
 on Apple Silicon macOS.

It is intentionally practical. The goal is to document the parts of the SA3
 backend that are likely to surprise users or developers while we continue to
 refine the MLX path.

## Current Runtime Shape

The macOS SA3 backend is a custom MLX inference path built from:

- the local `reference/sa3-mlx` staging work
- the upstream Stable Audio 3 repo
- additional `gary4local` service and LoRA integration

The current production defaults are defined in:

- [control-center/manifest/services.production.json](/Users/klgriffing/Documents/gary-localhost-installer-mac/control-center/manifest/services.production.json:234)
- [control-center/manifest/services.dev.json](/Users/klgriffing/Documents/gary-localhost-installer-mac/control-center/manifest/services.dev.json:198)

Important defaults:

- DiT runs at `float32`
- text, number, and autoencoder modules run at `float16`
- chunked decode is enabled
- loudness shaping is enabled by default
- continuation uses `regen_past` tail mode with a `6s` tail pad

That mix is deliberate, but the current takeaway is more nuanced than our
 earlier assumption. The DiT `float32` default still exists as a conservative
 MLX setting, and `gary4local` now exposes a user-facing `fp32`/`fp16` DiT
 toggle for testing. In current M4 testing, though, the most reliable loudness
 fix is still the backend's peak-normalize-plus-gentle-limiter path. The
 `fp16` DiT loudness issue only surfaced occasionally in practice, and was most
 noticeable when those loudness controls were disabled.

## What To Expect

### Cold Start

The first service load is expensive. The backend converts the runtime model
 stack to MLX and preloads any registered LoRAs before the service is ready.

If LoRAs are present, `model ready` time will be longer than the no-LoRA case.

### Warm Text-To-Audio

Plain text-to-audio is now the best-case path:

- negative conditioning is skipped when `cfg_scale == 1.0`
- progress reports a `conditioning` phase before sampling
- chunked decode reduces the long SAME-L decode tail

Short generations should feel much closer to the standalone `sa3-mlx`
 experiments than they did before the recent fixes.

### Source Audio And Continuation

Continuation and inpainting remain materially heavier than plain prompt
 generation.

The most important recent fix was on the source-audio encode path:

- we now encode only the real source clip length
- we do not pad source audio out to the full target duration before SAME-L
  encode
- latent padding happens after encode instead

That change lives in
[sa3/stable_audio_3/mlx/pipeline.py](/Users/klgriffing/Documents/gary-localhost-installer-mac/sa3/stable_audio_3/mlx/pipeline.py:720)
 and
[sa3/stable_audio_3/mlx/pipeline.py](/Users/klgriffing/Documents/gary-localhost-installer-mac/sa3/stable_audio_3/mlx/pipeline.py:944).

Even after that fix, continuation is still slower than pure text-to-audio,
 because the backend must:

- encode source audio into latents
- build and apply the inpaint mask
- run the diffusion pass over a longer latent window
- decode a longer result

Continuation also exposes a subtle tail-behavior tradeoff through
 `continuation_tail_pad`.

Practical meaning:

- lower values lean toward an output that feels like it really ends
- higher values lean toward an output that stops "inside" the music, which is
  more useful if you want to keep chaining continuations
- the difference is subtle and still somewhat experimental

### LoRA-Backed Generation

LoRAs are now much faster on repeat generations when the settings do not
 change.

The backend caches the applied MLX LoRA state by a runtime signature that
 includes:

- selected LoRAs
- strength values
- intervals
- layer filters

That cache lives in
[sa3/stable_audio_3/mlx/pipeline.py](/Users/klgriffing/Documents/gary-localhost-installer-mac/sa3/stable_audio_3/mlx/pipeline.py:38)
 and the apply/reuse logic lives around
[sa3/stable_audio_3/mlx/pipeline.py](/Users/klgriffing/Documents/gary-localhost-installer-mac/sa3/stable_audio_3/mlx/pipeline.py:372).

Practical meaning:

- first generation with a new LoRA setup can still be expensive
- second generation with the same LoRA strength should be much faster
- changing the LoRA strength slider is expected to trigger recomputation
- blending multiple LoRAs at different strengths is supported, but each new
  blend signature is a new applied state

For advanced API usage:

- fully static LoRA configs reuse the applied DiT state directly
- interval-based or layer-filtered LoRA configs still need the scheduled path
  and will remain more expensive

## User-Facing Quirks

### Saving A LoRA Is Not The Same As Building Prompt Dice Pools

Registering an SA3 LoRA checkpoint does not automatically make prompt dice
 pools ready for the VST.

If the user attaches a prompt folder, they still need to run the prompt build
 step so the backend emits the prompt JSON used by the dice flow.

We should keep improving the UI around this, because it is easy to assume that
 "save LoRA" also means "prompt pool ready."

### LoRA Registry Changes Rebuild The Warm Service State

When LoRAs are added, removed, or reloaded, the service rebuilds the loaded
 model state so the active registry stays consistent.

That is a reasonable macOS tradeoff right now. The current Apple Silicon
 backend is optimized more around one warm active service than around lazy
 multi-service model churn.

### Continuation Now Has Two Distinct Behaviors

The API supports both `continuation_mode=inpaint` and the experimental
 `continuation_mode=latent_prefix`.

`inpaint` regenerates the masked continuation region the way the rest of this
 document already describes.

`latent_prefix` is different: it forces the `pingpong` sampler and pins the
 encoded source audio as a fixed latent prefix before generating the rest of
 the clip. That makes it useful for certain iterative continuation workflows,
 but it is still a specialized mode rather than the default recommendation.

### Decode Still Owns A Visible Slice Of Total Time

The current progress path reserves sampling for `12% -> 80%` and decode for
 the `84% -> 91%+` range. That split is deliberate because SAME-L decode is
 still a meaningful part of total generation time, especially on longer clips.

### Loudness Controls Matter More Than The Precision Toggle

The current practical lesson from testing is:

- peak normalization plus a gentle limiter is still the single best loudness
  solution on the macOS MLX backend
- `fp32` DiT remains worth keeping around as an experimental fallback
- `fp16` DiT does not appear to be broadly broken, but it can occasionally
  show louder / less-controlled behavior when the loudness controls are turned
  off

## Operational Guidance

For the best current macOS experience:

- keep SA3 warm instead of repeatedly restarting the service
- iterate several times before changing LoRA strength
- expect continuation/inpainting to cost more than plain text-to-audio
- expect the first LoRA generation after a strength change to be slower than
  the second

If a run feels unexpectedly slow, the most useful log line is the per-request
 timing report from
[sa3/api.py](/Users/klgriffing/Documents/gary-localhost-installer-mac/sa3/api.py:1468):

```text
[session-id] generation timings first_step=...s generate=...s decode=...s postprocess=...s encode=...s total=...s
```

Interpretation:

- large `first_step`: front-loaded conditioning, source-audio encode, or first
  diffusion-step cost
- large `generate`: sampling path is the dominant cost
- large `decode`: SAME-L decode is still the bottleneck

## Known Follow-Up Work

- document the expected prompt-build flow better in the SA3 LoRA UI
- continue cherry-picking upstream decoder/runtime improvements without giving
  up LoRA support
- decide how much "keep one warm model active" should be hardcoded policy vs
  exposed setting on macOS
