# sa3 localhost backend

Local Stable Audio 3 service for gary4local.

Backend behavior, performance notes, and known quirks for the Apple Silicon
 MLX path now live in:

- [docs/sa3/README.md](/Users/klgriffing/Documents/gary-localhost-installer-mac/docs/sa3/README.md)

The Mac implementation uses an MLX-backed inference path built from the
validated `sa3-mlx` conversion layer while preserving the same HTTP contract
used by the Windows reference service.

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

`continuation_tail_pad` is about how the end of a continuation should feel:
lower values lean toward a more song-like ending, while higher values lean
toward stopping "inside" the music so repeated continuations chain more
naturally. The effect is subtle and still experimental.
