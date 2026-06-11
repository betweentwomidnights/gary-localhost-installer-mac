# MLX LoRA Training

The terminal-first MLX training prototype lives below the Gary UI layer so the
optimizer, checkpoint format, and memory behavior can be validated
independently.

## Current Milestone

- MLX `Linear` and `Conv1d` layers can be replaced with trainable LoRA, DoRA,
  BoRA, LoRA-XS, DoRA-XS, or BoRA-XS wrappers while base weights stay frozen.
- XS adapters compute deterministic frozen SVD bases from each base weight and
  train only the rank-by-rank `M_xs` core plus any DoRA/BoRA magnitudes.
- `mlx.nn.value_and_grad` and `mlx.optimizers.AdamW` update only adapters.
- Rectified-flow loss backpropagates through the SA3 Medium MLX DiT.
- Saved `.safetensors` use the same keys and metadata contract as the Windows
  Underfit trainer and the existing Mac inference loader.
- The terminal trainer can load cached SA3 Medium RF weights, recursively
  discover an audio folder, pre-encode each track through the MLX autoencoder,
  build frozen per-track prompt/duration conditioning, and randomly sample
  tracks and latent crops during training.
- Trainer defaults match Underfit: 2,000 steps, 47-second latent crops, and
  checkpoints every 500 steps.
- MLX pre-encoding uses the same half-precision default as the Windows path.
  Short clips are encoded only at their real duration, then padded in latent
  space with the padded region masked out of training.
- Sources longer than 30 seconds use overlapping chunked SAME-L encoding,
  matching Underfit's long-file pre-encoding strategy.
- A validation renderer can load the trained checkpoint through the existing MLX
  LoRA inference runtime and produce matched base-versus-LoRA WAVs.

Run one synthetic gradient step through the real SA3 Medium DiT structure:

```bash
cd sa3
python scripts/smoke_mlx_lora_training.py \
  --model medium \
  --adapter-type dora \
  --output /tmp/mlx_sa3_lora_gradient_smoke.safetensors
```

This command does not load base weights or audio. It validates the MLX
autodiff and checkpoint boundary.

## One-Audio Training Smoke

With `stable-audio-3-medium-RF.json` and
`stable-audio-3-medium-RF.safetensors` already present in the Hugging Face
cache, run:

```bash
cd sa3
python scripts/train_mlx_lora.py \
  --audio-path /path/to/example.wav \
  --output-dir "$HOME/Library/Application Support/GaryLocalhost/training/mlx-one-file-smoke" \
  --steps 500 \
  --crop-seconds 8 \
  --rank 4 \
  --adapter-type dora \
  --learning-rate 0.0001 \
  --save-every 100 \
  --log-every 10
```

The trainer defaults to DoRA rows, matching the Windows Gary trainer. Pass
`--adapter-type lora` to train a standard LoRA or `--adapter-type bora` to train
two-axis BoRA magnitudes instead. Use `--adapter-type lora-xs` for the primary
extra-small SVD variant; `dora-rows-xs`, `dora-cols-xs`, and `bora-xs` are also
available for Underfit parity. The default adapter target is all MLX `Linear`
layers in transformer blocks 20-23. The trainer keeps the Hugging Face snapshot
symlink filename intact so the shared loader recognizes `.safetensors`
checkpoints even when the cache blob itself has no extension.

For folder training, give each audio file an optional same-name `.txt` prompt
or JSON metadata sidecar and run:

```bash
cd sa3
python scripts/train_mlx_lora.py \
  --dataset-dir /path/to/audio-folder \
  --trigger-text garybell \
  --output-dir "$HOME/Library/Application Support/GaryLocalhost/training/mlx-folder-smoke" \
  --steps 500 \
  --crop-seconds 8 \
  --rank 16 \
  --adapter-type dora \
  --learning-rate 1e-4
```

JSON sidecars take precedence over text sidecars. The optional trigger is
prepended to every effective training prompt without duplicating an existing
prefix.

Render a matched comparison after training:

```bash
cd sa3
python scripts/validate_mlx_lora.py \
  --lora-path "$HOME/Library/Application Support/GaryLocalhost/training/mlx-one-file-smoke/gary-mlx-lora-final.safetensors" \
  --output-dir "$HOME/Library/Application Support/GaryLocalhost/training/mlx-one-file-smoke/validation" \
  --duration 8 \
  --steps 50 \
  --seed 20260607
```

## Control Center Integration

Select the `sa3` service in Gary Control Center and choose **Train LoRA...**.
The native MLX trainer accepts a folder of audio files and exposes:

- LoRA name, custom trigger text, and recursive folder selection
- A per-track `.txt` prompt editor with template fill and JSON-sidecar warnings
- LoRA, DoRA, BoRA, and XS adapter variants
- Step count, rank, crop duration, learning rate, and checkpoint interval
- Stable Audio 3/Underfit timestep parity by default: truncated logit-normal
  sampling followed by the model-configured training distribution shift
- Optional per-track latent-RMS loudness normalization with a `0.90`
  base-model target and iterative MLX encoding passes
- Persistent progress, log tail, cancellation, and completed-checkpoint reveal

Pre-encoding is performed once per source file and cached in the run directory.
Long files take proportionally longer because the full source is retained for
random crop selection. The experimental loudness fix can run several encoding
passes per track, so it remains opt-in.

Jobs live under:

```text
~/Library/Application Support/GaryLocalhost/sa3/training/jobs
```

On success, the job runner copies the final checkpoint into the configured
`SA3_LORA_DIR`, updates `SA3_LORA_REGISTRY`, and writes a matching prompt dice
file under `SA3_PROMPTS_DIR`. Restart or reload SA3 before using a newly trained
adapter if the inference model is already loaded.

The app always uses the base model's training timestep defaults. Direct CLI
runs may override the timestep sampler or distribution shift for research and
compatibility experiments.

## Current Gaps

1. Add production resume semantics and multi-file job recovery.
2. Add multi-file batching and optional adapter target presets.
3. Add validation-demo generation to the native training job.

Precomputed reusable SVD-basis files are not wired into the terminal trainer
yet, so XS injection currently computes bases from the loaded model weights.
