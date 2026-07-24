# SA3 MLX training and upstream integration handoff

Date: 2026-07-24  
Primary repository: `/Users/karenjessen/gary-localhost-installer-mac`  
Official upstream checkout used in this work: `/Users/karenjessen/stable-audio-3-official`  
PC sister project: `/Users/karenjessen/gary-localhost-installer`

## Read this first

This document records the current implementation and the evidence behind it. It
is intended to let a fresh agent continue without rediscovering the model
contract, training bugs, upstream differences, build procedure, or unresolved
inference/download questions.

The most important facts are:

1. **Training must use Stable Audio 3 `medium-base`. Inference applies the
   resulting adapter to Stable Audio 3 `medium`.** Do not train on the ARC /
   inference checkpoint.
2. Gary now supports two training DiT implementations:
   - `gary-generic` — the default, faster and lower-memory implementation.
   - `official-specialized` — the official optimized medium DiT, retained as an
     experimental fixed-crop A/B path.
3. Both training engines load the **same official hosted FP16 medium-base NPZ**
   from `stabilityai/stable-audio-3-optimized`.
4. The completed 2,000-step Billie listening test produced different audio from
   the same inference seed/prompt, but the user judged both adapters to be
   essentially the same quality at LoRA strength 2.
5. That listening test isolates the **DiT implementation only**. It does not
   compare Gary's trainer with the official standalone trainer. Both engines
   were driven by Gary's dataset, masks, loss, optimizer, layer selection, and
   checkpoint code.
6. Gary's current training graph is faster and uses less memory than the
   official specialized graph on this base M4 MacBook Air. There is no evidence
   yet that the heavier official graph buys better adapter quality.
7. Inference still constructs the pipeline through PyTorch and converts it to
   MLX at runtime. Moving inference to hosted MLX weights is the logical next
   project, but it should be incremental and reversible.
8. The working tree is intentionally dirty and contains all of this session's
   work plus earlier user-requested work. Do not reset, clean, or overwrite it.

## Repository state

At the time of this handoff:

- `gary-localhost-installer-mac`
  - Branch: `codex/ace-step-mlx-training`
  - HEAD before the uncommitted work: `b0a0442`
  - The SA3 and UI changes described here are uncommitted.
- `stable-audio-3-official`
  - Branch: `main`
  - HEAD: `124e8a7`
  - This checkout was clean.
- `gary-localhost-installer`
  - Branch: `main`
  - HEAD: `4eb7993` (`v0.2.0`)

There was no `/Users/karenjessen/stable-audio-3` Git checkout when this document
was written. The official checkout actually inspected and benchmarked was
`/Users/karenjessen/stable-audio-3-official`. Verify paths rather than assuming
the shorter name exists.

## Installed application state

The current A/B build is installed and running at:

```text
/Applications/gary4local.app
```

It remains bundle version `0.1.11`, build `111`; this was an ad-hoc local
production build, not a versioned release.

The immediately preceding fused-RoPE build is recoverable at:

```text
/Applications/gary4local.backup-pre-engine-ab-20260724.app
```

Earlier backups also exist:

```text
/Applications/gary4local.backup-pre-fast-rope-20260724.app
/Applications/gary4local.backup-pre-wired-memory-20260723.app
/Applications/gary4local.backup-pre-full-track-optimizations-20260723.app
```

The official optimized HF cache was already present under `~/.cache`. For this
Mac's installed app, a symlink was added so the app-specific HF cache can reuse
it without downloading another 2.91 GB:

```text
~/Library/Application Support/GaryLocalhost/cache/huggingface/hub/
  models--stabilityai--stable-audio-3-optimized
    -> ~/.cache/huggingface/hub/
       models--stabilityai--stable-audio-3-optimized
```

This is a development-machine convenience, not yet a product migration design.

## Features ported or implemented during this work

### Dataset preparation and prompt UX

- Ported BPM/key suggestion helpers from the PC/Carey flow.
- Added the SA3 **suggest BPM/key** action.
- Added **auto-label all** using ACE-Step 1.5's understand-music path, including
  the generated genre.
- Kept auto-label state management separate from training state.
- The user production-tested BPM/key suggestion and auto-labeling with the
  Ratatat dataset; both worked. The dependency path also worked without first
  pressing “rebuild environment.”
- Removed the unnecessary “prepend shared trigger phrase” control from the
  prompt editor.
- The shared trigger is now applied only when building the training prompt:
  - blank means no trigger;
  - it is not written into `.txt` sidecars;
  - it does not pollute prompt-dice entries.
- Training logs now distinguish source duration from encoding elapsed time.
- Training logs print an example final conditioning prompt before steps begin.

Important files:

```text
gary4local/gary4local/SA3DatasetPromptEditor.swift
gary4local/gary4local/SA3PromptMetadata.swift
ace-lego/wrapper/sa3_autolabel.py
sa3/scripts/analyze_audio.py
sa3/scripts/mlx_lora_dataset.py
```

### Full-track training

- Added the **train on full tracks** mode.
- The current maximum window is `285.35` seconds, beginning at `0:00`.
- Longer tracks are truncated after that window.
- Shorter tracks are extended with an encoded silence latent rather than raw
  zero latent padding.
- The fixed 285.35-second policy is deliberate but not sacred. It is a useful
  middle ground for full musical progression without forcing every short song
  to carry padding out to 380 seconds.
- Added variable full-track buckets rounded to 512 latent frames.
- Added per-bucket compiled training steps.
- Gradient checkpointing defaults on for full-track training and off for short
  random crops.
- The official-specialized experimental engine is currently restricted to
  random crops because the official model does not consume Gary's padding mask.

Important files:

```text
gary4local/gary4local/SA3LoraTrainingView.swift
sa3/scripts/train_mlx_lora.py
sa3/scripts/train_mlx_lora_job.py
sa3/stable_audio_3/mlx/training.py
```

### Correct medium-base contract and trainer parity fixes

The most serious regression investigated in this session was the familiar
“drone/hum at higher LoRA strength” failure mode. The critical model contract is:

```text
train:     stabilityai/stable-audio-3-medium-base
inference: stabilityai/stable-audio-3-medium
```

Hugging Face's repository layout changed: `medium-base` now has its own model
repository. Training asset resolution now fails closed unless both
`model_config.json` and `model.safetensors` come from that base repository and
the config is not ARC/distilled.

Other parity corrections made while investigating the hum:

- Added/used an encoded silence latent for padding.
- Restored the PyTorch training-style mixed inpainting conditioning:
  - 10% random segments;
  - 80% full generation;
  - 10% causal;
  - zero means generate, one means context;
  - context reconstruction loss weight 1.0.
- Uses `trunc_logit_normal` timestep sampling by default.
- Uses the model's full distribution shift and per-example effective sequence
  length.
- AdamW uses `(0.9, 0.95)`, epsilon `1e-8`, weight decay `0.01`, and bias
  correction.
- Adapter initialization is seeded before injection.
- Training PRNG is reset before the step loop.
- Adapter `lora_A` matrices are now initialized from a stable hash of
  `seed + normalized layer name`, so different module traversal orders cannot
  contaminate an engine A/B.

The name-derived initialization fix matters: before it, the generic and
specialized models found the same 228 layers but initialized all 228 `lora_A`
matrices differently because their traversal orders differ. After the fix, all
684 DiT adapter tensors are byte-identical at initialization. The seconds
conditioner contributes three more tensors in a complete rank-16 DoRA
checkpoint.

### Checkpoint registration and live adapter switching

- Exported checkpoints are automatically installed and entered into Gary's SA3
  LoRA registry.
- Registration is not tied to opening the “add LoRAs” UI.
- The LoRA dropdown includes Gary-trained checkpoints automatically.
- Switching a checkpoint while SA3 inference is resident triggers an automatic
  adapter/model reload. The user no longer needs to press restart.
- The gary4juce-visible catalog updates without first opening the manager.

Important files:

```text
gary4local/gary4local/ControlCenterView.swift
gary4local/gary4local/ControlCenterViewModel.swift
gary4local/gary4local/SA3LoraTraining.swift
sa3/api.py
sa3/scripts/train_mlx_lora_job.py
```

### MLX performance work

- Ported the non-materializing DoRA forward.
- Added MLX wired-memory configuration using the recommended working-set limit.
- Added active/cache/peak MLX memory telemetry.
- Added full-track gradient checkpointing.
- Added full-track length buckets and one compiled graph per bucket.
- Added fused `mx.fast.rope` with a guarded fallback to the prior manual path.
- Added a `SA3_MLX_NAIVE_ROPE=1` diagnostic escape hatch.

The fast-RoPE path is only used when query and key lengths match and there is no
separate key rotary embedding. Cross/asymmetric rotary cases retain the manual
implementation.

Fast-RoPE validation:

- FP32/FP16 forward parity tests.
- Gradient parity tests.
- Differential-attention forward/input-gradient tests.
- At 23-second training length, roughly 2.2% steady step improvement.
- At 1,024 latents, generic forward median improved from about `1.639 s` to
  `1.553 s` (about 5.3%).
- At 1,024 latents, fused-versus-manual relative L2 output difference was only
  about `0.069%`; the larger generic-versus-specialized drift was not caused by
  fused RoPE.

## Hosted medium-base weights and the two training engines

### Hosted artifact

Training now resolves:

```text
repo:     stabilityai/stable-audio-3-optimized
filename: MLX/dit_medium-base_f16.npz
size:     approximately 2.91 GB
```

Resolver:

```text
sa3/scripts/mlx_training_assets.py
```

### Gary generic hosted loader

The hosted NPZ maps cleanly into Gary's existing generic DiT:

- 522 DiT tensors total.
- 258 names match directly.
- The remaining 264 require only two mechanical translations:
  - Gary RMSNorm `.gamma` -> hosted `.weight`
  - Gary `.to_local_embed.N` -> hosted `.to_local_embed.seq.N`

Loading the hosted NPZ into Gary's generic model produced bit-for-bit identical
output to Gary's prior runtime-converted medium-base model in the direct
256-latent check. Direct DiT loading took about `1.26 s`.

Implementation:

```text
StableAudioMLXDiT.from_hosted_medium_npz(...)
sa3/stable_audio_3/mlx/dit.py
```

The method name currently says “medium,” and the training resolver says
“medium-base.” When inference is migrated, consider generalizing the loader
name because the same key mapping should be applicable to the hosted ARC
`dit_medium_f16.npz` after explicit validation.

### Official specialized engine

The official optimized medium implementation was vendored from:

```text
/Users/karenjessen/stable-audio-3-official/
  optimized/mlx/models/defs/dit_mlx_medium.py
```

Vendored target:

```text
sa3/stable_audio_3/mlx/dit_medium_official.py
```

It is available in the training UI as:

```text
official specialized (experimental)
```

The wrapper adapts the official channels-last local condition input while
keeping Gary's external training contract unchanged.

The official model has two generated values that are not loaded as ordinary
parameters in the same way as the generic model:

- `timestep_features.freqs`
- `transformer.rotary_pos_emb.inv_freq`

Its loader deliberately mirrors upstream's non-strict treatment for those
generated values.

### Checkpoint normalization

The official specialized model calls its local projection:

```text
transformer.layers.N.to_local_embed.seq.M
```

Gary inference expects:

```text
transformer.layers.N.to_local_embed.M
```

Before saving, specialized adapter `source_name` values are normalized to the
Gary form. A two-step specialized checkpoint was loaded into Gary's generic
inference DiT successfully:

- 228 DiT layers loaded;
- 228 applied;
- no skipped DiT layers;
- the seconds conditioner is applied separately by the pipeline, as expected.

## What the listening A/B did and did not prove

The user completed two 2,000-step, rank-16, 23-second-crop Billie runs:

- `billie-gary`
- `billie-official`

Persistent jobs and installed adapters:

```text
~/Library/Application Support/GaryLocalhost/sa3/training/jobs/
  20260724-080742-billie-gary
  20260724-090719-billie-official

~/Library/Application Support/GaryLocalhost/sa3/loras/
  billie-gary.safetensors
  billie-official.safetensors
```

At inference with the same prompt and seed, strength 2:

- outputs were different;
- subjective quality was essentially the same.

This is a useful result: it argues against replacing the generic engine merely
because the specialized graph is official.

However, this was a deliberately controlled **engine-only** A/B. Both runs used:

- the same hosted medium-base NPZ;
- Gary's 228 adapted DiT projections;
- Gary's seconds conditioner adapter;
- name-stable identical adapter initialization;
- Gary's dataset order and random crop selection;
- Gary's 10/80/10 mixed inpainting masks;
- Gary's noise/timestep/mask seeds;
- Gary's rectified-flow loss;
- Gary's AdamW configuration;
- Gary's checkpoint exporter.

It did **not** compare:

- Gary's trainer against
  `optimized/mlx/scripts/lora_train_mlx.py`;
- upstream's default 168-layer exclusion policy against Gary's 228 layers;
- upstream's all-ones local conditioning against Gary's mixed masks;
- upstream's per-step global grad/adapter norm telemetry and clipping against
  Gary's lighter step;
- PyTorch Lightning (intentionally out of scope).

Do not use this listening result to claim the entire official upstream trainer
is equivalent. It says the official specialized **DiT implementation** did not
sound better under an otherwise identical Gary training loop.

### Completed 2,000-step run metrics

Both jobs recorded:

```text
seed:                 20260607
trainable parameters: 21,584,768
latent bucket:        256
mask counts:          causal 196, full 1,607, random segments 197
```

| Metric | Billie Gary | Billie official |
|---|---:|---:|
| First loss | 0.799996 | 0.798332 |
| Final loss | 0.492590 | 0.489633 |
| Mean step time, excluding compile | 1.399 s | 1.799 s |
| Early mean, steps 2–100 | 1.210 s | 1.509 s |
| Late mean, steps 1501–2000 | 1.460 s | 1.837 s |
| Total training-loop time | 2,800.85 s | 3,600.84 s |
| Peak MLX memory | 8.60 GB | 12.35 GB |

The specialized engine took almost exactly 800 seconds longer over 2,000 steps
(about 28.6% more training-loop wall time) and used about 3.75 GB more peak MLX
memory. Its slightly lower final loss did not translate into an obvious
subjective quality advantage in this test.

## Performance evidence

Hardware: base Apple M4 MacBook Air, 32 GB, fanless.

### Model-only forward

| Latents | Gary generic median | Official specialized median | Result |
|---:|---:|---:|---|
| 256 | 0.453 s | 0.633 s | Gary about 28% faster |
| 1,024 | 1.553 s | 1.957 s | Gary about 21% faster |

Peak memory in the 256-latent forward check:

- Gary: about `2.94 GB`
- Official: about `3.62 GB`

At 256 latents, output comparison:

- cosine similarity: `0.999992`
- relative L2 difference: about `0.397%`

At 1,024 latents with intentionally random/high-amplitude synthetic
conditioning:

- cosine similarity: about `0.9992`
- relative L2 difference: about `5.08%`

The long-sequence synthetic drift remained with manual RoPE, so it is not a
fast-RoPE regression. It may be reduction-order amplification through 24 FP16
blocks under unrealistic random conditioning. If this is revisited, use
realistic cached T5 conditioning and latents, then compare both against the
official PyTorch parity harness.

### Two-step training smoke inside Gary's loop

Same cached “bad guy” latent, rank-16 DoRA, 23-second crop:

| Engine | Compile/first step | Steady second step | Peak MLX memory |
|---|---:|---:|---:|
| Gary generic | 2.40 s | 1.42 s | 8.58 GB |
| Official specialized | 3.77 s | 1.87 s | 12.35 GB |

Losses were close but not identical, as expected from FP16 reduction ordering:

```text
generic:     0.56404477, 0.37703255
specialized: 0.56209832, 0.37544620
```

### Official standalone trainer controls

Using upstream's own `lora_train_mlx.py` at 256 latents:

- default exclusions, 168 DiT layers:
  - steady about `1.64 s/step`
  - peak about `12.39 GB`
- empty exclusions, 228 DiT layers:
  - steady about `1.70 s/step`
  - peak about `12.52 GB`

This is not perfectly apples-to-apples because upstream computes global gradient
and adapter norms every step and the control enabled gradient clipping. It also
keeps T5Gemma resident during training. Gary precomputes prompt conditioning
then frees T5 and the VAE before the step loop.

### Thermal behavior

During the longer `billie-fast` observation, MLX memory stayed flat while steps
slowed from roughly `1.22–1.27 s` early to roughly `1.9 s` later. Active memory,
cache, and peak did not grow. On this fanless MacBook Air that strongly suggests
thermal throttling rather than an allocation leak.

For future benchmarks:

- compare early steady steps, not only the late average;
- record machine temperature/cooling state or allow a cooldown;
- run engine order both ways if timing differences matter;
- do not infer a memory leak from late slowdown without memory growth.

## Current inference architecture

Inference is **not yet using the hosted optimized MLX bundle directly**.

Current load path:

```text
sa3/api.py
  load_pipeline()
    StableAudioMLXPipeline.from_torch_pretrained(...)
```

That path:

1. resolves/downloads the PyTorch Stable Audio 3 model;
2. constructs the upstream Torch pipeline;
3. converts DiT, T5Gemma, seconds conditioner, and autoencoder to MLX;
4. keeps `torch_pipeline` because several helpers still read model metadata and
   length information from it;
5. preloads all registered LoRAs;
6. supports live adapter switching through `MLXLoRASet`.

Relevant code:

```text
sa3/api.py
sa3/stable_audio_3/mlx/pipeline.py
sa3/stable_audio_3/mlx/lora.py
```

The current training path also still constructs the PyTorch medium-base pipeline
for VAE encoding and T5/seconds conditioning before discarding the converted DiT
and replacing it with the hosted training DiT. In the two-step smoke runs:

- PyTorch pipeline construction/conversion took roughly 14 seconds;
- hosted DiT replacement took roughly 1.2 seconds.

Therefore, “training uses the hosted DiT” is true, but “training no longer needs
PyTorch assets” is **not** yet true.

## Recommended hosted-MLX inference migration

My recommendation is to preserve Gary's generic inference engine and migrate
the weight/component loading beneath it. Do not switch wholesale to the
official specialized runtime simply because it is upstream.

### Phase 1: direct hosted inference DiT behind a feature flag

Use the hosted ARC/inference artifact, not the training base:

```text
stabilityai/stable-audio-3-optimized
MLX/dit_medium_f16.npz
```

Keep:

- Gary's generic `StableAudioMLXDiT`;
- Gary's custom sampler behavior;
- Gary's continuation/inpainting/padding-mask support;
- Gary's dynamic LoRA strength, schedule, and switching code;
- current Torch-converted T5, conditioner, and SAME-L codec initially.

Replace only the inference DiT load, then verify:

1. all hosted ARC DiT keys map into the generic model;
2. direct-hosted generic output matches current runtime-converted generic output;
3. empty prompt/padding embedding behavior is preserved;
4. medium adapter application loads every intended DiT layer;
5. live adapter switching still reloads without a manual service restart;
6. identical prompt/seed/sampler settings produce equivalent decoded audio;
7. continuation and inpainting still honor padding masks;
8. model unload/reload and memory telemetry remain correct.

This is the lowest-risk next step because it establishes the direct inference
DiT loader without changing Gary's execution graph. By itself it will not yet
eliminate the PyTorch download or full Torch model construction: the current
factory loads the monolithic Torch pipeline to obtain T5, conditioner, codec,
and metadata. Treat Phase 1 primarily as a parity and LoRA compatibility
milestone. Startup and disk savings arrive only after the pipeline factory can
skip Torch DiT construction and load the remaining components independently.

### Phase 2: direct hosted conditioner and T5Gemma

Hosted files:

```text
MLX/t5gemma_f16.npz
cond.* entries bundled in MLX/dit_medium_f16.npz
```

The DiT NPZ contains the learned padding embedding and seconds conditioner.
Port/load them into Gary's conditioner classes, then remove the inference
pipeline's dependency on the Torch conditioner.

Verify:

- empty and short prompts use the learned padding embedding;
- prompt tokenization and padding are equivalent;
- text and seconds LoRA targets still resolve;
- trigger/dice prompts remain unchanged;
- conditioner outputs match the current path on a prompt fixture set.

### Phase 3: direct hosted SAME-L encoder/decoder

Hosted files:

```text
MLX/same_l_encoder_f32.npz
MLX/same_l_decoder_f32.npz
```

The official implementation intentionally runs SAME-L codec paths in FP32.
Preserve that unless parity testing justifies another dtype.

After direct codec loading, remove uses of `torch_pipeline` for:

- sample rate;
- downsampling ratio;
- audio sample/latent length calculations;
- any remaining config lookup.

Move those values into explicit fields on `StableAudioMLXPipeline` rather than
leaving a fake Torch object solely to satisfy `generation_ready`.

### Phase 4: training preparation without PyTorch

Training cannot discard the medium-base PyTorch repository until the hosted
components cover:

- SAME-L encoding for uncached datasets;
- T5Gemma conditioning;
- seconds conditioner;
- model config/distribution-shift metadata;
- silence-latent construction.

The official optimized repository already contains most of these pieces. Port
them incrementally into Gary's pipeline while retaining the content-addressed
latent cache and current training loop.

Only after an uncached dataset can be prepared and trained end to end without
touching PyTorch should medium-base PyTorch cleanup be considered.

## Download UI and cleanup recommendation

If hosted MLX becomes the production path, the SA3 download UI should become a
component inventory rather than a single opaque “download models” action.

Suggested inventory rows:

```text
Medium inference DiT       MLX/dit_medium_f16.npz
Medium training base DiT   MLX/dit_medium-base_f16.npz
T5Gemma                    MLX/t5gemma_f16.npz
SAME-L encoder             MLX/same_l_encoder_f32.npz
SAME-L decoder             MLX/same_l_decoder_f32.npz
Legacy PyTorch inference   stabilityai/stable-audio-3-medium
Legacy PyTorch training    stabilityai/stable-audio-3-medium-base
```

Show:

- required-for-inference / required-for-training labels;
- downloaded size and expected size;
- current runtime source (“hosted MLX” or “legacy conversion”);
- migration/verification state;
- a focused retry button for each missing component.

### Do not silently delete PyTorch models

The kind user experience is an **assisted cleanup**, not automatic deletion
without consent.

Recommended flow:

1. Download every required optimized MLX component.
2. Verify checksums and perform a real load/generation health check.
3. Switch runtime preference to hosted MLX.
4. Keep the legacy PyTorch cache through at least one successful app restart and
   generation.
5. Show exact reclaimable bytes and exact repositories.
6. Offer:
   - “keep legacy models for rollback”
   - “remove legacy inference model”
   - later, only when training is PyTorch-free, “remove legacy training model”
7. Move to Trash or use a recoverable cache operation where practical.
8. Never recursively delete a broad Hugging Face cache root.

Inference and training repositories must be treated separately. Direct hosted
inference may make `stable-audio-3-medium` unnecessary while
`stable-audio-3-medium-base` is still needed for uncached training preparation.

Also account for:

- users who deliberately retain both backends;
- interrupted downloads;
- symlinked snapshots;
- HF cache revisions/refs;
- free disk required during migration;
- rollback after an app update.

## Upstream areas still worth reviewing

Official source root:

```text
/Users/karenjessen/stable-audio-3-official/optimized/mlx
```

High-value files:

```text
models/defs/dit_mlx_medium.py
models/defs/lora.py
models/defs/latent_dataset.py
models/defs/training.py
models/defs/sa3_pipeline.py
models/defs/t5gemma_mlx.py
scripts/lora_train_mlx.py
scripts/pre_encode_mlx.py
scripts/sa3_mlx.py
scripts/weights.py
TRAINING_CONVENTIONS.md
README.md
```

Questions to keep open:

- Why is the specialized graph slower/heavier on this M4 Air despite upstream
  inference benchmarks?
- Can upstream's medium class be improved to match the generic graph without
  losing its excellent PyTorch parity?
- Should upstream free T5Gemma after conditioning during training?
- Should upstream's standalone trainer support Gary/PC's mixed inpainting masks
  rather than an all-ones local condition?
- Should global grad/adapter norm telemetry be optional because it adds a
  full-tree reduction every step?
- Which layer exclusions give the best quality/speed tradeoff?
- Can the official-specialized model gain padding-mask support without
  perturbing parity?
- Can Gary's full-track length buckets be contributed back to upstream and the
  PC implementation?
- Can the name-stable per-layer adapter initialization be contributed upstream
  for reproducible engine/filter comparisons?

PyTorch Lightning remains intentionally out of scope for the Mac trainer.

## Layer-exclusion experiment still pending

The official standalone trainer's product default excludes:

```text
to_timestep_embed
to_cond_embed
to_global_embed
to_local_embed
global_cond_embedder
project_in
project_out
preprocess_conv
postprocess_conv
```

That adapts 168 DiT layers rather than Gary's current 228.

The user expressed interest in exposing this as a toggle. Do not conflate this
with the engine selector. A good future UI would present a clearly named
training scope such as:

- **all projections (Gary current)** — 228 DiT layers;
- **attention + feed-forward only (upstream product default)** — 168 layers.

Run it as a separate quality/speed A/B with the same name-stable initialization,
not at the same time as an engine comparison.

## Production build-and-test loop used in this session

### Python tests

```bash
cd /Users/karenjessen/gary-localhost-installer-mac
PYTHONPATH=sa3 sa3/.venv/bin/pytest -q sa3/tests
```

Current result:

```text
66 passed
```

### Release build

```bash
xcodebuild \
  -project gary4local/gary4local.xcodeproj \
  -scheme gary4local \
  -configuration Release \
  -derivedDataPath /tmp/gary4local-ab-019f8d07 \
  ARCHS=arm64 \
  ONLY_ACTIVE_ARCH=NO \
  CODE_SIGNING_ALLOWED=NO \
  build
```

### Verify staged runtime

Before installation, confirm newly added Python files actually landed in the app:

```bash
test -f \
  /tmp/gary4local-ab-019f8d07/Build/Products/Release/gary4local.app/Contents/Resources/runtime/sa3/stable_audio_3/mlx/dit_medium_official.py
```

### Ad-hoc sign and verify

```bash
codesign --force --deep --sign - \
  /tmp/gary4local-ab-019f8d07/Build/Products/Release/gary4local.app

codesign --verify --deep --strict \
  /tmp/gary4local-ab-019f8d07/Build/Products/Release/gary4local.app
```

### Replace the production app safely

The working pattern was:

1. stop the exact running `/Applications/gary4local.app` process;
2. move the existing app to a uniquely named backup;
3. copy the new app with `ditto`;
4. verify its signature;
5. launch it;
6. confirm the process path;
7. run a real training/inference sanity check.

Do not overwrite the existing backup names. Do not delete the previous app
until the replacement has been exercised.

## Useful test artifacts and logs

Temporary artifacts may disappear after reboot, but were:

```text
/tmp/gary-medium-forward-parity-019f8d07
/tmp/gary-upstream-medium-default.log
/tmp/gary-upstream-medium-all-layers.log
/tmp/gary-ab-smoke-generic
/tmp/gary-ab-smoke-official
/tmp/gary-ab-init-generic.safetensors
/tmp/gary-ab-init-official.safetensors
```

Persistent Gary training jobs/logs:

```text
~/Library/Application Support/GaryLocalhost/sa3/training/jobs
~/Library/Application Support/GaryLocalhost/sa3/training/latent-cache
```

The Billie dataset used here:

```text
/Users/karenjessen/Downloads/billie
```

## Suggested next session order

1. Read this file and inspect the dirty Git diff; do not reset it.
2. Confirm the completed `billie-gary` and `billie-official` job logs,
   checkpoints, and registry entries.
3. Preserve the user listening conclusion: same apparent quality, different
   audio, specialized engine slower/heavier.
4. Add a feature-flagged direct hosted **inference ARC DiT** loader using Gary's
   generic model.
5. Run tensor/output parity and LoRA-switching tests before changing defaults.
6. Only then adapt the SA3 model download UI to expose optimized components.
7. Move T5/conditioner and SAME-L codec loading to hosted MLX incrementally.
8. Keep legacy PyTorch models as rollback until the hosted path survives app
   restart, generation, continuation/inpainting, adapter switching, and an
   uncached training preparation.
9. Add the layer-scope exclusion toggle as a separate experiment.
10. Once the implementation is stable, review and intentionally split/commit
    the large dirty working tree rather than creating one opaque commit.

## Bottom-line recommendation

The work was not in vain and should not be “untangled” by deleting Gary's
generic path. The evidence currently supports:

- **Gary generic DiT as the default execution graph**
- **official hosted MLX weights as the preferred weight source**
- **official specialized DiT retained as an experimental/parity reference**
- **incremental hosted-component migration**
- **assisted, opt-in cleanup of legacy PyTorch caches only after verified
  replacement**

That approach incorporates upstream's strongest contribution—the maintained,
hosted MLX artifacts—without giving up Gary's proven masking, LoRA switching,
training conventions, memory behavior, or performance.
