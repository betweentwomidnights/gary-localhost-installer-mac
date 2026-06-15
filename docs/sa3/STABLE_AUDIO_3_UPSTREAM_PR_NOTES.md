# Stable Audio 3 MLX Upstream PR Notes

> Internal project record. This document is deliberately more detailed and
> Gary-specific than the eventual public pull request description. It should
> not be copied into the upstream PR verbatim.

This document records how Gary4local's working MLX LoRA implementation was
adapted into a focused contribution for the official Stable Audio 3 repository.
It is intended to support maintainer review now and make a future return to
officially vendored SA3 and Underfit packages easier.

## Contribution Plan

The work is intentionally split into two Stable Audio 3 pull requests.

### PR 1: MLX Training and LoRA Primitives

The first PR adds reusable model-level functionality to the existing
`optimized/mlx` runtime:

- Trainable MLX `Linear` and `Conv1d` adapters
- LoRA, DoRA rows and columns, BoRA, and their XS variants
- Official SA3/Underfit-compatible safetensors loading and saving
- Fixed-strength MLX LoRA materialization for one-shot inference
- Rectified-flow training loss
- Training timestep samplers and distribution shifts
- Focused MLX and PyTorch parity tests

This PR does not add a dataset trainer, command-line training workflow, Gradio
backend, or Gary-specific application behavior. Its purpose is to establish the
smallest reusable public boundary that those integrations can consume.

### PR 2: MLX CLI and Gradio Integration

The follow-up PR will make the primitives user-facing through the official
runtime. Its likely scope is:

- LoRA checkpoint and strength flags for `optimized/mlx/sa3`
- Extraction of the monolithic MLX CLI into a reusable generation pipeline
- A maintainable MLX Gradio entry point using that pipeline
- Documentation and integration tests for terminal and Gradio inference

The existing Gradio interface cannot honestly support MLX through only a
`--backend mlx` flag. It currently reads PyTorch-specific model configuration,
sampler, pretransform, preview, and mutable LoRA state. A reusable MLX pipeline
must be established before a Gradio backend can follow those patterns without
duplicating the inference implementation or presenting incomplete parity.

## Downstream Provenance

The upstream work was derived from the implementation proven in
`betweentwomidnights/gary-localhost-installer-mac` on branch
`feature/train-sa3`, primarily:

- Commit `2ebcd13`: `Add SA3 MLX backend and tuning controls`
- Commit `be1a8d8`: `Add MLX LoRA training for Stable Audio 3`
- `sa3/stable_audio_3/mlx/lora.py`
- `sa3/stable_audio_3/mlx/training.py`
- `sa3/scripts/train_mlx_lora.py`
- `sa3/tests/test_mlx_training.py`

Gary's implementation remains the end-to-end reference: audio discovery and
pre-encoding, prompt conditioning, optimization, checkpointing, inference,
native job orchestration, and Control Center UI have all been exercised by
Gary4local users on Apple Silicon.

## Adaptations for Official SA3

The upstream contribution is not a direct copy of Gary's vendored backend.
The implementation was reshaped around the official repository's existing
optimized MLX architecture and dependency policy.

### Official Module Boundary

Gary owns a reusable application pipeline under
`sa3/stable_audio_3/mlx`. The upstream implementation instead targets the
official `optimized/mlx/models/defs` package so it works with the project's
existing small and medium optimized DiT definitions.

The contribution adds only `lora.py` and `training.py` primitives. It does not
introduce Gary's parallel model loader, autoencoder, conditioning, sampling,
pipeline, API, or runtime hierarchy.

### Runtime Dependencies

Gary can rely on the larger service environment it installs. The official
optimized runtime promises pure MLX inference without PyTorch,
`stable-audio-tools`, or a separate safetensors package at runtime.

The upstream checkpoint implementation therefore uses MLX's native
`load`, `save_safetensors`, and metadata support. PyTorch is used only in tests
to verify compatibility with the official loader and adapter math.

### Official Optimized Model Names

Checkpoint keys use the canonical PyTorch SA3 names while the optimized MLX
model has a small number of structural naming differences. The upstream loader
maps both directions between:

- `to_local_embed.0` and `to_local_embed.seq.0`
- `to_local_embed.2` and `to_local_embed.seq.2`

This lets checkpoints produced by the optimized runtime load through the
official PyTorch implementation and lets existing official-format checkpoints
target the optimized MLX model.

### Convolution Weight Layout

PyTorch `Conv1d` checkpoints store weights as `[out, in, kernel]`. MLX uses
`[out, kernel, in]`. The upstream adapter code explicitly converts between
these layouts before flattening or restoring LoRA-family updates.

This behavior has a dedicated parity test because shape-compatible transposes
can otherwise produce a valid checkpoint with incorrect audio behavior.

### Generic Layer Selection

Gary currently exposes product defaults, including late-transformer-block
targeting intended for approachable micro-training. The upstream primitive
accepts generic include and exclude patterns and does not encode Gary's target
preset, model choice, rank, crop length, step count, or learning rate.

Callers such as Underfit or Gary remain responsible for choosing those
policies.

### Adapter Coverage

The upstream implementation supports the complete official adapter family:

- LoRA
- DoRA rows
- DoRA columns
- BoRA
- LoRA-XS
- DoRA rows XS
- DoRA columns XS
- BoRA-XS

Aliases such as `dora`, `dora-xs`, and `xs` are normalized to canonical
checkpoint metadata. This is broader than Gary's simplified UI, which exposes
the useful product choices without requiring users to understand every
underlying variant.

### Inference Application

Gary's runtime supports dynamic, scheduled, multi-LoRA application for UI
sliders, timestep intervals, and layer filters. The first upstream PR uses a
smaller primitive: one or more checkpoints are materialized into a loaded MLX
model's in-memory weights at requested strengths.

This does not modify the base checkpoint on disk. It is a fixed-strength,
load-time operation for a model instance. Applying another strength to that
same instance would start from the already-adapted weights and therefore
compound the result. A caller must reload the base model before rematerializing
different strengths.

That tradeoff is intentional for the official optimized CLI's current
load-generate-exit lifecycle. It avoids retaining another copy of every targeted
base weight and avoids adapter dispatch during inference.

PR 2 must not build interactive sliders by repeatedly calling this helper. It
should introduce a persistent mutable adapter session, similar in responsibility
to Gary's `MLXLoRASet`, which:

- Loads multiple checkpoint tensors once
- Preserves canonical base values for targeted weights
- Recomputes the complete ordered adapter stack from those base values whenever
  strengths change
- Supports independent strengths for multiple simultaneously loaded LoRAs
- Allows every adapter to return cleanly to strength zero
- Never treats previously adapted weights as the next slider update's base

LoRA is additive, but DoRA and BoRA include normalization and magnitude
operations, so adapter order matters. Recomputing the ordered stack from the
same base is necessary for deterministic parity with the official PyTorch
parametrization behavior. This capability is a requirement for an MLX Gradio
backend because the existing interface supports persistent sessions and
independent strength controls for multiple LoRAs.

### Training Scope

Gary includes the complete training workflow:

- Recursive dataset discovery and prompt sidecars
- Trigger text
- SAME-L audio pre-encoding and latent caching
- Random latent crops and masks
- Optional iterative loudness normalization
- Optimizer loop, progress, cancellation, and checkpoint cadence
- Automatic LoRA registration and prompt installation

The first upstream PR includes only the reusable trainable adapters, timestep
sampling and shifting, rectified-flow loss, and checkpoint contract. Dataset
and optimization orchestration remain outside Stable Audio 3 until the
maintainers establish the preferred integration boundary, most likely through
Underfit.

## Compatibility Contract

The intended bridge between Gary, official SA3, and Underfit is the existing
safetensors format:

- Official parametrization-style tensor keys
- JSON `lora_config` metadata
- A single rank, alpha, and adapter type per checkpoint
- Correct LoRA-family magnitude tensors
- Correct XS `U`, `M_xs`, and `V` tensors
- Canonical official model layer names

Keeping this contract stable means Gary can continue moving quickly while
upstream review proceeds, and either side can consume checkpoints produced by
the other.

## Validation Completed

As of June 14, 2026, the local upstream branch has passed:

- Ruff lint and formatting checks
- 16 focused MLX primitive and parity tests
- The existing official PyTorch LoRA test
- A real optimized medium-DiT forward, backward, and AdamW step
- Save and load of a real MLX-generated adapter checkpoint
- Application of Gary's 500-step DoRA bell checkpoint to the official
  optimized medium model

The real Gary checkpoint applied to all 36 expected layers with no missing or
skipped targets and produced a finite forward pass at strength `0.65`.

A broader test run reached 44 passed and 5 skipped before entering an unrelated
Hugging Face model download. No test failure was observed before that run was
stopped.

## Cross-Backend Listening Smoke

On June 14, 2026, a matched listening smoke compared the official optimized
MLX runtime with Gary's MLX pipeline using:

- The ARC `rf_denoiser` medium model family
- Prompt `garybell, lofi hip hop beat, warm vinyl texture, dusty drums, mellow
  bass, stereo`
- Seed `20260608`
- Eight ping-pong steps
- CFG `1.0`
- Bell DoRA strength `1.0`
- Eight seconds of output
- FP16 DiT and FP32 decoder/autoencoder for the closest practical alignment

Both implementations applied all 36 expected DiT adapter layers with no skipped
targets. The adapter produced a similarly large output-level change in each
backend:

- Official optimized MLX RMS changed from `0.262` to `0.116`.
- Gary MLX RMS changed from `0.228` to `0.095`.

The generated waveforms are not expected to be identical. The official runtime
uses 87 latent tokens for this duration, while Gary aligns to 88 and supplies a
padding mask. The runtimes also advance MLX random state differently during
ping-pong sampling.

This smoke validates that both paths load and materially apply the same
checkpoint. It does not establish perceptual equivalence; listening remains
important when the generation pipelines differ. Subjectively, the outputs were
clearly different but sounded compatible enough to belong to the same musical
source. The random-state progression is the most likely reason for the large
waveform difference, while the extra aligned latent token may also affect the
tail and attention context.

The existing bell DoRA was trained against the trainable medium RF/base path
and applied to ARC inference. This is the intended contemporary SA3 LoRA
workflow rather than a parity defect: training uses the trainable base model,
while distilled ARC is used for fast inference. The comparison therefore
exercises the real deployment pattern.

Gary's shipping service defaults were deliberately not used for this aligned
comparison. The product currently uses FP32 DiT, FP16 autoencoder, chunked
decode, six seconds of duration padding, and output normalization/limiting.
Those settings should be evaluated separately as product-quality policy rather
than attributed to adapter-math differences.

## Deliberately Excluded from PR 1

- Changes to the official PyTorch CLI
- Changes to `run_gradio.py` or the existing Gradio interface
- A standalone MLX training executable
- Gary Control Center UI and native job management
- Application Support paths and model registries
- Gary presets and experimental loudness correction
- Sparkle, packaging, or release-build behavior
- Underfit interface changes

These exclusions should be stated directly in the PR so maintainers can review
the primitive API without needing to evaluate an application integration at
the same time.

## Public PR Framing

The public PR should be much shorter and centered on the official repository.
Gary is useful provenance and validation, not the subject of the contribution.

A reasonable public framing is:

- The primitives were developed while implementing and validating Apple
  Silicon LoRA training in an open-source downstream application.
- The contribution was rewritten around the official `optimized/mlx` model
  definitions and pure-MLX runtime constraints.
- Checkpoints and adapter math are tested directly against the official
  PyTorch implementation.
- The downstream branch can be linked once as evidence of an end-to-end
  integration, without describing Gary's UI, release system, or product
  roadmap.
- CLI and Gradio integration are intentionally deferred to a follow-up PR.

The public description should include the adaptations maintainers need to
review: dependency policy, checkpoint naming, convolution layout conversion,
adapter coverage, fixed-strength inference semantics, and test evidence. The
Gary migration checklist and application-specific exclusions should remain in
this internal record.

## Future Vendoring Checklist

After the Stable Audio 3 and Underfit contributions are accepted and released:

1. Compare the released SA3 primitive API with Gary's vendored
   `stable_audio_3/mlx/training.py` and `lora.py`.
2. Replace duplicated adapter math and checkpoint handling with official SA3.
3. Keep Gary's defaults, loudness processing, progress persistence, model
   registry, and native UI as thin product layers.
4. Move dataset and optimization orchestration to the official Underfit MLX
   backend when it provides equivalent cancellation and progress hooks.
5. Decide whether Gary should adopt the official optimized model pipeline or
   retain its reusable pipeline while importing official primitives.
6. Add cross-package checkpoint fixtures before removing the vendored fallback.
7. Remove local compatibility shims only after both inference and training
   smoke tests pass from a clean Gary installation.

The checkpoint contract is already the migration bridge. The largest remaining
architectural decision is the reusable MLX generation API that will support
both the official CLI/Gradio work and Gary's eventual upstream vendoring.
