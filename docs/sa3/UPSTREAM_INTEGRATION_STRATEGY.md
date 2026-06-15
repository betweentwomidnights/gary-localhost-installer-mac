# SA3 and Underfit Upstream Integration Strategy

Gary4local currently vendors the Stable Audio 3 and Underfit functionality it
needs so product development is not blocked by upstream release timing.

## Desired Upstream Shape

We would like the reusable MLX model, adapter, training, checkpoint, and
inference primitives to live in the official `stable-audio-3` repository.
Underfit could then add an Apple Silicon backend that consumes those public
primitives, just as its current PyTorch backend consumes the official SA3
package.

The intended contribution order is:

1. Submit focused MLX audio-encoding, LoRA training, and inference primitives to
   `stability-ai/stable-audio-3`.
2. Follow with official MLX CLI and Gradio integration built on a reusable
   generation pipeline.
3. Submit an Underfit MLX backend that uses the accepted SA3 API.
4. Replace Gary4local's corresponding vendored implementations when the
   upstream versions provide the behavior and iteration speed the app needs.

The detailed provenance, adaptation decisions, validation evidence, and
vendoring checklist are recorded in
[`STABLE_AUDIO_3_UPSTREAM_PR_NOTES.md`](STABLE_AUDIO_3_UPSTREAM_PR_NOTES.md).

## Current Product Policy

Upstream discussion and review must not block Gary4local development. Until the
relevant changes are accepted and released:

- Gary4local maintains its own MLX inference and training implementation.
- Checkpoints follow the existing SA3/Underfit safetensors key and metadata
  contract.
- Adapter behavior is tested against the PyTorch implementation for LoRA,
  DoRA, BoRA, and XS variants.
- Product-facing APIs remain local and may evolve faster than an upstream
  public API.

## Migration Boundary

Gary-specific concerns should remain outside an eventual upstream contribution:

- Native control-center UI and job orchestration
- Application Support paths and model registries
- Progress persistence, cancellation, and process recovery
- Gary defaults, presets, and experimental loudness processing
- Release packaging and Sparkle integration

The vendored MLX modules should stay sufficiently isolated that they can later
be replaced by official SA3 and Underfit packages without redesigning the
control-center workflow.

## Status

As of June 14, 2026, Gary4local users have confirmed end-to-end MLX training
and inference on Apple Silicon. The Stable Audio 3 maintainers have invited a
pull request for the reusable MLX training primitives.

The first local upstream branch is intentionally limited to waveform-to-latent
encoding, adapter, checkpoint, training-distribution, and rectified-flow-loss
primitives with focused parity tests. Dataset orchestration, CLI, and Gradio
integration will be proposed separately so the primitive API can be reviewed
without coupling it to a larger interface refactor.
