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

1. Submit focused MLX LoRA training and inference primitives to
   `stability-ai/stable-audio-3`.
2. Submit an Underfit MLX backend that uses the accepted SA3 API.
3. Replace Gary4local's corresponding vendored implementations when the
   upstream versions provide the behavior and iteration speed the app needs.

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

As of June 8, 2026, the Mac implementation has terminal smoke coverage for
standard LoRA, DoRA rows, BoRA, LoRA-XS, DoRA-XS, and BoRA-XS training and MLX
inference. Upstream interest has been raised with the Stable Audio 3
maintainers. Gary4local now also has a native one-audio MLX training workflow
with persistent progress, cancellation, automatic LoRA registration, and prompt
dice installation. Dataset-folder parity with the Windows Underfit workflow
remains local follow-up work.
