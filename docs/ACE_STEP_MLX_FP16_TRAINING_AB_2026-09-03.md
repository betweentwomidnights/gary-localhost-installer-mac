# ACE-Step XL-base FP16 artifact training A/B

## Question

Can one FP16 hosted artifact replace the legacy XL-base DiT for both inference
and adapter training without an audible training-quality regression?

## Preliminary result

Pure FP16 training is not currently viable. Both the source checkpoint cast to
FP16 and the pinned FP16 artifact produced a non-finite loss on the first step
of the Billie XL-base rank-64 balanced DoRA job.

The one-download mixed-storage path passed its first-step smoke test:

- base storage: pinned FP16 artifact revision
  `218ca5e51809821177bdfed70bf500020df6e958`;
- training compute: artifact weights upcast in memory to BF16;
- step 1 loss: `1.543200`;
- step time: 54.729 seconds;
- peak MLX memory: 13.056 GB.

The matching historical/source-BF16 step 1 loss is `1.544360`, with the same
13.056 GB peak. A single step establishes numerical viability, not quality.

## Controlled 30-epoch experiment

Both arms start from newly initialized adapters and use the same preprocessed
13-track Billie tensor set, sample ordering, seed, optimizer, and adapter
configuration:

- model: `acestep-v15-xl-base`;
- adapter: balanced DoRA, reference rank 64, alpha 128, 352 targets;
- optimizer: AdamW, learning rate `3e-4`, weight decay `0.01`;
- schedule: 30 epochs, 390 steps, batch size 1;
- seed: `20260620`;
- loss: Min-SNR, gamma 5.0;
- CFG dropout: 0.15;
- gradient checkpointing: enabled;
- training arithmetic: BF16.

Arm A loads the original XL-base source checkpoint and converts it to BF16.
Arm B loads the provenance-pinned FP16 artifact, then upcasts floating weights
to BF16 in memory. This isolates the effect of FP16 base-weight rounding while
preserving stable BF16 training arithmetic.

Experiment root:

`~/Library/Application Support/GaryLocalhost/carey/training/experiments/billie-xl-precision-ab`

Arm A runs first. Arm B is locally queued and starts only after Arm A exits and
has produced a complete `final/adapter_model.safetensors`, preventing GPU
contention and fail-open comparisons.

## Interpretation boundaries

The reused tensors provide the cleanest precision-only comparison, but they
were produced by an older preprocessing run that selected turbo's silence
latent for XL-base. Both arms receive the same tensors, so this does not bias
the A/B toward either precision. It does mean the experiment is not the final
artifact-only preprocessing qualification.

After both adapters finish, listening samples must be rendered with identical
prompts, seeds, duration, step count, CFG, base model, and adapter scale. Sample
labels should be blinded before listening. Artifact-only preprocessing remains
a separate release gate.
