# ACE-Step MLX cold-load audit

Date: 2026-08-31

## Decision

Keep Carey's hard unload after every generation. The load problem should be
fixed at the checkpoint boundary; keeping a 3-8+ GB MLX DiT plus PyTorch
duplicates resident is not the solution.

Do **not** make a new Hugging Face NPZ repository the first implementation
step. The first shippable work should be:

1. replace per-tensor PyTorch-to-NumPy conversion with bulk `mx.load` from the
   already-downloaded safetensors;
2. stop constructing/loading the PyTorch DiT when the native MLX DiT is the
   selected backend;
3. load LoRA/DoRA adapters directly into MLX, so an XL LoRA does not force the
   PyTorch DiT back into memory;
4. benchmark the resulting end-to-end `/v1/load` before deciding which hosted
   artifacts are still worth maintaining.

A hosted, revision-pinned FP16 artifact is likely worthwhile for XL. The
official regular checkpoints are already BF16, but the official XL checkpoints
are FP32: the XL decoder is 15.53 GiB at source and 7.86 GiB in FP16. If we host
artifacts, use MLX-native safetensors unless a cold-cache benchmark shows NPZ is
materially faster. MLX reads both, and safetensors gives us metadata, safer
validation, and a format already used by Gary's SA3 conversion cache.

## What Carey does today

The normal MLX load is not torch-free:

1. `AutoModel.from_pretrained(...)` constructs the complete ACE model on CPU.
2. On the MLX path, the service chooses `torch.float32` for those CPU modules.
3. `_init_mlx_dit()` constructs a second decoder and `convert_and_load()` walks
   the PyTorch decoder state dict. Every tensor is forced through
   `.detach().cpu().float().numpy()` before becoming an MLX array.
4. The full PyTorch model stays resident because Carey still uses its encoder,
   audio tokenizer/detokenizer, `prepare_condition()`, and optional lyric
   alignment methods.
5. The VAE is also loaded in PyTorch and converted to a second MLX VAE.
6. Qwen3-Embedding-0.6B remains a separate PyTorch text encoder.

The model is then hard-unloaded after the job. That lifecycle is intentional
and should remain the default.

### Checkpoint composition

Measured from the installed safetensors headers (tensor payload only):

| family | source dtype | full checkpoint | decoder | conditioning encoder | tokenizer + detokenizer |
|---|---:|---:|---:|---:|---:|
| regular | BF16 | 4.46 GiB | 2.94 GiB / 476 tensors | 1.13 GiB | 0.39 GiB |
| XL | FP32 | 18.58 GiB | 15.53 GiB / 628 tensors | 2.27 GiB | 0.78 GiB |

The current conversion doubles the regular decoder from BF16 to FP32 in MLX.
For XL it retains the already-large FP32 representation. An FP16 MLX decoder is
about 3.15 GB for regular and 8.44 GB for XL.

The auxiliary weights are not universally shareable between variants. Regular
base and turbo differed in every one of the 201 non-decoder tensors checked.
XL turbo and XL SFT had identical non-decoder tensors in the installed copies,
while XL base differed in nine tensors. Artifact identity therefore has to be
variant-specific unless equality is proven and recorded during export.

## Existing code we should use

`acestep/models/mlx/dit_convert.py` already contains a torch-free,
shard-streaming `load_decoder_safetensors()` implementation. The MLX trainer
uses it through `training_job.build_mlx_decoder()`, but inference does not.
This is the correct key normalization and validation authority for an exporter
or new loader.

That loader evaluates one tensor at a time through `safetensors.safe_open(...,
framework="pt")`. It avoids a full PyTorch model and controls peak memory, but
it is not the fastest load path. A bulk `mx.load` of each original safetensors
shard, followed by filtering, the two convolution layout transforms, one cast,
and one evaluation per shard was substantially faster in the audit.

The ACE MLX training code also already has MLX LoRA/DoRA layer wrappers and
PEFT-compatible save conventions. It lacks the inverse operation needed by
inference: read an existing PEFT adapter, validate its base/family, and either
merge it into MLX weights or install inference wrappers.

## Measurements

These are diagnostic timings on this Mac, not release claims. Runs were in
fresh Python processes, but later runs benefited from the macOS filesystem page
cache. End-to-end Carey also loads the VAE, Qwen, and the conditioning modules,
which these DiT-focused probes do not include.

| regular base path | time | MLX active memory | relevant peak |
|---|---:|---:|---:|
| current full PyTorch model + FP32 MLX conversion | 10.64 s | 6.30 GB | 16.21 GB macOS peak footprint |
| existing streaming loader to FP16, first observed read | 6.83 s | 3.15 GB | 3.55 GB macOS peak footprint |
| existing streaming loader to FP16, warm repeat | 2.42 s | 3.15 GB | 3.55 GB macOS peak footprint |
| prototype FP16 NPZ load, warm | 0.69 s | 3.15 GB | 3.21 GB macOS peak footprint |
| prototype bulk `mx.load` from the original BF16 safetensors, warm | **0.27 s** | 3.16 GB | 3.24 GB macOS peak footprint |

For XL base, bulk `mx.load` from the four original FP32 shards, converting and
evaluating once per shard, loaded all 628 decoder tensors into FP16 in 5.09 s
with 8.44 GB active MLX memory and 9.67 GB MLX peak memory. A preconverted FP16
artifact should reduce XL disk traffic and casting further; that benefit is
much less compelling for the already-BF16 regular checkpoints.

The 3.15 GB temporary regular NPZ used for this comparison was deleted after
the audit.

## What to copy from Stable Audio 3

The useful SA3 pattern is broader than the `.npz` suffix:

- direct MLX-native weight loading rather than constructing a disposable
  PyTorch model;
- one canonical manifest mapping logical components to hosted files;
- Hugging Face cache reuse rather than a second copied download;
- a pinned repository commit for production, with an intentional update and
  parity check before changing it;
- strict key/shape validation and fail-closed loading;
- deletion of source dictionaries before materializing model parameters to
  control peak memory;
- an atomic local conversion cache as a fallback, with a format version and
  complete-entry marker;
- explicit freeing of T5/DiT/codec stages instead of keeping every model
  resident.

Gary's SA3 service is also a better guide than the upstream demo for cache
correctness: `stable_audio_3/mlx/conversion_cache.py` uses atomic staging,
strict loads, a conversion format version, source identity, and separate
safetensors components. SA3 training pins the official optimized repo to a
specific commit in `scripts/mlx_training_assets.py`.

Do not copy the SA3 Gradio model-residency caches. Carey's lifecycle requirement
is the opposite. Also do not key ACE artifacts on an SA3-style runtime attention
mode without evidence: ACE's layer types/sliding window are part of its model
config, not a user-selectable conversion mode today.

## Existing hosted ACE-Step MLX weights

Checked on 2026-08-31:

- `mlx-community/ACE-Step1.5-MLX` exists, but its composite model is FP32
  (~9.58 GB) and targets the separate `mlx-audio` implementation. It is not a
  drop-in Carey decoder artifact.
- VoiceKit has regular base/SFT/turbo and an XL turbo safetensors repo. The
  files use MLX convolution layout and BF16, but have no model cards or explicit
  conversion provenance/parity contract. The checked XL set is incomplete for
  Carey's base/SFT switching.

These are valuable comparison inputs, not production dependencies yet. If one
is evaluated, pin its exact commit and compare every target tensor plus a fixed
forward pass. The upstream ACE-Step model card labels ACE-Step 1.5 as MIT, so
publishing transformed weights appears compatible with the stated license, but
our artifact repo should still carry the source attribution and MIT notice.

## Target architecture

### 1. MLX checkpoint resolver

Add a small resolver with an explicit policy, for example:

`ACESTEP_MLX_WEIGHT_SOURCE=auto|original|hosted|local-cache`

`auto` should prefer a verified local/hosted optimized artifact, then fall back
to bulk loading the original local safetensors. It must never silently download
a different variant. Log the source, revision/fingerprint, dtype, bytes, load
time, and whether a conversion/cast occurred.

The resolver should return the model config, decoder weights, and
`null_condition_emb`. It should validate the exact expected parameter set and
shapes before publishing the decoder to the handler.

### 2. Conditioning-only PyTorch shell

Carey needs a model-like object containing:

- `encoder`;
- `tokenizer` and `detokenizer`;
- `null_condition_emb`;
- the existing `tokenize()`, `detokenize()`, and `prepare_condition()` behavior.

It does not need a PyTorch DiT for the Gary localhost generation path. Build
this shell from config and stream only the non-decoder weights into it. Keep the
existing full `AutoModel` path as a fallback for non-MLX platforms, LyCORIS
LoKr until native support exists, and features that explicitly require
PyTorch decoder attentions. Gary's service-mode UI already disables automatic
score/LRC controls, but this boundary must be covered by tests rather than
assumed.

### 3. Native MLX LoRA/DoRA inference

This is required before the conditioning shell can become the default for the
reported XL-LoRA workload.

- Parse `adapter_config.json` and `adapter_model.safetensors` directly.
- Validate model family, target modules, ranks, key shapes, adapter type, and
  base-model identity.
- Prefer merging into the MLX base weights at load time. Carey discards the
  model after the job, so a permanent per-load merge avoids extra low-rank
  matmuls on every diffusion step.
- Implement DoRA with the same formula used by MLX training and compare against
  the current PEFT-merge result.
- Support scale changes without keeping a complete second decoder. A delta
  update or a base-weight reload is preferable to a full model backup.
- Keep the full PyTorch fallback for LoKr initially and report that fallback in
  load diagnostics.

### 4. VAE and text components

The VAE still performs a runtime PyTorch-to-MLX conversion. Once the DiT path is
fixed, measure it separately. A single shared, preconverted VAE safetensors file
is low-risk and likely useful because it is reused across variants.

Qwen3-Embedding-0.6B remains PyTorch. Moving it to MLX could improve inference
and remove another duplicate runtime, but it is a separate parity project and
should not block the DiT cold-load fix.

## Hosted artifact decision

Re-evaluate hosting after the bulk loader, conditioning shell, and native
adapter path are benchmarked end to end.

If hosting still wins, prefer one public, attributed repository with files such
as:

```text
manifest.json
MLX/acestep-v15-base-f16.safetensors
MLX/acestep-v15-sft-f16.safetensors
MLX/acestep-v15-turbo-f16.safetensors
MLX/acestep-v15-xl-base-f16.safetensors
MLX/acestep-v15-xl-sft-f16.safetensors
MLX/acestep-v15-xl-turbo-f16.safetensors
MLX/vae-f16.safetensors
```

The manifest/export metadata should include:

- conversion format version;
- source repository and immutable source commit;
- source checkpoint filename(s) and SHA-256;
- source config SHA-256;
- target key count, parameter count, dtype, and byte count;
- converter git commit and MLX version;
- artifact SHA-256;
- parity command/results.

Pin the optimized repository to a commit in Carey, as SA3 training does. Use
`hf_hub_download()` directly from the Hugging Face cache. A local conversion
fallback should use an atomic staging directory and a cache key based on the
source identity, conversion format, variant/config, and dtype. Do not include
attention mode unless conversion output actually depends on it.

## Implementation phases

### Phase A — instrumentation and bulk original-safetensors loader

- Add stage timings for graph construction, shard open, weight preparation,
  MLX materialization, static buffers, PyTorch auxiliaries, VAE, Qwen, LoRA,
  and total `/v1/load`.
- Add a tested bulk `mx.load` path beside the existing streaming fallback.
- Default the MLX DiT to FP16 behind an explicit feature flag first.
- Preserve strict key validation and the current PyTorch fallback.

This phase is useful for memory and establishes the loader/export authority,
but total load will not fall enough while the full PyTorch DiT is still read.

### Phase B — conditioning-only shell

- Extract/reuse conditioning behavior without a PyTorch decoder.
- Stream only auxiliary weights.
- Exercise text-to-music, cover, repaint, complete, lego, audio-code input, and
  CFG paths against the current implementation.
- Fail back to the full model for a feature that still requires the PyTorch
  decoder; never silently return degraded output.

### Phase C — native LoRA/DoRA

- Load and merge Gary-trained PEFT LoRA and DoRA in MLX.
- Cover adapter scale, load/unload, base/turbo/SFT/XL family validation, and
  consecutive jobs with different adapters.
- Keep LoKr on an explicit full-PyTorch fallback until it has a native path.

Only after this phase should the new shell become Carey's default.

### Phase D — hosted artifact A/B

- Export regular and XL FP16 artifacts reproducibly.
- Compare original-safetensors bulk load, local converted artifact, and hosted
  converted artifact after download is warm.
- Measure load time, first-generation time, peak RSS, MLX active/peak memory,
  and output parity.
- Publish only the artifacts that improve the complete workflow enough to
  justify their storage and maintenance. XL is the likely first candidate.

### Phase E — inference work exposed by the audit

- Compare merged native adapters with wrapper-based adapter forwards.
- Measure FP16 versus BF16 numerics and throughput per family.
- Profile the PyTorch↔NumPy↔MLX conditioning bridge and only then consider a
  native MLX condition encoder/Qwen path.
- Keep static-buffer materialization and `mx.compile` decisions separate in
  the report: moving work from first generation to `/v1/load` is not an
  end-to-end speedup.

## Validation gate

Before changing the default:

- exact expected key set and tensor shapes for every supported variant;
- converted tensor comparison against the current converter;
- fixed-input decoder forward parity in FP32 and the chosen shipping dtype;
- fixed-seed audio A/B for every Carey task family;
- PEFT LoRA and DoRA parity at scales 0, 0.5, and 1;
- consecutive base -> turbo -> SFT and adapter A -> adapter B jobs;
- hard unload verification: handler references cleared, garbage collection
  complete, and MLX active/cache memory returned to the expected floor;
- corrupt/missing/old artifact tests with a deterministic original-checkpoint
  fallback;
- no surprise download of an unrequested variant.

The benchmark harness should run each case in a subprocess and report both
cold-ish first observation and page-cache-warm repeats. Download time must be
reported separately from model materialization time.

## Post-audit artifact execution

The first Phase D canary was published on 2026-08-31:

- repository: `thepatch/acestep-1.5-mlx`;
- immutable repository revision:
  `b4543ad2ac4a5d6ac5f5bc24fc5fb2dc1b7f7a67`;
- artifact: `MLX/acestep-v15-xl-base-f16`;
- upstream revision: `220c1166efbdd9583eafcb12eb160594bbfcb241`;
- 628 decoder tensors plus the null-conditioning tensor;
- 8,337,798,272 FP16 tensor bytes across four safetensors shards;
- all 629 tensors exactly equal to an independent fresh conversion;
- source-parity validation: 23.05 seconds, 6.97 GB peak footprint, zero swaps;
- 201 decoder-free conditioning tensors (3,273,654,552 source-dtype bytes)
  plus the exact source `silence_latent.pt` in `conditioning/`;
- unauthenticated remote verification matched every added file size and LFS
  SHA-256 value at the immutable repository revision.

The guarded Carey integration is now implemented for XL-base. A real handler
run measured 6.98 seconds for the complete model stack (conditioning shell,
MLX DiT, Qwen, and native MLX VAE), retained no PyTorch decoder, and reached an
8.39 GB MLX peak. A fixed 5-second/two-step generation completed through native
MLX diffusion and VAE decode. The 352-target XL DoRA canary merged in 5.25
seconds; its checked projection matched the direct DoRA formula with zero
error, scale changes restored the pinned base rather than retaining a second
decoder, and unload restored exact base weights. Peak MLX memory stayed below
9.34 GB during adapter lifecycle validation.

The installed `0.3.0-parity.17 (217)` canary also passed the existing Mayer
PEFT 0.18 DoRA export end to end. Its unnamed tensor layout (no `.default`
segment and no `.weight` suffix on magnitude vectors) resolved all 1,056
tensors into 352 MLX targets, merged successfully, and generated a valid
5.12-second WAV. The pinned snapshot loaded from Gary's normal Hugging Face
cache in 5.31 seconds. Hard unload cleared the model list and reduced the API
process RSS to its post-unload floor. Carey cache/progress/audio files now use
`ACESTEP_CACHE_DIR` or `XDG_CACHE_HOME`, so generation no longer mutates the
signed application bundle.

The hosted format is not intrinsically inference-only. It contains the complete
MLX decoder plus the exact decoder-free PyTorch conditioning shell and silence
latent used by preprocessing. MLX training can already reuse the pinned decoder
when training precision matches artifact precision, so the current FP16
artifact is eligible for FP16 adapter training. The remaining two-copy behavior
comes from implementation policy: Carey's trainer defaults to BF16, rejects an
FP16 artifact for that job, resolves a legacy model directory/config before it
tries the artifact, and preprocessing still constructs the full legacy model
instead of loading the artifact's conditioning shell.

This changes the rollout criterion. The user-facing downloader and legacy
cleanup must remain blocked until one artifact per variant completes raw-audio
preprocessing, native MLX LoRA/DoRA training, adapter reload, and generation with
the legacy DiT shards absent. The first canary should use the already-published
FP16 artifact after removing those path assumptions. If FP16 training stability
or adapter quality is inadequate, a BF16 decoder artifact has the same storage
cost per parameter and should be evaluated as the single universal Mac artifact,
including inference parity and load time, rather than asking users to retain two
checkpoint families.

All six regular and XL variants are now published and pinned at artifact
revision `218ca5e51809821177bdfed70bf500020df6e958`. The repository catalog and
all 65 constituent file identities were verified remotely against local
provenance after upload.

The next rollout gate is therefore artifact-only training, followed by the
one-time storage migration. Existing testers can temporarily hold both the 18.6
GiB source checkpoint and the 11.6 GB optimized artifact, but this must not
become the product UX. Cleanup may occur only after artifact-only generation and
training pass, the complete pinned snapshot is verified, and hard unload clears
every model/adapter reference. It should remove only source shards whose hashes
match the recorded upstream provenance, preserve unknown files, and write a
per-variant migration marker. No automatic legacy deletion is enabled yet.

The migration target is strictly the upstream per-variant DiT weights and
index. The shared 322 MB VAE, optional ScragVAE, 1.1 GB Qwen encoder, and
captioner models are not duplicates of the optimized artifacts and must never
be included in a “legacy ACE-Step” cleanup. Until the single-artifact gate
passes, base and XL-base source checkpoints remain temporary compatibility
inputs for default BF16 DoRA training; that is a blocker to promotion, not a
reason to design a permanent two-download workflow.

Recent optimized XL logs separate the remaining load cost sufficiently to keep
VAE work out of migration v1: the complete load is about 6.3 seconds, with the
PyTorch VAE load and MLX conversion accounting for roughly 2–3 seconds. That is
worth a later torch-free VAE benchmark, but the VAE is small, shared, and still
needed regardless of which DiT format is selected.

### XL-base + Mayer performance follow-up

A realistic 10-second Complete request on the M4 Air, using XL-base, the
352-target Mayer DoRA, CFG 5, and 50 diffusion steps, separated the remaining
costs clearly:

- pinned model load: 3.14 seconds in the fastest observed warm-cache run;
- Mayer resolve and merge: 3.74 seconds after batching MLX evaluation in groups
  of 32, down from 8.56 seconds with one synchronization per target;
- source VAE encode and batch preparation: 0.68 seconds;
- first-request conditioning: 5.92 seconds before the non-cover fast path;
- diffusion: 70.47 seconds, or 1.409 seconds per step;
- VAE decode: 1.03 seconds;
- request total: 78.36 seconds.

The conditioning shell was still running its cover-only FSQ tokenizer,
quantizer, and detokenizer for Complete/repaint batches, then discarding those
hints because every `is_covers` value was false. The handler now directly runs
the condition encoder and constructs context latents for an all-non-cover
batch. A fresh-process eight-step XL-base + Mayer Complete validation reduced
condition preparation to 0.525 seconds and completed the request in 13.72
seconds, with a valid WAV. Cover and mixed-cover batches retain the original
upstream path.

Neither tested graph option should become the default. Static-buffer
materialization was neutral (1.419 versus 1.416 seconds per step). `mx.compile`
increased the first eight-step diffusion to 11.99 seconds (1.499 seconds per
step) versus 11.17 seconds (1.397 seconds per step) for the uncompiled control,
and it also disables the existing cross-attention KV cache. The shipping
setting should remain uncompiled with static-buffer materialization disabled.

The next hosted inference artifact should be XL-turbo, not another XL-base
loader revision. XL-base's remaining wall time is overwhelmingly its required
diffusion work: 50 steps cost about 70.5 seconds on this machine, while eight
steps cost about 11.2 seconds. A provenance-pinned XL-turbo artifact enables a
meaningful end-to-end latency change without keeping models resident. It must
pass the same artifact parity, Mayer family compatibility, fixed-seed audio,
and hard-unload gates before publication or default selection.

### Production-duration attention and quantization follow-up

The earlier 10-second benchmark understated the real workload. Carey's default
122-second request produces 3,050 latent frames and 1,525 patched DiT tokens;
XL-base then measured 13–16 seconds per diffusion step on this M4 Air. Its 32
layers alternate between 16 sliding-window layers (window 128) and 16 full
attention layers.

An exact grouped bidirectional sliding implementation now uses strided K/V halo
blocks and per-query boundary masks instead of passing a dense 1,525-square
mask to SDPA. It is bit-exact to the dense FP16 path at production geometry.
The isolated attention kernel improved from 34.3ms to 15.4ms, but a full
one-step control improved only from 16.52s to 15.74s because XL's linear and
SwiGLU projections dominate. The path remains independently feature-gated by
`ACESTEP_MLX_GROUPED_SLIDING_ATTENTION`.

CFG cross-attention K/V can also be cached safely because conditional and null
encoder states are concatenated once before diffusion. Cache invalidation at
the cover/non-cover switch was already present. Cache-on and cache-off
fixed-seed runs produced byte-identical WAV files, although the measured saving
was only a few tenths of a second per cached step.

Runtime MLX affine quantization exposed the larger practical lever. In matched
122-second one-step runs with grouped attention, FP16 took 15.02 seconds,
8-bit groups of 64 took 12.37 seconds, and 4-bit took 12.23 seconds. Eight-bit
output retained 0.9988 waveform correlation to FP16 (25.5dB error SNR), while
4-bit fell to 0.8754 (5.8dB), so parity.19 uses 8-bit as a listening canary.
Packing adds roughly 0.3–0.7 seconds to the current FP16 artifact load.

Packed weights cannot accept an in-place PEFT merge. Carey therefore restores
the exact pinned FP16 arrays before loading native LoRA/DoRA, then re-quantizes
the base after adapter unload. Mayer's 352-target DoRA passed this transition:
merge took 3.49 seconds and unload plus exact restore/repacking took 4.26
seconds. MLX training remains unquantized. A hosted pre-quantized artifact may
reduce download/storage further, but should wait for sustained 50-step listening
tests and would need a quantized-aware artifact schema and loader.
