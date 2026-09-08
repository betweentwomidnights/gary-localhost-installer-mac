# mac parity TODO

Untracked scratch list. Things we know need doing to get the mac build as
robust as gary-localhost-installer's v0.3.0, captured as they come up rather
than held in a conversation.

## Release-candidate sequence

- [ ] Checkpoint the current ACE-Step MLX, conditioning-shell, native-adapter,
      training-parity, lifecycle, and storage UI work before integrating remote
      changes. The working tree is intentionally broad and must not be rebased
      or cleaned while uncommitted.
- [ ] Integrate the already-pushed `origin/main` parity work: Gary/Terry seed
      handling plus SA3 decoder-LoRA and seamless-continuation changes. Resolve
      the Carey manifest/UI overlaps deliberately and run both upstream and
      local regression tests.
- [ ] Ship the Carey weight-source compatibility layer and optimized download
      rows before any legacy cleanup assistant. Existing installs must continue
      to generate and train from legacy checkpoints without downloading or
      deleting anything unless the user opts in.
- [ ] Produce an unsigned/local release build for end-to-end upgrade testing,
      then exercise legacy-only, optimized-only, and mixed installs. Full custom
      runtime storage remains out of scope for this release.

## ACE-Step lifecycle audit — findings (2026-08-30)

Read from the code, not yet measured against real memory use.

**1. Nothing is released after a generation.** The wrapper's `finally` in
main.py only unloads when `MANAGE_MODEL_LIFECYCLE and use_gpu_queue`, and the
GPU queue is Spark-only. Its comment says this is deliberate — "keep the model
loaded between jobs". So on localhost a 20GB xl DiT stays resident until the
service stops.

**2. The configured unload frees nothing anyway.** The manifest sets
`ACESTEP_UNLOAD_STRATEGY=soft`, and soft unload only sets `_initialized=False`
and `_soft_unloaded=True` — "Keep model objects resident for fast subsequent
/v1/load calls". Only `hard` drops the references, collects, and calls
`mps.empty_cache()`.

**3. Switching variants allocates the new model before freeing the old.**
`/v1/load?config_path=X` hard-unloads first when `_initialized` is true and the
config differs — good. But the wrapper calls `/v1/unload` (soft) *before*
loading, which sets `_initialized=False`, so that branch is skipped. Control
falls through to `initialize_service`, which only nils the old models on a
precheck *failure* — on the success path it loads the new DiT with the old one
still referenced. That is a double allocation on exactly the turbo → base → sft
switch this audit was worried about.

**4. Downloads are correctly scoped.** `ensure_dit_model` fetches only the named
config, and the wrapper picks the config from `ACESTEP_*_CONFIG_PATH`, which the
app sets to one consistent family. So xl-base does not pull v15-base.
`ACESTEP_ALLOW_MAIN_AUTODOWNLOAD=false` also means missing shared assets are a
hard error rather than a surprise download.

**5. Carey does NOT lazily fetch Qwen.** With autodownload off, missing shared
files (Qwen *and* the default VAE) fail the precheck with "Run the focused Carey
downloader first". So the earlier TODO item is answered: there is no lazy fetch
to rely on.

**6. The default VAE is required regardless of ScragVAE.** It is in
`required_shared_files` in init_service_downloads.py, so the precheck fails
without it whether or not ScragVAE is present. Whether ScragVAE *could* replace
it is a separate question, but today it does not.

## Model lifecycle audit (ACE-Step / Carey especially)

The principle: **free memory after every generation.** Keeping a model resident
sounds like good UX until you remember someone may be running several services
at once, and switching turbo → base → sft on the fly from gary4juce. Holding
weights for a model the user has moved on from is what makes the next service
fail to start.

- [x] Carey cleanup after each generation — done 2026-08-31. Hard unload after
      every job (opt out with `CAREY_KEEP_MODEL_LOADED=true`), manifests moved
      from soft to hard, and `_unload_model` now asks for hard explicitly so the
      variant-switch double allocation is gone. NOT yet measured against real
      memory use — worth watching peak RSS through a turbo → base → sft switch.
- [ ] Audit the OTHER services' cleanup after each generation. The PC repo had a thorough
      pass on this; the belief is the MLX path already releases more
      effectively, but that is untested here.
- [ ] Same audit for the other model services — the rule should be uniform, not
      per-service folklore.
- [ ] **Stop ACE-Step lazily downloading models it does not need.** It is
      notorious for this. The PC build did significant work so that only the
      models required for the *current* inference request are fetched: using
      xl-base must not pull v15-base. Check the same holds here.
- [ ] The Carey *training* UI needed its own hardening on PC for the same
      reason. Check the mac trainer.
- [ ] Carey should lazily fetch Qwen if it is missing, rather than failing.
      (Confirmed it does not today — see finding 5.)
- [ ] Observed 2026-08-30: starting the stable_audio service re-downloads
      stable-audio-open-small if it is missing, before serving /health (~172s
      startup). Correct in that jerry needs it, but it means removing SAOS is
      undone by the next service start, and the panel will show it downloaded
      again. Decide whether that is the wanted behaviour or whether removal
      should imply "do not refetch until asked".
- [ ] Confirm whether the default VAE is still required once ScragVAE is
      toggled and downloaded. If it is not, the panel should not imply both are
      mandatory. (This corrects an earlier assumption that removing the VAE
      always breaks Carey — probably not true.)

## ACE-Step MLX audit and model load cost

First load on an M4 Air is painful — worse than remembered with an xl-base
LoRA. Inference time itself is acceptable; it is getting the model resident
that hurts. Now that we release after every generation, this cost is paid far
more often, so it matters more than it did.

- [x] Audit completed 2026-08-31. Detailed findings, measurements, architecture,
      and phased plan: `docs/ACE_STEP_MLX_COLD_LOAD_AUDIT_2026-08-31.md`.
- [x] **First implementation: bulk-load pinned native safetensors with MLX.**
      The existing torch-free streaming loader is used by training but not
      inference. A prototype bulk `mx.load` of regular base reached `0.27s`
      warm versus `0.69s` for a preconverted NPZ and `10.64s` for today's full
      PyTorch load + FP32 MLX conversion. The guarded runtime now validates and
      directly loads the pinned native artifact without constructing PyTorch DiT.
- [x] **Remove the PyTorch DiT from the MLX critical path.** An NPZ alone does
      not help enough because Carey currently loads the complete PyTorch model
      first and retains it for conditioning/audio-code helpers. Build a
      conditioning-only shell and stream only encoder/tokenizer/detokenizer
      weights; keep the full model as an explicit fallback for features that
      require PyTorch decoder attentions. Done for XL-base; lyric attention
      scoring/timestamps now report their unsupported optimized-mode boundary.
- [x] **Load PEFT LoRA/DoRA natively in MLX.** The current adapter path mutates
      the PyTorch decoder and reconverts it, so this is required for the actual
      XL-LoRA workload before the conditioning-only shell becomes default.
      The native route supports one PEFT LoRA/DoRA at a time and restores from
      the pinned base artifact without retaining a second decoder. LoKr reports
      that it requires the full-PyTorch path. The existing Mayer PEFT 0.18 DoRA
      (352 targets) now passes native load and generation, including its newer
      unnamed tensor-key layout.
- [x] Published the first provenance-pinned XL FP16 canary at
      `thepatch/acestep-1.5-mlx`, remote revision
      `b4543ad2ac4a5d6ac5f5bc24fc5fb2dc1b7f7a67`. The four MLX-native
      safetensors contain 8,337,798,272 tensor bytes; all 629 artifact tensors
      compare exactly with a fresh conversion of upstream revision
      `220c1166efbdd9583eafcb12eb160594bbfcb241`. Public remote verification
      matched all file identities and LFS SHA-256 values. The same revision now
      includes 201 exact source-dtype conditioning tensors plus silence latent.
- [x] Added a reproducible, atomic exporter plus provenance/index validation
      and exact source-tensor parity tooling. The artifact pins source and
      output hashes, converter source hashes, MLX version, dtype, format
      version, tensor count, and upstream revision.
- [x] Integrate the pinned artifact into Carey's MLX load path and Hugging Face
      cache. Keep the original-checkpoint conversion as a deterministic
      fallback; reject unknown format versions or mismatched hashes. ACE
      attention mode remains model-config state, so do not copy SA3's
      attention-mode cache key blindly.
- [x] **Single-artifact generation + training release decision.** A Mac user can
      use one per-variant FP16 artifact for generation and for BF16-compute
      training. Pure FP16 training is intentionally unsupported after both the
      source-cast and pinned-artifact loaders produced non-finite loss on step 1.
      FP16-storage/BF16-compute matched source-BF16 step-1 loss and memory. The
      original 30-epoch Billie A/B exposed a trainer-recipe regression rather
      than destructive FP16 rounding: both old adapters sounded mangled, the
      no-adapter control and known-good Mayer adapter sounded clean, and the
      corrected parity recipe produced a musical adapter distinct from the
      control. That is sufficient for this release; do not spend another long
      M4 Air training run on this gate. Keep a short artifact-only
      preprocess/train/save/reload smoke in release validation.
- [ ] **Mac-only model inventory after the single-artifact gate.** Split Carey's
      download panel into “Apple MLX models” and “Shared components”; each of the
      six variants gets one primary Mac download that supports both generation
      and training. Existing legacy checkpoints remain a supported compatibility
      choice: do not force a redownload or hide their installed state. Recommend
      Apple MLX to new Mac installs without presenting both formats as downloads
      a new user needs. Show verified state and real reclaimable size while
      keeping Qwen, stock/Scrag VAE, and captioners out of the migration target.
      PC must report this capability as unavailable rather than inheriting Mac
      UI or artifact assumptions.
- [ ] **Explicit Carey weight-source policy.** Replace the boolean-only choice
      with an internal `auto | optimized | legacy` policy. `auto` must never
      download multi-GB weights during generation: use a verified local optimized
      artifact when present, otherwise use an installed legacy checkpoint.
      `optimized` fails clearly when its pinned artifact is absent; `legacy`
      always uses the existing checkpoint path. Persist the choice and restart
      Carey when it changes. Remove the launcher's current implicit
      `ACESTEP_MLX_OPTIMIZED_LOAD=1` default before shipping.
- [ ] **One-time legacy DiT migration assistant.** Detect per variant, then
      offer a calm, non-blocking “Faster-loading Apple MLX models are available”
      card. Its primary action downloads the selected optimized variant; “Keep
      current models” leaves legacy generation and training untouched and
      snoozes by artifact revision. Only after that exact optimized variant is
      verified, has generated successfully, and Carey has hard-unloaded should
      a separate “Free up storage used by legacy ACE-Step models?” action appear.
      List selectable variants and exact freeable bytes. Never make cleanup a
      condition of using the app or the optimized artifact.
- [ ] **Hash-gated selective cleanup and durable receipt.** Delete only the
      recognized upstream weight file(s) and index whose size and SHA-256 match
      artifact provenance; retain config, silence latent, shared components,
      and every unknown/custom file. Remove only exclusive backing HF cache
      blobs so APFS clone accounting is honest. Refuse while Carey, a download,
      or ACE training is active. Write a per-variant receipt containing artifact
      revision, source hashes, bytes actually freed, and timestamp.
- [ ] **New-user/offline flow.** Let each optimized row pre-download the pinned
      universal artifact without materializing it into `checkpoints/`. Any DiT
      download continues to require the shared Qwen + VAE assets. The shipped
      Mac runtime must generate, preprocess, and train with the optimized
      artifact alone; legacy-only and mixed installs remain compatibility paths,
      not the new-user recommendation.
- [ ] **VAE scope gate, not part of migration v1.** Stock VAE is only 322 MB and
      shared by every DiT; logs put PyTorch load plus MLX conversion around
      2–3 seconds within a ~6.3-second optimized XL load. Add stage timing and
      benchmark one torch-free native VAE load before deciding whether to host
      one shared VAE artifact. Never call the current VAE “legacy” or delete it
      during DiT cleanup. Treat native Qwen/condition encoding as a separate
      inference-parity project.
- [x] Re-measure the release-after-every-generation trade once load is cheaper.
      Full optimized handler initialization measured 6.98s with no PyTorch
      decoder (8.39GB MLX peak); a 5-second two-step generation completed
      end-to-end. Hard unload now also clears native-adapter references and the
      MLX cache. `CAREY_KEEP_MODEL_LOADED` remains an explicit opt-out.
- [x] Implement exact grouped/block-sparse attention for ACE's 16 sliding
      layers. At the production 122-second geometry (1,525 DiT tokens), the
      isolated kernel is about 2.2x faster and the full FP16 step improved from
      16.52s to 15.74s in the original control. Full-model gains are smaller
      because projections and SwiGLU dominate the XL decoder.
- [x] Enable exact CFG cross-attention KV caching. Cache-on and cache-off
      fixed-seed runs produced byte-identical WAVs; the gain is only a few
      tenths of a second per cached step. The cache is invalidated when
      conditioning switches and remains disabled under `mx.compile`.
- [x] Add an 8-bit MLX inference canary. On matched 122-second one-step runs,
      grouped FP16 took 15.02s, 8-bit took 12.37s, and 4-bit took 12.23s.
      Eight-bit retained 0.9988 waveform correlation to FP16 while four-bit
      fell to 0.8754. Native LoRA/DoRA transitions restore exact FP16 weights
      before merge and repack the base after unload; training stays FP16/BF16.
- [ ] Run listening and sustained 50-step quality/thermal checks on the 8-bit
      parity.19 canary before publishing pre-quantized hosted artifacts or
      making quantization a non-canary release default.
- [x] Keep runtime MLX quantization opt-in after the sustained parity.19 test
      returned to roughly 16 seconds per step. Do not publish quantized MLX
      artifacts while the product is preparing to move service inference to
      GGML backends; retain the flag only for controlled comparisons.
- [x] Profile the realistic XL-base + Mayer Complete path. Batched native DoRA
      evaluation reduced Mayer merge from 8.56s to about 3.7-4.4s. Detailed
      timing identified cover-only FSQ work in every non-cover request; the
      new fast path reduced cold condition preparation from 5.92s to 0.525s.
      Static buffers were neutral and `mx.compile` regressed the eight-step
      control, so both remain disabled. At 1.397-1.409s per XL-base diffusion
      step, the next high-value hosted inference artifact is XL-turbo.
- [x] Exported, exactly validated, and published all six provenance-pinned FP16
      artifacts at `thepatch/acestep-1.5-mlx` revision
      `218ca5e51809821177bdfed70bf500020df6e958`. XL-turbo completed local
      optimized load, generation, and hard unload; every remote artifact,
      conditioning, index, config, provenance, README, and manifest identity
      matches the local catalog. Adapter compatibility remains variant-gated;
      Mayer is an XL-base adapter and is not advertised for XL-turbo.

## Storage

- [ ] **UV cache cleanup.** PC exposes this in the storage UI and it was
      worthwhile there. Check whether the mac build accumulates a uv cache at
      all, and how big it gets, before building UI for it.
- [ ] Service environment removal — the mac equivalent of PC's per-env remove.
      `apfs_extents.py` gives honest sizing and `remove_model.py` has the
      containment/refusal pattern to copy.

## Model panels

- [ ] Decide whether shared assets (Qwen, VAE) need a stronger warning than
      "(shared)" in the row label — depends on the lifecycle audit above.

## Done

- Service parity fixes: carey silence latent, foundation BPM range, melodyflow
  attention notices, melodyflow native 48kHz.
- Captioner downloads (all three), availability checks in both trainers, SA3
  prompt-editor captioner picker.
- Storage: clone-on-download, duplicate reclaim, clone-aware measurement,
  model removal that frees real space.
- medium-base listed in the SA3 inventory.
- Storage panel measuring time — confirmed acceptable, closed.
- stable_audio removal: whole repos (SAOS, open-1.0) and individual finetune
  checkpoints, which are files inside a shared repo.
- Removed the Carey "download all" button and the model lifecycle controls from
  the download panel. Lifecycle belongs with the service, and missing models are
  fetched lazily at inference time anyway. `loadCareyModel`/`unloadCareyModel`
  are kept in the view model for when that moves to the service tab.
