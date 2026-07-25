# changelog

this is where the version history lives that used to sit at the top of the main
README. the README should stay focused on what gary4local is now; this file gets
to remember how we got here.

## v0.2.0

paired with
[gary4juce v4.0.7-mac](https://github.com/betweentwomidnights/gary4juce/releases/tag/v4.0.7-mac).
the version jumps from `0.1.11` to line up with the windows build's numbering.

almost entirely the Stable Audio 3 LoRA trainer.

**the training base model is pinned.** asset resolution fails closed unless both
`model_config.json` and `model.safetensors` come from
`stabilityai/stable-audio-3-medium-base`, and rejects ARC/distilled configs.
hugging face split the base model into its own repository; resolving the wrong
one is what produced the drone/hum at higher LoRA strength — **retrain any LoRA
that does that.** inference still applies the adapter to
`stable-audio-3-medium`. the hosted training NPZ is now pinned to a revision so
an upstream re-upload can't change weights silently.

**MLX conversion cache.** the converted DiT, T5Gemma, seconds conditioner, and
autoencoder are written to
`~/Library/Application Support/GaryLocalhost/sa3/mlx-cache`, keyed on checkpoint
identity, the four component dtypes, attention mode, and a format version. cold
start `24.6s` → `3.9s` warm, `4.85 GiB` on disk. writes are atomic, a corrupt
entry falls back to torch and rewrites, stale entries are pruned.
`SA3_MLX_CACHE=0` disables writing.

that needed inference decoupled from the torch pipeline first: sample rate,
downsampling ratio, and channel counts come from `model_config` now, and
`_adapt_sample_size` plus the LogSNR sampling shift are ported to MLX.
generation is byte-identical with `torch_pipeline` released.

**in-place LoRA refresh.** `/reload` rebuilds the adapter sets on the resident
pipeline instead of reconstructing it — about `20s` down to under a second.
`clear_lora` runs before `load_lora`, since `MLXLoRASet` snapshots pristine base
weights on first apply and would otherwise treat `base + delta` as its base.

**layer scope.** `attention-feedforward` (169 adapted layers) is the new
default; `all-projections` (229) stays available. a paired 2,000-step A/B on
identical data, seed, and adapter initialization measured a mean loss delta of
`+0.0005` (`+0.00009` over the last 500 steps), ~5% faster steps, and
`8.38` vs `8.60 GiB` peak. mac-only for now — windows and sa3.cpp later.

**trainer parity.** mixed inpainting conditioning restored (10% random segments,
80% full, 10% causal, context loss weight 1.0), encoded-silence padding,
`trunc_logit_normal` timesteps, AdamW `(0.9, 0.95)` / eps `1e-8` / weight decay
`0.01`. adapter `lora_A` matrices are seeded from a stable hash of the layer
name, so engine and scope comparisons start byte-identical instead of diverging
on module traversal order.

**gradient checkpointing defaults off** at every window. on a base M4 Air it
costs ~45% step time while reclaiming `0.05 GiB` at 256 latents and `2.0 GiB` of
a `30.2 GiB` peak at 2048.

**dataset prep.** auto-label a folder through ACE-Step's understand-music path,
and get BPM/key suggestions from local analysis. the shared trigger is applied
only when building the training prompt — never written to sidecars or dice
prompts.

smaller:

- experimental loudness fix, normalizing per-track encoded latent RMS.
- exported checkpoints self-register; no need to open **add loras** first.
- DiT engine picker and full-track toggle removed from the form, both still on
  the trainer CLI.
- SA3 and Carey trainer copy rewritten.

### deferred

**hosted MLX weights.** `MLX/dit_medium_f16.npz` loads into gary's generic DiT
with all 522 tensors bitwise identical and no measurable forward-speed
difference, so the win is install size — roughly 18 GB of pytorch could go, and
torch leaves the inference path. not shipping here: existing installs would need
a ~9 GB download plus a cleanup we should be asking permission for. wants its
own release with a component-level download screen.

**full-track training.** 190s (2048 latents) peaks ~`29 GiB` at ~`29 s/step` on
a 32 GB machine; 285s (3072) peaks ~`40 GiB` at ~`83 s/step` and swaps.
checkpointing reclaims ~`2 GiB` for +45% step time, which points at activations
not being released the way they are on the windows path. explaining that gap is
the prerequisite for the toggle coming back. meanwhile 23-second crops produce
LoRAs that generate several minutes cleanly, and are what the best mac LoRAs so
far were trained on.

## v0.1.11

`v0.1.11` is the mac release paired with
[gary4juce v4.0.4-mac](https://github.com/betweentwomidnights/gary4juce/releases/tag/v4.0.4-mac).

the main thing in this one is that you can now train an ACE-Step LoRA directly
inside `gary4local`, done in MLX.

the smaller follow-up fix in the same release is carey parity around
`genre_ratio` during training. on the mac path we now pre-encode the relevant
tracks twice so the per-epoch genre-vs-caption behavior matches the windows
flow and the dice button logic lines up with what the UI says it's doing.

older Sparkle release notes live under
[`docs/updates/gary4local/release-notes/`](docs/updates/gary4local/release-notes/).
