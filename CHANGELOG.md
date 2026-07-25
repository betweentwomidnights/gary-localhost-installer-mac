# changelog

this is where the version history lives that used to sit at the top of the main
README. the README should stay focused on what gary4local is now; this file gets
to remember how we got here.

## v0.2.0

the version number jumps from `0.1.11` to `0.2.0` to line up with the windows
build, which is already on `v0.2.0`. same project, same numbering from here on.

this one is almost entirely about the Stable Audio 3 LoRA trainer on the mac.

**if you have an SA3 LoRA that hums or drones when you turn the strength up,
retrain it.** training now pins itself to the stable-audio-3 *base* model and
refuses to start against anything else, including the distilled/ARC checkpoint.
hugging face moved the base model into its own repository at some point, and
picking up the wrong one is what produced that drone. the adapter you get out
still applies to the regular model you generate with — that part hasn't changed.

**the trainer got a lot faster to live with.**

- starting the SA3 service used to spend about 25 seconds converting pytorch
  weights to MLX every single time. we now cache the converted weights, so after
  the first run it's under 4 seconds. the cache lands in
  `~/Library/Application Support/GaryLocalhost/sa3/mlx-cache` and costs about
  4.9 GB. set `SA3_MLX_CACHE=0` if you'd rather not spend the disk.
- switching or adding a LoRA used to rebuild the whole model, which took about
  20 seconds. it's now under a second in the common case, because swapping an
  adapter doesn't actually change any model weights.

**there's a new "layer scope" choice, and it defaults to the smaller one.** the
default now trains 169 layers instead of 229. it finishes faster and gives you a
smaller file, and in a paired 2,000-step A/B on the same dataset and seed we
couldn't hear a difference — the loss difference was in the fourth decimal
place. "train everything" is still right there if you want it. this one is
mac-only for now; it'll come to the windows build and sa3.cpp later.

**dataset prep is much less tedious.** you can auto-label a whole folder using
ACE-Step's understand-music path, and have gary suggest BPM and key. the shared
trigger word also behaves properly now: it's added to the front of every caption
while training, and it no longer gets written into your sidecar files or your
dice prompts.

smaller things:

- an experimental loudness fix that evens out volume across a dataset, so one
  hot track doesn't dominate.
- exported checkpoints register themselves — no more opening the **add loras**
  window just to make a fresh LoRA appear.
- the DiT engine picker and the full-track toggle are gone from the training
  form. both are still reachable from the trainer CLI; see below for why.
- a lot of the copy in the SA3 and Carey trainers got rewritten to be less
  ML-brained.

### two things we're still working on

**moving to hugging face's hosted MLX models.** stability now publishes the same
MLX weights we produce by converting pytorch at runtime — we checked, and
they're byte-identical, all 522 tensors. switching to them would let us drop
about 18 GB of pytorch weights and a big chunk of the dependency tree. we're
*not* doing it in this release, because it would mean asking everyone who
already downloaded the pytorch models to pull down ~9 GB more, and then cleaning
up the old files on their behalf. that deserves its own release with a proper
download UI and a cleanup step you actually get to approve, rather than being
bolted onto this one.

**training on full tracks.** the windows build trains on whole songs, and we
want that here too. right now MLX is nowhere near memory-efficient enough for
it: a 190-second window peaks around 29 GiB and runs ~29 s/step on a 32 GB
machine, and gradient checkpointing barely helps — it reclaims about 2 GiB of
that while costing 45% more time per step. so the toggle is hidden for now and
crop length is capped, rather than shipping something that looks available and
then thrashes your machine. figuring out where that memory is actually going is
the next thing we want to dig into.

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
