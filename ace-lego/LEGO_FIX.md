# Lego Mode Fix

## Problem

Lego mode generated stems that ignored the source audio entirely — wrong tempo,
no harmonic relationship, silence at the end of the output.

### Root Cause

**File:** `acestep/core/generation/handler/padding_utils.py` lines 118–125

When `repainting_end=-1` (meaning "full track"), `padding_utils.py` converts the
value to the actual audio duration (e.g. `20.22`) before passing it downstream:

```python
# BEFORE (broken)
if repainting_end is None or repainting_end < 0:
    adjusted_end = src_audio_duration + padding_info_batch[0]["left_padding_duration"]
    repainting_end_batch = [adjusted_end] * actual_batch_size  # e.g. [20.22, 20.22]
```

That non-negative value causes `conditioning_masks.py` to enter the **repainting
branch** (condition: `end_sec > start_sec`). Inside the repainting branch,
`src_latent[0:end]` is replaced with silence — wiping out the **entire source
audio** from the conditioning tensor:

```python
# conditioning_masks.py — the repainting branch (wrong path for lego)
src_latent = target_latents[i].clone()
src_latent[start_latent:end_latent] = silence_latent_tiled[...]  # whole track silenced
```

The DiT received zero harmonic/rhythmic context from the guitar. It generated
freely from text + BPM + key alone, producing stems that didn't follow the source.

## Fix

**File:** `acestep/core/generation/handler/padding_utils.py`

For lego tasks with `repainting_end < 0`, return `None` instead of converting to
the audio duration. `None` causes `conditioning_masks.py` to take the **full-mask
branch**, which preserves `src_latents` as the complete source audio:

```python
# AFTER (fixed)
if repainting_end is None or repainting_end < 0:
    if is_lego_task:
        # Leave as None → conditioning_masks takes the full-mask branch,
        # which keeps src_latents = full source audio (guitar context intact).
        # Converting -1 to the audio duration here routes through the repainting
        # branch which silences the entire src_latents tensor.
        repainting_end_batch = None
    else:
        adjusted_end = src_audio_duration + padding_info_batch[0]["left_padding_duration"]
        repainting_end_batch = [adjusted_end] * actual_batch_size
```

The `is_lego_task` parameter is already available in `prepare_padding_info()`'s
signature — no other changes needed.

## Result

- Stems now follow the source audio harmonically and rhythmically
- Silence-at-end artifact eliminated
- Vocals (no-LM) produced pitch-accurate wordless performance over the guitar
- 44.1kHz source files work correctly alongside 48kHz files

## Additional Finding: LM Breaks Lego

Do **not** use `thinking=true` with lego mode. When the LM generates audio codes,
`task_utils.py` forces `is_cover_task=True`, which conflicts with the lego
repainting mask and produces garbled output. Always pass:

```
thinking=false
use_cot_caption=false
use_cot_language=false
```

The LM provides no benefit for lego — the source audio latents are the context.

## What Still Needs Investigation

- **Bass / piano / keyboard / strings** generated poor results even after the fix.
  Likely a training data imbalance — the model may have seen fewer examples of
  isolated harmonic stems in lego-style conditioning vs. drums and vocals.
- **Vocals with lyrics** requires `ACESTEP_INIT_LLM=true` and a working LM. The
  LM's audio codes now have a non-conflicting path (src_latents preserved) so this
  is worth testing once the LM is stable.
- **Partial lego** (e.g. `repainting_start=5.0, repainting_end=-1`) may need
  further work — the current fix sets `repainting_end_batch=None` for all
  full-track lego, which triggers a full mask. A partial mask with preserved
  src_latents would require passing `is_lego_task` into `conditioning_masks.py`
  and skipping the silence-replacement step there.
