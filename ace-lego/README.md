# ace-lego

ACE-Step 1.5 deployment fork for lego mode (stem generation over source audio), wrapped for T4 GPU with load-on-demand VRAM management.

**Not ready for upstream fork or PR yet** — experimental.

---

## The Core Fix

`acestep/core/generation/handler/padding_utils.py` — when `is_lego_task=True`, return `None` for `repainting_end_batch` so conditioning routes through the full-mask branch in `conditioning_masks.py`, preserving `src_latents` as the full source audio rather than silencing it.

Additional patches:
- `acestep/llm_inference.py` — force `enforce_eager=True` on CUDA to skip CUDA graph capture (prevents RNG contamination of the sampler on GB10/T4)
- `acestep/third_parts/nano-vllm/nanovllm/engine/model_runner.py` — graceful fallback if graph capture ever runs
- `acestep/api_server.py` — added `POST /v1/load` and `POST /v1/unload` for VRAM lifecycle management

---

## Structure

```
ACE-Step-1.5/   patched fork of ace-step/ACE-Step-1.5
wrapper/        FastAPI service: POST /lego handles gpu-queue-service + model load/unload
```

See `AGENTS_T4.md` for the full T4 deployment plan and VRAM profile.
