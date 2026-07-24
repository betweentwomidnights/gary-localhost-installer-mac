# SA3 SAME-L Decoder A/B: Gary Chunked vs Official Optimized

Date: 2026-06-29  
Machine: Apple M4 macOS  
Purpose: verify whether the official optimized MLX SAME-L decoder is actually faster than the
current `gary4local` SA3 backend decoder path for a UX-relevant 120 second generation.

## Result

The official optimized SAME-L decoder was **not faster** in this controlled Gary-backend test.

Current Gary chunked SAME-L decode was faster on both checks:

| check | Gary chunked decode | official optimized decoder | result |
|---|---:|---:|---|
| full 120s generation A/B, decode portion | 18.53s | 23.66s | Gary faster by 5.12s |
| full 120s generation A/B, sampling + decode | 41.40s | 47.70s | Gary faster by 6.30s |
| decode-only replay, measured pass | 20.79s | 23.79s | Gary faster by 3.00s |

Conclusion: do **not** switch Gary to the official optimized decoder as-is. The earlier standalone
benchmark that made the official decoder look faster was not the same runtime shape as Gary's
production path.

## Test Shape

Both measured full generations used:

- Prompt: `upbeat funk groove with slap bass`
- Duration: `120s`
- Seed: `0`
- Steps: `8`
- CFG scale: `1.0`
- Sampler: `pingpong`
- Gary production-style internal duration padding: `6s`
- Latent shape: `[1, 256, 1358]`
- Decode chunks: size `128`, overlap `32`, batch size `1`
- Gary pipeline: `StableAudioMLXPipeline.from_torch_pretrained("medium", ...)`
- Dtypes matching Gary production defaults:
  - DiT: `float32`
  - text conditioner: `float16`
  - number conditioner: `float16`
  - autoencoder: `float16`
- Official decoder weights:
  `/Users/karenjessen/stable-audio-3-official/optimized/mlx/models/mlx/same_l_decoder_f32.npz`

The benchmark used Gary's vendored pipeline for both runs and changed only the final decoder path:

- Gary path: `pipe.autoencoder.decode_audio(..., chunked=True, chunk_size=128, overlap=32, chunk_batch_size=1)`
- Official path: `models.defs.same_l_decoder.decode_chunked(...)` plus
  `models.defs.sa3_pipeline.patched_decode(...)`

## Full Generation A/B

Before the measured 120s runs, each decoder path got one 12s warmup.

| decoder | sampling | decode | sampling + decode | output |
|---|---:|---:|---:|---|
| Gary chunked | 22.86s | 18.53s | 41.40s | `bench/sa3-decoder-ab-2026-06-29/gary_chunked_duration120_seed0.wav` |
| official optimized | 24.04s | 23.66s | 47.70s | `bench/sa3-decoder-ab-2026-06-29/official_same_l_duration120_seed0.wav` |

The sampled latents were identical:

- latent cosine: `1.0`
- latent max absolute diff: `0.0`

The decoded audio was similar but not identical:

- raw waveform cosine: `0.915257`
- RMS envelope cosine: `0.999814`
- log-mag spectrogram cosine: `0.939896`

## Decode-Only Replay

To check whether the official decoder was penalized by first long-shape setup, I also sampled one
120s latent tensor with Gary's pipeline, then decoded it through both decoders twice.

| phase | decoder | decode time |
|---|---|---:|
| warmup | Gary chunked | 18.94s |
| warmup | official optimized | 23.46s |
| measured | Gary chunked | 20.79s |
| measured | official optimized | 23.79s |

This replay supports the same conclusion: for Gary's current runtime shape, the official optimized
decoder is slower as a drop-in decoder.

## Artifacts

Generated benchmark artifacts live under:

```text
bench/sa3-decoder-ab-2026-06-29/
```

Files:

- `decoder_ab_report.json`
- `decoder_replay_report.json`
- `gary_chunked_duration120_seed0.wav`
- `official_same_l_duration120_seed0.wav`

## Follow-Up

The official decoder may still contain implementation ideas worth studying, but the next step should
not be a direct Gary backend swap. Better follow-ups:

- inspect the official decoder shape for ideas that can be selectively ported without regressing Gary
  production settings
- repeat only if we test a materially different official path, such as a compiled decoder mode or a
  full official pipeline rather than only the decoder
- return to `sa3.cpp` work: port promising SAME-L graph structure to ggml and evaluate
  `GGML_OP_FLASH_ATTN_EXT` for compatible attention subgraphs
