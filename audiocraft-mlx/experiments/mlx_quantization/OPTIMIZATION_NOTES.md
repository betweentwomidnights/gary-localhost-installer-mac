# Continuation Speed Roadmap

## Current wiring and hotspots

- API entrypoint: `/Users/karenjessen/audiocraft-mlx/audiocraft/g4l_localhost.py`
- MLX bridge: `/Users/karenjessen/audiocraft-mlx/audiocraft/g4laudio_mlx.py`
- Generation loop: `/Users/karenjessen/audiocraft-mlx/audiocraft/mlx_continuation/mlx_musicgen.py`

Main cost center is autoregressive decoding in:

- `MusicGenContinuation._generate_tokens_with_prompt(...)`

## Options ranked by expected impact

1. Quantize decoder linears (4-bit or 8-bit)
   - Expected: major memory drop, meaningful generation speedup on large models.
   - Risk: quality loss increases as bits go down (4-bit > 8-bit risk).
   - Status: ready to test with `bench_quant_continuation.py`.

2. Cast float weights to float16 before generation
   - Expected: medium speedup, lower memory pressure.
   - Risk: some quality degradation possible, but often milder than 4-bit quantization.
   - Status: supported in benchmark with `--cast-dtype float16`.

3. Avoid unnecessary second decode in continuation path
   - In `/Users/karenjessen/audiocraft-mlx/audiocraft/mlx_continuation/mlx_musicgen.py`,
     `generate_continuation(...)` always decodes full and continuation audio, even when only full is used by `/Users/karenjessen/audiocraft-mlx/audiocraft/g4laudio_mlx.py`.
   - Expected: small-to-medium end-to-end improvement.
   - Risk: low; straightforward API adjustment.

4. Prompt prefill optimization
   - Current prompt tokens are walked step-by-step in the same loop as generated tokens.
   - Expected: moderate speedup when prompt is long (for example 6s prompt in a 30s target).
   - Risk: medium; requires careful cache/prefill logic changes.

5. Reduce generated duration or CFG usage
   - `OUTPUT_DURATION_S` is fixed at 30s in `/Users/karenjessen/audiocraft-mlx/audiocraft/g4l_models.py`.
   - CFG doubles forward batch size when text conditioning is active.
   - Expected: very large speedups, but changes UX/behavior.
   - Risk: product-level tradeoff.

## Suggested execution order

1. Run quantization matrix in this folder and pick best speed/quality point.
   - Keep seed fixed across variants and compare exported WAVs.
   - For empty text prompts, keep CFG disabled unless you explicitly want to test otherwise.
2. If needed, add float16 cast to the chosen quantization setup.
3. Apply low-risk decode cleanup in production path.
4. Only then evaluate prompt prefill changes.
