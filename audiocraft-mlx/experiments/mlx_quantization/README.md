# MLX MusicGen Quantization Experiments

This folder is isolated from the production localhost backend so we can test aggressive speed changes without touching:

- `/Users/karenjessen/audiocraft-mlx/audiocraft/g4l_localhost.py`
- `/Users/karenjessen/audiocraft-mlx/audiocraft/g4laudio_mlx.py`

## Why this exists

Your current continuation path is:

1. `/Users/karenjessen/audiocraft-mlx/audiocraft/g4l_localhost.py`
2. `/Users/karenjessen/audiocraft-mlx/audiocraft/g4laudio_mlx.py`
3. `/Users/karenjessen/audiocraft-mlx/audiocraft/mlx_continuation/mlx_musicgen.py`

The slowest stage is autoregressive token generation in `MusicGenContinuation._generate_tokens_with_prompt(...)`, especially on large finetunes with ~30s outputs.

## Benchmark script

`bench_quant_continuation.py` measures:

- model load time
- optional dtype cast time
- optional quantization time
- prompt encode time
- token generation time
- decode time
- real-time factor and peak memory

It supports in-memory quantization via `mlx.nn.quantize(...)` with tunable scope, bits, group size, and mode.

## Quick start

From repo root:

```bash
cd /Users/karenjessen/audiocraft-mlx
source .venv/bin/activate

python experiments/mlx_quantization/bench_quant_continuation.py \
  --model thepatch/keygen-gary-v2-large-16 \
  --base-model facebook/musicgen-large \
  --prompt /absolute/path/to/prompt.wav \
  --prompt-seconds 6 \
  --output-seconds 30 \
  --seed 12345 \
  --no-progress
```

Decoder-linears quantized 4-bit:

```bash
python experiments/mlx_quantization/bench_quant_continuation.py \
  --model thepatch/keygen-gary-v2-large-16 \
  --base-model facebook/musicgen-large \
  --prompt /absolute/path/to/prompt.wav \
  --quantize \
  --quant-scope decoder-linears \
  --q-bits 4 \
  --q-group-size 64 \
  --q-mode affine \
  --seed 12345 \
  --no-progress
```

If text is empty, CFG is disabled by default (recommended for MLX continuation quality).  
Override only if you explicitly want it:

```bash
--allow-empty-text-cfg
```

## Recommended first matrix

Run these and compare `timings_s.token_generate_s` and subjective audio quality:

1. Baseline (no quantization)
2. `--cast-dtype float16` only
3. `--quantize --quant-scope decoder-linears --q-bits 8 --q-group-size 64`
4. `--quantize --quant-scope decoder-linears --q-bits 4 --q-group-size 64`
5. `--quantize --quant-scope decoder-linears+emb --q-bits 4 --q-group-size 64`

Use the same `--seed` for all variants when doing listening tests (best-effort determinism).

The matrix runner now saves WAVs for each run so you can A/B directly:

```bash
./experiments/mlx_quantization/run_matrix.sh \
  /absolute/path/to/prompt.wav \
  thepatch/keygen-gary-v2-large-16 \
  facebook/musicgen-large \
  12345
```

Outputs land in:

- `/Users/karenjessen/audiocraft-mlx/experiments/mlx_quantization/reports/<timestamp>/`

Prompt slicing defaults to the **start** of the source file to match `process_audio`.
If you want continuation-style tail prompts, add:

```bash
--prompt-from-end
```

## Notes on expected behavior

- Quantization feasibility: Yes, supported by your installed MLX (`0.30.5`) through `mlx.nn.quantize` and quantized linear/embedding layers.
- Warmup: quantization increases first-run setup time but can reduce steady-state generation time and memory.
- Quality tradeoff: 8-bit is usually safer; 4-bit gives larger speed/memory wins but can degrade detail/transients.

## Next integration step (after benchmarks)

If results look good, add a guarded opt-in path in `/Users/karenjessen/audiocraft-mlx/audiocraft/g4laudio_mlx.py` behind env flags, defaulting to current behavior:

- `G4L_MLX_QUANTIZE=0|1`
- `G4L_MLX_Q_BITS=4|8`
- `G4L_MLX_Q_GROUP_SIZE=64`
- `G4L_MLX_Q_SCOPE=decoder-linears`

That keeps your current UX stable while enabling fast A/B testing in the plugin.
