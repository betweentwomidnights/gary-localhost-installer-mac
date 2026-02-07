#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <prompt_wav_path> <model_repo> [base_model_repo] [seed]"
  echo "Example: $0 /tmp/prompt.wav thepatch/keygen-gary-v2-large-16 facebook/musicgen-large 12345"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BENCH="$ROOT_DIR/experiments/mlx_quantization/bench_quant_continuation.py"
PROMPT="$1"
MODEL="$2"
BASE_MODEL="${3:-}"
SEED="${4:-12345}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$ROOT_DIR/experiments/mlx_quantization/reports/$TIMESTAMP"
mkdir -p "$OUT_DIR"

source "$ROOT_DIR/.venv/bin/activate"

COMMON=(
  --model "$MODEL"
  --prompt "$PROMPT"
  --prompt-seconds 6
  --output-seconds 30
  --text "${TEXT:-}"
  --top-k "${TOP_K:-250}"
  --temperature "${TEMPERATURE:-1.0}"
  --guidance-coef "${GUIDANCE_COEF:-3.0}"
  --seed "$SEED"
  --no-progress
)

if [[ -n "$BASE_MODEL" ]]; then
  COMMON+=(--base-model "$BASE_MODEL")
fi

if [[ "${ALLOW_EMPTY_TEXT_CFG:-0}" == "1" ]]; then
  COMMON+=(--allow-empty-text-cfg)
fi

echo "Running baseline..."
python "$BENCH" "${COMMON[@]}" \
  --save-prefix "$OUT_DIR/01_baseline" \
  --report-json "$OUT_DIR/01_baseline.json" | tee "$OUT_DIR/01_baseline.log"

echo "Running float16 cast..."
python "$BENCH" "${COMMON[@]}" \
  --cast-dtype float16 \
  --save-prefix "$OUT_DIR/02_cast_float16" \
  --report-json "$OUT_DIR/02_cast_float16.json" | tee "$OUT_DIR/02_cast_float16.log"

echo "Running 8-bit decoder-linears..."
python "$BENCH" "${COMMON[@]}" \
  --quantize \
  --quant-scope decoder-linears \
  --q-bits 8 \
  --q-group-size 64 \
  --q-mode affine \
  --save-prefix "$OUT_DIR/03_q8_decoder_linears" \
  --report-json "$OUT_DIR/03_q8_decoder_linears.json" | tee "$OUT_DIR/03_q8_decoder_linears.log"

echo "Running 4-bit decoder-linears..."
python "$BENCH" "${COMMON[@]}" \
  --quantize \
  --quant-scope decoder-linears \
  --q-bits 4 \
  --q-group-size 64 \
  --q-mode affine \
  --save-prefix "$OUT_DIR/04_q4_decoder_linears" \
  --report-json "$OUT_DIR/04_q4_decoder_linears.json" | tee "$OUT_DIR/04_q4_decoder_linears.log"

echo "Running 4-bit decoder-linears+emb..."
python "$BENCH" "${COMMON[@]}" \
  --quantize \
  --quant-scope decoder-linears+emb \
  --q-bits 4 \
  --q-group-size 64 \
  --q-mode affine \
  --save-prefix "$OUT_DIR/05_q4_decoder_linears_emb" \
  --report-json "$OUT_DIR/05_q4_decoder_linears_emb.json" | tee "$OUT_DIR/05_q4_decoder_linears_emb.log"

echo "Done. Reports in: $OUT_DIR"
