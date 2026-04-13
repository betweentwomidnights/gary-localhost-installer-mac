#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CAREY_PYTHON="${CAREY_PYTHON:-python3}"
CAREY_API_ROOT="${CAREY_API_ROOT:-${ROOT_DIR}/ACE-Step-1.5}"
CAREY_WRAPPER_ROOT="${CAREY_WRAPPER_ROOT:-${SCRIPT_DIR}}"
CAREY_API_HOST="${CAREY_API_HOST:-127.0.0.1}"
CAREY_API_PORT="${CAREY_API_PORT:-8001}"

# Wrapper consumes ACESTEP_URL + WRAPPER_PORT.
export ACESTEP_URL="${ACESTEP_URL:-http://${CAREY_API_HOST}:${CAREY_API_PORT}}"
export WRAPPER_PORT="${WRAPPER_PORT:-8003}"

# Carey performance defaults:
# - Prefer native MLX DiT/VAE by default for Apple Silicon throughput.
# - Keep an explicit opt-out for torch+MPS compatibility checks.
# - Respect caller-provided ACESTEP_USE_MLX_* settings from the control center.
# - Use FP16 MLX VAE by default for faster encode/decode.
if [[ -n "${ACESTEP_USE_MLX_DIT:-}" || -n "${ACESTEP_USE_MLX_VAE:-}" ]]; then
  export ACESTEP_USE_MLX_DIT="${ACESTEP_USE_MLX_DIT:-1}"
  export ACESTEP_USE_MLX_VAE="${ACESTEP_USE_MLX_VAE:-1}"
elif [[ "${ACESTEP_FORCE_TORCH_MPS:-0}" == "1" ]]; then
  export ACESTEP_USE_MLX_DIT=0
  export ACESTEP_USE_MLX_VAE=0
else
  export ACESTEP_USE_MLX_DIT=1
  export ACESTEP_USE_MLX_VAE=1
fi

if [[ "${ACESTEP_USE_MLX_VAE}" == "1" ]]; then
  export ACESTEP_MLX_VAE_FP16="${ACESTEP_MLX_VAE_FP16:-1}"
fi

echo "[carey] ACESTEP_USE_MLX_DIT=${ACESTEP_USE_MLX_DIT}"
echo "[carey] ACESTEP_USE_MLX_VAE=${ACESTEP_USE_MLX_VAE}"
if [[ -n "${ACESTEP_MLX_VAE_FP16:-}" ]]; then
  echo "[carey] ACESTEP_MLX_VAE_FP16=${ACESTEP_MLX_VAE_FP16}"
fi

API_PID=""
WRAPPER_PID=""

cleanup() {
  if [[ -n "${WRAPPER_PID}" ]] && kill -0 "${WRAPPER_PID}" 2>/dev/null; then
    kill "${WRAPPER_PID}" 2>/dev/null || true
    wait "${WRAPPER_PID}" 2>/dev/null || true
  fi
  if [[ -n "${API_PID}" ]] && kill -0 "${API_PID}" 2>/dev/null; then
    kill "${API_PID}" 2>/dev/null || true
    wait "${API_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

cd "${CAREY_API_ROOT}"
"${CAREY_PYTHON}" -m uvicorn acestep.api_server:app \
  --host "${CAREY_API_HOST}" \
  --port "${CAREY_API_PORT}" \
  --workers 1 \
  --log-level warning &
API_PID=$!

for i in $(seq 1 60); do
  if curl -fsS "http://${CAREY_API_HOST}:${CAREY_API_PORT}/health" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "${API_PID}" 2>/dev/null; then
    echo "carey api exited during startup" >&2
    exit 1
  fi
  if [[ "${i}" -eq 60 ]]; then
    echo "carey api health check timed out" >&2
    exit 1
  fi
  sleep 1
done

cd "${CAREY_WRAPPER_ROOT}"
"${CAREY_PYTHON}" main.py &
WRAPPER_PID=$!
wait "${WRAPPER_PID}"
