#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
RECREATE=0
UPGRADE_BUILD_TOOLS=1

usage() {
  cat <<'USAGE'
Usage: scripts/rebuild_venvs.sh [options]

Rebuilds local Python virtualenvs for:
  - audiocraft-mlx
  - melodyflow
  - stable-audio-tools

Options:
  --python <path>     Python interpreter to use (default: python3.11 or $PYTHON_BIN)
  --recreate          Delete existing .venv folders before rebuilding
  --no-upgrade-tools  Skip pip/setuptools/wheel upgrade
  -h, --help          Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      shift
      [[ $# -gt 0 ]] || { echo "Missing value for --python" >&2; exit 1; }
      PYTHON_BIN="$1"
      ;;
    --recreate)
      RECREATE=1
      ;;
    --no-upgrade-tools)
      UPGRADE_BUILD_TOOLS=0
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
  shift
done

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python interpreter not found: $PYTHON_BIN" >&2
  exit 1
fi

if ! "$PYTHON_BIN" -c 'import sys; print(sys.version_info >= (3, 11))' | grep -q "True"; then
  echo "Expected Python 3.11+ but got: $("$PYTHON_BIN" -V 2>&1)" >&2
  exit 1
fi

SERVICES=(
  "audiocraft-mlx|audiocraft/requirements.txt"
  "melodyflow|requirements.txt"
  "stable-audio-tools|requirements.txt"
)

echo "Using Python: $("$PYTHON_BIN" -V 2>&1)"
echo

for item in "${SERVICES[@]}"; do
  service_dir="${item%%|*}"
  requirements_rel="${item##*|}"
  service_path="${ROOT_DIR}/${service_dir}"
  requirements_path="${service_path}/${requirements_rel}"
  venv_path="${service_path}/.venv"
  venv_python="${venv_path}/bin/python"

  echo "=== ${service_dir} ==="
  if [[ ! -f "$requirements_path" ]]; then
    echo "Missing requirements file: $requirements_path" >&2
    exit 1
  fi

  if [[ $RECREATE -eq 1 && -d "$venv_path" ]]; then
    rm -rf "$venv_path"
  fi

  if [[ ! -d "$venv_path" ]]; then
    "$PYTHON_BIN" -m venv "$venv_path"
  fi

  if [[ ! -x "$venv_python" ]]; then
    echo "Missing venv python at: $venv_python" >&2
    exit 1
  fi

  if [[ $UPGRADE_BUILD_TOOLS -eq 1 ]]; then
    "$venv_python" -m pip install --upgrade pip setuptools wheel
  fi

  "$venv_python" -m pip install -r "$requirements_path"
  "$venv_python" -m pip --version
  echo
done

echo "All virtualenvs rebuilt successfully."
