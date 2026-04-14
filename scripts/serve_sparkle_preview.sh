#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PREVIEW_ROOT="${ROOT_DIR}/build-artifacts/sparkle-preview"
PORT="${PORT:-8000}"
BIND_ADDRESS="${BIND_ADDRESS:-127.0.0.1}"

fail() { printf "error: %s\n" "$1" >&2; exit 1; }

usage() {
  cat <<'USAGE'
Usage: scripts/serve_sparkle_preview.sh [options]

Serves a local Sparkle preview directory over HTTP for update testing.

Options:
  --root <path>         Directory containing preview.xml and the staged archive
  --port <num>          Port to bind (default: 8000)
  --bind <address>      Bind address (default: 127.0.0.1)
  -h, --help            Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      shift
      [[ $# -gt 0 ]] || fail "Missing value for --root"
      PREVIEW_ROOT="$1"
      ;;
    --port)
      shift
      [[ $# -gt 0 ]] || fail "Missing value for --port"
      PORT="$1"
      ;;
    --bind)
      shift
      [[ $# -gt 0 ]] || fail "Missing value for --bind"
      BIND_ADDRESS="$1"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "Unknown option: $1"
      ;;
  esac
  shift
done

[[ -d "$PREVIEW_ROOT" ]] || fail "Preview root not found: $PREVIEW_ROOT"
command -v python3 >/dev/null 2>&1 || fail "python3 not found"

cd "$PREVIEW_ROOT"
printf "serving Sparkle preview feed at http://%s:%s/\n" "$BIND_ADDRESS" "$PORT"
exec python3 -m http.server "$PORT" --bind "$BIND_ADDRESS"
