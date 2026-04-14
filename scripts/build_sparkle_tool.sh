#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

SCHEME=""
CONFIGURATION="${CONFIGURATION:-Release}"
DERIVED_DATA_PATH=""

info() { printf "\n==> %s\n" "$1" >&2; }
fail() { printf "error: %s\n" "$1" >&2; exit 1; }

usage() {
  cat <<'USAGE'
Usage: scripts/build_sparkle_tool.sh --scheme <name> [options]

Builds a Sparkle helper tool from the Sparkle source checkout that Xcode resolved
for the gary4local project and prints the resulting binary path.

Supported schemes:
  generate_keys
  sign_update
  generate_appcast

Options:
  --scheme <name>           Sparkle scheme to build
  --configuration <name>    Xcode configuration (default: Release)
  --derived-data-path <p>   Derived data output directory
  -h, --help                Show this help
USAGE
}

find_sparkle_project() {
  local project_path
  project_path="$(
    find "${HOME}/Library/Developer/Xcode/DerivedData" \
      -path '*/SourcePackages/checkouts/Sparkle/Sparkle.xcodeproj' \
      -print \
      -quit 2>/dev/null || true
  )"

  [[ -n "$project_path" ]] || fail "Unable to locate Sparkle source checkout. Open gary4local.xcodeproj in Xcode and resolve package dependencies first."
  printf '%s\n' "$project_path"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scheme)
      shift
      [[ $# -gt 0 ]] || fail "Missing value for --scheme"
      SCHEME="$1"
      ;;
    --configuration)
      shift
      [[ $# -gt 0 ]] || fail "Missing value for --configuration"
      CONFIGURATION="$1"
      ;;
    --derived-data-path)
      shift
      [[ $# -gt 0 ]] || fail "Missing value for --derived-data-path"
      DERIVED_DATA_PATH="$1"
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

[[ -n "$SCHEME" ]] || fail "--scheme is required"

case "$SCHEME" in
  generate_keys|sign_update|generate_appcast)
    ;;
  *)
    fail "Unsupported Sparkle scheme: $SCHEME"
    ;;
esac

if [[ -z "$DERIVED_DATA_PATH" ]]; then
  DERIVED_DATA_PATH="${ROOT_DIR}/build-artifacts/sparkle-tools/${SCHEME}"
fi

SPARKLE_PROJECT="$(find_sparkle_project)"
OUTPUT_BINARY="${DERIVED_DATA_PATH}/Build/Products/${CONFIGURATION}/${SCHEME}"

mkdir -p "$(dirname "$DERIVED_DATA_PATH")"

info "Building Sparkle tool ${SCHEME}"
xcodebuild \
  -project "$SPARKLE_PROJECT" \
  -scheme "$SCHEME" \
  -configuration "$CONFIGURATION" \
  -derivedDataPath "$DERIVED_DATA_PATH" \
  build >&2

[[ -x "$OUTPUT_BINARY" ]] || fail "Expected Sparkle tool not found: $OUTPUT_BINARY"

printf '%s\n' "$OUTPUT_BINARY"
