#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

CONFIGURATION="${CONFIGURATION:-Release}"
SCHEME="${SCHEME:-gary4local}"
VOL_NAME="${VOL_NAME:-gary4local}"
OUTPUT_NAME="${OUTPUT_NAME:-gary4local-arm64-adhoc.dmg}"
BUILD_ROOT="${BUILD_ROOT:-/tmp/gary4local-adhoc-build}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${ROOT_DIR}/build-artifacts}"
ARCHIVE_PATH="${BUILD_ROOT}/${SCHEME}.xcarchive"
DERIVED_DATA_PATH="${BUILD_ROOT}/DerivedData"
STAGE_DIR="${BUILD_ROOT}/dmg-root"
OUTPUT_DMG="${ARTIFACT_DIR}/${OUTPUT_NAME}"
APP_PATH="${ARCHIVE_PATH}/Products/Applications/${SCHEME}.app"
APP_BINARY="${APP_PATH}/Contents/MacOS/${SCHEME}"

info() { printf "\n==> %s\n" "$1"; }
fail() { printf "error: %s\n" "$1" >&2; exit 1; }

usage() {
  cat <<'USAGE'
Usage: scripts/build_gary4local_adhoc_dmg.sh [options]

Builds an arm64 ad hoc DMG suitable for trusted Apple Silicon testing.

Options:
  --output-name <filename>   Output DMG filename inside build-artifacts/
  --build-root <path>        Temporary working directory (default: /tmp/gary4local-adhoc-build)
  --configuration <name>     Xcode configuration (default: Release)
  --vol-name <name>          DMG volume name (default: gary4local)
  -h, --help                 Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-name)
      shift
      [[ $# -gt 0 ]] || fail "Missing value for --output-name"
      OUTPUT_NAME="$1"
      OUTPUT_DMG="${ARTIFACT_DIR}/${OUTPUT_NAME}"
      ;;
    --build-root)
      shift
      [[ $# -gt 0 ]] || fail "Missing value for --build-root"
      BUILD_ROOT="$1"
      ARCHIVE_PATH="${BUILD_ROOT}/${SCHEME}.xcarchive"
      DERIVED_DATA_PATH="${BUILD_ROOT}/DerivedData"
      STAGE_DIR="${BUILD_ROOT}/dmg-root"
      APP_PATH="${ARCHIVE_PATH}/Products/Applications/${SCHEME}.app"
      APP_BINARY="${APP_PATH}/Contents/MacOS/${SCHEME}"
      ;;
    --configuration)
      shift
      [[ $# -gt 0 ]] || fail "Missing value for --configuration"
      CONFIGURATION="$1"
      ;;
    --vol-name)
      shift
      [[ $# -gt 0 ]] || fail "Missing value for --vol-name"
      VOL_NAME="$1"
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

command -v xcodebuild >/dev/null 2>&1 || fail "xcodebuild not found"
command -v hdiutil >/dev/null 2>&1 || fail "hdiutil not found"
command -v codesign >/dev/null 2>&1 || fail "codesign not found"

mkdir -p "$ARTIFACT_DIR"
rm -rf "$BUILD_ROOT"
mkdir -p "$BUILD_ROOT"

info "Archiving ${SCHEME} for arm64"
xcodebuild archive \
  -project "${ROOT_DIR}/gary4local/gary4local.xcodeproj" \
  -scheme "$SCHEME" \
  -configuration "$CONFIGURATION" \
  -archivePath "$ARCHIVE_PATH" \
  -derivedDataPath "$DERIVED_DATA_PATH" \
  ARCHS=arm64 \
  ONLY_ACTIVE_ARCH=NO \
  CODE_SIGNING_ALLOWED=NO

[[ -f "$APP_BINARY" ]] || fail "Archive missing app binary: $APP_BINARY"

if ! lipo -info "$APP_BINARY" | grep -q "arm64"; then
  fail "Archived app is not arm64"
fi

info "Preparing DMG staging directory"
mkdir -p "$STAGE_DIR"
ditto "$APP_PATH" "${STAGE_DIR}/${SCHEME}.app"
ln -s /Applications "${STAGE_DIR}/Applications"

info "Applying ad hoc signature"
codesign --force --deep -s - "${STAGE_DIR}/${SCHEME}.app"

info "Creating DMG"
rm -f "$OUTPUT_DMG"
hdiutil create \
  -volname "$VOL_NAME" \
  -srcfolder "$STAGE_DIR" \
  -ov \
  -format UDZO \
  "$OUTPUT_DMG"

info "Build complete"
ls -lh "$OUTPUT_DMG"
shasum -a 256 "$OUTPUT_DMG"
