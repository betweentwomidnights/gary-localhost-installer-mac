#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

CONFIGURATION="${CONFIGURATION:-Release}"
SCHEME="${SCHEME:-gary4local}"
VOL_NAME="${VOL_NAME:-gary4local}"
OUTPUT_NAME="${OUTPUT_NAME:-gary4local-arm64-adhoc.dmg}"
MARKETING_VERSION_OVERRIDE=""
CURRENT_PROJECT_VERSION_OVERRIDE=""
BUILD_ROOT="${BUILD_ROOT:-/tmp/gary4local-adhoc-build}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${ROOT_DIR}/build-artifacts}"
ICON_PATH="${ICON_PATH:-${ROOT_DIR}/gary4local/icon.icns}"
SKIP_LAYOUT=0
ARCHIVE_PATH="${BUILD_ROOT}/${SCHEME}.xcarchive"
DERIVED_DATA_PATH="${BUILD_ROOT}/DerivedData"
STAGE_DIR="${BUILD_ROOT}/dmg-root"
TEMP_DMG="${BUILD_ROOT}/${SCHEME}-temp.dmg"
OUTPUT_DMG="${ARTIFACT_DIR}/${OUTPUT_NAME}"
APP_PATH="${ARCHIVE_PATH}/Products/Applications/${SCHEME}.app"
APP_BINARY="${APP_PATH}/Contents/MacOS/${SCHEME}"
MOUNT_POINT="/Volumes/${VOL_NAME}"

info() { printf "\n==> %s\n" "$1"; }
fail() { printf "error: %s\n" "$1" >&2; exit 1; }

append_xcodebuild_overrides() {
  if ((${#XCODEBUILD_OVERRIDES[@]})); then
    xcodebuild_args+=("${XCODEBUILD_OVERRIDES[@]}")
  fi
}

set_custom_icon() {
  local target="$1"
  local icon="$2"
  [[ -f "$icon" ]] || return 0

  osascript -e "
    use framework \"AppKit\"
    set iconImage to current application's NSImage's alloc()'s initWithContentsOfFile:\"$icon\"
    set result to current application's NSWorkspace's sharedWorkspace()'s setIcon:iconImage forFile:\"$target\" options:0
  " >/dev/null || true
}

configure_dmg_window() {
  local volume_name="$1"
  local app_name="$2"
  local dest_name="$3"

  sleep 2

  osascript <<EOF
tell application "Finder"
    tell disk "$volume_name"
        open
        delay 1
        set current view of container window to icon view
        set toolbar visible of container window to false
        set statusbar visible of container window to false
        set bounds of container window to {200, 200, 660, 470}
        set theViewOptions to the icon view options of container window
        set arrangement of theViewOptions to not arranged
        set icon size of theViewOptions to 112
        set text size of theViewOptions to 12
        set position of item "$app_name" of container window to {120, 115}
        set position of item "$dest_name" of container window to {340, 115}
        close
        open
        delay 0.5
        close
    end tell
end tell
EOF
}

cleanup_mount() {
  if [[ -d "$MOUNT_POINT" ]]; then
    hdiutil detach "$MOUNT_POINT" -force >/dev/null 2>&1 || true
  fi
}

usage() {
  cat <<'USAGE'
Usage: scripts/build_gary4local_adhoc_dmg.sh [options]

Builds an arm64 ad hoc DMG suitable for trusted Apple Silicon testing.

Options:
  --output-name <filename>   Output DMG filename inside build-artifacts/
  --build-root <path>        Temporary working directory (default: /tmp/gary4local-adhoc-build)
  --configuration <name>     Xcode configuration (default: Release)
  --marketing-version <ver>  Override MARKETING_VERSION for this build
  --build-number <num>       Override CURRENT_PROJECT_VERSION for this build
  --vol-name <name>          DMG volume name (default: gary4local)
  --skip-layout              Skip Finder window layout customization
  -h, --help                 Show this help
USAGE
}

XCODEBUILD_OVERRIDES=()

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
      TEMP_DMG="${BUILD_ROOT}/${SCHEME}-temp.dmg"
      APP_PATH="${ARCHIVE_PATH}/Products/Applications/${SCHEME}.app"
      APP_BINARY="${APP_PATH}/Contents/MacOS/${SCHEME}"
      ;;
    --configuration)
      shift
      [[ $# -gt 0 ]] || fail "Missing value for --configuration"
      CONFIGURATION="$1"
      ;;
    --marketing-version)
      shift
      [[ $# -gt 0 ]] || fail "Missing value for --marketing-version"
      MARKETING_VERSION_OVERRIDE="$1"
      ;;
    --build-number)
      shift
      [[ $# -gt 0 ]] || fail "Missing value for --build-number"
      CURRENT_PROJECT_VERSION_OVERRIDE="$1"
      ;;
    --vol-name)
      shift
      [[ $# -gt 0 ]] || fail "Missing value for --vol-name"
      VOL_NAME="$1"
      MOUNT_POINT="/Volumes/${VOL_NAME}"
      ;;
    --skip-layout)
      SKIP_LAYOUT=1
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

if [[ -n "$MARKETING_VERSION_OVERRIDE" ]]; then
  XCODEBUILD_OVERRIDES+=("MARKETING_VERSION=${MARKETING_VERSION_OVERRIDE}")
fi

if [[ -n "$CURRENT_PROJECT_VERSION_OVERRIDE" ]]; then
  XCODEBUILD_OVERRIDES+=("CURRENT_PROJECT_VERSION=${CURRENT_PROJECT_VERSION_OVERRIDE}")
fi

command -v xcodebuild >/dev/null 2>&1 || fail "xcodebuild not found"
command -v hdiutil >/dev/null 2>&1 || fail "hdiutil not found"
command -v codesign >/dev/null 2>&1 || fail "codesign not found"

trap cleanup_mount EXIT

mkdir -p "$ARTIFACT_DIR"
rm -rf "$BUILD_ROOT"
mkdir -p "$BUILD_ROOT"

info "Archiving ${SCHEME} for arm64"
xcodebuild_args=(
  archive
  -project "${ROOT_DIR}/gary4local/gary4local.xcodeproj"
  -scheme "$SCHEME"
  -configuration "$CONFIGURATION"
  -archivePath "$ARCHIVE_PATH"
  -derivedDataPath "$DERIVED_DATA_PATH"
  ARCHS=arm64
  ONLY_ACTIVE_ARCH=NO
  CODE_SIGNING_ALLOWED=NO
)
append_xcodebuild_overrides
xcodebuild "${xcodebuild_args[@]}"

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

info "Creating writable DMG"
rm -f "$TEMP_DMG" "$OUTPUT_DMG"
hdiutil create \
  -volname "$VOL_NAME" \
  -srcfolder "$STAGE_DIR" \
  -ov \
  -format UDRW \
  "$TEMP_DMG"

if [[ $SKIP_LAYOUT -eq 0 ]]; then
  if [[ -d "$MOUNT_POINT" ]]; then
    hdiutil detach "$MOUNT_POINT" -force >/dev/null 2>&1 || true
    sleep 1
  fi

  info "Configuring DMG window layout"
  hdiutil attach "$TEMP_DMG" -mountpoint "$MOUNT_POINT"
  configure_dmg_window "$VOL_NAME" "${SCHEME}.app" "Applications"
  osascript -e "tell application \"Finder\" to eject disk \"$VOL_NAME\"" >/dev/null 2>&1 || true
  sleep 2
  cleanup_mount
fi

info "Converting to compressed DMG"
hdiutil convert \
  "$TEMP_DMG" \
  -format UDZO \
  -imagekey zlib-level=9 \
  -o "$OUTPUT_DMG"

set_custom_icon "$OUTPUT_DMG" "$ICON_PATH"

info "Build complete"
ls -lh "$OUTPUT_DMG"
shasum -a 256 "$OUTPUT_DMG"
