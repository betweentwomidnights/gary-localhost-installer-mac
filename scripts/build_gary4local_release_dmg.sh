#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PROJECT_DIR="${ROOT_DIR}/gary4local"
XCODEPROJ="${PROJECT_DIR}/gary4local.xcodeproj"

SCHEME="${SCHEME:-gary4local}"
CONFIGURATION="${CONFIGURATION:-Release}"
IDENTITY="${IDENTITY:-Developer ID Application: Kevin Griffing (P8L4LGS728)}"
KEYCHAIN_PROFILE="${KEYCHAIN_PROFILE:-gary-profile}"
VOL_NAME="${VOL_NAME:-gary4local}"
TEAM_ID="${TEAM_ID:-P8L4LGS728}"
BUILD_ROOT="${BUILD_ROOT:-/tmp/gary4local-release-build}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${ROOT_DIR}/build-artifacts}"
ICON_PATH="${ICON_PATH:-${PROJECT_DIR}/icon.icns}"

SKIP_NOTARIZE=0
SKIP_LAYOUT=0
OUTPUT_NAME=""
MARKETING_VERSION_OVERRIDE=""
CURRENT_PROJECT_VERSION_OVERRIDE=""

ARCHIVE_PATH="${BUILD_ROOT}/${SCHEME}.xcarchive"
DERIVED_DATA_PATH="${BUILD_ROOT}/DerivedData"
STAGE_DIR="${BUILD_ROOT}/dmg-root"
TEMP_DMG="${BUILD_ROOT}/${SCHEME}-temp.dmg"
APP_PATH="${ARCHIVE_PATH}/Products/Applications/${SCHEME}.app"
STAGED_APP_PATH="${STAGE_DIR}/${SCHEME}.app"
MOUNT_POINT="/Volumes/${VOL_NAME}"

info() { printf "\n==> %s\n" "$1"; }
ok() { printf "ok: %s\n" "$1"; }
fail() { printf "error: %s\n" "$1" >&2; exit 1; }

append_xcodebuild_overrides() {
  if ((${#XCODEBUILD_OVERRIDES[@]})); then
    xcodebuild_args+=("${XCODEBUILD_OVERRIDES[@]}")
  fi
}

usage() {
  cat <<'USAGE'
Usage: scripts/build_gary4local_release_dmg.sh [options]

Builds, signs, notarizes, and staples a Developer ID DMG for gary4local.

Options:
  --output-name <filename>    Output DMG filename inside build-artifacts/
  --build-root <path>         Temporary working directory
  --configuration <name>      Xcode configuration (default: Release)
  --marketing-version <ver>   Override MARKETING_VERSION for this build
  --build-number <num>        Override CURRENT_PROJECT_VERSION for this build
  --identity <name>           Code signing identity
  --keychain-profile <name>   notarytool keychain profile
  --vol-name <name>           DMG volume name (default: gary4local)
  --skip-notarize             Build and sign the DMG but skip notarization/stapling
  --skip-layout               Skip Finder window layout customization
  -h, --help                  Show this help
USAGE
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

resolve_version() {
  if [[ -n "$MARKETING_VERSION_OVERRIDE" ]]; then
    printf '%s\n' "$MARKETING_VERSION_OVERRIDE"
    return
  fi

  local version
  version="$(
    xcodebuild \
      -project "$XCODEPROJ" \
      -scheme "$SCHEME" \
      -configuration "$CONFIGURATION" \
      -showBuildSettings 2>/dev/null |
      awk -F ' = ' '/MARKETING_VERSION = / { print $2; exit }'
  )"

  if [[ -z "$version" ]]; then
    version="1.0"
  fi

  printf '%s\n' "$version"
}

XCODEBUILD_OVERRIDES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-name)
      shift
      [[ $# -gt 0 ]] || fail "Missing value for --output-name"
      OUTPUT_NAME="$1"
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
      STAGED_APP_PATH="${STAGE_DIR}/${SCHEME}.app"
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
    --identity)
      shift
      [[ $# -gt 0 ]] || fail "Missing value for --identity"
      IDENTITY="$1"
      ;;
    --keychain-profile)
      shift
      [[ $# -gt 0 ]] || fail "Missing value for --keychain-profile"
      KEYCHAIN_PROFILE="$1"
      ;;
    --vol-name)
      shift
      [[ $# -gt 0 ]] || fail "Missing value for --vol-name"
      VOL_NAME="$1"
      MOUNT_POINT="/Volumes/${VOL_NAME}"
      ;;
    --skip-notarize)
      SKIP_NOTARIZE=1
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

VERSION="$(resolve_version)"
if [[ -z "$OUTPUT_NAME" ]]; then
  OUTPUT_NAME="${SCHEME}-v${VERSION}-mac-arm64.dmg"
fi

OUTPUT_DMG="${ARTIFACT_DIR}/${OUTPUT_NAME}"
APP_BINARY="${APP_PATH}/Contents/MacOS/${SCHEME}"

command -v xcodebuild >/dev/null 2>&1 || fail "xcodebuild not found"
command -v hdiutil >/dev/null 2>&1 || fail "hdiutil not found"
command -v codesign >/dev/null 2>&1 || fail "codesign not found"
command -v xcrun >/dev/null 2>&1 || fail "xcrun not found"
command -v lipo >/dev/null 2>&1 || fail "lipo not found"
command -v security >/dev/null 2>&1 || fail "security not found"

trap cleanup_mount EXIT

mkdir -p "$ARTIFACT_DIR"
rm -rf "$BUILD_ROOT"
mkdir -p "$BUILD_ROOT"

security find-identity -v -p codesigning | grep -Fq "$IDENTITY" || fail "Signing identity not found: $IDENTITY"

info "Archiving ${SCHEME} for arm64"
xcodebuild_args=(
  archive
  -project "$XCODEPROJ"
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
lipo -info "$APP_BINARY" | grep -q "arm64" || fail "Archived app is not arm64"

info "Preparing DMG staging directory"
mkdir -p "$STAGE_DIR"
ditto "$APP_PATH" "$STAGED_APP_PATH"
ln -s /Applications "${STAGE_DIR}/Applications"

info "Codesigning app bundle"
codesign \
  --force \
  --deep \
  --options runtime \
  --timestamp \
  --sign "$IDENTITY" \
  "$STAGED_APP_PATH"

codesign --verify --deep --strict --verbose=2 "$STAGED_APP_PATH"
ok "App signature verified"

info "Creating writable DMG"
rm -f "$TEMP_DMG"
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
rm -f "$OUTPUT_DMG"
hdiutil convert \
  "$TEMP_DMG" \
  -format UDZO \
  -imagekey zlib-level=9 \
  -o "$OUTPUT_DMG"

set_custom_icon "$OUTPUT_DMG" "$ICON_PATH"

info "Codesigning DMG"
codesign --force --sign "$IDENTITY" --timestamp "$OUTPUT_DMG"
codesign --verify --verbose=2 "$OUTPUT_DMG"
ok "DMG signature verified"

if [[ $SKIP_NOTARIZE -eq 0 ]]; then
  info "Submitting DMG for notarization"
  xcrun notarytool submit "$OUTPUT_DMG" \
    --keychain-profile "$KEYCHAIN_PROFILE" \
    --wait

  info "Stapling notarization ticket"
  xcrun stapler staple "$OUTPUT_DMG"
  xcrun stapler validate "$OUTPUT_DMG"
  ok "DMG notarized and stapled"
fi

info "Build complete"
ls -lh "$OUTPUT_DMG"
shasum -a 256 "$OUTPUT_DMG"
