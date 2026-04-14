#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

ARCHIVE_PATH=""
SHORT_VERSION=""
BUILD_NUMBER=""
BASE_URL="http://127.0.0.1:8000"
OUTPUT_DIR="${ROOT_DIR}/build-artifacts/sparkle-preview"
ACCOUNT="${ACCOUNT:-gary4local}"
MINIMUM_SYSTEM_VERSION="14.6"
APPCAST_FILENAME="preview.xml"
RELEASE_NOTES_FILENAME="release-notes.html"
TITLE_PREFIX="gary4local"

info() { printf "\n==> %s\n" "$1"; }
fail() { printf "error: %s\n" "$1" >&2; exit 1; }

usage() {
  cat <<'USAGE'
Usage: scripts/generate_sparkle_preview_feed.sh [options]

Creates a self-contained preview Sparkle feed directory for local testing.
The output directory will contain:
  - preview.xml
  - a copy of the DMG archive
  - a simple HTML release notes page

Options:
  --archive <path>          Path to the newer DMG/archive that Sparkle should offer
  --short-version <ver>     CFBundleShortVersionString of the newer build
  --build-number <num>      CFBundleVersion of the newer build
  --base-url <url>          Base URL the preview server will use (default: http://127.0.0.1:8000)
  --output-dir <path>       Output directory for preview.xml and copied archive
  --account <name>          Sparkle keychain account for sign_update (default: gary4local)
  --minimum-system-version  Minimum macOS version to publish in the appcast (default: 14.6)
  -h, --help                Show this help
USAGE
}

xml_escape() {
  local value="$1"
  value="${value//&/&amp;}"
  value="${value//</&lt;}"
  value="${value//>/&gt;}"
  value="${value//\"/&quot;}"
  value="${value//\'/&apos;}"
  printf '%s' "$value"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --archive)
      shift
      [[ $# -gt 0 ]] || fail "Missing value for --archive"
      ARCHIVE_PATH="$1"
      ;;
    --short-version)
      shift
      [[ $# -gt 0 ]] || fail "Missing value for --short-version"
      SHORT_VERSION="$1"
      ;;
    --build-number)
      shift
      [[ $# -gt 0 ]] || fail "Missing value for --build-number"
      BUILD_NUMBER="$1"
      ;;
    --base-url)
      shift
      [[ $# -gt 0 ]] || fail "Missing value for --base-url"
      BASE_URL="$1"
      ;;
    --output-dir)
      shift
      [[ $# -gt 0 ]] || fail "Missing value for --output-dir"
      OUTPUT_DIR="$1"
      ;;
    --account)
      shift
      [[ $# -gt 0 ]] || fail "Missing value for --account"
      ACCOUNT="$1"
      ;;
    --minimum-system-version)
      shift
      [[ $# -gt 0 ]] || fail "Missing value for --minimum-system-version"
      MINIMUM_SYSTEM_VERSION="$1"
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

[[ -n "$ARCHIVE_PATH" ]] || fail "--archive is required"
[[ -f "$ARCHIVE_PATH" ]] || fail "Archive not found: $ARCHIVE_PATH"
[[ -n "$SHORT_VERSION" ]] || fail "--short-version is required"
[[ -n "$BUILD_NUMBER" ]] || fail "--build-number is required"

SIGN_UPDATE_BIN="$("${ROOT_DIR}/scripts/build_sparkle_tool.sh" --scheme sign_update)"
[[ -x "$SIGN_UPDATE_BIN" ]] || fail "sign_update tool missing: $SIGN_UPDATE_BIN"

mkdir -p "$OUTPUT_DIR"

ARCHIVE_BASENAME="$(basename "$ARCHIVE_PATH")"
STAGED_ARCHIVE_PATH="${OUTPUT_DIR}/${ARCHIVE_BASENAME}"
cp -f "$ARCHIVE_PATH" "$STAGED_ARCHIVE_PATH"

ARCHIVE_URL="${BASE_URL%/}/${ARCHIVE_BASENAME}"
RELEASE_NOTES_URL="${BASE_URL%/}/${RELEASE_NOTES_FILENAME}"
APPCAST_PATH="${OUTPUT_DIR}/${APPCAST_FILENAME}"
RELEASE_NOTES_PATH="${OUTPUT_DIR}/${RELEASE_NOTES_FILENAME}"

info "Signing preview archive"
ARCHIVE_SIGNATURE="$(
  "$SIGN_UPDATE_BIN" --account "$ACCOUNT" -p "$STAGED_ARCHIVE_PATH" | tr -d '\r\n'
)"
[[ -n "$ARCHIVE_SIGNATURE" ]] || fail "sign_update returned an empty signature"

ARCHIVE_LENGTH="$(stat -f%z "$STAGED_ARCHIVE_PATH")"
PUB_DATE="$(LC_ALL=C date -u +"%a, %d %b %Y %H:%M:%S +0000")"

cat > "$RELEASE_NOTES_PATH" <<EOF
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>${TITLE_PREFIX} ${SHORT_VERSION} preview notes</title>
    <style>
      body {
        font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        margin: 24px;
        line-height: 1.5;
        color: #111;
      }
      h1 { margin-bottom: 0.25rem; }
      p { margin-top: 0; color: #444; }
      ul { padding-left: 1.2rem; }
    </style>
  </head>
  <body>
    <h1>${TITLE_PREFIX} ${SHORT_VERSION}</h1>
    <p>Preview Sparkle update feed for local validation.</p>
    <ul>
      <li>Build number: ${BUILD_NUMBER}</li>
      <li>Archive: ${ARCHIVE_BASENAME}</li>
      <li>Minimum macOS: ${MINIMUM_SYSTEM_VERSION}</li>
    </ul>
  </body>
</html>
EOF

cat > "$APPCAST_PATH" <<EOF
<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0"
     xmlns:sparkle="http://www.andymatuschak.org/xml-namespaces/sparkle"
     xmlns:dc="http://purl.org/dc/elements/1.1/">
  <channel>
    <title>${TITLE_PREFIX} preview updates</title>
    <link>https://github.com/betweentwomidnights/gary-localhost-installer-mac</link>
    <description>Preview Sparkle appcast for ${TITLE_PREFIX}.</description>
    <language>en</language>
    <item>
      <title>Version $(xml_escape "$SHORT_VERSION")</title>
      <pubDate>${PUB_DATE}</pubDate>
      <sparkle:version>$(xml_escape "$BUILD_NUMBER")</sparkle:version>
      <sparkle:shortVersionString>$(xml_escape "$SHORT_VERSION")</sparkle:shortVersionString>
      <sparkle:minimumSystemVersion>$(xml_escape "$MINIMUM_SYSTEM_VERSION")</sparkle:minimumSystemVersion>
      <sparkle:releaseNotesLink>$(xml_escape "$RELEASE_NOTES_URL")</sparkle:releaseNotesLink>
      <description><![CDATA[Preview update $(xml_escape "$SHORT_VERSION")]]></description>
      <enclosure
        url="$(xml_escape "$ARCHIVE_URL")"
        sparkle:edSignature="$(xml_escape "$ARCHIVE_SIGNATURE")"
        sparkle:os="macos"
        length="${ARCHIVE_LENGTH}"
        type="application/x-apple-diskimage" />
    </item>
  </channel>
</rss>
EOF

info "Signing preview appcast"
"$SIGN_UPDATE_BIN" --account "$ACCOUNT" "$APPCAST_PATH" >/dev/null

info "Preview feed ready"
printf "output dir: %s\n" "$OUTPUT_DIR"
printf "appcast: %s\n" "$APPCAST_PATH"
printf "archive url: %s\n" "$ARCHIVE_URL"
printf "feed override: %s/%s\n" "${BASE_URL%/}" "$APPCAST_FILENAME"
