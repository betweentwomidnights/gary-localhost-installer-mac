# gary4local macOS Update Feed

This document defines the intended updater contract for `gary4local` on macOS.

It is modeled after the documented Windows/Tauri flow in `gary-localhost-installer`, but adapted for a direct-download Swift app using Sparkle instead of Tauri's native updater.

## Status

- Sparkle is now wired into `gary4local` with a baked-in feed URL, public key, and `check for updates...` menu command.
- This document defines the feed shape and maintainer workflow we want to preserve across future sessions.
- The remaining validation step is end-to-end preview/stable update testing on Apple Silicon.

## Goal

The operator workflow should feel as close as possible to the existing Tauri flow:

1. Build the release artifact.
2. Push the GitHub release.
3. Update one stable feed on GitHub Pages.
4. Walk away knowing installed apps will notice.

## Distribution Model

`gary4local` is distributed outside the Mac App Store as a Developer ID signed + notarized download.

That means:

- GitHub Releases hosts the actual downloadable artifact.
- GitHub Pages hosts the stable update feed.
- The app embeds a stable public feed URL and Sparkle public key.

## Production Endpoints

Recommended stable URLs:

- `https://betweentwomidnights.github.io/gary-localhost-installer-mac/updates/gary4local/stable.xml`
- `https://betweentwomidnights.github.io/gary-localhost-installer-mac/updates/gary4local/preview.xml`

Recommended supporting paths:

- `docs/updates/gary4local/stable.xml`
- `docs/updates/gary4local/preview.xml`
- `docs/updates/gary4local/release-notes/`

The exact filenames can change, but the stable production appcast URL should remain constant once shipped.

## Artifact Strategy

Use the same notarized DMG already built for public distribution:

- `scripts/build_gary4local_release_dmg.sh`

Rationale:

- one public artifact for website downloads and in-app updates
- release assets remain easy to inspect manually
- the release process stays aligned with current notarized DMG packaging

## Security Model

Two separate trust layers are involved:

1. Apple trust:
   - app bundle signed with `Developer ID Application`
   - DMG signed and notarized
2. Sparkle trust:
   - update archive signed with Sparkle EdDSA
   - app embeds the Sparkle public key

Important rules:

- the Sparkle private key must never be committed
- the Sparkle public key is safe to ship in the app
- the appcast is public and can live in GitHub Pages

## Feed Contents

Each appcast item should include at least:

- machine-readable version
- human-readable version
- publish date
- release notes link or embedded notes
- download URL pointing at the GitHub Release asset
- archive length
- Sparkle EdDSA signature

Optional but useful:

- minimum supported macOS version
- Apple Silicon hardware requirement
- preview/stable channel separation

## Release Notes Strategy

Recommended approach:

- keep concise release notes in `docs/updates/gary4local/release-notes/`
- publish one release-notes file per version
- link that file from the appcast item

This keeps the appcast compact and makes the release notes readable in GitHub.

## Preview Channel

Keep a preview feed separate from stable:

- stable users read `stable.xml`
- internal/test users can point at `preview.xml`

This mirrors the stable/preview separation already documented in the Tauri repo and keeps release testing sane.

## What Gets Baked Into the App

Once Sparkle is integrated, production builds should embed:

- stable appcast URL
- Sparkle public EdDSA key

Preview/testing overrides can exist, but the production defaults should be baked in so end users do not need environment variables or manual setup.

Current runtime testing override:

- `GARY4LOCAL_SPARKLE_FEED_URL`

If set at launch, this overrides the baked-in feed URL and is suitable for local or preview appcast testing.

## Expected Maintainer Workflow

At release time:

1. build, sign, notarize, and staple the DMG
2. create or update the GitHub Release and upload the DMG
3. generate/update the Sparkle appcast item that points to that DMG
4. publish the updated appcast to GitHub Pages
5. confirm an installed app sees the new version from the stable feed

The detailed maintainer checklist lives in `docs/releasing/SPARKLE_RELEASE.md`.

## Local Preview Testing

Because `gary4local` is usually built on an Intel development Mac but tested on an Apple Silicon machine, the recommended validation loop is:

1. Install an older `gary4local` build on the Apple Silicon test machine.
2. Build a newer preview DMG with a higher version/build number.
3. Generate a local preview appcast that points at that newer DMG.
4. Serve the preview directory over `http://127.0.0.1:<port>/`.
5. Launch the installed app with `GARY4LOCAL_SPARKLE_FEED_URL` pointing at that local preview appcast.
6. Use `check for updates...` from the app menu and verify Sparkle surfaces the update UI.

Helpful scripts:

- `scripts/build_gary4local_release_dmg.sh --marketing-version <ver> --build-number <num>`
- `scripts/build_gary4local_adhoc_dmg.sh --marketing-version <ver> --build-number <num>`
- `scripts/generate_sparkle_preview_feed.sh`
- `scripts/serve_sparkle_preview.sh`

Example:

```bash
./scripts/build_gary4local_release_dmg.sh \
  --marketing-version 1.0.1-preview \
  --build-number 2 \
  --output-name gary4local-v1.0.1-preview.dmg

./scripts/generate_sparkle_preview_feed.sh \
  --archive build-artifacts/gary4local-v1.0.1-preview.dmg \
  --short-version 1.0.1-preview \
  --build-number 2 \
  --base-url http://127.0.0.1:8000

./scripts/serve_sparkle_preview.sh
```

Then on the Apple Silicon machine:

```bash
GARY4LOCAL_SPARKLE_FEED_URL=http://127.0.0.1:8000/preview.xml \
  /Applications/gary4local.app/Contents/MacOS/gary4local
```

Sparkle logging is printed to the launching terminal and will report when the appcast loads, when an update is found, or when no update is available.
