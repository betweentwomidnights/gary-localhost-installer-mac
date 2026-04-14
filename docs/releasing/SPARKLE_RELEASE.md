# Sparkle Release Flow for gary4local

This document describes the intended maintainer workflow for shipping `gary4local` updates on macOS using Sparkle.

It is deliberately written in the same spirit as the updater docs in `gary-localhost-installer`: public enough to reproduce the workflow in fresh sessions, but without committing secrets.

## Status

- Sparkle is the active in-app updater design for `gary4local`.
- This is the release checklist we want to standardize around.
- If the wiring or packaging details change, update this file instead of relying on memory.

## High-Level Model

The macOS updater should work like this:

1. `gary4local` ships with a stable Sparkle appcast URL and Sparkle public key.
2. The public appcast lives on GitHub Pages.
3. The downloadable update artifact lives on GitHub Releases.
4. A new release becomes visible to installed apps when the stable appcast is updated.

## Stable Files and URLs

Recommended public URLs:

- stable appcast:
  - `https://betweentwomidnights.github.io/gary-localhost-installer-mac/updates/gary4local/stable.xml`
- preview appcast:
  - `https://betweentwomidnights.github.io/gary-localhost-installer-mac/updates/gary4local/preview.xml`

Recommended repo paths:

- `docs/updates/gary4local/stable.xml`
- `docs/updates/gary4local/preview.xml`
- `docs/updates/gary4local/release-notes/`

## Secrets and Local State

Do not commit:

- Apple ID / notary credentials
- Sparkle private EdDSA signing key

Current local key setup:

- Sparkle key generated via `generate_keys --account gary4local`
- private key material remains in the macOS keychain under that account

Safe to commit or ship:

- Sparkle public key
- appcast XML
- release-notes files

## Release Artifact

Primary artifact:

- notarized `gary4local` DMG built by `scripts/build_gary4local_release_dmg.sh`

Why reuse the DMG:

- it matches the public download artifact
- it keeps the release process simple
- it avoids creating a second distribution format unless Sparkle integration proves that necessary

If future Sparkle testing shows a `.zip` or `.tar.xz` archive is operationally cleaner, update this document and the build scripts together.

## Preflight

Before cutting a release:

1. Confirm the marketing version and build number are correct in Xcode.
2. Confirm the app launches locally from a clean build.
3. Confirm the release DMG script still signs, notarizes, staples, and validates successfully.
4. Confirm you have local access to:
   - the Developer ID identity
   - the notarytool keychain profile
   - the Sparkle private signing key

## Release Checklist

### 1. Build the notarized DMG

From repo root:

```bash
./scripts/build_gary4local_release_dmg.sh
```

Expected result:

- notarized, stapled DMG in `build-artifacts/`

Recommended sanity checks:

```bash
spctl -a -vv build-artifacts/gary4local-v<version>-mac-arm64.dmg
shasum -a 256 build-artifacts/gary4local-v<version>-mac-arm64.dmg
```

### 2. Create the GitHub Release

Create or update a GitHub Release tagged `v<version>`.

Upload:

- the notarized DMG

The GitHub Release asset URL becomes the appcast download URL.

### 3. Prepare release notes

Add a versioned notes file under:

- `docs/updates/gary4local/release-notes/`

Recommended filename pattern:

- `v<version>.md`

Keep it concise and user-facing.

### 4. Sign the update archive for Sparkle

Using the Sparkle tooling on the release DMG:

```bash
sign_update /path/to/gary4local-v<version>-mac-arm64.dmg
```

Or use Sparkle's `generate_appcast` helper if that becomes the standard local tool.

Expected output includes:

- Sparkle EdDSA signature
- archive length

Those values are required in the appcast item.

### 5. Update the appcast

Update either:

- `docs/updates/gary4local/stable.xml`
- or `docs/updates/gary4local/preview.xml`

The release item should point to:

- the GitHub Release DMG URL
- the versioned release-notes file
- the Sparkle signature and archive length

Important rule:

- the stable appcast URL must remain stable
- only its contents change from release to release

That is what lets installed apps "notice" the update without any per-release app reconfiguration.

### 6. Publish GitHub Pages changes

Commit and push the updated files under `docs/`.

Then verify the live Pages URL serves the new appcast contents.

### 7. End-to-end verification

On a machine with an older installed build:

1. launch `gary4local`
2. verify it sees the new update from the stable appcast
3. verify install/relaunch behavior
4. verify the relaunched app reports the expected version

## What Must Be Implemented in the App

For this release flow to work, `gary4local` needs Sparkle wired in with:

- a baked-in `SUFeedURL`
- a baked-in `SUPublicEDKey`
- a user-visible "Check for Updates…" action
- background/automatic update checks using Sparkle defaults unless product requirements say otherwise

Because the current Xcode project uses `GENERATE_INFOPLIST_FILE = YES`, these values will likely need to be injected through build settings or a custom `Info.plist` rather than assumed to already exist.

## Recommended Automation Follow-Up

Manual release steps are acceptable initially, but the end state should include a small helper that:

1. takes a version and DMG path
2. generates the Sparkle signature metadata
3. writes or updates the appcast item
4. optionally writes a preview or stable feed

That would give this repo the same "documented and reproducible" feel as the Tauri updater flow.

Current helper scripts in this repo:

- `scripts/build_sparkle_tool.sh`
- `scripts/generate_sparkle_preview_feed.sh`
- `scripts/serve_sparkle_preview.sh`

## Decision Notes

These choices are intentional:

- GitHub Releases remains the canonical host for downloadable artifacts
- GitHub Pages remains the stable feed host
- installed apps only need to know one stable appcast URL
- updater secrets stay local
- public documentation remains sufficient to reproduce the flow in a fresh session
