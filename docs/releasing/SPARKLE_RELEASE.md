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
2. Confirm `About gary4local` shows the intended recommended `gary4juce` companion version and the exact matching GitHub release tag URL by updating `gary4local/gary4local/AppReleaseInfo.swift` if needed.
3. Confirm the app launches locally from a clean build.
4. Confirm the release DMG script still signs, notarizes, staples, and validates successfully.
5. Confirm you have local access to:
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
spctl -a -t open --context context:primary-signature -v build-artifacts/gary4local-v<version>-mac-arm64.dmg
shasum -a 256 build-artifacts/gary4local-v<version>-mac-arm64.dmg
```

Note: plain `spctl -a -vv <dmg>` reports `rejected (the code is valid but
does not seem to be an app)` for a perfectly good notarized DMG — `spctl -a`
defaults to an app-execute assessment, which doesn't apply to a disk image.
Use the `-t open --context context:primary-signature` form above, or trust
the build script's own `stapler validate` output, which is authoritative.

### 2. Create the GitHub Release

Create or update a GitHub Release tagged `v<version>`.

Upload:

- the notarized DMG

The GitHub Release asset URL becomes the appcast download URL.

Important:

- when using `gh release create`, do not pass shell-escaped `\n` sequences in `--notes`
- prefer a real multiline notes file with `--notes-file` so the published release body does not end up with literal `\n\n`
- after publishing, spot-check the rendered GitHub release body once in the browser or via `gh release view`
- if the paired `gary4juce` release introduced licensing/compliance packaging changes, mention that briefly in the release body so the cross-repo pairing stays explicit

### 3. Prepare release notes

Add a versioned notes file under:

- `docs/updates/gary4local/release-notes/`

Recommended filename pattern:

- `v<version>.html`

Keep it concise and user-facing.

Companion-version rule:

- if the release notes mention the recommended `gary4juce` build, always link to the exact release tag URL, not the generic releases index
- if the paired `gary4juce` release changed its shipped licensing/compliance materials, note that briefly here as well

### 4. Sign the update archive for Sparkle

Using the Sparkle tooling on the release DMG (after stapling — signing before
stapling signs the wrong bytes, since stapling modifies the DMG):

```bash
sign_update /path/to/gary4local-v<version>-mac-arm64.dmg --account gary4local
```

Expected output includes:

- Sparkle EdDSA signature
- archive length

Those values are required in the appcast item.

If any of the Sparkle CLI tools (`sign_update`, `generate_appcast`,
`generate_keys`) were just rebuilt via `scripts/build_sparkle_tool.sh`, the
first invocation against the Keychain-stored private key will pop a macOS
"wants to use your confidential information" prompt that needs an interactive
click (Always Allow) — a freshly-built binary has a new code signature, so it
doesn't inherit the trust decision from a previous build. If a call to one of
these tools hangs with no output, this is almost certainly why; it is not
hanging on network or notarization work.

### 5. Update the appcast

Update either:

- `docs/updates/gary4local/stable.xml`
- or `docs/updates/gary4local/preview.xml`

Recommended way to build the new item — use `generate_appcast` for the fields
it computes correctly from the actual archive, then hand-add the fields it
doesn't manage, then re-sign with `sign_update`:

1. Stage the notarized DMG and a copy of the current appcast in one directory:
   ```bash
   mkdir -p /tmp/appcast-stage
   cp build-artifacts/gary4local-v<version>-mac-arm64.dmg /tmp/appcast-stage/
   cp docs/updates/gary4local/stable.xml /tmp/appcast-stage/
   ```
2. Run `generate_appcast` pointed at that directory with `--account gary4local`
   and `--download-url-prefix` set to the GitHub Release download URL for that
   tag. This computes `sparkle:version`, `sparkle:shortVersionString`,
   `sparkle:minimumSystemVersion`, `sparkle:hardwareRequirements` (correctly
   inferred as `arm64` from the binary — worth keeping, since it stops an
   Intel Mac from ever being offered an arm64-only build), and the enclosure's
   `edSignature`/`length`, by actually inspecting the DMG rather than by hand.
3. Copy the result into `docs/updates/gary4local/stable.xml`, then hand-add:
   - `<sparkle:releaseNotesLink>` pointing at the versioned HTML page.
     **Do not use `--full-release-notes-url`** — it produces
     `<sparkle:fullReleaseNotesLink>`, and the vendored Sparkle 2.9.1 client
     only parses `<sparkle:releaseNotesLink>` (see `SUAppcastItem.m`). Notes
     would silently never show in the update dialog.
   - `<description><![CDATA[...]]></description>` — a one-line fallback blurb,
     matching the style of existing items.
   - `sparkle:os="macos"` and `type="application/x-apple-diskimage"` on the
     `<enclosure>`, matching existing items (`generate_appcast` alone leaves
     these off).
4. Re-sign the file **with `sign_update <file> --account gary4local` run
   directly on the appcast XML**, not by re-running `generate_appcast`.
   `sign_update` detects the `sparkle-sign-warning` marker and updates just
   the trailing signature comment in place. Re-running `generate_appcast`
   instead will recompute every managed field from the current CLI flags and
   silently strip the `releaseNotesLink`/`description`/`os`/`type` fields you
   just hand-added back out — it is not a safe no-op re-sign for a file with
   manual edits.

The release item should point to:

- the GitHub Release DMG URL
- the versioned release-notes file
- the Sparkle signature and archive length

Important rule:

- the stable appcast URL must remain stable
- only its contents change from release to release

That is what lets installed apps "notice" the update without any per-release app reconfiguration.

Aside: the whole-file trailer signature (the `<!-- sparkle-signatures: -->`
comment) is currently cosmetic for this app — `SURequireSignedFeed` isn't set
in `gary4local/Info.plist`, so the client never validates it, only the
per-item `enclosure` `edSignature` matters at runtime. Still worth keeping
correct in case that ever gets turned on, but a slightly-stale trailer
signature is not a shipping blocker the way a wrong enclosure signature
would be.

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
