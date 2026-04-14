# Code Signing + Notarization (gary4local DMG)

Use this flow for outside-App-Store macOS distribution.

## Distribution Target

- Target: notarized DMG download.
- Not target (yet): Mac App Store sandbox distribution.

Updater design and maintainer release-flow docs:

- `docs/updates/README.md`
- `docs/releasing/SPARKLE_RELEASE.md`

## Certificates You Need

For a notarized DMG, you only need:

- `Developer ID Application` (required, signs `.app` and can sign `.dmg`)
- `Developer ID Installer` (only needed for signed `.pkg`, optional for DMG-only flow)

Provisioning profiles are generally **not required** for Developer ID DMG distribution.
Profiles are mainly relevant for App Store/TestFlight/iOS style workflows or specific entitlements/capabilities.

## Xcode Project Settings (gary4local)

- Team: `P8L4LGS728`
- Signing: Automatic is fine for local release builds.
- Release config:
  - `ENABLE_HARDENED_RUNTIME = YES`
  - `ENABLE_APP_SANDBOX = NO` (DMG target)

## Runtime Payload Layout

The app stages backend source into app resources at build time:

- `Contents/Resources/runtime/audiocraft-mlx`
- `Contents/Resources/runtime/melodyflow`
- `Contents/Resources/runtime/stable-audio-tools`
- `Contents/Resources/runtime/foundation`
- `Contents/Resources/manifest/services.production.json`

Virtualenvs and caches remain user-local:

- `~/Library/Application Support/GaryLocalhost/venvs`
- `~/Library/Application Support/GaryLocalhost/cache`
- `~/Library/Logs/GaryLocalhost`

## Preflight Checks

Verify signing identities exist in local keychain:

```bash
security find-identity -v -p codesigning
```

You should see at least one `Developer ID Application: <Name> (<TeamID>)` identity.

For this project, the expected release identity is:

```bash
Developer ID Application: Kevin Griffing (P8L4LGS728)
```

## Build + Sign

From Xcode:

1. Select `gary4local` target, `Release` configuration.
2. Product -> Archive.
3. Distribute App -> Copy App (or direct export).

Or CLI:

```bash
xcodebuild -project gary4local/gary4local.xcodeproj \
  -scheme gary4local \
  -configuration Release \
  -destination 'platform=macOS' \
  build
```

For a repeatable Developer ID DMG from the terminal:

```bash
cd /path/to/gary-localhost-installer-mac
./scripts/build_gary4local_release_dmg.sh
```

Defaults:

- identity: `Developer ID Application: Kevin Griffing (P8L4LGS728)`
- notary profile: `gary-profile`
- output: `build-artifacts/gary4local-v<MARKETING_VERSION>-mac-arm64.dmg`

Useful flags:

```bash
./scripts/build_gary4local_release_dmg.sh --skip-notarize
./scripts/build_gary4local_release_dmg.sh --skip-layout
./scripts/build_gary4local_release_dmg.sh --output-name gary4local.dmg
```

## Trusted Test Build From Intel Mac

This is the quickest handoff flow for a trusted Apple Silicon tester when notarization is out of scope.

It produces:

- an `arm64` app binary
- an ad hoc signed app bundle
- a compressed DMG suitable for direct handoff

Run:

```bash
cd /path/to/gary-localhost-installer-mac
./scripts/build_gary4local_adhoc_dmg.sh
```

Output:

- `build-artifacts/gary4local-arm64-adhoc.dmg`

What the script does:

1. Archives `gary4local` in `Release` with `ARCHS=arm64` and `CODE_SIGNING_ALLOWED=NO`.
2. Verifies the archived app binary is actually `arm64`.
3. Copies the app into a DMG staging folder and adds an `Applications` symlink.
4. Applies an ad hoc signature with `codesign --force --deep -s -`.
5. Builds a compressed DMG with `hdiutil create`.

Notes:

- This flow works from an Intel Mac even if the host cannot run the built app locally.
- The app still targets macOS `14.6+`.
- This is not a notarized distribution artifact.
- Gatekeeper may require `Control-click -> Open` on first launch.
- If quarantine still blocks launch on the tester machine:

```bash
xattr -dr com.apple.quarantine /Applications/gary4local.app
```

## Developer ID Upgrade Path

When you do want a proper release artifact, the `gary4juce` DMG script is the right pattern to follow:

1. Sign the `.app` with `Developer ID Application` using `--options runtime --timestamp`.
2. Build the DMG.
3. Sign the DMG with the same identity.
4. Submit with `xcrun notarytool submit ... --wait`.
5. Staple with `xcrun stapler staple`.

Reference implementation:

- `/Users/klgriffing/Documents/gary4juce/build-dmg.sh`
- `scripts/build_gary4local_release_dmg.sh`

## Notarization (notarytool)

1. Store credentials once:

```bash
xcrun notarytool store-credentials "gary4local-notary" \
  --apple-id "<apple-id-email>" \
  --team-id "P8L4LGS728" \
  --password "<app-specific-password>"
```

2. Zip or DMG the signed app, then submit:

```bash
xcrun notarytool submit /path/to/gary4local.dmg \
  --keychain-profile "gary4local-notary" \
  --wait
```

3. Staple ticket:

```bash
xcrun stapler staple /path/to/gary4local.app
xcrun stapler staple /path/to/gary4local.dmg
```

4. Validate:

```bash
spctl -a -vv /path/to/gary4local.app
```

## First-Run Runtime Notes

- On first use, users run `build all environments` to create service venvs.
- `uv` bootstraps automatically if missing.
- `python3.11` must exist on user machine (or be supplied by your installer/bootstrap in a future pass).

## Recommended Next Hardening Pass

- Add explicit python bootstrap/install guidance for machines missing `python3.11`.
- Move toward wheelhouse/bundled runtime strategy for deterministic dependency installs.
