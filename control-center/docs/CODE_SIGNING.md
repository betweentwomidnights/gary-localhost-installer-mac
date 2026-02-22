# Code Signing + Notarization (gary4local DMG)

Use this flow for outside-App-Store macOS distribution.

## Distribution Target

- Target: notarized DMG download.
- Not target (yet): Mac App Store sandbox distribution.

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
