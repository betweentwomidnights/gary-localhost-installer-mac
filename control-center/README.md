# GaryControlCenter (Swift)

Prototype macOS menu-bar control center for managing local Gary backend services.

## What It Currently Does

- Loads service config from `manifest/services.dev.json`
- Starts/stops/restarts each backend process
- Rebuilds each service Python environment from manifest-defined bootstrap config
- Guides Stable Audio Hugging Face access setup and stores token in macOS Keychain
- Polls each service `health_check` URL
- Streams process `stdout/stderr` into per-service log files
- Shows logs in UI

## Development Build

```bash
cd /path/to/gary-localhost-installer-mac/control-center
swift build
swift run
```

If needed, point to a custom manifest:

```bash
GARY_SERVICE_MANIFEST=/absolute/path/to/services.json swift run
```

## Open In Xcode

Open this package folder in Xcode:

- `/path/to/gary-localhost-installer-mac/control-center`

Then run the `GaryControlCenter` executable target.

## Code Signing / Distribution Plan

For a distributable `.app`/DMG, the next step is to migrate this package code into a dedicated macOS App project in Xcode and set:

- Team
- Bundle Identifier
- Signing Certificate
- Hardened Runtime
- Notarization flow

Recommended process:

1. Create a new macOS App target in Xcode (`GaryControlCenter.app`).
2. Copy `Sources/` Swift files into that target.
3. Add manifest loading path for App Support directory.
4. Add archive/export pipeline with your Developer ID cert.
5. Notarize and staple before DMG distribution.

This package is the functional service-management core we will carry into that app target.

See also:
- `docs/CODE_SIGNING.md`
