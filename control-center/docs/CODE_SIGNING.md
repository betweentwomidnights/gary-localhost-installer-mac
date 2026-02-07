# Code Signing Checklist (macOS Control Center)

Use this when moving from Swift package prototype to a signed `.app` in Xcode.

## 1) Create App Target

- In Xcode, create a macOS App target (SwiftUI/AppKit hybrid is fine).
- Use bundle id like: `com.betweentwomidnights.gary.localhost.controlcenter`.

## 2) Team + Signing

- Set your Apple Developer Team.
- Signing Certificate:
  - Development for local testing.
  - Developer ID Application for outside-App-Store distribution.

## 3) Entitlements

Recommended starting point:
- App Sandbox: OFF (unless you deliberately harden for sandboxed distribution).
- Hardened Runtime: ON for release builds.

## 4) Runtime Paths

- Install backend files under:
  - `~/Library/Application Support/GaryLocalhost/`
- Generate manifest there and set:
  - `GARY_SERVICE_MANIFEST=/Users/<user>/Library/Application Support/GaryLocalhost/manifest/services.json`

## 5) Notarization

For release:
- Archive in Xcode.
- Export signed app.
- Notarize with your Developer ID credentials.
- Staple ticket to app (and DMG, if used).

## 6) Login Item (Optional)

If you want auto-launch at login:
- Add login item support with `SMAppService` in the app target.
