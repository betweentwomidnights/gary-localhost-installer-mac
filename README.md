# Gary Localhost Installer (macOS)

This repository combines the local backend environments used by the Gary plugin stack on Apple Silicon macOS.

It is intended to pair with [gary4juce](https://github.com/betweentwomidnights/gary4juce), and will be the base for a Swift menu-bar control center + installer workflow.

## Monorepo Layout

- `audiocraft-mlx/`: MusicGen continuation localhost backend (MLX path)
- `melodyflow/`: MelodyFlow localhost backend (custom MPS-enabled AudioCraft fork)
- `stable-audio-tools/`: Stable Audio localhost backend (custom MPS-enabled fork)
- `control-center/`: Swift menu bar control-center prototype (manifest-driven service manager)

## Current Status

### Working Now

- All three Python environments can be rebuilt in-place from a fresh clone.
- Control-center runs from Xcode and shows service controls + live logs.
- Per-service `Start` / `Stop` / `Restart` / `Rebuild Env` and global `Rebuild All Environments` are wired.
- Startup/restart flow handles conflicting listeners on service ports more safely.
- Log viewer now uses a bounded tail window to keep UI/resource usage stable as logs grow.
- Stable Audio setup supports Hugging Face token save/read/delete in macOS Keychain.
- Stable Audio start is gated on token presence, with setup links and Step 2 screenshot hover preview.
- Audiocraft MLX + MelodyFlow have been rebuilt/tested from this repo and validated with the JUCE plugin flow.

### Still To Do

- Migrate the Swift package prototype to a full macOS App target in Xcode (app lifecycle, assets, release settings).
- Add proper app/menu bar icon assets (Gary logo).
- UI polish pass (copy style, naming, spacing, and service labels such as `gary` / `terry` / `jerry`).
- Define packaging/release flow for a one-click install experience (DMG + signing/notarization path).
- Keep refining first-run bootstrap UX and docs for end users.

## Rebuild Python Environments

From a fresh clone, rebuild all three service virtualenvs with:

```bash
cd /path/to/gary-localhost-installer-mac
./scripts/rebuild_venvs.sh
```

Optional flags:
- `--python /path/to/python3.11` to pin a specific interpreter (Python `3.11+` required)
- `--recreate` to delete and recreate existing `.venv` folders
- `--no-upgrade-tools` to skip `pip/setuptools/wheel` upgrades
