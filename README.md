# Gary Localhost Installer (macOS)

This repository combines the local backend environments used by the Gary plugin stack on Apple Silicon macOS.

It pairs with [gary4juce](https://github.com/betweentwomidnights/gary4juce) and now includes a working macOS Swift app target (`gary4local`) that manages local services from a window + menu bar control center.

## Monorepo Layout

- `audiocraft-mlx/`: MusicGen continuation localhost backend (MLX path)
- `melodyflow/`: MelodyFlow localhost backend (custom MPS-enabled AudioCraft fork)
- `stable-audio-tools/`: Stable Audio localhost backend (custom MPS-enabled fork)
- `control-center/`: earlier Swift package prototype + manifest/docs
- `gary4local/`: active macOS app target in Xcode

## Staging Repos

These staging repos are used to validate MLX integrations before promoting minimal runtime code into this repo:

- [stable-audio-mlx](https://github.com/betweentwomidnights/stable-audio-mlx)
- [melodyflow-mlx](https://github.com/betweentwomidnights/melodyflow-mlx)

## Current Status

### Working Now

- All three Python environments can be rebuilt in-place from a fresh clone.
- `gary4local` launches from Xcode as a macOS app with:
  - main control-center window
  - menu bar extra with per-service controls
  - live log tail display and rebuild controls
- Per-service `Start` / `Stop` / `Restart` / `Rebuild Env` and global `Rebuild All Environments` are wired.
- Service startup/restart handles conflicting listeners on service ports.
- Log viewer uses a bounded tail window to keep UI/resource usage stable.
- Stable Audio setup supports Hugging Face token save/read/delete in macOS Keychain.
- Stable Audio start is gated on token presence, with setup links and Step 2 hover screenshot reference.
- Stable Audio now includes optional MLX backend support (toggleable with MPS in `gary4local`), implemented from [stable-audio-mlx](https://github.com/betweentwomidnights/stable-audio-mlx).
- MelodyFlow now includes optional backend switching in `gary4local`:
  - `mps` (baseline torch path)
  - `mlx_native_torch_codec` (MLX flow + torch codec)
  - `mlx_native_mlx_codec` (end-to-end MLX flow + MLX codec)
  Implemented from the staging flow in [melodyflow-mlx](https://github.com/betweentwomidnights/melodyflow-mlx).
- Audiocraft MLX + MelodyFlow rebuild/run have been validated from this repo with the JUCE plugin flow.
- App icon asset set now uses the repository-provided Gary icon.

### Next Production Pass

- Menu bar branding:
  - replace default slider symbol with a custom Gary-tray icon
  - finalize dock/menu-bar icon consistency
- App menu polish:
  - define what stays in `About`, `Help`, and top-level menus
  - add project links (GitHub, Discord) under `Help`
  - remove/limit default menu items that are not useful for end users
- UI polish:
  - service naming/copy pass (`gary` / `terry` / `jerry`)
  - visual style alignment with gary4juce theme (while keeping native macOS clarity)
- Packaging/release:
  - bundle manifest/assets for distribution
  - signing + notarization
  - DMG installer workflow

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
