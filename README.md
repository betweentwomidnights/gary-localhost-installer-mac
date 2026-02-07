# Gary Localhost Installer (macOS)

This repository combines the local backend environments used by the Gary plugin stack on Apple Silicon macOS.

It is intended to pair with [gary4juce](https://github.com/betweentwomidnights/gary4juce), and will be the base for a Swift menu-bar control center + installer workflow.

## Monorepo Layout

- `audiocraft-mlx/`: MusicGen continuation localhost backend (MLX path)
- `melodyflow/`: MelodyFlow localhost backend (custom MPS-enabled AudioCraft fork)
- `stable-audio-tools/`: Stable Audio localhost backend (custom MPS-enabled fork)
- `control-center/`: Swift menu bar control-center prototype (manifest-driven service manager)

## Current Status

- All three environments were cleaned for production-oriented localhost runtime use.
- Runtime requirements and lock snapshots are included per environment.
- Next phase: Swift macOS control center (service lifecycle + logs UI) and packaging into a distributable app/DMG.

## Rebuild Python Environments

From a fresh clone, rebuild all three service virtualenvs with:

```bash
cd /path/to/gary-localhost-installer-mac
./scripts/rebuild_venvs.sh
```

Optional flags:
- `--recreate` to delete and recreate existing `.venv` folders
- `--python /path/to/python3.11` to pin a specific interpreter
