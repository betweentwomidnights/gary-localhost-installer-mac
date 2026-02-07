# Gary localhost backend (MelodyFlow)

This folder is the cleaned MelodyFlow localhost backend used by the JUCE plugin on macOS Apple Silicon.

## Runtime files
- `localhost_melodyflow.py`
- `variations.py`
- `audiocraft/`

## Dependencies
- `requirements.txt` -> production install entrypoint
- `requirements.runtime.txt` -> pinned runtime dependencies
- `requirements.lock.txt` -> full lock snapshot from known-working venv

## Install
```bash
cd /Users/karenjessen/gary-localhost-installer-mac/melodyflow
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run localhost backend
```bash
cd /Users/karenjessen/gary-localhost-installer-mac/melodyflow
python localhost_melodyflow.py
```

## Notes
- Backend listens on port `8002`.
- Model is loaded via local `audiocraft.models.MelodyFlow` and prefers MPS on Apple Silicon.
- Progress snapshots are written to a shared temp directory for relay into the control center polling flow.
