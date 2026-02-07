# Gary localhost backend (Stable Audio)

This folder is the cleaned Stable Audio localhost backend used by the JUCE plugin on macOS Apple Silicon.

## Runtime files
- `api.py`
- `model_loader_enhanced.py`
- `riff_manager.py`
- `stable_audio_tools/`
- `riffs/`

## Dependencies
- `requirements.txt` -> production install entrypoint
- `requirements.runtime.txt` -> pinned runtime dependencies
- `requirements.lock.txt` -> full lock snapshot from known-working venv

## Install
```bash
cd /Users/karenjessen/gary-localhost-installer-mac/stable-audio-tools
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run localhost backend
```bash
cd /Users/karenjessen/gary-localhost-installer-mac/stable-audio-tools
python api.py
```

## Notes
- Backend defaults to port `8004`.
- `sample_rf_guided` support is already integrated in `stable_audio_tools/inference/sampling.py`.
