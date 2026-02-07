# Gary localhost backend (MLX)

This folder is the cleaned runtime backend used by the JUCE plugin on macOS Apple Silicon.

## Runtime files
- `g4l_localhost.py`
- `g4laudio_mlx.py`
- `g4l_models.py`
- `mlx_continuation/`

## Dependencies
- `requirements.txt` -> production install entrypoint
- `requirements.runtime.txt` -> pinned runtime dependencies
- `requirements.lock.txt` -> full lock snapshot from known-working venv

## Install
```bash
cd /Users/karenjessen/gary-localhost-installer-mac/audiocraft-mlx
source .venv/bin/activate
python -m pip install -r audiocraft/requirements.txt
```

## Run localhost backend
```bash
cd /Users/karenjessen/gary-localhost-installer-mac/audiocraft-mlx/audiocraft
python g4l_localhost.py
```

## Notes
- Quantization modes are handled in `g4laudio_mlx.py`.
- Optional Redis support is auto-detected; the backend falls back to in-memory sessions if Redis is unavailable.
