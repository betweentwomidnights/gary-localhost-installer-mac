# stable-audio-tools cleanup notes (macOS localhost backend)

## Runtime keep set
- `api.py`
- `model_loader_enhanced.py`
- `riff_manager.py`
- `stable_audio_tools/` (custom MPS-enabled modifications)
- `riffs/`
- `requirements.txt`
- `requirements.runtime.txt`
- `requirements.lock.txt`

## Dependency files
- `requirements.txt`: production install entrypoint
- `requirements.runtime.txt`: pinned runtime deps for localhost backend
- `requirements.lock.txt`: full lock snapshot from known-working venv (2026-02-07)

## Notes
- `sample_rf_guided` patch is already present directly in `stable_audio_tools/inference/sampling.py`.
- `patch_generation.py` is no longer required for runtime once that in-tree patch is present.
