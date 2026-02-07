# audiocraft-mlx cleanup notes (macOS localhost backend)

## Runtime keep set
- `g4l_localhost.py`
- `g4laudio_mlx.py`
- `g4l_models.py`
- `mlx_continuation/`
- `requirements.txt`
- `requirements.runtime.txt`
- `requirements.lock.txt`

## Dependency files
- `requirements.txt`: production install entrypoint (includes `requirements.runtime.txt`)
- `requirements.runtime.txt`: minimal pinned runtime dependencies for backend usage
- `requirements.lock.txt`: full `pip freeze` snapshot from known-working venv (2026-02-07)

## Notes
- `mlx_continuation/continue.py` no longer imports `audiocraft.models.encodec`; `--prompt-encodec torch` is now a backward-compatible alias to the HF Encodec path.
- Redis remains optional in `g4l_localhost.py` and falls back to in-memory session storage if not installed.
