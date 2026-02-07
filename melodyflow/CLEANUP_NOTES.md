# melodyflow cleanup notes (macOS localhost backend)

## Runtime keep set
- `localhost_melodyflow.py`
- `variations.py`
- `audiocraft/` (custom MPS-enabled fork used by localhost runtime)
- `requirements.txt`
- `requirements.runtime.txt`
- `requirements.lock.txt`

## Dependency files
- `requirements.txt`: production install entrypoint
- `requirements.runtime.txt`: pinned runtime deps for localhost backend
- `requirements.lock.txt`: full `pip freeze` snapshot from known-working venv (2026-02-07)

## Notes
- Redis is optional in `localhost_melodyflow.py`; if unavailable, progress snapshots are still written to shared temp files for relay by `g4l_localhost.py`.
