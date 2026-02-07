# MLX MusicGen Continuation (AudioCraft-compatible)

This folder vendors a minimal MLX implementation of MusicGen and adds **audio-prompt continuation**.

## Install
Use your existing venv and install from the runtime requirements in this repo:

```bash
cd /path/to/gary-localhost-installer-mac/audiocraft-mlx
source .venv/bin/activate
python -m pip install -r audiocraft/requirements.runtime.txt
```

## Run continuation
```bash
cd /path/to/gary-localhost-installer-mac/audiocraft-mlx/audiocraft
python -m mlx_continuation.continue \
  --model facebook/musicgen-small \
  --prompt /path/to/prompt.wav \
  --prompt-seconds 4 \
  --continuation-seconds 8 \
  --out-dir outputs
```

Outputs:
- `outputs/out_full.wav`
- `outputs/out_continuation_only.wav`

## Notes
- `--max-steps` overrides `--continuation-seconds` and expects **frames**, not delayed steps.
- Prompt audio is resampled to the model sample rate and channels before encoding.
- The prompt tokens are trimmed to the expected frame count computed from `frame_rate`.
- Debug helpers:
- `--dump-prompt-recon` writes `out_prompt_recon.wav` so you can sanity-check Encodec.
- `--prompt-encodec hf` uses Hugging Face's Encodec encoder (CPU) for comparison.
- `--prompt-encodec torch` is a backward-compatible alias for `hf`.
- `--no-trim-prompt` disables prompt token trimming.
- `--text-only` generates text-to-audio only and writes `out_text.wav`.
