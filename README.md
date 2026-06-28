# gary4local (macOS)

This repository combines the local backend environments used by the Gary plugin stack on Apple Silicon macOS.

It pairs with [gary4juce](https://github.com/betweentwomidnights/gary4juce) and now includes a working macOS Swift app target (`gary4local`) that manages local services from a window + menu bar control center.

## Maintainer Note

this README is currently somewhat of a placeholder, and mostly maintained by robots. i am currently feeling pretty overwhelmed by the size of this project as a solo developer, and proper human documentation has not been top of mind on this particular repo. apologies.

## Current Note

- fair warning: i am not super happy with training and generation time for ace-step on the MLX path right now, but the machine we use for testing is a MacBook Air with 32GB of unified memory. my hope is that the experience is better on more powerful Apple Silicon MacBooks, and we will keep looking for sensible optimizations.
- Auto-updater is now implemented in `gary4local` using Sparkle 2.
- Recommended companion build for the current SA3 macOS release train: [gary4juce v4.0.2-mac](https://github.com/betweentwomidnights/gary4juce/releases/tag/v4.0.2-mac).
- `foundation-1` has now been successfully added to `gary4local`, including model download flow, prompt randomization support, text generation, and audio2audio on the macOS MLX path.
- Carey now includes the regular and XL `ace-step-v15-{base,sft,turbo}` model family toggle in `gary4local`, aligned with the Windows `gary-localhost-installer` flow.
- Stable Audio 3 now includes local LoRA training on macOS, built as an MLX path with parity to the [Dada Bots `underfit`](https://github.com/dada-bots/underfit) trainer while we work toward upstreaming the implementation.
- TODO: investigate surfacing runtime generation failures, especially Carey stack traces, through the same popup/reporting UX currently used for build failures.
- TODO: defer Stable Audio Hugging Face token lookup until the jerry service is actually used, or replace the current launch-time Keychain read with a less intrusive storage/auth flow.

## Planned Auto-Update Flow

`gary4local` is distributed outside the Mac App Store, so update hosting and installation will remain developer-managed.

The target production flow is:

- GitHub Releases hosts the notarized `gary4local` DMG asset.
- GitHub Pages hosts a stable Sparkle appcast URL that installed apps poll.
- Releasing a new version means:
  - build/sign/notarize the DMG
  - upload it to GitHub Releases
  - update the stable appcast on GitHub Pages

Design and maintainer docs live here:

- `docs/updates/README.md`
- `docs/releasing/SPARKLE_RELEASE.md`
- `docs/sa3/README.md`

## Monorepo Layout

- `ace-lego/`: Carey (ACE-Step) localhost backend + wrapper used by `gary4local`
- `audiocraft-mlx/`: MusicGen continuation localhost backend (MLX path)
- `melodyflow/`: MelodyFlow localhost backend (custom MPS-enabled AudioCraft fork)
- `stable-audio-tools/`: Stable Audio localhost backend (custom MPS-enabled fork)
- `control-center/`: earlier Swift package prototype + manifest/docs
- `gary4local/`: active macOS app target in Xcode

## Staging Repos

These staging repos are used to validate MLX integrations before promoting minimal runtime code into this repo:

- [ace-lego](https://github.com/betweentwomidnights/ace-lego) (staging ground for the `ace-lego/` folder vendored in this repository)
- [stable-audio-mlx](https://github.com/betweentwomidnights/stable-audio-mlx)
- [melodyflow-mlx](https://github.com/betweentwomidnights/melodyflow-mlx)

## Current Status

### Shipping Now

- `gary4local` is now a signed + notarized Apple Silicon macOS app with Sparkle-based in-app updates.
- The Swift control center and menu bar app handle per-service start/stop/restart/rebuild, bounded live logs, setup flows, and local model/runtime management.
- Service startup still clears conflicting listeners on the expected ports so restart behavior is a little more forgiving than a bare terminal workflow.
- The shipped service stack currently includes:
  - Carey (ACE-Step), including the regular / XL `base`, `sft`, and `turbo` model-family flow now used on the current Gary side.
  - `foundation-1` on the MLX path, including model download flow, prompt randomization, text generation, and audio2audio.
  - MelodyFlow with backend switching between `mps`, `mlx_native_torch_codec`, and `mlx_native_mlx_codec`.
  - Stable Audio 3 on the Apple Silicon MLX path, including gated model download flow, generation/continuation, LoRA registration, prompt-pool building, and user-facing loudness / DiT precision controls.
- Stable Audio 3 also now includes local LoRA training on macOS, built as an MLX path with parity to the [Dada Bots `underfit`](https://github.com/dada-bots/underfit) workflow. That includes dataset prompt editing, persistent job state, log tailing, cancellation, and automatic `.safetensors` adapter registration for `gary4juce`.
- The release flow is live now, not theoretical: Developer ID signing, notarization, GitHub Releases, Sparkle appcasts, and release-notes pages are all part of the shipping path.

### Near-Term Cleanup

- proper cross-repo README cleanup across `gary-localhost-installer`, `gary-localhost-installer-mac`, and `gary4juce`
- better surfacing of runtime failures, especially places where backend stack traces still hide in logs instead of showing up in the app
- continued UX cleanup around SA3-specific quirks like token setup, prompt-pool building, continuation modes, and training expectations
- continued trimming of bundle-only clutter while preserving the local runtimes we actually need to ship

### Runtime Packaging

- Release builds stage the backend runtime trees into app resources during build:
  - `Contents/Resources/runtime/ace-lego`
  - `Contents/Resources/runtime/audiocraft-mlx`
  - `Contents/Resources/runtime/melodyflow`
  - `Contents/Resources/runtime/sa3`
  - `Contents/Resources/runtime/stable-audio-tools`
  - `Contents/Resources/runtime/foundation`
- The production manifest is staged to:
  - `Contents/Resources/manifest/services.production.json`
- Mutable user data remains outside the app bundle:
  - venvs/caches/models: `~/Library/Application Support/GaryLocalhost/`
  - logs: `~/Library/Logs/GaryLocalhost/`
- Build helpers:
  - trusted Apple Silicon handoff build from an Intel Mac: `./scripts/build_gary4local_adhoc_dmg.sh`
  - notarized Developer ID DMG: `./scripts/build_gary4local_release_dmg.sh`
- Maintainer docs for the release/update path live in:
  - `docs/releasing/SPARKLE_RELEASE.md`
  - `docs/updates/README.md`

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
