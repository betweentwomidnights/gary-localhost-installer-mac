# gary4local (macOS)

this is the macOS control center and localhost stack for running the gary model
backends directly on Apple Silicon.

it pairs with [gary4juce](https://github.com/betweentwomidnights/gary4juce)
and now ships as a signed Swift app target, also called `gary4local`, that
handles service lifecycle, logs, setup flows, model downloads, and update
checks from a normal desktop app instead of a pile of terminal tabs.

## this app is retired

gary4local is one app now, on Windows, macOS and Linux:
[gary4local v0.7.6](https://github.com/betweentwomidnights/gary-localhost-installer/releases/tag/v0.7.6).
it runs every service on native C++ with Metal instead of Python and MLX, and
it's signed and notarized. this repo won't get any more updates.

### moving to the new gary4local

1. quit this app with Command-Q. that stops its services.
2. download the `.dmg` from the
   [v0.7.6 release](https://github.com/betweentwomidnights/gary-localhost-installer/releases/tag/v0.7.6)
   and drag gary4local into Applications. both apps are called gary4local, so
   Finder asks to replace this one. that's expected.
3. update gary4juce to
   [v5.0.7](https://github.com/betweentwomidnights/gary4juce/releases/tag/v5.0.7)
   at the same time.
4. in the new app, press `set up all runtimes`, then download models from each
   service's Models panel. models don't carry over: the C++ runtimes use their
   own model files, so plan for the downloads.

### jerry and gary4juce

gary4juce v5.0.7 is built for the new gary4local, but this app's last update,
v0.4.2, keeps local jerry working with it too: generate and smart loops, on the
base model and on finetunes. Sparkle offers v0.4.2 from inside the app.

- v0.4.1 and earlier don't work with v5.0.7's jerry. they were built for the
  older Mac gary4juce (v5.0.4-mac and before).
- finetunes still load here: pick one in jerry's model panel and press
  `use model`. gary4juce v5.0.7 lists whatever this app has loaded.

### cleaning up

the new gary4local keeps its files in its own folder and never touches this
app's, so it can't clean them up for you. almost everything this app
downloaded lives in `~/Library/Application Support/GaryLocalhost`: Python
environments, models, and your SA3 and carey LoRAs. that's usually most of the
space.

this script lists what's left, offers to keep your LoRAs, prompts and training
runs in `~/Documents/gary4local-mac-loras`, and moves the rest to the Trash:

```sh
curl -fsSLO https://raw.githubusercontent.com/betweentwomidnights/gary-localhost-installer-mac/main/scripts/remove-old-gary4local.sh
bash remove-old-gary4local.sh --dry-run   # only lists
bash remove-old-gary4local.sh
```

nothing is deleted until you empty the Trash. we haven't tested whether LoRAs
trained in this app load in the new one.

to do it by hand instead, quit this app and move these to the Trash:

- `~/Library/Application Support/GaryLocalhost`
- `~/Library/Application Support/gary4local` (this app's copy of `uv`)
- `~/Library/Logs/GaryLocalhost`
- `~/Library/Caches/thecollabagepatch.gary4local`
- `~/Library/Caches/com.apple.python/Applications/gary4local.app`
- `~/Library/HTTPStorages/thecollabagepatch.gary4local`
- `~/Library/WebKit/thecollabagepatch.gary4local`
- `~/Library/Saved Application State/thecollabagepatch.gary4local.savedState`
- `~/Library/Preferences/thecollabagepatch.gary4local.plist`

if you gave this app a hugging face token, it's still in your keychain. the
script leaves it alone. to remove it, open Keychain Access, search for
`gary.localhost.controlcenter`, and delete `stable-audio-hf-token`, or run:

```sh
security delete-generic-password -s com.betweentwomidnights.gary.localhost.controlcenter -a stable-audio-hf-token
```

don't touch `~/Library/Application Support/com.betweentwomidnights.gary4local`.
that's the new app.

## version history

version history moved to [`CHANGELOG.md`](CHANGELOG.md) so this file can stay
focused on what gary4local is right now.

older Sparkle release notes live under
[`docs/updates/gary4local/release-notes/`](docs/updates/gary4local/release-notes/).

## what lives here

- `gary4local/`
  the active macOS Swift app target and Xcode project.
- `ace-lego/`
  the Carey / ACE-Step localhost backend, plus the wrapper flow used by the mac
  app.
- `audiocraft-mlx/`
  the MLX path used for local Audiocraft / MusicGen continuation work.
- `melodyflow/`
  the MelodyFlow localhost backend.
- `sa3/`
  the Stable Audio 3 localhost backend and mac training flow.
- `stable-audio-tools/`
  the vendored Stable Audio runtime pieces still used on this stack.
- `foundation/`
  the local `foundation-1` runtime path.
- `control-center/`
  an older Swift package prototype plus manifest and packaging docs that are
  still useful as reference material.

## what ships right now

- a signed + notarized Apple Silicon macOS app with Sparkle-based in-app
  updates.
- a local control center that can start, stop, restart, rebuild, and monitor
  the backend services without dropping straight into shell scripts.
- carey on the mac path, including the regular and XL
  `ace-step-v15-{base,sft,turbo}` model family flow.
- `foundation-1` on the MLX path, including model download flow, prompt
  randomization support, text generation, and audio2audio.
- MelodyFlow with backend switching between `mps`,
  `mlx_native_torch_codec`, and `mlx_native_mlx_codec`.
- Stable Audio 3 on Apple Silicon, including local LoRA training, prompt-pool
  building, continuation, adapter registration, and the related UI inside
  `gary4local`.
- Yuey through its native universal Metal/CPU runtime, with explicit GGUF
  downloads and unload-after-generation behavior.

the main localhost endpoints in the production manifest are:

- `audiocraft-mlx`: `http://127.0.0.1:8000`
- `melodyflow`: `http://127.0.0.1:8002`
- `carey`: `http://127.0.0.1:8003`
- `sa3`: `http://127.0.0.1:8006`
- `yuey`: `http://127.0.0.1:8007`
- `foundation-1`: `http://127.0.0.1:8015`

## repo layout notes

- development runs directly from this repo.
- release builds stage the runtime trees into the app bundle under
  `Contents/Resources/runtime/`.
- mutable runtime data stays outside the app bundle.
- app support, venvs, models, and caches live under
  `~/Library/Application Support/GaryLocalhost/`.
- logs live under `~/Library/Logs/GaryLocalhost/`.
- the production service manifest lives at
  `control-center/manifest/services.production.json` and gets staged into the
  app resources at build time.

## rough edges

- i am still not especially happy with ACE-Step training and generation speed
  on the MLX path. most of the testing here has been on a MacBook Air with
  `32 GB` of unified memory, so some of this may just be "don't expect miracles
  from the air."
- some runtime failures, especially Carey-side failures, still explain
  themselves better in logs than they do in the UI.
- the SA3 auth and token flow is better than it was, but it still has enough
  weirdness that i don't consider that part "done."
- full-track training is hidden for now. MLX wants far more memory for it than
  the windows build does, and we want to know why first. short crops still train
  LoRAs that generate several minutes fine.
- SA3 converts pytorch weights to MLX at runtime instead of using stability's
  hosted MLX models. they're byte-identical, so it's an install-size win waiting
  on its own release.

## auto-updater

this repo now uses Sparkle 2.

the production flow is:

- GitHub Releases hosts the notarized `gary4local` DMG.
- GitHub Pages hosts the stable Sparkle appcast.
- a new mac release becomes visible to installed apps when the stable appcast
  is updated.

maintainer docs for that live in:

- [`docs/releasing/SPARKLE_RELEASE.md`](docs/releasing/SPARKLE_RELEASE.md)
- [`docs/updates/README.md`](docs/updates/README.md)

## useful commands

rebuild all Python environments from a fresh clone:

```bash
./scripts/rebuild_venvs.sh
```

build an ad hoc Apple Silicon DMG for trusted handoff testing:

```bash
./scripts/build_gary4local_adhoc_dmg.sh
```

build the signed + notarized release DMG:

```bash
./scripts/build_gary4local_release_dmg.sh
```

## staging repos

these repos are where some of the MLX and localhost-specific work gets proven
out before smaller runtime slices get promoted back into this one:

- [ace-lego](https://github.com/betweentwomidnights/ace-lego)
- [stable-audio-mlx](https://github.com/betweentwomidnights/stable-audio-mlx)
- [melodyflow-mlx](https://github.com/betweentwomidnights/melodyflow-mlx)

## related repos

- [gary4juce](https://github.com/betweentwomidnights/gary4juce)
- [gary-localhost-installer](https://github.com/betweentwomidnights/gary-localhost-installer)
- [gary-lora-examples](https://github.com/betweentwomidnights/gary-lora-examples)
