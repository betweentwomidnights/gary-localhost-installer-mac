#!/bin/bash
# Removes the files the old Python/MLX gary4local (thecollabagepatch.gary4local,
# this repo) left behind, after you've moved to the unified gary4local.
#
# Everything goes to the Trash, so nothing is gone until you empty it. The new
# gary4local (com.betweentwomidnights.gary4local) keeps its files elsewhere and
# is never touched.
#
#   bash remove-old-gary4local.sh            # list what would go, then ask
#   bash remove-old-gary4local.sh --dry-run  # only list
set -euo pipefail

OLD_ID="thecollabagepatch.gary4local"
SUPPORT="$HOME/Library/Application Support"
DATA="$SUPPORT/GaryLocalhost"
KEEP="$HOME/Documents/gary4local-mac-loras"
DRY_RUN=false
[ "${1:-}" = "--dry-run" ] && DRY_RUN=true

[ "$(uname -s)" = "Darwin" ] || { echo "this script is for macOS." >&2; exit 1; }

to_trash() {  # path, name in the Trash
  mv "$1" "$HOME/.Trash/$2" 2>/dev/null && return 0
  # Without Full Disk Access, Terminal may not write to ~/.Trash; ask Finder.
  osascript -e 'on run argv' -e 'tell application "Finder" to delete (POSIX file (item 1 of argv) as alias)' -e 'end run' "$1" >/dev/null
}

if lsappinfo list 2>/dev/null | grep -q "\"$OLD_ID\""; then
  echo "the old gary4local is running. quit it first (Command-Q), then run this again." >&2
  exit 1
fi
if pgrep -f "Application Support/GaryLocalhost" >/dev/null 2>&1; then
  echo "an old gary4local service is still running:" >&2
  pgrep -fl "Application Support/GaryLocalhost" >&2 || true
  echo "quit the old app, or stop those processes, then run this again." >&2
  exit 1
fi

candidates=(
  "$DATA"
  "$SUPPORT/gary4local"
  "$HOME/Library/Logs/GaryLocalhost"
  "$HOME/Library/Caches/$OLD_ID"
  "$HOME/Library/HTTPStorages/$OLD_ID"
  "$HOME/Library/Saved Application State/$OLD_ID.savedState"
  "$HOME/Library/Preferences/$OLD_ID.plist"
)
# Installing the new gary4local replaces /Applications/gary4local.app, so only
# remove that bundle if it's still the old app.
for app in "/Applications/gary4local.app" "$HOME/Applications/gary4local.app"; do
  if [ -d "$app" ] && [ "$(defaults read "$app/Contents/Info" CFBundleIdentifier 2>/dev/null)" = "$OLD_ID" ]; then
    candidates+=("$app")
  fi
done

found=()
for path in "${candidates[@]}"; do
  [ -e "$path" ] && found+=("$path")
done
if [ ${#found[@]} -eq 0 ]; then
  echo "nothing from the old gary4local is left. you're done."
  exit 0
fi

echo "the old gary4local left these behind:"
for path in "${found[@]}"; do
  printf '  %8s  %s\n' "$(du -sh "$path" 2>/dev/null | cut -f1)" "$path"
done
echo

loras=()
for dir in "$DATA/sa3" "$DATA/carey"; do
  [ -d "$dir" ] && loras+=("$dir")
done

if $DRY_RUN; then
  [ ${#loras[@]} -gt 0 ] && echo "your SA3/carey LoRAs, prompts and training runs are in: ${loras[*]}"
  echo "dry run: nothing was moved."
  exit 0
fi

if [ ${#loras[@]} -gt 0 ]; then
  read -r -p "keep your SA3/carey LoRAs, prompts and training runs in $KEEP? [Y/n] " answer
  if [ "${answer:-y}" != "n" ] && [ "${answer:-y}" != "N" ]; then
    [ -e "$KEEP" ] && { echo "$KEEP already exists. move it aside and run this again." >&2; exit 1; }
    mkdir -p "$KEEP"
    for dir in "${loras[@]}"; do
      mv "$dir" "$KEEP/"
    done
    # SA3's MLX cache is rebuilt from the checkpoint; it isn't worth keeping.
    [ -d "$KEEP/sa3/mlx-cache" ] && to_trash "$KEEP/sa3/mlx-cache" "sa3 mlx-cache (old gary4local)"
    echo "kept in $KEEP"
  fi
fi

read -r -p "move everything listed above to the Trash? [y/N] " answer
if [ "$answer" != "y" ] && [ "$answer" != "Y" ]; then
  echo "nothing else was moved."
  exit 0
fi

stamp="$(date +%Y%m%d-%H%M%S)"
for path in "${found[@]}"; do
  [ -e "$path" ] || continue
  to_trash "$path" "$(basename "$path") (old gary4local $stamp)"
done
defaults delete "$OLD_ID" >/dev/null 2>&1 || true
echo "done. empty the Trash to get the disk space back."
