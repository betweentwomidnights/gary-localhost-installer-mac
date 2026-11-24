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
  # Several folders share a name (Caches, HTTPStorages and WebKit all use the
  # bundle id), and mv would nest one inside another, so number repeats.
  local dest="$HOME/.Trash/$2" n=2
  while [ -e "$dest" ] || [ -L "$dest" ]; do
    dest="$HOME/.Trash/$2 $n"
    n=$((n + 1))
  done
  mv "$1" "$dest" 2>/dev/null && return 0
  # Without Full Disk Access, Terminal may not write to ~/.Trash; ask Finder.
  # A big folder can take Finder longer than AppleScript's 2-minute default.
  osascript -e 'on run argv' -e 'with timeout of 3600 seconds' \
    -e 'tell application "Finder" to delete (POSIX file (item 1 of argv) as alias)' \
    -e 'end timeout' -e 'end run' "$1" >/dev/null
}

# This app saved SA3 LoRAs with DiT names like "transformer.layers.0...". The
# new gary4local's converter wants the upstream export's "model." prefix, and
# without it imports little or nothing of the LoRA. Rewrite only the header;
# the tensor data is copied as it is.
sa3_lora_for_new_app() {  # source.safetensors, destination.safetensors
  /usr/bin/perl -MJSON::PP -e '
    open(my $in, "<:raw", $ARGV[0]) or die "$ARGV[0]: $!\n";
    read($in, my $size, 8) == 8 or die "$ARGV[0]: not a safetensors file\n";
    my $n = unpack("Q<", $size);
    read($in, my $header, $n) == $n or die "$ARGV[0]: truncated header\n";
    my $json = JSON::PP->new->utf8->canonical;
    my $tensors = $json->decode($header);
    my %renamed;
    for my $key (keys %$tensors) {
      my $new = ($key eq "__metadata__" || $key =~ /^(model|conditioners)\./) ? $key : "model.$key";
      $renamed{$new} = $tensors->{$key};
    }
    $header = $json->encode(\%renamed);
    $header .= " " x ((8 - length($header) % 8) % 8);
    open(my $out, ">:raw", $ARGV[1]) or die "$ARGV[1]: $!\n";
    print $out pack("Q<", length $header), $header;
    local $/ = \(1 << 20);
    print $out $_ while <$in>;
    close($out) or die "$ARGV[1]: $!\n";
  ' "$1" "$2"
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
  "$HOME/Library/Caches/com.apple.python/Applications/gary4local.app"
  "$HOME/Library/HTTPStorages/$OLD_ID"
  "$HOME/Library/WebKit/$OLD_ID"
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

# The hugging face token stays in the keychain; that's for you to remove.
KEYCHAIN_SERVICE="com.betweentwomidnights.gary.localhost.controlcenter"
KEYCHAIN_ACCOUNT="stable-audio-hf-token"
if security find-generic-password -s "$KEYCHAIN_SERVICE" -a "$KEYCHAIN_ACCOUNT" >/dev/null 2>&1; then
  echo "your hugging face token from the old gary4local is still in the keychain. this script leaves it."
  echo "to remove it: security delete-generic-password -s $KEYCHAIN_SERVICE -a $KEYCHAIN_ACCOUNT"
  echo
fi

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
    if [ -d "$KEEP/sa3/mlx-cache" ]; then
      to_trash "$KEEP/sa3/mlx-cache" "sa3 mlx-cache (old gary4local)" ||
        echo "couldn't move $KEEP/sa3/mlx-cache to the Trash; you can delete it yourself." >&2
    fi
    echo "kept in $KEEP"
    if [ -d "$KEEP/sa3/loras" ]; then
      ready="$KEEP/sa3-for-new-gary4local"
      mkdir -p "$ready"
      for lora in "$KEEP/sa3/loras"/*.safetensors; do
        [ -f "$lora" ] || continue
        if sa3_lora_for_new_app "$lora" "$ready/$(basename "$lora")"; then
          echo "  ready for the new gary4local: $ready/$(basename "$lora")"
        else
          rm -f "$ready/$(basename "$lora")"
          echo "  couldn't prepare $(basename "$lora"); the original is still in $KEEP/sa3/loras" >&2
        fi
      done
      echo "add SA3 LoRAs to the new gary4local from $ready, not from sa3/loras."
    fi
  fi
fi

read -r -p "move everything listed above to the Trash? [y/N] " answer
if [ "$answer" != "y" ] && [ "$answer" != "Y" ]; then
  echo "nothing else was moved."
  exit 0
fi

stamp="$(date +%Y%m%d-%H%M%S)"
failed=()
for path in "${found[@]}"; do
  [ -e "$path" ] || continue
  to_trash "$path" "$(basename "$path") (old gary4local $stamp)" || failed+=("$path")
done
defaults delete "$OLD_ID" >/dev/null 2>&1 || true
if [ ${#failed[@]} -gt 0 ]; then
  echo "couldn't move these to the Trash. move them yourself in Finder:" >&2
  printf '  %s\n' "${failed[@]}" >&2
  exit 1
fi
echo "done. empty the Trash to get the disk space back."
