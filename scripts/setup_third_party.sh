#!/usr/bin/env bash
# Clone the three reference repos into third_party/ at the pinned commits
# used for docs/code_analysis.md. Idempotent: skips repos already cloned,
# and re-pins them to the recorded commit if they exist.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TP="$ROOT/third_party"
mkdir -p "$TP"

# name | url | branch | pinned commit
REPOS=(
  "MAD|https://github.com/top-yun/MAD|main|e6f5072fcc81cfa8946b427b903f840ea8437646"
  "AVCD|https://github.com/kaistmm/AVCD|main|92aeaf1656e6ec7760e3cd3be4a287c88a30d37c"
  "VideoLLaMA2|https://github.com/DAMO-NLP-SG/VideoLLaMA2|audio_visual|8cedfbda23e779246112cb963416b95415448a91"
)

for entry in "${REPOS[@]}"; do
  IFS='|' read -r name url branch commit <<< "$entry"
  dest="$TP/$name"
  if [ ! -d "$dest/.git" ]; then
    echo "[setup_third_party] cloning $name ($branch)..."
    git clone --branch "$branch" "$url" "$dest"
  else
    echo "[setup_third_party] $name already cloned."
  fi
  current="$(git -C "$dest" rev-parse HEAD)"
  if [ "$current" != "$commit" ]; then
    echo "[setup_third_party] pinning $name to $commit (was $current)"
    git -C "$dest" fetch origin "$branch" || git -C "$dest" fetch --unshallow origin "$branch" || true
    git -C "$dest" checkout "$commit" \
      || { echo "ERROR: cannot checkout pinned commit $commit for $name" >&2; exit 1; }
  fi
  echo "[setup_third_party] $name @ $(git -C "$dest" rev-parse --short HEAD) OK"
done

echo "[setup_third_party] done."
