#!/usr/bin/env bash
# Downloads the Stockfish 17 NNUE evaluation networks required by chesskit-engine.
# They are looked up in Bundle.main by name at engine start, so they must be
# bundled as app resources. ~70 MB total; gitignored, run this once after clone.
set -euo pipefail

DEST="$(cd "$(dirname "$0")/.." && pwd)/App/Resources/NNUE"
mkdir -p "$DEST"

NETS=(nn-1111cefa1111.nnue nn-37f18f62d772.nnue)

for net in "${NETS[@]}"; do
  if [[ -f "$DEST/$net" ]]; then
    echo "✓ $net already present"
    continue
  fi
  echo "Downloading $net ..."
  curl -fL --retry 3 -o "$DEST/$net" "https://tests.stockfishchess.org/api/nn/$net"
  echo "✓ $net"
done

ls -lh "$DEST"
