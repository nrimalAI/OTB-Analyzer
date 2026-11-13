#!/bin/bash
# Score one model the standard way, on the CPU (a training run usually has
# the GPU and the machine has 16 GB), into runs/scores/<label>.txt.
#
#   bash score.sh pieces  <label> <weights.pt>   piece model with the labelled corners:
#                                                Fenit board B, board A, ChessReD test
#   bash score.sh corners <label> <weights.pt>   corner model: corner error and the
#                                                boards it gives with the labelled boxes
#
# Run it detached for a long one:  nohup bash score.sh ... > /dev/null 2>&1 &
cd "$(dirname "$0")" || exit 1
kind="$1"; label="$2"; weights="$3"
if [ -z "$weights" ] || { [ "$kind" != pieces ] && [ "$kind" != corners ]; }; then
  echo "usage: bash score.sh pieces|corners <label> <weights.pt>"; exit 2
fi
POS="--positions datasets/merged/positions.json --source fenit"
CR="--annotations datasets/chessred2k/annotations.json --images datasets/chessred2k/images"
mkdir -p runs/scores
out="runs/scores/$label.txt"
if [ "$kind" = pieces ]; then
  { echo "=== $label, labelled corners, fenit B"
    ./venv/bin/python evaluate.py --pieces-model "$weights" --gt-corners $POS --group B --device cpu
    echo "=== $label, labelled corners, fenit A (all)"
    ./venv/bin/python evaluate.py --pieces-model "$weights" --gt-corners $POS --group A --split all --device cpu
    echo "=== $label, labelled corners, chessred test"
    ./venv/bin/python evaluate.py --pieces-model "$weights" --gt-corners $CR --split test --device cpu
  } > "$out" 2>&1
else
  { echo "=== $label fenit"
    ./venv/bin/python corner_error.py --corners-model "$weights" --source fenit --device cpu
    echo "=== $label chessred test"
    ./venv/bin/python corner_error.py --corners-model "$weights" --source chessred2k --split test --device cpu
  } > "$out" 2>&1
fi
grep -E "^===|per-square|exact|photos|corner order|corner error" "$out"
