#!/bin/bash
# Score one model the standard way, on the CPU (a training run usually has
# the GPU and the machine has 16 GB), into runs/scores/<label>.txt.
#
#   bash score.sh pieces  <label> <weights.pt> [evaluate.py options]
#                                                piece model with the labelled corners, one box per piece:
#                                                Fenit board B, board A, ChessReD test
#   bash score.sh corners <label> <weights.pt>   corner model: corner error and the
#                                                boards it gives with the labelled boxes
#
# Run it detached for a long one:  nohup bash score.sh ... > /dev/null 2>&1 &
cd "$(dirname "$0")" || exit 1
kind="$1"; label="$2"; weights="$3"
shift 3 2>/dev/null
# Anything after the weights goes to evaluate.py or corner_error.py as it is,
# e.g. --crop-board for a piece model trained on board crops.
if [ -z "$weights" ] || { [ "$kind" != pieces ] && [ "$kind" != corners ]; }; then
  echo "usage: bash score.sh pieces|corners <label> <weights.pt>"; exit 2
fi
# The photos scored are the merged dataset's 1280-px copies (--copies), not
# the full-size originals: ultralytics shrinks a 12 MP photo to 640 without
# smoothing, and a model trained on the copies then reads ChessReD test at
# 62% of boards instead of 96% (pieces_s_norf, 2026-11-13). The app draws a
# photo into at most 1280 px before the models see it, so the copies are the
# closer test; the final word is the exported model through app_check.
POS="--positions datasets/merged/positions.json --copies"
# One box per piece, as the exported piece model gives (its NMS runs across
# classes). Without it a pawn that also scores as a bishop comes back twice,
# the second box lands on a neighbouring square, and a model looks far worse
# than it is in the app.
ONE="--one-piece-per-spot 0.7"
mkdir -p runs/scores
out="runs/scores/$label.txt"
if [ "$kind" = pieces ]; then
  { echo "=== $label, labelled corners, fenit B $*"
    ./venv/bin/python evaluate.py --pieces-model "$weights" --gt-corners $POS --source fenit --group B --device cpu $ONE "$@"
    echo "=== $label, labelled corners, fenit A (all)"
    ./venv/bin/python evaluate.py --pieces-model "$weights" --gt-corners $POS --source fenit --group A --split all --device cpu $ONE "$@"
    echo "=== $label, labelled corners, chessred test"
    ./venv/bin/python evaluate.py --pieces-model "$weights" --gt-corners $POS --source chessred2k --split test --device cpu $ONE "$@"
  } > "$out" 2>&1
else
  { echo "=== $label fenit"
    ./venv/bin/python corner_error.py --corners-model "$weights" --source fenit --device cpu
    echo "=== $label chessred test"
    ./venv/bin/python corner_error.py --corners-model "$weights" --source chessred2k --split test --device cpu
  } > "$out" 2>&1
fi
grep -E "^===|per-square|exact|photos|corner order|corner error" "$out"
