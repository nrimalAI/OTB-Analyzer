#!/bin/bash
# Scores a corner model and a piece model on every part of the second held-out
# test set (datasets/heldout2, see README.md), on the CPU so that it can run
# next to a training job. Score finished models only: nothing is to be chosen
# or tuned on these photos.
#
#   ./score_heldout2.sh <tag> <corners model> <pieces model> [more evaluate.py arguments]
#   ./score_heldout2.sh current baseline/corners_yolo11s_exp3.pt \
#       baseline/pieces_yolo11n_chessred2k.pt --one-piece-per-spot 0.7
#
# Writes runs/scores/heldout2_<tag>.txt, one evaluate.py report per part.
cd "$(dirname "$0")" || exit 1
if [ $# -lt 3 ]; then sed -n '2,11p' "$0"; exit 1; fi
tag=$1; corners=$2; pieces=$3; shift 3
PY=${PY:-./venv/bin/python}
mkdir -p runs/scores
out=runs/scores/heldout2_$tag.txt
: > "$out"
for part in "chessq" "ylichess --group wood180" "ylichess --group wood180-layout" \
            "chesscog_real" "samryan" "chesscv_games"; do
  echo "== $part" >> "$out"
  # $part is split on purpose: a source and, for some, its --group.
  OMP_NUM_THREADS=4 nice -n 10 "$PY" evaluate.py --corners-model "$corners" --pieces-model "$pieces" \
      --positions datasets/heldout2/positions.json --split test --source $part \
      --device cpu "$@" >> "$out" 2>&1
done
echo "== done" >> "$out"
echo "wrote $out"
