"""How a corner model fails, on photos whose corners and boxes are labelled.

    python corner_error.py --corners-model runs/pose/corners/weights/best.pt \
                           --source fenit --group B
    python corner_error.py --corners-model baseline/corners_yolo11n_chessred2k.pt \
                           --source chessred2k --split test

evaluate.py only says how many squares end up wrong. This separates the two
ways the corners can be wrong:
  location  how far each predicted corner is from a labelled one, in squares
  order     whether the corner called a1 is the labelled a1, or the board is
            found but turned by 90 or 180 degrees
and scores the boards with the labelled boxes, once with the corners as
predicted and once with the best of the four rotations, to show how much a
right order alone would recover.

Baseline (yolo11n, ChessReD2K only), 2026-11: ChessReD test median 0.31
squares, order always right. Fenit: median 2.3-2.7 squares, over one square
in 95% of photos: on a new board it does not find the board at all.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from chess_pipeline import (
    FEN_LETTER, assign_squares, fen_placement_to_board, homography, project,
)
from evaluate import merged_labels

HERE = Path(__file__).resolve().parent
BOARD_CORNERS = np.array([[0, 0], [0, 8], [8, 8], [8, 0]], float)  # a1 a8 h8 h1


def squares_wrong(corners, boxes, truth, pull):
    predicted = {sq: FEN_LETTER[cls]
                 for sq, (cls, _) in assign_squares(homography(corners), boxes, pull).items()}
    return sum(truth.get((f, r)) != predicted.get((f, r))
               for f in range(8) for r in range(8))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corners-model", required=True)
    ap.add_argument("--positions", default=str(HERE / "datasets/merged/positions.json"))
    ap.add_argument("--source", help="only this source, e.g. fenit")
    ap.add_argument("--group", help="only this group, e.g. Fenit board A or B")
    ap.add_argument("--split", default="all", choices=["train", "val", "test", "all"])
    ap.add_argument("--pull", type=float, default=0.30)
    ap.add_argument("--device", default="cpu",
                    help="cpu by default, so it can run next to a training job")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.corners_model)
    stats = defaultdict(lambda: defaultdict(list))
    for row in json.loads(Path(args.positions).read_text()):
        if args.source and row["source"] != args.source:
            continue
        if args.group and row.get("group") != args.group:
            continue
        if args.split != "all" and row["split"] != args.split:
            continue
        src = Path(row.get("original") or row["image"])
        corners, boxes = merged_labels(row, src)
        if not corners or not boxes:
            continue
        s = stats[row["source"] + (f" {row['group']}" if row.get("group") else "")]
        result = model(str(src), imgsz=640, verbose=False, device=args.device)[0]
        if result.keypoints is None or len(result.keypoints) == 0:
            s["not found"].append(1)
            continue
        best = int(result.boxes.conf.argmax())
        predicted = np.array(result.keypoints.xy[best].tolist())
        # Predicted corners in board coordinates, via the labelled corners.
        to_board = homography(corners)
        on_board = np.array([project(to_board, p) for p in predicted])
        # Rotation k compares predicted corner i with labelled corner (i + k) % 4.
        errors = [np.linalg.norm(on_board - np.roll(BOARD_CORNERS, -k, axis=0), axis=1)
                  for k in range(4)]
        k = int(np.argmin([e.mean() for e in errors]))
        s["rotation"].append(k)
        s["error"].append(float(errors[k].mean()))
        truth = fen_placement_to_board(row["placement"])
        s["as predicted"].append(squares_wrong(predicted, boxes, truth, args.pull))
        s["best rotation"].append(min(
            squares_wrong(np.roll(predicted, j, axis=0), boxes, truth, args.pull)
            for j in range(4)))

    for name in sorted(stats):
        s = stats[name]
        n, missed = len(s["rotation"]), len(s["not found"])
        print(f"{name}: {n + missed} photos, board not found in {missed}")
        if not n:
            continue
        turns, e = Counter(s["rotation"]), np.array(s["error"])
        print(f"  corner order: right {turns[0]}, turned 90 degrees {turns[1] + turns[3]}, "
              f"turned 180 degrees {turns[2]}")
        print(f"  corner error with the order matched, in squares: median {np.median(e):.2f}, "
              f"90th percentile {np.percentile(e, 90):.2f}; within 0.25: {np.mean(e < 0.25):.0%}, "
              f"within 0.5: {np.mean(e < 0.5):.0%}, over 1: {np.mean(e > 1):.0%}")
        for key in ("as predicted", "best rotation"):
            wrong = np.array(s[key])
            print(f"  with the labelled boxes, corners {key}: "
                  f"{1 - wrong.sum() / (64 * n):.2%} per-square, {np.mean(wrong == 0):.1%} exact")


if __name__ == "__main__":
    main()
