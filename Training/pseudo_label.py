"""Turn the ~8,700 ChessReD photos that have no bounding boxes into piece
training data, using the current models and the known positions as a check.

ChessReD labels every photo with the position (which piece is on which
square) but only the 2K subset with boxes. For each remaining photo:

  1. run the corner and piece models, assign detections to squares
  2. accept only if the occupied squares match the known position exactly
     and at least --min-class-agreement of the piece classes are right
  3. write the boxes with the KNOWN classes (so a detected-but-misclassified
     piece becomes a corrected label, which is the useful part)

Only photos from ChessReD's train split are used, and never ones in the 2K
subset, so val/test stay clean.

Corner labels are NOT written by default: they would be the corner model's
own predictions, and corner precision is exactly what needs to improve, so
training on them teaches nothing new. --with-corners is there for experiments.

    python build_dataset.py fetch chessred_full        # 24.5 GB
    python pseudo_label.py --corners-model ../runs/pose/corners/weights/best.pt \
                           --pieces-model ../runs/detect/pieces/weights/best.pt
    python build_dataset.py merge                      # picks up chessred_pseudo
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from ultralytics import YOLO

from build_dataset import CLASS_ID, LETTER_TO_CLASS, find_image
from chess_pipeline import FEN_LETTER, PIECE_CLASSES, Detection, assign_detections, homography
from evaluate import detect_corners, ground_truth_board
from prepare_data import image_index, link, normalize_category

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE / "datasets/chessred"),
                    help="folder holding annotations.json and the images")
    ap.add_argument("--corners-model", required=True)
    ap.add_argument("--pieces-model", required=True)
    ap.add_argument("--out", default=str(HERE / "datasets/chessred_pseudo"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--pull", type=float, default=0.30)
    ap.add_argument("--min-class-agreement", type=float, default=0.9)
    ap.add_argument("--with-corners", action="store_true")
    ap.add_argument("--allow-labeled", action="store_true",
                    help="also process the 2K subset (for checking this script)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out)
    data = json.loads((root / "annotations.json").read_text())
    categories = {c["id"]: normalize_category(c["name"]) for c in data["categories"]}
    imgs = image_index(data)
    splits = data["splits"]
    in_2k = {i for s in splits["chessred2k"].values() for i in s["image_ids"]}
    candidates = [i for i in splits["train"]["image_ids"]
                  if args.allow_labeled or i not in in_2k]

    by_image = defaultdict(list)
    for p in data["annotations"]["pieces"]:
        by_image[p["image_id"]].append(p)

    tasks = ["pieces"] + (["corners"] if args.with_corners else [])
    for task in tasks:
        (out / task / "images").mkdir(parents=True, exist_ok=True)
        (out / task / "labels").mkdir(parents=True, exist_ok=True)

    corner_model, piece_model = YOLO(args.corners_model), YOLO(args.pieces_model)
    tally = Counter()
    for image_id in sorted(candidates):
        stem = f"img{image_id:06d}"
        if (out / "pieces" / "labels" / f"{stem}.txt").exists():
            tally["already done"] += 1
            continue
        meta = imgs[image_id]
        src = find_image(root, meta["path"])
        if src is None:
            tally["image not on disk"] += 1
            continue
        truth = ground_truth_board(by_image[image_id], categories)
        if not truth:
            continue

        kpts = detect_corners(corner_model, src)
        if kpts is None:
            tally["rejected: board not found"] += 1
            continue
        result = piece_model(str(src), conf=args.conf, verbose=False)[0]
        detections = [
            Detection(PIECE_CLASSES[int(c)], float(conf), tuple(box))
            for box, c, conf in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(),
                                    result.boxes.conf.tolist())
        ] if result.boxes is not None else []
        placed = assign_detections(homography(kpts), detections, pull=args.pull)

        if set(placed) != set(truth):
            tally["rejected: occupied squares differ"] += 1
            continue
        right = sum(FEN_LETTER[placed[sq].cls] == truth[sq] for sq in truth)
        if right < args.min_class_agreement * len(truth):
            # Usually the board read upside down: same squares, wrong pieces.
            tally["rejected: piece classes differ"] += 1
            continue

        w, h = meta["width"], meta["height"]
        lines = []
        for sq, det in placed.items():
            x1, y1, x2, y2 = det.bbox
            lines.append(f"{CLASS_ID[LETTER_TO_CLASS[truth[sq]]]} "
                         f"{(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                         f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
        (out / "pieces" / "labels" / f"{stem}.txt").write_text("\n".join(lines))
        link(src, out / "pieces" / "images" / f"{stem}{src.suffix}")
        if args.with_corners:
            xs, ys = [p[0] / w for p in kpts], [p[1] / h for p in kpts]
            pad = 0.03
            x1, x2 = max(min(xs) - pad, 0), min(max(xs) + pad, 1)
            y1, y2 = max(min(ys) - pad, 0), min(max(ys) + pad, 1)
            (out / "corners" / "labels" / f"{stem}.txt").write_text(
                f"0 {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f} "
                + " ".join(f"{x:.6f} {y:.6f} 2" for x, y in zip(xs, ys)))
            link(src, out / "corners" / "images" / f"{stem}{src.suffix}")
        tally["accepted"] += 1
        tally["accepted with corrected classes"] += right < len(truth)

        if args.limit and tally["accepted"] >= args.limit:
            break

    for k, v in tally.items():
        print(f"{v:6}  {k}")
    print(f"labels in {out}")


if __name__ == "__main__":
    main()
