"""End-to-end evaluation with product metrics.

Runs corner + piece models over a split of ChessReD, assembles each board via
chess_pipeline, and compares against the ground-truth placement derived from
the piece annotations (`chessboard_position` per piece).

Reports:
  - per-square accuracy (64 * n squares)
  - full-board exact-match rate  <- the number that matters to users
  - per-class confusion counts

Ground truth uses ALL annotated images in the split (algebraic positions exist
for every ChessReD image), so you can also evaluate on images that have no
bbox/corner labels — the models don't need them at inference time.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from ultralytics import YOLO

from chess_pipeline import (
    Detection, PIECE_CLASSES, FEN_LETTER,
    assign_squares, homography,
)
from prepare_data import image_index, normalize_category, official_splits


def ground_truth_board(piece_anns, categories):
    """{(file, rank): fen_letter} from chessboard_position annotations."""
    board = {}
    for p in piece_anns:
        pos = p.get("chessboard_position")
        if not pos:
            continue
        name = categories[p["category_id"]]
        if name not in FEN_LETTER:  # e.g. the dataset's "empty" category
            continue
        file = ord(pos[0]) - ord("a")
        rank = int(pos[1]) - 1
        board[(file, rank)] = FEN_LETTER[name]
    return board


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corners-model", required=True)
    ap.add_argument("--pieces-model", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--pull", type=float, default=0.30,
                    help="anchor pull toward board center (see chess_pipeline)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    data = json.loads(Path(args.annotations).read_text())
    categories = {c["id"]: normalize_category(c["name"]) for c in data["categories"]}
    imgs = image_index(data)
    split_by_id = official_splits(data)

    by_image = defaultdict(list)
    for p in data["annotations"]["pieces"]:
        by_image[p["image_id"]].append(p)

    corner_model = YOLO(args.corners_model)
    piece_model = YOLO(args.pieces_model)

    images_root = Path(args.images)
    total_squares = correct_squares = 0
    boards = exact = corner_failures = flip_suspects = 0
    confusion = Counter()

    items = sorted(by_image.items())
    for image_id, piece_anns in items:
        meta = imgs[image_id]
        if args.split != "all" and split_by_id.get(image_id) != args.split:
            continue
        src = images_root / meta["path"]
        if not src.exists():
            alt = images_root / Path(*Path(meta["path"]).parts[1:])
            if not alt.exists():
                continue
            src = alt

        gt = ground_truth_board(piece_anns, categories)
        if not gt:
            continue

        # 1. corners
        cr = corner_model(str(src), verbose=False)[0]
        if cr.keypoints is None or len(cr.keypoints) == 0:
            corner_failures += 1
            boards += 1
            total_squares += 64
            continue
        best = int(cr.boxes.conf.argmax()) if cr.boxes is not None and len(cr.boxes) else 0
        kpts = cr.keypoints.xy[best].tolist()  # [[x,y] * 4] in a1,a8,h8,h1 order
        H = homography(kpts)

        # 2. pieces
        pr = piece_model(str(src), conf=args.conf, verbose=False)[0]
        detections = []
        if pr.boxes is not None:
            for box, cls, conf in zip(
                    pr.boxes.xyxy.tolist(), pr.boxes.cls.tolist(), pr.boxes.conf.tolist()):
                detections.append(
                    Detection(PIECE_CLASSES[int(cls)], float(conf), tuple(box)))

        predicted = assign_squares(H, detections, pull=args.pull)
        pred_letters = {sq: FEN_LETTER[cls] for sq, (cls, _) in predicted.items()}

        # 3. score all 64 squares
        board_ok = True
        wrong_here = 0
        for f in range(8):
            for r in range(8):
                total_squares += 1
                want = gt.get((f, r))
                got = pred_letters.get((f, r))
                if want == got:
                    correct_squares += 1
                else:
                    board_ok = False
                    wrong_here += 1
                    confusion[(want or "·", got or "·")] += 1
        boards += 1
        exact += board_ok

        # Diagnostic: would the 180°-rotated prediction have matched better?
        # A high count here means the corner model confuses a1 with h8
        # (chessboards are 2-fold rotationally symmetric).
        if not board_ok:
            rotated = {(7 - f, 7 - r): v for (f, r), v in pred_letters.items()}
            rot_wrong = sum(
                1 for f in range(8) for r in range(8)
                if gt.get((f, r)) != rotated.get((f, r)))
            if rot_wrong < wrong_here / 2:
                flip_suspects += 1

        if args.limit and boards >= args.limit:
            break

    if boards == 0:
        print("No images evaluated — check --split / paths.")
        return

    print(f"boards evaluated:        {boards}")
    print(f"corner detection failed: {corner_failures}")
    print(f"per-square accuracy:     {correct_squares / total_squares:.4%}")
    print(f"full-board exact match:  {exact / boards:.2%}")
    print(f"180-degree flip suspects: {flip_suspects}")
    print("\ntop confusions (truth -> predicted):")
    for (want, got), n in confusion.most_common(15):
        print(f"  {want} -> {got}: {n}")


if __name__ == "__main__":
    main()
