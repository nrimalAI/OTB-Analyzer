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


def detect_corners(model, src, imgsz=640, two_pass=False):
    """Predicted (a1, a8, h8, h1) image points, or None.

    With `two_pass`, re-runs the model on a crop around the first-pass board —
    the board then fills most of the model input, sharpening keypoints ~3×
    without retraining. Mirrors what the app does with two Vision requests.
    """
    import cv2

    def run(img):
        r = model(img, imgsz=imgsz, verbose=False)[0]
        if r.keypoints is None or len(r.keypoints) == 0:
            return None
        best = int(r.boxes.conf.argmax()) if r.boxes is not None and len(r.boxes) else 0
        return r.keypoints.xy[best].tolist()

    first = run(str(src))
    if first is None or not two_pass:
        return first

    img = cv2.imread(str(src))
    h, w = img.shape[:2]
    xs = [p[0] for p in first]
    ys = [p[1] for p in first]
    margin = 0.15 * max(max(xs) - min(xs), max(ys) - min(ys))
    x1 = int(max(min(xs) - margin, 0))
    y1 = int(max(min(ys) - margin, 0))
    x2 = int(min(max(xs) + margin, w))
    y2 = int(min(max(ys) + margin, h))
    if x2 - x1 < 32 or y2 - y1 < 32:
        return first

    second = run(img[y1:y2, x1:x2])
    if second is None:
        return first
    return [[p[0] + x1, p[1] + y1] for p in second]


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
    ap.add_argument("--corner-imgsz", type=int, default=640,
                    help="corner-model inference resolution (kpt precision lever)")
    ap.add_argument("--two-pass", action="store_true",
                    help="re-run corner model on a board crop for sharper keypoints")
    ap.add_argument("--gt-corners", action="store_true",
                    help="use annotated ground-truth corners (bottleneck diagnostic)")
    args = ap.parse_args()

    data = json.loads(Path(args.annotations).read_text())
    categories = {c["id"]: normalize_category(c["name"]) for c in data["categories"]}
    imgs = image_index(data)
    split_by_id = official_splits(data)

    by_image = defaultdict(list)
    for p in data["annotations"]["pieces"]:
        by_image[p["image_id"]].append(p)
    gt_corners_by_image = {
        c["image_id"]: c["corners"] for c in data["annotations"].get("corners", [])
    }

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
        if args.gt_corners:
            gt_c = gt_corners_by_image.get(image_id)
            kpts = ([gt_c[k] for k in ("bottom_left", "top_left", "top_right", "bottom_right")]
                    if gt_c else None)
        else:
            kpts = detect_corners(
                corner_model, src, imgsz=args.corner_imgsz, two_pass=args.two_pass)
        if kpts is None:
            corner_failures += 1
            boards += 1
            total_squares += 64
            continue
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
