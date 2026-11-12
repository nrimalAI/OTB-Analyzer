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

To score other data (e.g. the held-out Fenit photos), pass the positions.json
that build_dataset.py merge writes:
    python evaluate.py ... --positions datasets/merged/positions.json \
                           --source fenit --split test

To see which model is at fault, swap one for the labels: --gt-corners uses the
labelled corners (no corner model needed), --gt-boxes the labelled boxes (no
piece model needed). With both, no model runs at all and the score is what
chess_pipeline's geometry alone allows, e.g. for trying --pull values.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from chess_pipeline import (
    Detection, PIECE_CLASSES, FEN_LETTER,
    assign_squares, fen_placement_to_board, homography,
)
from prepare_data import image_index, image_size, normalize_category, official_splits


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

    first = run(src if not isinstance(src, Path) else str(src))
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


def stretched(src, side=640):
    """The photo as the app hands it to the models: scaled to side x side
    without keeping its shape (Vision's .scaleFill). Returns the image and the
    factors that take model coordinates back to photo pixels."""
    import cv2
    img = cv2.imread(str(src))
    h, w = img.shape[:2]
    return cv2.resize(img, (side, side), interpolation=cv2.INTER_AREA), w / side, h / side


def chessred_items(args):
    """(image, true board, annotated corners or None) for a ChessReD split."""
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

    images_root = Path(args.images)
    for image_id, piece_anns in sorted(by_image.items()):
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
        gt_c = gt_corners_by_image.get(image_id)
        boxes = [Detection(categories[p["category_id"]], 1.0,
                           (p["bbox"][0], p["bbox"][1],
                            p["bbox"][0] + p["bbox"][2], p["bbox"][1] + p["bbox"][3]))
                 for p in piece_anns
                 if p.get("bbox") and categories[p["category_id"]] in FEN_LETTER]
        yield src, gt, gt_c and [
            gt_c[k] for k in ("bottom_left", "top_left", "top_right", "bottom_right")], boxes


def merged_labels(row, src):
    """(corners, boxes) in pixels of `src` from the merged dataset's label
    files for a positions.json row; either is None when it has no label."""
    merged = Path(row["image"])
    if len(merged.parents) < 4:
        return None, None
    root, split = merged.parents[3], merged.parent.name
    corner_file = root / "corners" / "labels" / split / f"{merged.stem}.txt"
    box_file = root / "pieces" / "labels" / split / f"{merged.stem}.txt"
    if not (corner_file.exists() or box_file.exists()):
        return None, None
    w, h = image_size(src)  # after EXIF rotation, like the detections
    corners = boxes = None
    if corner_file.exists():
        v = [float(x) for x in corner_file.read_text().split()[5:17]]
        corners = [[v[i] * w, v[i + 1] * h] for i in range(0, 12, 3)]
    if box_file.exists():
        boxes = []
        for line in box_file.read_text().splitlines():
            c, cx, cy, bw, bh = line.split()[:5]
            cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
            boxes.append(Detection(PIECE_CLASSES[int(c)], 1.0,
                                   (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)))
    return corners, boxes


def position_items(args):
    """The same tuples from a positions.json written by build_dataset.py. The
    photo scored is the original one, not a resized training copy."""
    for row in json.loads(Path(args.positions).read_text()):
        if args.split != "all" and row["split"] != args.split:
            continue
        if args.source and row["source"] != args.source:
            continue
        if args.group and row.get("group") != args.group:
            continue
        src = Path(row.get("original") or row["image"])
        corners, boxes = (merged_labels(row, src) if args.gt_corners or args.gt_boxes
                          else (None, None))
        yield src, fen_placement_to_board(row["placement"]), corners, boxes


def wilson(hits, n, z=1.96):
    """95% interval for a rate measured on n boards. On 100 boards it is
    about +-9 points wide in the middle: differences of a few boards between
    two models are noise."""
    if n == 0:
        return 0.0, 0.0
    p = hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return max(centre - half, 0.0), min(centre + half, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corners-model", help="not needed with --gt-corners")
    ap.add_argument("--pieces-model", help="not needed with --gt-boxes")
    ap.add_argument("--annotations", help="ChessReD annotations.json")
    ap.add_argument("--images", help="ChessReD image root")
    ap.add_argument("--positions",
                    help="positions.json from build_dataset.py merge, instead of ChessReD")
    ap.add_argument("--source", help="with --positions: only this source, e.g. fenit")
    ap.add_argument("--group", help="with --positions: only this group, e.g. Fenit board A or B")
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
                    help="use the labelled corners instead of the corner model")
    ap.add_argument("--gt-boxes", action="store_true",
                    help="use the labelled piece boxes instead of the piece model")
    ap.add_argument("--stretch", action="store_true",
                    help="feed the models the photo stretched to 640x640, as the app "
                         "does (.scaleFill), instead of letterboxed")
    args = ap.parse_args()

    if args.stretch and args.two_pass:
        ap.error("--stretch and --two-pass cannot be combined")
    if not args.gt_corners and not args.corners_model:
        ap.error("give --corners-model or --gt-corners")
    if not args.gt_boxes and not args.pieces_model:
        ap.error("give --pieces-model or --gt-boxes")
    if args.positions:
        items = position_items(args)
    elif args.annotations and args.images:
        items = chessred_items(args)
    else:
        ap.error("give --annotations and --images, or --positions")

    if not (args.gt_corners and args.gt_boxes):
        from ultralytics import YOLO
    corner_model = None if args.gt_corners else YOLO(args.corners_model)
    piece_model = None if args.gt_boxes else YOLO(args.pieces_model)

    total_squares = correct_squares = 0
    boards = exact = corner_failures = flip_suspects = unlabelled = 0
    confusion = Counter()

    for src, gt, gt_kpts, gt_boxes in items:
        # A photo without the label that was asked for cannot take part.
        if (args.gt_corners and not gt_kpts) or (args.gt_boxes and not gt_boxes):
            unlabelled += 1
            continue
        photo, sx, sy = stretched(src) if args.stretch else (src, 1.0, 1.0)
        # 1. corners
        if args.gt_corners:
            kpts = gt_kpts
        else:
            kpts = detect_corners(
                corner_model, photo, imgsz=args.corner_imgsz, two_pass=args.two_pass)
            if kpts is not None:
                kpts = [[x * sx, y * sy] for x, y in kpts]
        if kpts is None:
            corner_failures += 1
            boards += 1
            total_squares += 64
            continue
        H = homography(kpts)

        # 2. pieces
        detections = []
        if args.gt_boxes:
            detections = gt_boxes
        else:
            pr = piece_model(photo if args.stretch else str(src), conf=args.conf,
                             verbose=False)[0]
            if pr.boxes is not None:
                for box, cls, conf in zip(pr.boxes.xyxy.tolist(), pr.boxes.cls.tolist(),
                                          pr.boxes.conf.tolist()):
                    x1, y1, x2, y2 = box
                    detections.append(Detection(PIECE_CLASSES[int(cls)], float(conf),
                                                (x1 * sx, y1 * sy, x2 * sx, y2 * sy)))

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
    if unlabelled:
        print(f"left out, no such label: {unlabelled}")
    print(f"corner detection failed: {corner_failures}")
    print(f"per-square accuracy:     {correct_squares / total_squares:.4%}")
    low, high = wilson(exact, boards)
    print(f"full-board exact match:  {exact / boards:.2%}"
          f"  (95% interval {low:.0%}-{high:.0%} on {boards} boards)")
    print(f"180-degree flip suspects: {flip_suspects}")
    print("\ntop confusions (truth -> predicted):")
    for (want, got), n in confusion.most_common(15):
        print(f"  {want} -> {got}: {n}")


if __name__ == "__main__":
    main()
