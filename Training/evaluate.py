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
    assign_squares, board_crop, colour_patch, fen_placement_to_board, homography,
    one_king_a_side, orient_by_pieces, sides_from_colour,
)
from canonical_corners import a1_order
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


def placement_of(board):
    """{(file, rank): fen_letter} -> FEN placement field."""
    rows = []
    for rank in range(7, -1, -1):
        row, empty = "", 0
        for file in range(8):
            letter = board.get((file, rank))
            if letter is None:
                empty += 1
                continue
            row += (str(empty) if empty else "") + letter
            empty = 0
        rows.append(row + (str(empty) if empty else ""))
    return "/".join(rows)


def srgb_to_lab(bgr):
    """CIE Lab (D65, L* 0-100) of 8-bit sRGB pixels given as an (n, 3) array
    in OpenCV's B, G, R order, by the formula (OpenCV's own constants). Not
    cv2.cvtColor: on float input that interpolates a table and is off by up
    to half a unit, which is enough to move a piece between two groups, and
    the Swift port computes the formula."""
    import numpy as np
    c = bgr.astype(np.float64)[:, ::-1] / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.412453, 0.357580, 0.180423],
                  [0.212671, 0.715160, 0.072169],
                  [0.019334, 0.119193, 0.950227]])
    xyz = lin @ m.T / np.array([0.950456, 1.0, 1.088754])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    y = xyz[:, 1]
    lightness = np.where(y > 216 / 24389, 116 * f[:, 1] - 16, 903.3 * y)
    return np.stack([lightness, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], axis=1)


def piece_colours(src, detections):
    """Median CIE Lab of the middle of each detection's box in the photo
    (chess_pipeline.colour_patch; the median of each of L*, a*, b* over the
    patch's pixels), or None for an empty patch."""
    import cv2
    import numpy as np
    img = cv2.imread(str(src))
    if img is None:
        return [None] * len(detections)
    h, w = img.shape[:2]
    out = []
    for det in detections:
        patch = colour_patch(det.bbox, w, h)
        if patch is None:
            out.append(None)
            continue
        xa, ya, xb, yb = patch
        lab = srgb_to_lab(img[ya:yb, xa:xb].reshape(-1, 3))
        out.append(tuple(float(v) for v in np.median(lab, axis=0)))
    return out


def detect_corners(model, src, imgsz=640, two_pass=False, device=None):
    """Predicted (a1, a8, h8, h1) image points, or None.

    With `two_pass`, re-runs the model on a crop around the first-pass board —
    the board then fills most of the model input, sharpening keypoints ~3×
    without retraining. Mirrors what the app does with two Vision requests.
    """
    import cv2

    def run(img):
        r = model(img, imgsz=imgsz, verbose=False, device=device)[0]
        if r.keypoints is None or len(r.keypoints) == 0:
            return None
        best = int(r.boxes.conf.argmax()) if r.boxes is not None and len(r.boxes) else 0
        cls = r.boxes.cls[best] if r.boxes is not None and len(r.boxes) else 0
        # A model trained with canonical_corners.py says where a1 is by class.
        return a1_order(r.keypoints.xy[best].tolist(), cls, r.names)

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


def box_iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def kind_scores_of(detections, every_box, max_iou=0.7):
    """For each detection, the model's score for every kind it considered for
    that piece: the best confidence among all boxes (of either side, down to
    a low confidence) that overlap the detection by more than max_iou. This
    stands in for the class scores of one box, which ultralytics does not
    hand out and Vision does."""
    out = []
    for det in detections:
        scores = {}
        for other in every_box:
            if box_iou(det.bbox, other.bbox) > max_iou:
                kind = other.cls.split("_", 1)[1]
                scores[kind] = max(scores.get(kind, 0.0), other.confidence)
        out.append(scores)
    return out


def one_per_spot(detections, max_iou):
    """Detections without the ones that overlap a more confident detection
    (of any class) by more than max_iou."""
    iou = box_iou
    kept = []
    for det in sorted(detections, key=lambda d: -d.confidence):
        if all(iou(det.bbox, k.bbox) <= max_iou for k in kept):
            kept.append(det)
    return kept


def stretched(src, side=640):
    """The photo as the app hands it to the models: scaled to side x side
    without keeping its shape (Vision's .scaleFill). Returns the image and the
    factors that take model coordinates back to photo pixels."""
    import cv2
    img = cv2.imread(str(src))
    h, w = img.shape[:2]
    return cv2.resize(img, (side, side), interpolation=cv2.INTER_AREA), w / side, h / side


def shrunk(src, max_side):
    """The photo no longer than max_side on its long side, shrunk with
    smoothing (INTER_AREA), and the factors that take model coordinates back
    to photo pixels. What the merged datasets' copies are, made on the fly
    for folders that have none; a photo already that small is returned as it
    is."""
    import cv2
    img = cv2.imread(str(src))
    h, w = img.shape[:2]
    s = max(h, w) / max_side
    if s <= 1:
        return img, 1.0, 1.0
    small = cv2.resize(img, (round(w / s), round(h / s)), interpolation=cv2.INTER_AREA)
    return small, w / small.shape[1], h / small.shape[0]


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
    photo scored is the original one, or with --copies the dataset's resized
    copy of it."""
    if args.copies:
        # A folder merged without --max-side has no copies: its "image" is the
        # full-size photo. (With --max-side a photo that is already small
        # enough is its own copy, so the rows cannot tell.)
        manifest = Path(args.positions).parent / "manifest.json"
        if manifest.exists() and not json.loads(manifest.read_text()).get("max_side"):
            raise SystemExit(f"--copies: {args.positions} was merged without --max-side "
                             "and has no copies. Use --max-side 1280 instead.")
    for row in json.loads(Path(args.positions).read_text()):
        if args.split != "all" and row["split"] != args.split:
            continue
        if args.source and row["source"] != args.source:
            continue
        if args.group and row.get("group") != args.group:
            continue
        if args.copies:
            src = Path(row["image"])
        else:
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
    ap.add_argument("--one-piece-per-spot", type=float, default=None, metavar="IOU",
                    help="drop a detected box that overlaps a more confident one of "
                         "any class by more than IOU (the piece model's own NMS is "
                         "per class, so one piece can come back as two classes)")
    ap.add_argument("--one-king", action="store_true",
                    help="exactly one king a side, from every kind the model scored "
                         "for each piece (chess_pipeline.one_king_a_side)")
    ap.add_argument("--crop-board", action="store_true",
                    help="give the piece model only the part of the photo round the "
                         "board (chess_pipeline.board_crop), for a piece model trained "
                         "on such crops (crop_boards.py)")
    ap.add_argument("--sides-from-colour", action="store_true",
                    help="decide White and Black from the colours of the detected "
                         "pieces where the model's answer does not fit them "
                         "(chess_pipeline.sides_from_colour)")
    ap.add_argument("--grid-corners", action="store_true",
                    help="replace the corners by the classical grid fit "
                         "(grid_corners.py) where it accepts its own result; the "
                         "rotation stays the one nearest the model's or labelled corners")
    ap.add_argument("--a1-from-pieces", type=float, nargs="?", const=0.0, default=None,
                    metavar="PRIOR",
                    help="choose which corner is a1 from the detected pieces "
                         "(chess_pipeline.orient_by_pieces); PRIOR is how much the "
                         "corners' own order counts (default 0)")
    ap.add_argument("--stretch", action="store_true",
                    help="feed the models the photo stretched to 640x640, as the app "
                         "does (.scaleFill), instead of letterboxed")
    ap.add_argument("--copies", action="store_true",
                    help="with --positions: score the dataset's own copies of the photos "
                         "(1280 px on the long side, cleanly downscaled) instead of the "
                         "full-size originals. ultralytics shrinks a 12 MP photo to 640 "
                         "without smoothing, which a model trained on the copies reads "
                         "badly and the app does not do (it draws the photo into at most "
                         "1280 px first): the copies are the fairer test of such a model")
    ap.add_argument("--max-side", type=int, default=0, metavar="PX",
                    help="shrink each photo, with smoothing, to at most PX on its long "
                         "side before the models see it (1280: what --copies gives, for "
                         "folders that have no copies, e.g. datasets/heldout2)")
    ap.add_argument("--device", default=None,
                    help="where the models run, e.g. cpu, so as not to take GPU "
                         "memory from a training job (default: ultralytics picks)")
    ap.add_argument("--dump-list", metavar="FILE",
                    help="run no model: write the chosen photos and their true "
                         "placements as JSON, the list app_check (the app's own "
                         "recognizer, see app_check/) takes")
    args = ap.parse_args()

    if args.dump_list:
        if args.positions:
            items = position_items(args)
        elif args.annotations and args.images:
            items = chessred_items(args)
        else:
            ap.error("give --annotations and --images, or --positions")
        photos = {}
        for src, gt, _, _ in items:
            photos[str(Path(src).resolve())] = placement_of(gt)
            if args.limit and len(photos) >= args.limit:
                break
        Path(args.dump_list).parent.mkdir(parents=True, exist_ok=True)
        Path(args.dump_list).write_text(json.dumps(photos, indent=0, sort_keys=True))
        print(f"{len(photos)} photos written to {args.dump_list}")
        return
    if args.stretch and args.two_pass:
        ap.error("--stretch and --two-pass cannot be combined")
    if args.stretch and args.crop_board:
        ap.error("--stretch and --crop-board cannot be combined")
    if args.max_side and (args.stretch or args.two_pass):
        ap.error("--max-side cannot be combined with --stretch or --two-pass")
    if args.copies and not args.positions:
        ap.error("--copies needs --positions")
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
    exact_or_flipped = grid_rejected = grid_alone = sides_changed = kings_changed = 0
    confusion = Counter()

    for src, gt, gt_kpts, gt_boxes in items:
        # A photo without the label that was asked for cannot take part.
        if (args.gt_corners and not gt_kpts) or (args.gt_boxes and not gt_boxes):
            unlabelled += 1
            continue
        if args.stretch:
            photo, sx, sy = stretched(src)
        elif args.max_side:
            photo, sx, sy = shrunk(src, args.max_side)
        else:
            photo, sx, sy = src, 1.0, 1.0
        as_array = args.stretch or bool(args.max_side)
        # 1. corners
        if args.gt_corners:
            kpts = gt_kpts
        else:
            kpts = detect_corners(
                corner_model, photo, imgsz=args.corner_imgsz, two_pass=args.two_pass,
                device=args.device)
            if kpts is not None:
                kpts = [[x * sx, y * sy] for x, y in kpts]
        # With the grid fit, a board the corner model does not find can still
        # be read: the fit needs no model. Its corners then come in no
        # particular order, so a1 is taken from the pieces alone below.
        model_found = kpts is not None
        if args.grid_corners:
            import cv2
            import numpy as np
            from grid_corners import find_grid
            grid, _ = find_grid(cv2.imread(str(src)))
            if grid is None:
                grid_rejected += 1
            elif model_found:
                grid, old = np.array(grid, float), np.array(kpts, float)
                turn = min(range(4), key=lambda j: np.linalg.norm(
                    np.roll(grid, -j, axis=0) - old, axis=1).sum())
                kpts = np.roll(grid, -turn, axis=0).tolist()
            else:
                kpts = np.array(grid, float).tolist()
                grid_alone += 1
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
            piece_input, dx, dy = (photo if as_array else str(src)), 0.0, 0.0
            if args.crop_board:
                # The piece model sees only the board (chess_pipeline.board_crop).
                import cv2
                whole = photo if as_array else cv2.imread(str(src))
                rect = board_crop([[x / sx, y / sy] for x, y in kpts],
                                  whole.shape[1], whole.shape[0])
                if rect is not None:
                    piece_input = whole[rect[1]:rect[3], rect[0]:rect[2]]
                    dx, dy = rect[0] * sx, rect[1] * sy
            # For --one-king every box down to a low confidence is wanted: the
            # ones below --conf only say what else each piece might be.
            pr = piece_model(piece_input, conf=min(args.conf, 0.02) if args.one_king else args.conf,
                             verbose=False, device=args.device)[0]
            if pr.boxes is not None:
                for box, cls, conf in zip(pr.boxes.xyxy.tolist(), pr.boxes.cls.tolist(),
                                          pr.boxes.conf.tolist()):
                    x1, y1, x2, y2 = box
                    detections.append(Detection(PIECE_CLASSES[int(cls)], float(conf),
                                                (x1 * sx + dx, y1 * sy + dy,
                                                 x2 * sx + dx, y2 * sy + dy)))

        every_box = detections
        detections = [d for d in detections if d.confidence >= args.conf or args.gt_boxes]
        if args.one_piece_per_spot is not None and not args.gt_boxes:
            detections = one_per_spot(detections, args.one_piece_per_spot)
        if args.sides_from_colour and not args.gt_boxes:
            before = [d.cls for d in detections]
            detections = sides_from_colour(detections, piece_colours(src, detections))
            sides_changed += any(d.cls != c for d, c in zip(detections, before))
        if args.one_king and not args.gt_boxes:
            before = [d.cls for d in detections]
            detections = one_king_a_side(detections, kind_scores_of(detections, every_box))
            kings_changed += any(d.cls != c for d, c in zip(detections, before))
        if not model_found:
            # No model opinion on a1 to weigh against the pieces.
            H = homography(orient_by_pieces(kpts, detections, args.pull, prior=0.0))
        elif args.a1_from_pieces is not None:
            H = homography(orient_by_pieces(kpts, detections, args.pull,
                                            prior=args.a1_from_pieces))
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
        flipped = {(7 - f, 7 - r): v for (f, r), v in pred_letters.items()}
        exact_or_flipped += board_ok or all(
            gt.get((f, r)) == flipped.get((f, r)) for f in range(8) for r in range(8))

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
    low, high = wilson(exact_or_flipped, boards)
    print(f"exact, or exact once turned 180 degrees (one tap in the app): "
          f"{exact_or_flipped / boards:.2%}  ({low:.0%}-{high:.0%})")
    if args.grid_corners:
        print(f"grid fit gave no result (model corners kept, if any): {grid_rejected}")
        print(f"grid fit used alone, the model found no board (a1 from the pieces): {grid_alone}")
    if args.sides_from_colour:
        print(f"sides taken from the pieces' colours on: {sides_changed} boards")
    if args.one_king:
        print(f"a king made or unmade on: {kings_changed} boards")
    print(f"180-degree flip suspects: {flip_suspects}")
    print("\ntop confusions (truth -> predicted):")
    for (want, got), n in confusion.most_common(15):
        print(f"  {want} -> {got}: {n}")


if __name__ == "__main__":
    main()
