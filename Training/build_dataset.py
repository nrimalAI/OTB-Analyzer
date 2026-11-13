"""Download several chess datasets and merge them into the two YOLO datasets
we train (pieces: detect, 12 classes; corners: pose, 4 keypoints).

    python build_dataset.py list                 # sources, sizes, what is on disk
    python build_dataset.py fetch                # download everything not optional
    python build_dataset.py fetch chesscog       # ...or just the named sources
    python build_dataset.py merge                # -> datasets/merged/{pieces,corners}
    python build_dataset.py merge --shippable    # leave out NonCommercial data
    python build_dataset.py merge --out datasets/heldout2 \
        --sources chessq ylichess chesscog_real samryan chesscv_games   # the test-only set

Sources are declared in sources.yaml. `merge` uses whatever is on disk and
skips the rest, so it works with a partial download.

What each source contributes:
  chessred   boxes + corners (ChessReD2K subset), official train/val/test split
  chesscog   boxes + corners (rendered), its own train/val/test split
  fenit      boxes + corners (real photos); 20% held out as test, 10% val
  roboflow   boxes only; near-duplicate photos across projects are dropped;
             90% train / 10% val by hash
  native     whatever labels are present, train only

Sources marked `test_only` in sources.yaml (the second held-out test set) are
never part of a training folder: `merge` leaves them out unless they are
named, refuses to mix them with other sources, and writes every photo as
`test` into a folder of their own. Most give a position and no boxes or
corners; evaluate.py needs no more than the position.

Corner labels need to say WHICH corner is a1. chesscog and Fenit do not, so
the order is recovered from where the pieces are known to stand (see
orient_corners). Images where that is ambiguous keep their boxes and lose
their corners.

Outputs, next to the two datasets:
  manifest.json   per-source counts, licenses, what was skipped
  positions.json  the true position of every image that has one, so
                  evaluate.py can score whole boards on non-ChessReD data
"""

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml

from chess_pipeline import (
    FEN_LETTER, PIECE_CLASSES, board_square, board_to_fen_placement,
    fen_placement_to_board, homography,
)
from prepare_data import (
    extract_corners, image_index, image_size, link, normalize_category,
    official_splits,
)

HERE = Path(__file__).resolve().parent
CLASS_ID = {name: i for i, name in enumerate(PIECE_CLASSES)}
LETTER_TO_CLASS = {letter: name for name, letter in FEN_LETTER.items()}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")

# Labels in third-party exports that are not pieces and can be dropped without
# leaving an unlabelled piece in the image.
NON_PIECE = {"board", "chessboard", "chess_board", "corner", "corners", "empty",
             "square", "hand"}


@dataclass
class Sample:
    source: str
    stem: str
    image: Path
    split: str
    boxes: list = None     # [(class_id, cx, cy, w, h)] normalized, or None
    corners: list = None   # [(x, y)] * 4 normalized, order a1 a8 h8 h1, or None
    placement: str = None  # FEN placement field, when the position is known
    group: str = None      # a sub-set evaluate.py can pick out, e.g. Fenit's board


# --------------------------------------------------------------------------
# Class names

_PIECE_WORDS = {
    "king": "king", "queen": "queen", "rook": "rook", "castle": "rook",
    "tower": "rook", "bishop": "bishop", "knight": "knight", "horse": "knight",
    "pawn": "pawn",
}
_COLOR_WORDS = {"white": "white", "w": "white", "light": "white",
                "black": "black", "b": "black", "dark": "black"}
_LETTERS = {"k": "king", "q": "queen", "r": "rook", "b": "bishop",
            "n": "knight", "p": "pawn"}


def canonical_class(name):
    """Map a third-party label ('white-king', 'bP', 'Knight_black', 'N') to one
    of PIECE_CLASSES, or None if it is not a colored piece."""
    raw = str(name).strip()
    if len(raw) == 1 and raw.lower() in _LETTERS:            # FEN letter
        color = "white" if raw.isupper() else "black"
        return f"{color}_{_LETTERS[raw.lower()]}"
    m = re.fullmatch(r"([wb])[-_ ]?([kqrbnp])", raw, re.I)   # wK, b_p
    if m:
        return f"{_COLOR_WORDS[m[1].lower()]}_{_LETTERS[m[2].lower()]}"
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", raw)        # camelCase
    tokens = [t for t in re.split(r"[^a-z]+", spaced.lower()) if t]
    piece = next((_PIECE_WORDS[t] for t in tokens if t in _PIECE_WORDS), None)
    color = next((_COLOR_WORDS[t] for t in tokens if t in _COLOR_WORDS), None)
    return f"{color}_{piece}" if piece and color else None


# --------------------------------------------------------------------------
# Geometry

def _pick_rotation(points, n, hits, threshold):
    """Try the four rotations of the clockwise corner ring; return the one
    with the most hits, or None when it is not clear-cut."""
    cx = sum(p[0] for p in points) / 4
    cy = sum(p[1] for p in points) / 4
    # Ascending atan2 with y pointing down = clockwise on screen.
    ring = sorted(points, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    scored = []
    for k in range(4):
        order = ring[k:] + ring[:k]
        scored.append((hits(homography(order)), k, order))
    scored.sort(key=lambda s: -s[0])
    if scored[0][0] < max(2, threshold * n) or scored[0][0] == scored[1][0]:
        return None
    return scored[0][2]


def _anchor(box):
    x1, _, x2, y2 = box
    return ((x1 + x2) / 2, y2)


def orient_corners(points, pieces):
    """Order four board corners as (a1, a8, h8, h1).

    points: four (x, y) image points in any order.
    pieces: [(square, (x1, y1, x2, y2))], e.g. ("e4", box) — known squares.

    Seen from above, a1 -> a8 -> h8 -> h1 runs clockwise in the image, so only
    the four rotations of the clockwise polygon are candidates. The winner is
    the one that puts the most pieces on their labelled squares. Returns None
    when that is not clear-cut (too few pieces, or a symmetric position).
    """
    def hits(H):
        return sum(
            board_square(H, _anchor(box)) == (ord(sq[0]) - ord("a"), int(sq[1]) - 1)
            for sq, box in pieces)
    return _pick_rotation(points, len(pieces), hits, 0.7)


def orient_corners_by_position(points, pieces, placement):
    """Same, when the boxes carry no squares but the whole position is known.

    pieces: [(fen_letter, (x1, y1, x2, y2))]; placement: FEN placement field.
    A hit is a box that lands on a square holding that same piece.
    """
    board = fen_placement_to_board(placement)

    def hits(H):
        return sum(board.get(board_square(H, _anchor(box))) == letter
                   for letter, box in pieces)
    return _pick_rotation(points, len(pieces), hits, 0.6)


def occupied_share(ordered, pieces, placement):
    """Share of boxes that land on a square the position has a piece on,
    whatever the piece — a check on the corners that a wrong class label on a
    box does not disturb."""
    board = fen_placement_to_board(placement)
    H = homography(ordered)
    return sum(board_square(H, _anchor(box)) in board for _, box in pieces) / len(pieces)


def dhash(path):
    """64-bit difference hash; equal for re-encoded or resized copies."""
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_REDUCED_GRAYSCALE_4)
    if img is None:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    small = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


# --------------------------------------------------------------------------
# Source adapters: each yields Samples and counts what it skipped

def load_chessred(name, cfg, root, skipped):
    ann_path = root / "annotations.json"
    data = json.loads(ann_path.read_text())
    categories = {c["id"]: normalize_category(c["name"]) for c in data["categories"]}
    imgs = image_index(data)
    split_by_id = official_splits(data)

    boxes = defaultdict(list)
    for p in data["annotations"]["pieces"]:
        if p.get("bbox"):
            boxes[p["image_id"]].append(p)
    corners = {}
    for c in data["annotations"].get("corners", []):
        pts = extract_corners(c)
        if pts is not None:
            corners[c["image_id"]] = pts

    for image_id in sorted(set(boxes) & set(corners)):
        meta = imgs[image_id]
        split = split_by_id.get(image_id)
        src = find_image(root, meta["path"])
        if split is None or src is None or not meta["width"] or not meta["height"]:
            skipped["missing image or split"] += 1
            continue
        w, h = meta["width"], meta["height"]
        out, board = [], {}
        for p in boxes[image_id]:
            cls = categories[p["category_id"]]
            if cls in CLASS_ID:
                x, y, bw, bh = p["bbox"]
                out.append((CLASS_ID[cls], (x + bw / 2) / w, (y + bh / 2) / h,
                            bw / w, bh / h))
                pos = p.get("chessboard_position")
                if pos:
                    board[(ord(pos[0]) - ord("a"), int(pos[1]) - 1)] = (cls, 1.0)
        yield Sample(name, f"img{image_id:06d}", src, split, out,
                     [(x / w, y / h) for x, y in corners[image_id]],
                     board_to_fen_placement(board) if len(board) == len(out) else None)


def find_image(root, rel):
    """Annotation paths look like images/0/G000_IMG000.jpg; zips unpack with
    or without an extra top-level folder."""
    rel = Path(rel)
    for base in (root, root / "images", root / "chessred2k", root / "chessred"):
        for candidate in (base / rel, base / Path(*rel.parts[1:])):
            if candidate.is_file():
                return candidate
    return None


def load_chesscog(name, cfg, root, skipped):
    for split in SPLITS:
        files = sorted((root / split).glob("*.json"))
        cap = cfg.get("max_train")
        if split == "train" and cap and len(files) > cap:
            # One rendered board should not outweigh every real photo.
            skipped[f"note: train capped at {cap} of {len(files)} (max_train)"] = len(files) - cap
            files = sorted(random.Random(0).sample(files, cap))
        for js in files:
            img = js.with_suffix(".png")
            if not img.exists():
                skipped["missing image or split"] += 1
                continue
            meta = json.loads(js.read_text())
            w, h = image_size(img)
            boxes, placed = [], []
            for p in meta["pieces"]:
                x, y, bw, bh = p["box"]
                boxes.append((CLASS_ID[LETTER_TO_CLASS[p["piece"]]],
                              (x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h))
                placed.append((p["square"], (x, y, x + bw, y + bh)))
            ordered = orient_corners([tuple(c) for c in meta["corners"]], placed)
            if ordered is None:
                skipped["corner order ambiguous (boxes kept)"] += 1
            yield Sample(name, f"{split}_{js.stem}", img, split, boxes,
                         ordered and [(x / w, y / h) for x, y in ordered],
                         meta["fen"].split()[0])


# Class order in Fenit's data.yaml, used if that file is missing.
FENIT_NAMES = ["Board", "White-King", "White-Pawn", "Black-Pawn", "Black-King",
               "White-Queen", "White-Bishop", "White-Knight", "White-Rook",
               "Black-Bishop", "Black-Rook", "Black-Knight", "Black-Queen"]


def placement_from_name(stem):
    """Fenit names most photos after their position: the eight FEN rows joined
    by '_' or '-', sometimes followed by ' w' or ' b'. Returns the placement,
    or None (some photos are named by timestamp instead)."""
    rows = re.split(r"[_-]", stem.split()[0]) if stem.strip() else []
    return valid_placement("/".join(rows))


def valid_placement(text):
    """The FEN placement field at the start of `text` (a full FEN is fine), or
    None if it is not eight rows of eight squares."""
    placement = text.split()[0] if text and text.strip() else ""
    rows = placement.split("/")
    if len(rows) != 8:
        return None
    for row in rows:
        if not re.fullmatch(r"[KQRBNPkqrbnp1-8]+", row):
            return None
        if sum(int(c) if c.isdigit() else 1 for c in row) != 8:
            return None
    return placement


def fenit_board(stem):
    """Fenit photographs two boards (see README). Board B's photos are the ones
    named by timestamp or with '-' between the FEN rows; the rest are board A."""
    head = stem.split()[0] if stem.strip() else ""
    if re.fullmatch(r"\d{8}_\d{6}", head) or len(head.split("-")) == 8:
        return "B"
    return "A"


def load_fenit(name, cfg, root, skipped):
    """Fenit (Mendeley Data, CC BY 4.0): real photos, one YOLO-pose .txt each.
    The 'Board' line carries the four corners as keypoints; piece lines carry a
    box (their own keypoints are ignored). Corner order is not documented, so
    it is recovered from the position in the file name.

    There is no official split, and photos of one position must stay together,
    so the split is a hash of the position. With `holdout: B` (or A) in
    sources.yaml every photo of that board is the test split instead, a board
    the models never see, and the other board is train/val only."""
    names = FENIT_NAMES
    data_yaml = next(iter(sorted(root.rglob("data.yaml"))), None)
    if data_yaml is not None:
        names = yaml.safe_load(data_yaml.read_text()).get("names") or names
    if isinstance(names, list):
        names = dict(enumerate(names))
    class_map = {int(i): canonical_class(n) for i, n in names.items()}
    images = {p.stem: p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS}
    test_cut = 100 * cfg.get("test_fraction", 0.2)
    val_cut = test_cut + 100 * cfg.get("val_fraction", 0.1)

    for label in sorted(root.rglob("*.txt")):
        img = images.get(label.stem)
        if img is None:
            if placement_from_name(label.stem):
                skipped["missing image or split"] += 1
            continue
        w, h = image_size(img)
        boxes, lettered, board_pts = [], [], None
        for line in label.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cls, v = int(float(parts[0])), [float(x) for x in parts[1:]]
            piece = class_map.get(cls)
            if piece is None:
                if board_pts is None and len(v) >= 16:
                    board_pts = [(v[4 + 3 * i] * w, v[5 + 3 * i] * h) for i in range(4)]
                continue
            cx, cy, bw, bh = v[:4]
            boxes.append((CLASS_ID[piece], cx, cy, bw, bh))
            lettered.append((FEN_LETTER[piece], ((cx - bw / 2) * w, (cy - bh / 2) * h,
                                                 (cx + bw / 2) * w, (cy + bh / 2) * h)))
        if not boxes:
            skipped["no pieces labelled"] += 1
            continue

        placement = placement_from_name(label.stem)
        corners = None
        if board_pts is None:  # mostly close-ups of a single piece on the floor
            skipped["no board outline labelled (boxes kept)"] += 1
        elif placement is None:
            skipped["file name is not a position (boxes kept, no corners)"] += 1
        else:
            ordered = orient_corners_by_position(board_pts, lettered, placement)
            if ordered is None:
                skipped["corner order ambiguous (boxes kept)"] += 1
            elif occupied_share(ordered, lettered, placement) < 0.8:
                # Seen where the board runs out of the frame and the outline
                # was clicked on the image edge instead of the real corner.
                skipped["corners do not fit the position (boxes kept)"] += 1
            else:
                corners = [(x / w, y / h) for x, y in ordered]

        # The name is the more reliable of the two: where it lists more pieces
        # than there are boxes, a piece on the board was left unlabelled, and
        # training on that teaches the detector to miss pieces.
        if placement is not None:
            pieces = len(fen_placement_to_board(placement))
            if len(boxes) < pieces:
                skipped["a piece has no box (boxes dropped, corners kept)"] += 1
                boxes = None
            elif len(boxes) > pieces:
                skipped["note: more boxes than pieces in the name (kept)"] += 1

        bucket = int(hashlib.md5((placement or label.stem).encode()).hexdigest(), 16) % 100
        board = fenit_board(label.stem)
        if cfg.get("holdout"):
            split = ("test" if board == cfg["holdout"]
                     else "val" if bucket < val_cut - test_cut else "train")
        else:
            split = "test" if bucket < test_cut else "val" if bucket < val_cut else "train"
        yield Sample(name, re.sub(r"[^A-Za-z0-9_.-]+", "-", label.stem), img, split,
                     boxes, corners, placement, board)


def load_roboflow(name, cfg, root, skipped, seen):
    """root holds one YOLO export per sub-folder. `seen` carries de-dup state
    across projects, since Universe is full of forks of the same photos: a
    photo is a repeat if it has the same original file name, or the same image
    hash and the same labels.

    Per-project settings in sources.yaml: `max_images` (default: the source's
    `max_per_project`) keeps a fixed random sample, so one large project of
    video frames cannot swamp the rest; `only_splits` reads just those of the
    export's train/valid/test folders, for exports whose train folder holds
    augmented copies; `exclude_prefixes` drops files whose names start with
    one of them, for projects that re-upload other datasets (copies of our own
    test photos would otherwise end up in training); `license` overrides what
    the export states."""
    settings = {p["id"].split("/")[-1]: p for p in cfg.get("projects", [])}
    for project in sorted(p for p in root.iterdir() if p.is_dir()):
        data_yaml = next(iter(sorted(project.rglob("data.yaml"))), None)
        if data_yaml is None:
            continue
        meta = yaml.safe_load(data_yaml.read_text())
        names = meta.get("names", {})
        if isinstance(names, list):
            names = dict(enumerate(names))
        class_map = {i: canonical_class(n) for i, n in names.items()}
        is_pose = bool(meta.get("kpt_shape"))
        ignorable = {i for i, n in names.items()
                     if normalize_category(str(n)).lower() in NON_PIECE}
        mine = settings.get(project.name, {})
        license_ = (mine.get("license") or (meta.get("roboflow") or {}).get("license")
                    or read_license(project))
        cfg.setdefault("project_licenses", {})[project.name] = license_
        if cfg.get("only_shippable") and not is_shippable(license_):
            skipped[f"{project.name}: license '{license_}' not shippable"] += 1
            continue

        labels = sorted(data_yaml.parent.rglob("labels/*.txt"))
        if mine.get("only_splits"):
            labels = [l for l in labels if l.parent.parent.name in mine["only_splits"]]
        prefixes = tuple(mine.get("exclude_prefixes") or ())
        if prefixes:
            copies = [l for l in labels if l.name.startswith(prefixes)]
            if copies:
                skipped[f"{project.name}: copy of another dataset"] += len(copies)
                labels = [l for l in labels if not l.name.startswith(prefixes)]
        cap = mine.get("max_images", cfg.get("max_per_project"))
        if cap and len(labels) > cap:
            random.Random(0).shuffle(labels)  # a fixed order; stop once `cap` are kept
        else:
            cap = None
        kept = 0
        for label in labels:
            if cap and kept >= cap:
                break
            img = next((label.parent.parent / "images" / (label.stem + ext)
                        for ext in IMAGE_EXTS
                        if (label.parent.parent / "images" / (label.stem + ext)).exists()),
                       None)
            if img is None:
                skipped["missing image or split"] += 1
                continue
            boxes, unknown = [], False
            for line in label.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls = int(float(parts[0]))
                vals = [float(v) for v in parts[1:]]
                if cls in ignorable:
                    continue
                if class_map.get(cls) is None:
                    unknown = True
                    break
                if is_pose:  # box followed by keypoints
                    vals = vals[:4]
                elif len(vals) > 4:  # segmentation polygon -> its bounding box
                    xs, ys = vals[0::2], vals[1::2]
                    vals = [(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2,
                            max(xs) - min(xs), max(ys) - min(ys)]
                boxes.append((CLASS_ID[class_map[cls]], *vals))
            if unknown:
                # e.g. a colorless "bishop": dropping just that box would leave
                # an unlabelled piece, which trains the model to miss pieces.
                skipped["has a label we cannot map to a colored piece"] += 1
                continue
            if not boxes:
                skipped["no pieces labelled"] += 1
                continue

            # Roboflow names files <original>_jpg.rf.<hash>.jpg; augmented
            # copies and forks share <original>.
            original = re.sub(r"\.rf\.[0-9a-zA-Z]+$", "", label.stem)
            original = re.sub(r"_(jpe?g|png)$", "", original, flags=re.I)
            # Same original name and the same pieces: an augmented copy or a
            # fork. The name alone is not enough -- exports are full of
            # unrelated photos called "10" or "IMG_1".
            name_key = (original, tuple(sorted(box[0] for box in boxes)))
            image_hash = dhash(img)
            if image_hash is None:
                skipped["image cannot be read"] += 1
                continue
            if name_key in seen["names"] or any(
                    same_labels(boxes, other) for other in seen["hashes"][image_hash]):
                skipped["duplicate of another image"] += 1
                continue
            seen["names"].add(name_key)
            seen["hashes"][image_hash].append(boxes)
            bucket = int(hashlib.md5(original.encode()).hexdigest(), 16) % 10
            kept += 1
            yield Sample(name, f"{project.name}_{label.stem}", img,
                         "val" if bucket == 0 else "train", boxes, None)
        skipped[f"note: {project.name} gives {kept} of {len(labels)} labelled images"] = kept


def same_labels(a, b, tolerance=0.02):
    """Two box lists that describe the same photo: same classes at the same
    places. The image hash alone is not enough to call two images duplicates:
    photos taken from a tripod differ only in a few small pieces and hash
    alike (it dropped 100 of 289 different photos in one Roboflow project)."""
    if len(a) != len(b):
        return False
    left = list(b)
    for box in a:  # each box takes the nearest unclaimed box of its class
        near = [o for o in left if o[0] == box[0]
                and abs(o[1] - box[1]) < tolerance and abs(o[2] - box[2]) < tolerance]
        if not near:
            return False
        left.remove(min(near, key=lambda o: abs(o[1] - box[1]) + abs(o[2] - box[2])))
    return True


def read_license(project):
    for readme in project.glob("README*.txt"):
        m = re.search(r"License:\s*(.+)", readme.read_text(errors="ignore"))
        if m:
            return m[1].strip()
    return "unknown"


def load_native(name, cfg, root, skipped):
    """root/{pieces,corners}/{images,labels}/ in our own label format."""
    found = {}
    for task in ("pieces", "corners"):
        for label in sorted((root / task / "labels").glob("*.txt")):
            img = next((root / task / "images" / (label.stem + ext)
                        for ext in IMAGE_EXTS
                        if (root / task / "images" / (label.stem + ext)).exists()), None)
            if img is None:
                skipped["missing image or split"] += 1
                continue
            sample = found.setdefault(label.stem, Sample(name, label.stem, img, "train"))
            rows = [[float(v) for v in line.split()]
                    for line in label.read_text().splitlines() if line.strip()]
            if task == "pieces":
                sample.boxes = [(int(r[0]), *r[1:5]) for r in rows]
            elif rows:
                k = rows[0][5:]
                sample.corners = [(k[i], k[i + 1]) for i in range(0, 12, 3)]
    yield from found.values()


# --------------------------------------------------------------------------
# Test-only sources: the second held-out test set. Every Sample is `test`.

def clean_stem(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text)


def load_chessq(name, cfg, root, skipped):
    """ChessQ Test (Hugging Face joelseytre/chessq-test): 50 phone photos of
    15 kinds of chess set, positions reviewed by hand. metadata.jsonl has one
    row per photo with the placement (`fen`) and the same thing square by
    square (`labels`, a8..h1, 0 empty, 1-6 PNBRQK, 7-12 pnbrqk); a row whose
    two labels disagree is left out. Which way round the board is follows the
    pieces, not the coordinates printed on some of the boards. The group is
    the kind of chess set."""
    letters = ".PNBRQKpnbrqk"
    for meta in sorted(root.rglob("metadata.jsonl")):
        for line in meta.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            img = meta.parent / row["file_name"]
            placement = valid_placement(row.get("fen", ""))
            if not img.exists() or placement is None:
                skipped["missing image or position"] += 1
                continue
            board = fen_placement_to_board(placement)
            squares = [board.get((f, r), ".") for r in range(7, -1, -1) for f in range(8)]
            if squares != [letters[i] for i in row["labels"]]:
                skipped["the two labels of the photo disagree"] += 1
                continue
            yield Sample(name, clean_stem(row["id"]), img, "test", placement=placement,
                         group=row.get("chess_set_type"))


def looks_like_a_game(placement):
    """Whether the position could come from a game with enough pawns left to
    show which way the board is played: one king a side, no pawn on the first
    or last rank, at most eight pawns and sixteen pieces a side, and at least
    six pawns on the board. Nothing a recognizer computes is used here, so
    which photos count as game positions does not depend on what is being
    tested. Y-LIChess also photographs piece layouts made to cover the board
    (two knights; all 32 pieces packed into four ranks), where a1 is only a
    convention."""
    board = fen_placement_to_board(placement)
    letters = list(board.values())
    white = [x for x in letters if x.isupper()]
    black = [x for x in letters if x.islower()]
    if letters.count("K") != 1 or letters.count("k") != 1:
        return False
    if any(x in "Pp" and rank in (0, 7) for (_, rank), x in board.items()):
        return False
    if len(white) > 16 or len(black) > 16 or white.count("P") > 8 or black.count("p") > 8:
        return False
    return white.count("P") + black.count("p") >= 6


# Class ids 1-12 in Y-LIChess's annots/, as in its FEN_generator.py.
YLICHESS_LETTERS = dict(zip(range(1, 13), "bknpqrBKNPQR"))


def load_ylichess(name, cfg, root, skipped):
    """Y-LIChess multi-board dataset: per board `<board>_train/` and
    `<board>_valid/` (the authors' split; all of it is test here), each with
    captures/ (the 1920x1080 webcam frame), crops/ (the board region), annots/
    (YOLO boxes in the crop, classes 1-12) and board_info*.json: the outline
    of the playing area in the frame (`board_px`: top-left, top-right,
    bottom-right, bottom-left) and the padding the crop was cut with. A folder
    with several calibrations names each after the photo numbers it covers.

    Only the boards listed under `boards` in sources.yaml are read: on the
    other three the boxes are a detector's output, not labels (pieces without
    a box, kings labelled queen, two boxes on one piece).

    There is no position file. The position is where the boxes stand, as the
    dataset's own FEN_generator.py derives it: a photo is left out unless
    every box lands on a square of its own, and on the same square for a
    range of anchor pulls (a box near a square's edge is not a label to trust).

    Which corner is a1 is not recorded either. The camera and the board do not
    move; photos numbered up to `white_far_until` have White on the far side
    (a1 is the top-right corner, which is also what the stickers on the
    board's frame say), the later ones are set up the other way round. The
    group is the board, or `<board>-layout` where the pieces do not form a
    game position and a1 is that convention only (score those with
    evaluate.py's "exact once turned 180 degrees")."""
    for board_name, rule in (cfg.get("boards") or {}).items():
        for folder in sorted(f for f in root.glob(f"{board_name}_*") if f.is_dir()):
            calibrations = []
            for info in sorted(folder.glob("board_info*.json")):
                if "original" in info.stem:
                    continue
                m = re.search(r"\((\d+)-(\d+)\)", info.name)
                calibrations.append(((int(m[1]), int(m[2])) if m else None,
                                     json.loads(info.read_text())))
            for img in sorted((folder / "captures").glob("*.jpg")):
                number = int(re.search(r"\d+", img.stem)[0])
                fits = [c for span, c in calibrations
                        if span is None or span[0] <= number <= span[1]]
                label = folder / "annots" / f"{img.stem}.txt"
                crop = folder / "crops" / img.name
                if len(fits) != 1 or not label.exists() or not crop.exists():
                    skipped["no single calibration, label or crop for the photo"] += 1
                    continue
                info = fits[0]
                w, h = image_size(img)
                pts = [tuple(p) for p in info["board_px"]]
                pad = info["padding_px"]
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                x0, y0 = max(min(xs) - pad["left"], 0), max(min(ys) - pad["top"], 0)
                x1, y1 = min(max(xs) + pad["right"], w), min(max(ys) + pad["bottom"], h)
                cw, ch = image_size(crop)
                if (cw, ch) != (x1 - x0, y1 - y0) or [w, h] != list(info["frame_size_px"]):
                    skipped["crop does not match the calibration"] += 1
                    continue

                boxes, placed = [], []
                for line in label.read_text().splitlines():
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    piece = LETTER_TO_CLASS[YLICHESS_LETTERS[int(parts[0])]]
                    cx, cy, bw, bh = (float(v) for v in parts[1:])
                    cx, cy, bw, bh = x0 + cx * cw, y0 + cy * ch, bw * cw, bh * ch
                    boxes.append((CLASS_ID[piece], cx / w, cy / h, bw / w, bh / h))
                    placed.append((piece, (cx, cy + bh / 2)))
                if not boxes:
                    skipped["no pieces labelled"] += 1
                    continue

                tl, tr, br, bl = pts
                white_far = number <= rule["white_far_until"]
                corners = [tr, br, bl, tl] if white_far else [bl, tl, tr, br]
                H = homography(corners)
                readings = set()
                for pull in (0.15, 0.30, 0.45):
                    squares = [board_square(H, anchor, pull) for _, anchor in placed]
                    if None in squares or len(set(squares)) != len(squares):
                        readings.add(None)
                    else:
                        readings.add(board_to_fen_placement(
                            {sq: (piece, 1.0) for sq, (piece, _) in zip(squares, placed)}))
                if len(readings) != 1 or None in readings:
                    skipped["a box is off the board, shares a square or sits on an edge"] += 1
                    continue
                placement = readings.pop()
                yield Sample(name, f"{folder.name}_{img.stem}", img, "test", boxes,
                             [(x / w, y / h) for x, y in corners], placement,
                             board_name if looks_like_a_game(placement)
                             else f"{board_name}-layout")


def load_fen_json(name, cfg, root, skipped):
    """One <photo>.json beside each photo with {"fen": ...} (chesscog's real
    photos; its "white_turn" says whose move it is and is not used)."""
    for label in sorted(root.rglob("*.json")):
        if "__MACOSX" in label.parts:
            continue
        img = next((label.with_suffix(ext) for ext in sorted(IMAGE_EXTS)
                    if label.with_suffix(ext).exists()), None)
        try:
            placement = valid_placement(json.loads(label.read_text()).get("fen", ""))
        except (ValueError, AttributeError):
            placement = None
        if img is None or placement is None:
            skipped["missing image or position"] += 1
            continue
        yield Sample(name, clean_stem("_".join(label.relative_to(root).with_suffix("").parts)),
                     img, "test", placement=placement)


def load_fen_names(name, cfg, root, skipped):
    """Photos named after their position, the eight FEN rows joined by '-'
    (samryan18/chess-dataset); a repeat of a position ends in _2, _3, ..."""
    folder = root / cfg.get("subdir", "")
    for img in sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS):
        placement = placement_from_name(re.sub(r"_\d+$", "", img.stem))
        if placement is None:
            skipped["file name is not a position"] += 1
            continue
        yield Sample(name, clean_stem(img.stem), img, "test", placement=placement)


def load_pgn_games(name, cfg, root, skipped):
    """Games replayed on a board and photographed after every move
    (Rizo-R/chess-cv): games/<game>/orig/<N>.jpg is the board after ply N of
    pgns/<game>.pgn seen from White's side, games/<game>/rev/<N>.jpg the same
    from Black's side (the group). Needs python-chess to replay the moves."""
    try:
        import chess.pgn
    except ImportError:
        sys.exit(f"{name}: replaying the PGN files needs python-chess (pip install chess)")
    for pgn in sorted((root / "pgns").glob("*.pgn")):
        with open(pgn, encoding="utf-8", errors="replace") as handle:
            game = chess.pgn.read_game(handle)
        if game is None or game.errors:
            skipped[f"{pgn.name} cannot be replayed"] += 1
            continue
        board, after = game.board(), {}
        for ply, move in enumerate(game.mainline_moves(), 1):
            board.push(move)
            after[ply] = board.board_fen()
        for side, group in (("orig", "white-side"), ("rev", "black-side")):
            photos = [p for p in (root / "games" / pgn.stem / side).glob("*")
                      if p.suffix.lower() in IMAGE_EXTS and p.stem.isdigit()]
            for img in sorted(photos, key=lambda p: int(p.stem)):
                if int(img.stem) not in after:
                    skipped["photo number beyond the game's last move"] += 1
                    continue
                yield Sample(name, f"{pgn.stem}_{side}_{int(img.stem):03d}", img, "test",
                             placement=after[int(img.stem)], group=group)


# --------------------------------------------------------------------------
# Licenses

def is_shippable(license_):
    text = (license_ or "").lower()
    if "nc" in re.split(r"[^a-z]+", text) or "noncommercial" in text.replace("-", ""):
        return False
    return text == "own" or bool(
        re.search(r"\b(cc by 4|cc0|public domain|mit|apache)\b", text))


def source_shippable(cfg):
    if "shippable" in cfg:
        return bool(cfg["shippable"])
    return is_shippable(cfg.get("license"))


# --------------------------------------------------------------------------
# merge

def training_image(src, stem, max_side, cache):
    """The file to put in the dataset: the photo itself, or with max_side a
    JPEG copy no longer than that on its long side. Phone photos are 9-24 MP
    and training uses 640 px, so decoding them is wasted time. Copies are kept
    in `cache` so the next merge does not make them again."""
    if not max_side:
        return src
    w, h = image_size(src)
    if max(w, h) <= max_side:
        return src
    small = cache / f"{stem}.jpg"
    if not small.exists() or small.stat().st_mtime < src.stat().st_mtime:
        import cv2
        # JPEG can decode straight to 1/2 or 1/4 size, which is much faster.
        reduce = next((r for r in (4, 2) if max(w, h) / r >= max_side), 1)
        flag = {4: cv2.IMREAD_REDUCED_COLOR_4, 2: cv2.IMREAD_REDUCED_COLOR_2,
                1: cv2.IMREAD_COLOR}[reduce]
        img = cv2.imread(str(src), flag)
        if img is None:
            return src
        scale = max_side / max(img.shape[:2])
        img = cv2.resize(img, (round(img.shape[1] * scale), round(img.shape[0] * scale)),
                         interpolation=cv2.INTER_AREA)
        cache.mkdir(parents=True, exist_ok=True)
        partial = cache / f"{stem}.partial.jpg"
        cv2.imwrite(str(partial), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        partial.rename(small)
    return small


def write_sample(sample, out, counts, positions, max_side=0, cache=None):
    stem = f"{sample.source}__{sample.stem}"
    image = training_image(sample.image, stem, max_side, cache)
    if sample.placement:
        # A photo whose labels were both unusable is still a photo to score
        # (often a hard one); it is listed by its own path.
        task = "pieces" if sample.boxes else "corners" if sample.corners else None
        merged = out / task / "images" / sample.split / f"{stem}{image.suffix}" if task else sample.image
        row = {"id": stem,
               "image": str(merged),
               # evaluate.py scores the photo as taken, not the training copy
               "original": str(sample.image),
               "source": sample.source, "split": sample.split,
               "placement": sample.placement}
        if sample.group:
            row["group"] = sample.group
        positions.append(row)
    if sample.boxes:
        base = out / "pieces"
        (base / "labels" / sample.split / f"{stem}.txt").write_text("\n".join(
            f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in sample.boxes))
        link(image, base / "images" / sample.split / f"{stem}{image.suffix}")
        counts[(sample.source, "pieces", sample.split)] += 1
    if sample.corners:
        xs = [p[0] for p in sample.corners]
        ys = [p[1] for p in sample.corners]
        pad = 0.03
        x1, x2 = max(min(xs) - pad, 0), min(max(xs) + pad, 1)
        y1, y2 = max(min(ys) - pad, 0), min(max(ys) + pad, 1)
        kpts = " ".join(f"{x:.6f} {y:.6f} 2" for x, y in sample.corners)
        base = out / "corners"
        (base / "labels" / sample.split / f"{stem}.txt").write_text(
            f"0 {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f} {kpts}")
        link(image, base / "images" / sample.split / f"{stem}{image.suffix}")
        counts[(sample.source, "corners", sample.split)] += 1


def readers(out):
    """Running jobs that read something inside `out`: processes whose command
    line names it (a training run started with --data <out>/corners/data.yaml,
    absolute or as datasets/<name>/...), and the run queue.sh has under way,
    whose command line names only a checkpoint once it has been continued
    after a restart."""
    inside = {f"{out}/", f"{out.parent.name}/{out.name}/"}
    try:
        inside.add(f"{out.relative_to(HERE)}/")
    except ValueError:  # out is reached through a link, e.g. from a worktree
        pass
    listing = subprocess.run(["ps", "-axo", "pid=,command="],
                             capture_output=True, text=True).stdout
    busy = [line.strip() for line in listing.splitlines()
            if any(path in line for path in inside)
            and line.split(None, 1)[0] != str(os.getpid())]
    current = out.parent.parent / "runs" / "queue.current"
    if current.exists():
        line = current.read_text().strip()
        if any(path in line for path in inside):
            busy.append(f"queue.sh: {line}")
    return busy


def merge(args, sources):
    # Test-only sources never share a folder with data a model is trained on:
    # the training scripts read every photo and every positions.json row of
    # the folder they are given.
    unknown = [n for n in args.sources or [] if n not in sources]
    if unknown:
        sys.exit(f"not in sources.yaml: {', '.join(unknown)}")
    test_only = [n for n in args.sources or [] if sources[n].get("test_only")]
    testing = bool(test_only)
    if testing and len(test_only) != len(args.sources):
        sys.exit("test-only sources (" + ", ".join(test_only) + ") cannot be merged "
                 "together with training sources; merge them into a folder of their own.")
    out = (HERE / (args.out or "datasets/merged")).resolve()
    if testing and (not args.out or out.name.startswith("merged")):
        # datasets/merged* is where the training scripts look by default.
        sys.exit("a test-only merge needs a folder of its own that is not called "
                 "merged*, e.g. --out datasets/heldout2")
    if out.exists() and any(out.iterdir()):
        if not (out / "manifest.json").exists():
            sys.exit(f"{out} exists and was not written by this script; "
                     "pick another --out or remove it yourself.")
        was_testing = bool(json.loads((out / "manifest.json").read_text()).get("test_only"))
        if was_testing != testing:
            sys.exit(f"{out} holds a {'test-only' if was_testing else 'training'} set; "
                     "pick another --out.")
        busy = readers(out)
        if busy and not args.force:
            # merge deletes and rebuilds the folder: files would vanish under the job.
            sys.exit(f"{out} is in use by a running job:\n  " + "\n  ".join(
                line[:150] for line in busy)
                + "\nUse another --out, wait for it, or pass --force.")
        shutil.rmtree(out)
    for task in ("pieces", "corners"):
        for kind in ("images", "labels"):
            for split in SPLITS:
                (out / task / kind / split).mkdir(parents=True, exist_ok=True)

    counts, manifest, positions = Counter(), {}, []
    seen = {"names": set(), "hashes": defaultdict(list)}  # roboflow de-dup state
    for name, cfg in sources.items():
        if args.sources and name not in args.sources:
            continue
        kind, root = cfg["kind"], HERE / cfg["dir"]
        if kind == "raw":
            continue
        if cfg.get("test_only") and not testing:
            print(f"- {name}: test only, left out (see sources.yaml)")
            continue
        if kind != "roboflow" and args.shippable and not source_shippable(cfg):
            print(f"- {name}: left out (license {cfg.get('license')})")
            continue
        if not root.exists():
            print(f"- {name}: not on disk ({cfg['dir']}), skipped")
            continue
        skipped = Counter()
        cfg["only_shippable"] = args.shippable
        loader = {"chessred": load_chessred, "chesscog": load_chesscog,
                  "fenit": load_fenit, "native": load_native,
                  "chessq": load_chessq, "ylichess": load_ylichess,
                  "fen_json": load_fen_json, "fen_names": load_fen_names,
                  "pgn_games": load_pgn_games}.get(kind)
        if loader is None and kind != "roboflow":
            sys.exit(f"{name}: no adapter for kind '{kind}'")
        samples = (load_roboflow(name, cfg, root, skipped, seen) if kind == "roboflow"
                   else loader(name, cfg, root, skipped))
        n, stems = 0, set()
        for sample in samples:
            if sample.stem in stems:
                # Writing it would pair the first photo with these labels.
                skipped["file name collides with another image"] += 1
                continue
            stems.add(sample.stem)
            if testing and sample.split != "test":
                sys.exit(f"{name}: test-only, but {sample.stem} is split {sample.split}")
            write_sample(sample, out, counts, positions, args.max_side,
                         HERE / "datasets" / "_resized" / str(args.max_side))
            n += 1
        print(f"- {name}: {n} images" + "".join(
            f"\n    {k}" if k.startswith("note: ") and " gives " in k
            else f"\n    skipped {v}: {k}" for k, v in skipped.items()))
        manifest[name] = {
            "license": cfg.get("project_licenses") or cfg.get("license"),
            "images": n,
            "skipped": dict(skipped),
            **{task: {s: counts[(name, task, s)] for s in SPLITS}
               for task in ("pieces", "corners")},
        }

    (out / "pieces" / "data.yaml").write_text(
        f"path: {out / 'pieces'}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(PIECE_CLASSES)))
    (out / "corners" / "data.yaml").write_text(
        f"path: {out / 'corners'}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "kpt_shape: [4, 3]\nflip_idx: [0, 1, 2, 3]\nnames:\n  0: board\n")
    (out / "manifest.json").write_text(json.dumps(
        {"shippable_only": args.shippable, "max_side": args.max_side,
         "test_only": testing, "sources": manifest}, indent=2))
    (out / "positions.json").write_text(json.dumps(positions))

    print("\ntotals:")
    for task in ("pieces", "corners"):
        print(f"  {task:8}" + "  ".join(
            f"{s} {sum(v for (_, t, sp), v in counts.items() if t == task and sp == s)}"
            for s in SPLITS))
    print(f"\nwrote {out}")


# --------------------------------------------------------------------------
# fetch

def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024


def download(url, dest, size=None):
    have = dest.stat().st_size if dest.exists() else 0
    if dest.exists() and (size is None or have == size):
        return
    need = max((size or 0) - have, 0)
    free = shutil.disk_usage(dest.parent).free
    if need > free:
        sys.exit(f"Not enough disk for {dest.name}: need about {human(need)}, "
                 f"{human(free)} free.")
    print(f"downloading {dest.name}" + (f" ({human(size)})" if size else ""))
    command = ["curl", "-L", "--fail", "-C", "-", "-o", str(dest), url]
    if subprocess.run(command).returncode:
        # A resume the server refuses (e.g. the local file is already longer
        # than the remote one) can never succeed: start this file over once.
        dest.unlink(missing_ok=True)
        subprocess.run(command, check=True)
    if size and dest.stat().st_size != size:
        print(f"note: {dest.name} is {dest.stat().st_size} bytes, sources.yaml says "
              f"{size}; update sources.yaml if the publisher changed the file.")


def fetch_files(cfg, root):
    root.mkdir(parents=True, exist_ok=True)
    for item in cfg.get("downloads", []):
        dest = root / item["file"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        marker = root / f".{item['file']}.extracted"
        files = [item] + item.get("parts", [])
        on_disk = [f for f in files if (root / f["file"]).exists()]
        if marker.exists():
            continue
        # A folder unpacked by hand, with no archive left beside it, counts as
        # done. With an archive present the unpack may have been interrupted,
        # so it is done again.
        if item.get("provides") and (root / item["provides"]).exists() and not on_disk:
            continue
        # Peak disk use: what is left to download, a joined copy of a split
        # archive, and the unpacked files (about the archive's size again).
        total = sum(f.get("size", 0) for f in files)
        left = sum(max(f.get("size", 0) - (root / f["file"]).stat().st_size, 0)
                   if (root / f["file"]).exists() else f.get("size", 0) for f in files)
        need = left + (total * (2 if item.get("parts") else 1) if dest.suffix == ".zip" else 0)
        free = shutil.disk_usage(root).free
        if need > free:
            sys.exit(f"Not enough disk for {item['file']}: need about {human(need)} "
                     f"at peak, {human(free)} free.")
        for part in item.get("parts", []):
            download(part["url"], root / part["file"], part.get("size"))
        download(item["url"], dest, item.get("size"))
        if dest.suffix != ".zip":
            continue
        archive = dest
        if item.get("parts"):  # join a split archive before unpacking
            archive = root / f"joined-{item['file']}"
            subprocess.run(["zip", "-q", "-s", "0", str(dest), "--out", str(archive)],
                           check=True)
        # -o: a file cut short by an interrupted unpack is written again.
        subprocess.run(["unzip", "-q", "-o", str(archive), "-d", str(root)], check=True)
        marker.touch()
        if archive != dest:
            archive.unlink()
        print(f"unpacked {item['file']} (the .zip can be deleted to save space)")


def http_json(url):
    """GET JSON with curl rather than urllib: curl uses the system certificates,
    which python.org builds of Python often lack. The URL goes in on stdin so
    an API key in it never shows up in the process list."""
    done = subprocess.run(
        ["curl", "-sS", "--fail", "-L", "-H", "Accept: application/json", "-K", "-"],
        input=f'url = "{url}"\n', capture_output=True, text=True)
    if done.returncode:
        raise RuntimeError(done.stderr.strip() or f"curl exited {done.returncode}")
    return json.loads(done.stdout)


def roboflow_api(path, key):
    return http_json(f"https://api.roboflow.com/{path}?"
                     + urllib.parse.urlencode({"api_key": key}))


def fetch_mendeley(cfg, root, dry_run):
    """A Mendeley Data dataset is a flat list of files, fetched one by one."""
    info = http_json(f"https://data.mendeley.com/public-api/datasets/{cfg['mendeley_id']}")
    fetch_list(root, [(f["content_details"]["download_url"], root / Path(f["filename"]).name,
                       f.get("size")) for f in info["files"]], dry_run)


def fetch_hf(cfg, root, dry_run):
    """The files of one folder of a Hugging Face dataset, at a fixed commit.
    Works without a token for a public dataset that is not gated."""
    repo, revision = cfg["hf_dataset"], cfg.get("hf_revision", "main")
    folder = cfg.get("hf_path", "").strip("/")
    listing = http_json(f"https://huggingface.co/api/datasets/{repo}/tree/{revision}/{folder}")
    fetch_list(root, [(f"https://huggingface.co/datasets/{repo}/resolve/{revision}/"
                       + urllib.parse.quote(f["path"]), root / f["path"], f.get("size"))
                      for f in listing if f.get("type") == "file"], dry_run)


def fetch_git(cfg, root):
    """Some folders of a git repository (`git_paths`) and the files at its top
    level, without its history or its other folders. The clone's .git is
    removed afterwards: it holds a second copy of every file."""
    marker = root / ".fetched"
    if marker.exists():
        return
    partial = root.parent / f"{root.name}.partial"
    if partial.exists():
        shutil.rmtree(partial)  # an interrupted fetch of ours
    subprocess.run(["git", "clone", "--quiet", "--depth", "1", "--filter=blob:none",
                    "--sparse", cfg["git"], str(partial)], check=True)
    subprocess.run(["git", "-C", str(partial), "sparse-checkout", "set",
                    *cfg.get("git_paths", [])], check=True)
    shutil.rmtree(partial / ".git")
    root.mkdir(parents=True, exist_ok=True)
    for entry in partial.iterdir():
        if not (root / entry.name).exists():
            entry.rename(root / entry.name)
    shutil.rmtree(partial)
    marker.touch()
    print(f"{root.name}: cloned {', '.join(cfg.get('git_paths', []))} of {cfg['git']}")


def fetch_list(root, files, dry_run):
    """Download (url, destination, size or None) files, several at a time."""
    todo = []
    for url, dest, size in files:
        if dest.exists() and (size is None or dest.stat().st_size == size):
            continue
        todo.append((url, dest, size or 0))
    total = sum(size for _, _, size in todo)
    print(f"{root.name}: {len(todo)} of {len(files)} files to download "
          f"({human(total)})")
    if dry_run or not todo:
        return
    root.mkdir(parents=True, exist_ok=True)
    for _, dest, _ in todo:
        dest.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(root).free
    if total > free:
        sys.exit(f"Not enough disk: need {human(total)}, {human(free)} free.")

    def quoted(text):
        return str(text).replace("\\", "\\\\").replace('"', '\\"')
    config = "".join(f'url = "{quoted(u)}"\noutput = "{quoted(d)}"\n' for u, d, _ in todo)
    subprocess.run(["curl", "-L", "--fail", "--retry", "3", "--parallel",
                    "--parallel-max", "6", "-K", "-"], input=config, text=True)
    missing = [d.name for _, d, size in todo
               if not d.exists() or (size and d.stat().st_size != size)]
    if missing:
        sys.exit(f"{len(missing)} files did not download completely "
                 f"(e.g. {missing[0]}); run fetch again to retry them.")


def fetch_roboflow(cfg, root):
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        print("roboflow: set ROBOFLOW_API_KEY (free account -> Settings -> API keys), "
              f"or unzip YOLOv8 exports by hand into {cfg['dir']}/<name>/. Skipped.")
        return
    root.mkdir(parents=True, exist_ok=True)
    for project in cfg.get("projects", []):
        workspace, slug = project["id"].split("/")
        dest = root / slug
        if any(dest.rglob("data.yaml")):
            continue
        try:
            version = project.get("version")
            if not version:
                info = roboflow_api(f"{workspace}/{slug}", key)
                if not info.get("versions"):
                    # Only a published version can be exported.
                    print(f"roboflow: {project['id']} has no published version "
                          f"({info.get('project', {}).get('images', 0)} images in the "
                          "project); nothing to download. Skipped.")
                    continue
                version = max(int(str(v["id"]).rsplit("/", 1)[-1]) for v in info["versions"])
            export = roboflow_api(f"{workspace}/{slug}/{version}/yolov8", key)
            # The first request for a format nobody has exported yet only
            # starts the export; the link appears a little later.
            for _ in range(12):
                if "export" in export:
                    break
                time.sleep(10)
                export = roboflow_api(f"{workspace}/{slug}/{version}/yolov8", key)
            if "export" not in export:
                print(f"roboflow: {project['id']} v{version} is still being prepared "
                      "by Roboflow; run fetch again in a few minutes.")
                continue
            dest.mkdir(parents=True, exist_ok=True)
            archive = dest / "export.zip"
            archive.unlink(missing_ok=True)  # size unknown: a leftover may be partial
            download(export["export"]["link"], archive)
            subprocess.run(["unzip", "-q", "-o", str(archive), "-d", str(dest)], check=True)
            archive.unlink()
            print(f"roboflow: fetched {project['id']} v{version}")
        except Exception as e:  # one bad project should not stop the rest
            print(f"roboflow: {project['id']} failed: {type(e).__name__}: {e}")


def fetch(args, sources):
    for name, cfg in sources.items():
        if args.sources and name not in args.sources:
            continue
        if not args.sources and cfg.get("optional"):
            continue
        root = HERE / cfg["dir"]
        if cfg.get("mendeley_id"):
            fetch_mendeley(cfg, root, args.dry_run)
        elif cfg.get("hf_dataset"):
            fetch_hf(cfg, root, args.dry_run)
        elif cfg.get("git"):
            if not args.dry_run:
                fetch_git(cfg, root)
            elif not (root / ".fetched").exists():
                print(f"{name}: would clone {', '.join(cfg.get('git_paths', []))} "
                      f"of {cfg['git']}")
        elif args.dry_run:
            pending = [i["file"] for i in cfg.get("downloads", [])
                       if not (root / f".{i['file']}.extracted").exists()
                       and not (i.get("provides") and (root / i["provides"]).exists()
                                and not (root / i["file"]).exists())
                       and not (not i["file"].endswith(".zip") and (root / i["file"]).exists()
                                and (root / i["file"]).stat().st_size == i.get("size"))]
            pending = pending or [p["id"] for p in cfg.get("projects", [])]
            if pending:
                print(f"{name}: would fetch {', '.join(pending)}")
        elif cfg["kind"] == "roboflow":
            fetch_roboflow(cfg, root)
        elif cfg.get("downloads"):
            fetch_files(cfg, root)


def list_sources(sources):
    for name, cfg in sources.items():
        root = HERE / cfg["dir"]
        size = cfg.get("approx_bytes", 0) + sum(
            i.get("size", 0) + sum(p.get("size", 0) for p in i.get("parts", []))
            for i in cfg.get("downloads", []))
        print(f"{name:16} {cfg['kind']:9} {human(size) if size else '-':>9}  "
              f"{'on disk' if root.exists() else 'missing':8} "
              f"{'TEST ONLY  ' if cfg.get('test_only') else ''}"
              f"{'optional  ' if cfg.get('optional') else ''}{cfg.get('license', 'per project')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=str(HERE / "sources.yaml"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    f = sub.add_parser("fetch")
    f.add_argument("sources", nargs="*")
    f.add_argument("--dry-run", action="store_true", help="show what would be downloaded")
    m = sub.add_parser("merge")
    m.add_argument("--out", help="default: datasets/merged; required for test-only sources")
    m.add_argument("--sources", nargs="*",
                   help="default: every source on disk that is not test-only")
    m.add_argument("--shippable", action="store_true",
                   help="only data a commercially distributed model may be trained on")
    m.add_argument("--force", action="store_true",
                   help="rebuild --out even though a running job is reading it")
    m.add_argument("--max-side", type=int, default=0, metavar="PX",
                   help="write JPEG copies no longer than PX on the long side instead "
                        "of links to the full-size photos (e.g. 1280); faster training")
    args = ap.parse_args()

    sources = yaml.safe_load(Path(args.config).read_text())["sources"]
    {"list": lambda: list_sources(sources),
     "fetch": lambda: fetch(args, sources),
     "merge": lambda: merge(args, sources)}[args.cmd]()


if __name__ == "__main__":
    main()
