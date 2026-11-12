"""Download several chess datasets and merge them into the two YOLO datasets
we train (pieces: detect, 12 classes; corners: pose, 4 keypoints).

    python build_dataset.py list                 # sources, sizes, what is on disk
    python build_dataset.py fetch                # download everything not optional
    python build_dataset.py fetch chesscog       # ...or just the named sources
    python build_dataset.py merge                # -> datasets/merged/{pieces,corners}
    python build_dataset.py merge --shippable    # leave out NonCommercial data

Sources are declared in sources.yaml. `merge` uses whatever is on disk and
skips the rest, so it works with a partial download.

What each source contributes:
  chessred   boxes + corners (ChessReD2K subset), official train/val/test split
  chesscog   boxes + corners (rendered), its own train/val/test split
  fenit      boxes + corners (real photos); 20% held out as test, 10% val
  roboflow   boxes only; near-duplicate photos across projects are dropped;
             90% train / 10% val by hash
  native     whatever labels are present, train only

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
import re
import shutil
import subprocess
import sys
import urllib.parse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml

from chess_pipeline import (
    FEN_LETTER, PIECE_CLASSES, board_square, fen_placement_to_board, homography,
)
from prepare_data import (
    extract_corners, image_index, link, normalize_category, official_splits,
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


def image_size(path):
    from PIL import Image
    with Image.open(path) as im:
        return im.size


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
        out = []
        for p in boxes[image_id]:
            cls = categories[p["category_id"]]
            if cls in CLASS_ID:
                x, y, bw, bh = p["bbox"]
                out.append((CLASS_ID[cls], (x + bw / 2) / w, (y + bh / 2) / h,
                            bw / w, bh / h))
        yield Sample(name, f"img{image_id:06d}", src, split, out,
                     [(x / w, y / h) for x, y in corners[image_id]])


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
        for js in sorted((root / split).glob("*.json")):
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
    """Fenit names each photo after its position: the eight FEN rows joined by
    '_', sometimes followed by ' w' or ' b'. Returns the placement, or None."""
    rows = stem.split()[0].split("_") if stem.strip() else []
    if len(rows) != 8:
        return None
    for row in rows:
        if not re.fullmatch(r"[KQRBNPkqrbnp1-8]+", row):
            return None
        if sum(int(c) if c.isdigit() else 1 for c in row) != 8:
            return None
    return "/".join(rows)


def load_fenit(name, cfg, root, skipped):
    """Fenit (Mendeley Data, CC BY 4.0): real photos, one YOLO-pose .txt each.
    The 'Board' line carries the four corners as keypoints; piece lines carry a
    box (their own keypoints are ignored). Corner order is not documented, so
    it is recovered from the position in the file name.

    There is no official split, and photos of one position must stay together,
    so the split is a hash of the position."""
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
        if placement is None:
            skipped["file name is not a position (boxes kept, no corners)"] += 1
        elif board_pts is None:
            skipped["no board outline labelled (boxes kept)"] += 1
        else:
            if len(boxes) != len(fen_placement_to_board(placement)):
                skipped["note: box count differs from the position in the name"] += 1
            ordered = orient_corners_by_position(board_pts, lettered, placement)
            if ordered is None:
                skipped["corner order ambiguous (boxes kept)"] += 1
            else:
                corners = [(x / w, y / h) for x, y in ordered]

        bucket = int(hashlib.md5((placement or label.stem).encode()).hexdigest(), 16) % 100
        split = "test" if bucket < test_cut else "val" if bucket < val_cut else "train"
        yield Sample(name, re.sub(r"[^A-Za-z0-9_.-]+", "-", label.stem), img, split,
                     boxes, corners, placement)


def load_roboflow(name, cfg, root, skipped, seen):
    """root holds one YOLO export per sub-folder. `seen` carries de-dup state
    across projects, since Universe is full of forks of the same photos."""
    for project in sorted(p for p in root.iterdir() if p.is_dir()):
        data_yaml = next(iter(sorted(project.rglob("data.yaml"))), None)
        if data_yaml is None:
            continue
        meta = yaml.safe_load(data_yaml.read_text())
        names = meta.get("names", {})
        if isinstance(names, list):
            names = dict(enumerate(names))
        class_map = {i: canonical_class(n) for i, n in names.items()}
        ignorable = {i for i, n in names.items()
                     if normalize_category(str(n)).lower() in NON_PIECE}
        license_ = (meta.get("roboflow") or {}).get("license") or read_license(project)
        cfg.setdefault("project_licenses", {})[project.name] = license_
        if cfg.get("only_shippable") and not is_shippable(license_):
            skipped[f"{project.name}: license '{license_}' not shippable"] += 1
            continue

        for label in sorted(data_yaml.parent.rglob("labels/*.txt")):
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
                if len(vals) > 4:  # segmentation polygon -> its bounding box
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
            keys = {("name", original, len(boxes)), ("hash", dhash(img))}
            if keys & seen:
                skipped["duplicate of another image"] += 1
                continue
            seen |= keys
            bucket = int(hashlib.md5(original.encode()).hexdigest(), 16) % 10
            yield Sample(name, f"{project.name}_{label.stem}", img,
                         "val" if bucket == 0 else "train", boxes, None)


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
# Licenses

def is_shippable(license_):
    text = (license_ or "").lower()
    if "nc" in re.split(r"[^a-z]+", text) or "noncommercial" in text.replace("-", ""):
        return False
    return text == "own" or any(
        k in text for k in ("cc by 4", "cc0", "public domain", "mit", "apache"))


def source_shippable(cfg):
    if "shippable" in cfg:
        return bool(cfg["shippable"])
    return is_shippable(cfg.get("license"))


# --------------------------------------------------------------------------
# merge

def write_sample(sample, out, counts, positions):
    stem = f"{sample.source}__{sample.stem}"
    if sample.boxes:
        base = out / "pieces"
        (base / "labels" / sample.split / f"{stem}.txt").write_text("\n".join(
            f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in sample.boxes))
        merged = base / "images" / sample.split / f"{stem}{sample.image.suffix}"
        link(sample.image, merged)
        counts[(sample.source, "pieces", sample.split)] += 1
        if sample.placement:
            positions.append({"image": str(merged), "source": sample.source,
                              "split": sample.split, "placement": sample.placement})
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
        link(sample.image, base / "images" / sample.split / f"{stem}{sample.image.suffix}")
        counts[(sample.source, "corners", sample.split)] += 1


def merge(args, sources):
    out = (HERE / args.out).resolve()
    if out.exists() and any(out.iterdir()):
        if not (out / "manifest.json").exists():
            sys.exit(f"{out} exists and was not written by this script; "
                     "pick another --out or remove it yourself.")
        shutil.rmtree(out)
    for task in ("pieces", "corners"):
        for kind in ("images", "labels"):
            for split in SPLITS:
                (out / task / kind / split).mkdir(parents=True, exist_ok=True)

    counts, manifest, seen, positions = Counter(), {}, set(), []
    for name, cfg in sources.items():
        if args.sources and name not in args.sources:
            continue
        kind, root = cfg["kind"], HERE / cfg["dir"]
        if kind == "raw":
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
                  "fenit": load_fenit, "native": load_native}.get(kind)
        samples = (load_roboflow(name, cfg, root, skipped, seen) if kind == "roboflow"
                   else loader(name, cfg, root, skipped))
        n = 0
        for sample in samples:
            write_sample(sample, out, counts, positions)
            n += 1
        print(f"- {name}: {n} images" + "".join(
            f"\n    skipped {v}: {k}" for k, v in skipped.items()))
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
        {"shippable_only": args.shippable, "sources": manifest}, indent=2))
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
    # Room for the rest of the download plus the unpacked copy.
    need = ((size or 0) - have) + (size or 0)
    free = shutil.disk_usage(dest.parent).free
    if need > free:
        sys.exit(f"Not enough disk for {dest.name}: need about {human(need)}, "
                 f"{human(free)} free.")
    print(f"downloading {dest.name}" + (f" ({human(size)})" if size else ""))
    subprocess.run(["curl", "-L", "--fail", "-C", "-", "-o", str(dest), url], check=True)


def fetch_files(cfg, root):
    root.mkdir(parents=True, exist_ok=True)
    for item in cfg.get("downloads", []):
        dest = root / item["file"]
        marker = root / f".{item['file']}.extracted"
        if marker.exists() or (item.get("provides") and (root / item["provides"]).exists()):
            continue
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
        subprocess.run(["unzip", "-q", "-n", str(archive), "-d", str(root)], check=True)
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
    todo = []
    for f in info["files"]:
        dest, size = root / Path(f["filename"]).name, f.get("size")
        if dest.exists() and (size is None or dest.stat().st_size == size):
            continue
        todo.append((f["content_details"]["download_url"], dest, size or 0))
    total = sum(size for _, _, size in todo)
    print(f"{root.name}: {len(todo)} of {len(info['files'])} files to download "
          f"({human(total)})")
    if dry_run or not todo:
        return
    root.mkdir(parents=True, exist_ok=True)
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
                version = max(int(str(v["id"]).rsplit("/", 1)[-1]) for v in info["versions"])
            export = roboflow_api(f"{workspace}/{slug}/{version}/yolov8", key)
            dest.mkdir(parents=True, exist_ok=True)
            archive = dest / "export.zip"
            download(export["export"]["link"], archive)
            subprocess.run(["unzip", "-q", "-n", str(archive), "-d", str(dest)], check=True)
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
        elif args.dry_run:
            pending = [i["file"] for i in cfg.get("downloads", [])
                       if not (root / f".{i['file']}.extracted").exists()
                       and not (i.get("provides") and (root / i["provides"]).exists())
                       and not ((root / i["file"]).exists()
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
    m.add_argument("--out", default="datasets/merged")
    m.add_argument("--sources", nargs="*")
    m.add_argument("--shippable", action="store_true",
                   help="only data a commercially distributed model may be trained on")
    args = ap.parse_args()

    sources = yaml.safe_load(Path(args.config).read_text())["sources"]
    {"list": lambda: list_sources(sources),
     "fetch": lambda: fetch(args, sources),
     "merge": lambda: merge(args, sources)}[args.cmd]()


if __name__ == "__main__":
    main()
