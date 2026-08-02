"""Convert ChessReD2K annotations into the two YOLO datasets we train.

ChessReD's annotations.json is COCO-flavored:
  images:      [{id, path/file_name, width, height, ...}]
  annotations: {pieces:  [{image_id, category_id, chessboard_position, bbox?}],
                corners: [{image_id, corners: {a1,a8,h8,h1} or similar}]}
  categories:  [{id, name}]   # e.g. "white_pawn"

Only the ChessReD2K subset carries bboxes + corners; images without them are
skipped automatically. Field names are probed defensively — run with --inspect
to dump the actual schema if conversion complains.

Outputs (YOLO layout):
  out/pieces/{images,labels}/{train,val,test}  + data.yaml   (detect, 12 classes)
  out/corners/{images,labels}/{train,val,test} + data.yaml   (pose, 1 class, 4 kpts)

Splits are by game id (from the image path) so no position leaks across splits:
game % 10 in {0..6} -> train, {7} -> val, {8,9} -> test.
"""

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from chess_pipeline import PIECE_CLASSES

CORNER_ORDER = ["a1", "a8", "h8", "h1"]


def inspect(data):
    def describe(obj, depth=0, name="root"):
        pad = "  " * depth
        if isinstance(obj, dict):
            print(f"{pad}{name}: dict keys={list(obj.keys())[:12]}")
            for k, v in list(obj.items())[:6]:
                describe(v, depth + 1, k)
        elif isinstance(obj, list):
            print(f"{pad}{name}: list len={len(obj)}")
            if obj:
                describe(obj[0], depth + 1, f"{name}[0]")
        else:
            print(f"{pad}{name}: {type(obj).__name__} = {str(obj)[:60]}")

    describe(data)


def image_index(data):
    idx = {}
    for im in data["images"]:
        path = im.get("path") or im.get("file_name")
        idx[im["id"]] = {
            "path": path,
            "width": im.get("width"),
            "height": im.get("height"),
        }
    return idx


def game_of(path):
    # Paths look like images/G000/G000_IMG000.jpg — the game id is the G-number.
    for part in Path(path).parts:
        if part.upper().startswith("G") and part[1:4].isdigit():
            return int(part[1:4])
    # Fallback: hash the parent directory.
    return abs(hash(Path(path).parent.name)) % 1000


def split_of(game):
    m = game % 10
    if m <= 6:
        return "train"
    if m == 7:
        return "val"
    return "test"


def extract_corners(entry):
    """Return [(x,y) for a1,a8,h8,h1] from a corners annotation, best-effort."""
    c = entry.get("corners", entry)
    if isinstance(c, dict):
        try:
            return [tuple(c[k]) for k in CORNER_ORDER]
        except KeyError:
            pass
        # Some variants use bottom_left/top_left/... relative to White.
        alias = {
            "a1": ["bottom_left", "bl"], "a8": ["top_left", "tl"],
            "h8": ["top_right", "tr"], "h1": ["bottom_right", "br"],
        }
        out = []
        for key in CORNER_ORDER:
            for a in [key] + alias[key]:
                if a in c:
                    out.append(tuple(c[a]))
                    break
        if len(out) == 4:
            return out
    if isinstance(c, list) and len(c) == 4:
        return [tuple(p) for p in c]
    if isinstance(c, list) and len(c) == 8:
        return [(c[i], c[i + 1]) for i in range(0, 8, 2)]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--images", help="root dir the annotation paths are relative to")
    ap.add_argument("--out", default="datasets/yolo")
    ap.add_argument("--inspect", action="store_true", help="dump schema and exit")
    args = ap.parse_args()

    data = json.loads(Path(args.annotations).read_text())
    if args.inspect:
        inspect(data)
        return

    if not args.images:
        ap.error("--images is required (unless --inspect)")

    images_root = Path(args.images)
    out = Path(args.out)
    categories = {c["id"]: c["name"] for c in data["categories"]}
    imgs = image_index(data)

    ann = data["annotations"]
    piece_anns = ann.get("pieces", [])
    corner_anns = ann.get("corners", [])

    # Group piece boxes per image (only those that actually have a bbox).
    boxes = defaultdict(list)
    for p in piece_anns:
        if "bbox" in p and p["bbox"]:
            boxes[p["image_id"]].append(p)

    corners = {}
    for c in corner_anns:
        pts = extract_corners(c)
        if pts is not None:
            corners[c["image_id"]] = pts

    usable = sorted(set(boxes) & set(corners))
    if not usable:
        print("No images with both bboxes and corners found.", file=sys.stderr)
        print("Run with --inspect and check the annotation schema.", file=sys.stderr)
        sys.exit(1)
    print(f"{len(usable)} images with boxes + corners "
          f"({len(boxes)} with boxes, {len(corners)} with corners)")

    class_id = {name: i for i, name in enumerate(PIECE_CLASSES)}
    counts = defaultdict(int)
    skipped_missing = 0

    for split in ("train", "val", "test"):
        for task in ("pieces", "corners"):
            (out / task / "images" / split).mkdir(parents=True, exist_ok=True)
            (out / task / "labels" / split).mkdir(parents=True, exist_ok=True)

    for image_id in usable:
        meta = imgs[image_id]
        src = images_root / meta["path"]
        if not src.exists():
            # Annotation paths sometimes already include the images/ prefix.
            alt = images_root / Path(*Path(meta["path"]).parts[1:])
            if alt.exists():
                src = alt
            else:
                skipped_missing += 1
                continue
        w, h = meta["width"], meta["height"]
        if not w or not h:
            skipped_missing += 1
            continue

        split = split_of(game_of(meta["path"]))
        stem = f"img{image_id:06d}"
        counts[split] += 1

        # --- pieces (detect) ---
        lines = []
        for p in boxes[image_id]:
            name = categories[p["category_id"]]
            if name not in class_id:
                continue
            x, y, bw, bh = p["bbox"]  # COCO xywh
            cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
            lines.append(
                f"{class_id[name]} {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}")
        (out / "pieces" / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))
        link(src, out / "pieces" / "images" / split / f"{stem}{src.suffix}")

        # --- corners (pose): one "board" instance, 4 keypoints ---
        pts = corners[image_id]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        pad = 0.03
        x1 = max(min(xs) / w - pad, 0)
        x2 = min(max(xs) / w + pad, 1)
        y1 = max(min(ys) / h - pad, 0)
        y2 = min(max(ys) / h + pad, 1)
        kpts = " ".join(f"{px / w:.6f} {py / h:.6f} 2" for px, py in pts)
        line = (f"0 {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} "
                f"{x2 - x1:.6f} {y2 - y1:.6f} {kpts}")
        (out / "corners" / "labels" / split / f"{stem}.txt").write_text(line)
        link(src, out / "corners" / "images" / split / f"{stem}{src.suffix}")

    if skipped_missing:
        print(f"skipped {skipped_missing} images (file or size missing)")
    print("split sizes:", dict(counts))

    (out / "pieces" / "data.yaml").write_text(
        f"path: {(out / 'pieces').resolve()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(PIECE_CLASSES)))

    (out / "corners" / "data.yaml").write_text(
        f"path: {(out / 'corners').resolve()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        "kpt_shape: [4, 3]\n"
        "flip_idx: [0, 1, 2, 3]\n"  # no horizontal flip augmentation (see train.py)
        "names:\n  0: board\n")
    print(f"wrote YOLO datasets under {out}/")


def link(src, dst):
    if dst.exists():
        return
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


if __name__ == "__main__":
    main()
