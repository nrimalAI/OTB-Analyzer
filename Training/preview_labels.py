"""Draw the merged dataset's labels on a few images per source, to check them
by eye after a download or an adapter change.

    python preview_labels.py                       # 6 images per source
    python preview_labels.py --per-source 12 --sources fenit
    python preview_labels.py --check               # numbers only, every image

Each preview shows the piece boxes with their FEN letter (white pieces in
yellow, black in blue), the board outline with the corner names, the square
grid those corners imply, and the squares a1 and h8 named. Where the position
is known (positions.json) it is printed above the picture.

A photo that has a position and no boxes or corners (most of the test-only
sources) is shown beside two diagrams of that position, one as White sees the
board and one as Black does, to compare with the photo by eye.

--check maps every labelled box through the labelled corners and counts how
many land on a square that holds that piece in the known position. Low numbers
mean wrong corners, wrong corner order, or a position that is not the photo's.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from chess_pipeline import (
    FEN_LETTER, PIECE_CLASSES, board_square, fen_placement_to_board, homography,
)
from prepare_data import image_size

HERE = Path(__file__).resolve().parent
CORNER_NAMES = ("a1", "a8", "h8", "h1")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def read_boxes(path, w, h):
    """[(class_name, (x1, y1, x2, y2))] in pixels."""
    out = []
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cx, cy, bw, bh = (float(v) for v in p[1:5])
        out.append((PIECE_CLASSES[int(p[0])],
                    ((cx - bw / 2) * w, (cy - bh / 2) * h,
                     (cx + bw / 2) * w, (cy + bh / 2) * h)))
    return out


def read_corners(path, w, h):
    """[(x, y)] * 4 in pixels, order a1 a8 h8 h1, or None."""
    if not path.exists():
        return None
    p = path.read_text().split()
    if len(p) < 17:
        return None
    k = [float(v) for v in p[5:17]]
    return [(k[i] * w, k[i + 1] * h) for i in range(0, 12, 3)]


def collect(merged):
    """{source: [(stem, image, pieces label, corners label)]} over all splits."""
    by_source = defaultdict(list)
    for split in ("train", "val", "test"):
        for task in ("pieces", "corners"):
            for label in sorted((merged / task / "labels" / split).glob("*.txt")):
                image = next((merged / task / "images" / split / (label.stem + e)
                              for e in IMAGE_EXTS
                              if (merged / task / "images" / split / (label.stem + e)).exists()),
                             None)
                if image is None:
                    continue
                by_source[label.stem.split("__")[0]].append((label.stem, split, task, image, label))
    out = defaultdict(dict)
    for source, rows in by_source.items():
        for stem, split, task, image, label in rows:
            entry = out[source].setdefault(stem, {"split": split, "image": image})
            entry[task] = label
    return out


def hit_count(boxes, corners, placement):
    """(boxes on a square holding that piece, boxes) for the given corners."""
    board = fen_placement_to_board(placement)
    H = homography(corners)
    hits = sum(board.get(board_square(H, ((b[0] + b[2]) / 2, b[3]))) == FEN_LETTER[cls]
               for cls, b in boxes)
    return hits, len(boxes)


def draw(entry, stem, placement, width):
    img = cv2.imread(str(entry["image"]))
    h0, w0 = img.shape[:2]
    scale = width / w0
    img = cv2.resize(img, (width, round(h0 * scale)), interpolation=cv2.INTER_AREA)
    font = cv2.FONT_HERSHEY_SIMPLEX

    def text(s, org, color, size=0.6):
        org = (int(org[0]), int(org[1]))
        cv2.putText(img, s, org, font, size, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, s, org, font, size, color, 1, cv2.LINE_AA)

    corners = read_corners(entry["corners"], w0, h0) if "corners" in entry else None
    if corners:
        Hinv = np.linalg.inv(homography(corners))

        def to_image(bx, by):
            p = Hinv @ np.array([bx, by, 1.0])
            return int(p[0] / p[2] * scale), int(p[1] / p[2] * scale)

        for i in range(9):
            cv2.line(img, to_image(i, 0), to_image(i, 8), (0, 255, 0), 1, cv2.LINE_AA)
            cv2.line(img, to_image(0, i), to_image(8, i), (0, 255, 0), 1, cv2.LINE_AA)
        for name, (x, y) in zip(CORNER_NAMES, corners):
            cv2.circle(img, (int(x * scale), int(y * scale)), 7, (0, 0, 255), -1)
            text(name, (x * scale + 9, y * scale - 9), (0, 0, 255), 0.9)
        text("a1", to_image(0.2, 0.35), (0, 255, 0), 0.5)
        text("h8", to_image(7.2, 7.35), (0, 255, 0), 0.5)

    boxes = read_boxes(entry["pieces"], w0, h0) if "pieces" in entry else []
    for cls, (x1, y1, x2, y2) in boxes:
        color = (0, 220, 255) if cls.startswith("white") else (255, 120, 0)
        cv2.rectangle(img, (int(x1 * scale), int(y1 * scale)),
                      (int(x2 * scale), int(y2 * scale)), color, 2)
        text(FEN_LETTER[cls], (x1 * scale + 2, y1 * scale - 4), color, 0.7)

    note = f"{entry['split']}  boxes={len(boxes)}  corners={'yes' if corners else 'NO'}"
    if placement and corners and boxes:
        hits, n = hit_count(boxes, corners, placement)
        note += f"  on-square={hits}/{n}"
    bar = np.zeros((66, img.shape[1], 3), np.uint8)
    cv2.putText(bar, stem[:110], (6, 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(bar, note, (6, 40), font, 0.5, (160, 255, 160), 1, cv2.LINE_AA)
    cv2.putText(bar, f"position: {placement or '-'}", (6, 60), font, 0.5,
                (160, 200, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def diagram(placement, side, from_black=False):
    """The position as a square picture `side` pixels wide: white pieces are
    light discs, black pieces dark ones, each with its letter."""
    cell = side // 9
    img = np.full((side, side, 3), 40, np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    board = fen_placement_to_board(placement)
    for f in range(8):
        for r in range(8):
            col, row = (7 - f, r) if from_black else (f, 7 - r)
            x, y = cell // 2 + col * cell, row * cell
            light = (f + r) % 2 == 1
            cv2.rectangle(img, (x, y), (x + cell, y + cell),
                          (200, 220, 235) if light else (95, 135, 170), -1)
            letter = board.get((f, r))
            if letter:
                white = letter.isupper()
                centre = (x + cell // 2, y + cell // 2)
                cv2.circle(img, centre, int(cell * 0.38), (255, 255, 255) if white else (0, 0, 0), -1)
                cv2.circle(img, centre, int(cell * 0.38), (0, 0, 0) if white else (255, 255, 255), 1)
                size = cell / 55
                (tw, th), _ = cv2.getTextSize(letter.upper(), font, size, 2)
                cv2.putText(img, letter.upper(), (centre[0] - tw // 2, centre[1] + th // 2), font,
                            size, (0, 0, 0) if white else (255, 255, 255), 2, cv2.LINE_AA)
    for i in range(8):
        file, rank = ("hgfedcba" if from_black else "abcdefgh")[i], (i + 1 if from_black else 8 - i)
        cv2.putText(img, file, (cell // 2 + i * cell + cell // 3, 8 * cell + cell // 2), font,
                    cell / 80, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, str(rank), (cell // 8, i * cell + cell * 2 // 3), font,
                    cell / 80, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "from Black's side" if from_black else "from White's side",
                (cell // 2, side - cell // 8), font, cell / 90, (160, 255, 160), 1, cv2.LINE_AA)
    return img


def draw_position_only(row, width):
    """The photo with its position drawn beside it from both sides."""
    img = cv2.imread(str(row.get("original") or row["image"]))
    h0, w0 = img.shape[:2]
    photo_w = width * 2 // 3
    img = cv2.resize(img, (photo_w, round(h0 * photo_w / w0)), interpolation=cv2.INTER_AREA)
    side = width - photo_w
    panel = np.vstack([diagram(row["placement"], side), diagram(row["placement"], side, True)])
    height = max(img.shape[0], panel.shape[0])
    canvas = np.zeros((height, width, 3), np.uint8)
    canvas[:img.shape[0], :photo_w] = img
    canvas[:panel.shape[0], photo_w:] = panel
    font = cv2.FONT_HERSHEY_SIMPLEX
    bar = np.zeros((66, width, 3), np.uint8)
    cv2.putText(bar, row["id"][:110], (6, 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(bar, f"{row['split']}  position only" + (f"  group={row['group']}" if row.get("group") else ""),
                (6, 40), font, 0.5, (160, 255, 160), 1, cv2.LINE_AA)
    cv2.putText(bar, f"position: {row['placement']}", (6, 60), font, 0.5,
                (160, 200, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, canvas])


def check(by_source, placements):
    for source, entries in sorted(by_source.items()):
        total = boxes_n = hits_n = both = low = 0
        worst = []
        for stem, entry in entries.items():
            total += 1
            placement = placements.get(stem)
            if not (placement and "pieces" in entry and "corners" in entry):
                continue
            w, h = image_size(entry["image"])
            boxes = read_boxes(entry["pieces"], w, h)
            hits, n = hit_count(boxes, read_corners(entry["corners"], w, h), placement)
            both += 1
            boxes_n += n
            hits_n += hits
            want = len(fen_placement_to_board(placement))
            if hits < n or n != want:
                low += 1
                worst.append((hits / max(n, 1), hits, n, want, stem))
        print(f"{source}: {total} images, {both} with boxes + corners + position")
        if both:
            print(f"  boxes on their square: {hits_n}/{boxes_n} ({hits_n / boxes_n:.2%})")
            print(f"  images with a miss or a box-count mismatch: {low}")
            for rate, hits, n, want, stem in sorted(worst)[:8]:
                print(f"    {hits}/{n} on square, position has {want} pieces: {stem}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--merged", default="datasets/merged")
    ap.add_argument("--out", default=None, help="default: <merged>/preview")
    ap.add_argument("--per-source", type=int, default=6)
    ap.add_argument("--sources", nargs="*")
    ap.add_argument("--stems", nargs="*", help="draw these images instead of a sample")
    ap.add_argument("--width", type=int, default=1100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true", help="print numbers, draw nothing")
    args = ap.parse_args()

    merged = (HERE / args.merged).resolve()
    by_source = collect(merged)
    rows = json.loads((merged / "positions.json").read_text())
    placements = {Path(r["image"]).stem: r["placement"] for r in rows}
    # Photos with a position and no label files: nothing to draw on them.
    position_only = defaultdict(dict)
    for r in rows:
        r.setdefault("id", f"{r['source']}__{Path(r['image']).stem}")
        if Path(r["image"]).stem not in by_source.get(r["source"], {}):
            position_only[r["source"]][r["id"]] = r
    if args.sources:
        by_source = {k: v for k, v in by_source.items() if k in args.sources}
        position_only = {k: v for k, v in position_only.items() if k in args.sources}
    if args.check:
        check(by_source, placements)
        return

    out = Path(args.out) if args.out else merged / "preview"
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    for source, entries in sorted(by_source.items()):
        stems = sorted(entries)
        if args.stems:
            picked = [s for s in stems if s in args.stems]
        else:
            picked = rng.sample(stems, min(args.per_source, len(stems)))
        for stem in picked:
            cv2.imwrite(str(out / f"{stem}.jpg"),
                        draw(entries[stem], stem, placements.get(stem), args.width),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"{source}: wrote {len(picked)} previews")
    for source, entries in sorted(position_only.items()):
        stems = sorted(entries)
        if args.stems:
            picked = [s for s in stems if s in args.stems]
        else:
            picked = rng.sample(stems, min(args.per_source, len(stems)))
        for stem in picked:
            cv2.imwrite(str(out / f"{stem}.jpg"), draw_position_only(entries[stem], args.width),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"{source}: wrote {len(picked)} previews (position only)")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
