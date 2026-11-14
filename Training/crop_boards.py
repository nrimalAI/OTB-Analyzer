"""Cut every photo of a merged dataset down to its board, for a piece model
that looks at the board only.

    python crop_boards.py --src datasets/merged_all --out datasets/merged_all_crop

Why: the piece model sees the whole photo at 640 px, and on a phone photo
the board is often half of it, so a piece is 20-40 px wide. What it gets
wrong on an unseen set is small dark pieces seen from above. The app knows
where the board is before it looks for pieces (corner model, grid fit), so
it can hand the piece model the board alone: chess_pipeline.board_crop. This
script makes the training data for that: for each photo of <src>/pieces
that has a corner label in <src>/corners (same split, same file name), the
crop round those corners, with the piece boxes moved and clipped to it.
Photos without a corner label are left out (Roboflow photos, renders whose
board is cut by the frame). Boxes that lose more than 60% of their area to
the crop are dropped.

evaluate.py --crop-board scores a model trained on the result.
"""

import argparse
from pathlib import Path

import cv2
import yaml

from chess_pipeline import board_crop

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="a folder written by build_dataset.py merge")
    ap.add_argument("--out", required=True)
    ap.add_argument("--quality", type=int, default=92)
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    spec = yaml.safe_load((src / "pieces" / "data.yaml").read_text())

    counts = {}
    for split in ("train", "val", "test"):
        images = src / "pieces" / "images" / split
        if not images.is_dir():
            continue
        (out / "pieces" / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "pieces" / "labels" / split).mkdir(parents=True, exist_ok=True)
        done = skipped = 0
        for image in sorted(images.iterdir()):
            if image.suffix.lower() not in IMAGE_EXTS:
                continue
            corner_file = src / "corners" / "labels" / split / f"{image.stem}.txt"
            box_file = src / "pieces" / "labels" / split / f"{image.stem}.txt"
            if not corner_file.exists() or not box_file.exists():
                skipped += 1
                continue
            target = out / "pieces" / "images" / split / f"{image.stem}.jpg"
            label = out / "pieces" / "labels" / split / f"{image.stem}.txt"
            if target.exists() and label.exists():
                done += 1
                continue
            img = cv2.imread(str(image))
            if img is None:
                skipped += 1
                continue
            h, w = img.shape[:2]
            v = [float(x) for x in corner_file.read_text().split()[5:17]]
            corners = [(v[i] * w, v[i + 1] * h) for i in range(0, 12, 3)]
            rect = board_crop(corners, w, h)
            if rect is None:
                skipped += 1
                continue
            x0, y0, x1, y1 = rect
            cw, ch = x1 - x0, y1 - y0
            lines = []
            for line in box_file.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cx, cy, bw, bh = (float(p) for p in parts[1:5])
                a, b, c, d = (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h
                ca, cb, cc, cd = max(a, x0), max(b, y0), min(c, x1), min(d, y1)
                if cc <= ca or cd <= cb or (cc - ca) * (cd - cb) < 0.4 * (c - a) * (d - b):
                    continue
                lines.append(f"{parts[0]} {((ca + cc) / 2 - x0) / cw:.6f} {((cb + cd) / 2 - y0) / ch:.6f} "
                             f"{(cc - ca) / cw:.6f} {(cd - cb) / ch:.6f}")
            cv2.imwrite(str(target), img[y0:y1, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, args.quality])
            label.write_text("\n".join(lines) + ("\n" if lines else ""))
            done += 1
        counts[split] = (done, skipped)
        print(f"{split}: {done} photos cropped, {skipped} left out (no corner label)")

    data = {"path": str((out / "pieces").resolve()), "train": "images/train", "val": "images/val",
            "names": spec["names"]}
    if "test" in counts:
        data["test"] = "images/test"
    (out / "pieces" / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
