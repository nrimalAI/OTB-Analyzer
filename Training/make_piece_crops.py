"""Cut every labelled piece out of a piece training set into an image folder
for a piece classifier (a second stage after the piece detector).

    python make_piece_crops.py --list datasets/merged_all4_shape1/pieces/train.txt \
        --val-dir datasets/merged_all4/pieces/images/val --out datasets/crops1 \
        --render-keep 0.3 --jitter 2

Each crop is a square around the box (side = the box's longer side times
--margin), shifted and scaled a little at random (--jitter crops per piece;
the detector's boxes in the app are not the labelled ones), cut from the photo
as the training set has it (1280-px copies; the app draws a photo into at most
1280 px too) and resized to --size. Out: <out>/train/<class>/*.jpg and
<out>/val/<class>/*.jpg, the layout `yolo classify` trains on.

--render-keep keeps that share of our render photos (render_*; a photo is kept
or dropped whole), so the
real photos are not drowned. Nothing from Fenit board B or the second test set
is in the piece training sets, so nothing of them gets in here either.
"""
import argparse
import os
import random
from multiprocessing import Pool
from pathlib import Path

import cv2

NAMES = ["white_king", "white_queen", "white_rook", "white_bishop", "white_knight", "white_pawn",
         "black_king", "black_queen", "black_rook", "black_bishop", "black_knight", "black_pawn"]


def label_path(img):
    s = str(img)
    i = s.rfind("/images/")
    return Path(s[:i] + "/labels/" + s[i + 8:]).with_suffix(".txt")


def crops_of(job):
    img_path, out_dir, args_d, seed = job
    rng = random.Random(seed)
    lp = label_path(img_path)
    if not lp.exists():
        return 0
    rows = [r.split() for r in lp.read_text().splitlines() if r.strip()]
    if not rows:
        return 0
    is_render = Path(img_path).name.startswith("render_")
    if is_render and rng.random() > args_d["render_keep"]:
        return 0
    img = cv2.imread(str(img_path))
    if img is None:
        return 0
    H, W = img.shape[:2]
    n = 0
    stem = Path(img_path).stem
    for k, r in enumerate(rows):
        cls = int(r[0])
        cx, cy, w, h = (float(v) for v in r[1:5])
        cx, cy, w, h = cx * W, cy * H, w * W, h * H
        if max(w, h) < 6:
            continue
        for j in range(args_d["jitter"]):
            jit = args_d["jitter_amount"] if j > 0 else 0.0
            side = max(w, h) * args_d["margin"] * (1 + rng.uniform(-jit, jit))
            x0 = cx + rng.uniform(-jit, jit) * w - side / 2
            y0 = cy + rng.uniform(-jit, jit) * h - side / 2
            x0i, y0i = int(round(x0)), int(round(y0))
            s = int(round(side))
            # pad with the border's colours where the square runs off the photo
            pad = max(0, -x0i, -y0i, x0i + s - W, y0i + s - H)
            src = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE) if pad else img
            crop = src[y0i + pad:y0i + pad + s, x0i + pad:x0i + pad + s]
            if crop.size == 0:
                continue
            interp = cv2.INTER_AREA if s > args_d["size"] else cv2.INTER_LINEAR
            crop = cv2.resize(crop, (args_d["size"], args_d["size"]), interpolation=interp)
            d = Path(out_dir) / NAMES[cls]
            cv2.imwrite(str(d / f"{stem}_{k}_{j}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True, help="train.txt of a piece set (one image path a line)")
    ap.add_argument("--val-dir", required=True, help="the set's images/val folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--margin", type=float, default=1.2)
    ap.add_argument("--jitter", type=int, default=2, help="crops per piece in train (the first unjittered)")
    ap.add_argument("--jitter-amount", type=float, default=0.12)
    ap.add_argument("--render-keep", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    train = sorted(set(l.strip() for l in open(args.list) if l.strip()))
    val = sorted(str(p) for p in Path(args.val_dir).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    for split in ("train", "val"):
        for name in NAMES:
            (out / split / name).mkdir(parents=True, exist_ok=True)
    common = {"size": args.size, "margin": args.margin, "jitter_amount": args.jitter_amount}
    jobs = [(p, out / "train", {**common, "jitter": args.jitter, "render_keep": args.render_keep},
             args.seed * 1000003 + i) for i, p in enumerate(train)]
    jobs += [(p, out / "val", {**common, "jitter": 1, "render_keep": 1.0}, i) for i, p in enumerate(val)]
    with Pool(args.workers) as pool:
        done = 0
        for i, n in enumerate(pool.imap_unordered(crops_of, jobs, chunksize=64)):
            done += n
            if i % 2000 == 0:
                print(f"{i}/{len(jobs)} photos, {done} crops", flush=True)
    for split in ("train", "val"):
        counts = {name: len(os.listdir(out / split / name)) for name in NAMES}
        print(split, sum(counts.values()), counts)


if __name__ == "__main__":
    main()
