"""What the side of single pieces (chess_pipeline.odd_sides) does to pieces
whose side is known: the labelled boxes of a merged dataset's real photos.

    python side_rule_check.py --src datasets/merged_all5 --dump runs/dumps/truth_all5.jsonl

A photo's labels give every piece's box and its true side, so every piece
the rule changes there is changed wrongly, and a white piece relabelled
Black on purpose shows how often the rule puts one back. Our own renders
are left out (the file names that start with `render__`). The colours are
measured as evaluate.py measures them (piece_colours); they are written to
--dump once and read from it afterwards, since measuring 9,000 photos takes
a few minutes.

Prints, for each source: the photos, on how many of them a right piece is
changed, and of one white piece a photo relabelled Black (chosen with
--seed) how many are put back.
"""

import argparse
import collections
import json
import random
import re
from pathlib import Path

from chess_pipeline import Detection, PIECE_CLASSES, odd_sides


def measure(src, out):
    """One JSON line a photo with labelled pieces: its source, and every
    piece's class, box in pixels and colour."""
    import cv2
    from evaluate import piece_colours
    root = Path(src) / "pieces"
    n = 0
    with open(out, "w") as f:
        for split in ("train", "val", "test"):
            for img in sorted((root / "images" / split).glob("*")):
                if img.name.startswith("render__"):
                    continue
                source = img.name.split("__")[0]
                if source == "roboflow":
                    m = re.match(r"roboflow__([a-zA-Z0-9-]+?)_", img.name)
                    source = "roboflow:" + (m.group(1) if m else "?")
                label = root / "labels" / split / (img.stem + ".txt")
                picture = cv2.imread(str(img)) if label.exists() else None
                if picture is None:
                    continue
                h, w = picture.shape[:2]
                dets = []
                for line in label.read_text().splitlines():
                    p = line.split()
                    if len(p) < 5:
                        continue
                    cx, cy, bw, bh = (float(v) for v in p[1:5])
                    dets.append(Detection(PIECE_CLASSES[int(p[0])], 1.0,
                                          ((cx - bw / 2) * w, (cy - bh / 2) * h,
                                           (cx + bw / 2) * w, (cy + bh / 2) * h)))
                if not dets:
                    continue
                colours = piece_colours(img, dets)
                f.write(json.dumps({"photo": img.name, "source": source, "split": split,
                                    "pieces": [{"cls": d.cls, "bbox": list(d.bbox), "lab": c}
                                               for d, c in zip(dets, colours)]}) + "\n")
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", help="a folder written by build_dataset.py merge (needed to make --dump)")
    ap.add_argument("--dump", required=True, help="the measured colours: made if it is not there")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--nearer", type=float)
    ap.add_argument("--apart", type=float)
    ap.add_argument("--chroma", type=float)
    ap.add_argument("--cover", type=float)
    args = ap.parse_args()
    if not Path(args.dump).exists():
        if not args.src:
            ap.error(f"{args.dump} is not there: give --src to make it")
        print(f"{measure(args.src, args.dump)} photos measured into {args.dump}")
    settings = {k: v for k, v in (("nearer", args.nearer), ("apart", args.apart),
                                  ("chroma", args.chroma), ("cover", args.cover)) if v is not None}
    rng = random.Random(args.seed)
    stat = collections.defaultdict(collections.Counter)
    for line in open(args.dump):
        row = json.loads(line)
        st = stat[row["source"]]
        dets = [Detection(p["cls"], 1.0, tuple(p["bbox"])) for p in row["pieces"]]
        colours = [tuple(p["lab"]) if p["lab"] else None for p in row["pieces"]]
        changed = sum(a.cls != b.cls for a, b in zip(dets, odd_sides(dets, colours, **settings)))
        st["photos"] += 1
        st["pieces changed"] += changed
        st["photos changed"] += changed > 0
        whites = [k for k, d in enumerate(dets) if d.cls.startswith("white_")]
        if whites:
            k = rng.choice(whites)
            wrong = list(dets)
            wrong[k] = Detection("black_" + dets[k].cls.split("_", 1)[1], 1.0, dets[k].bbox)
            st["relabelled"] += 1
            st["put back"] += odd_sides(wrong, colours, **settings)[k].cls == dets[k].cls
    total = collections.Counter()
    for source, st in sorted(stat.items(), key=lambda item: -item[1]["photos"]) + [("all", total)]:
        if source != "all":
            total.update(st)
        print(f"{source:28s} {st['photos']:5d} photos; a right piece changed on {st['photos changed']:4d} "
              f"({st['photos changed'] / st['photos']:.2%}; {st['pieces changed']} pieces); "
              f"of {st['relabelled']} white pieces relabelled Black, one a photo, "
              f"{st['put back'] / max(1, st['relabelled']):.0%} put back")


if __name__ == "__main__":
    main()
