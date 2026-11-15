"""A training set for the piece model made as a list of photos, without
copying any: a merged dataset's training photos, the real ones listed more
than once if asked, plus render batches taken straight from datasets/render.

    python train_list.py --base datasets/merged_all4 --real-times 3 \
        --out datasets/merged_all4_real3
    python train_list.py --base datasets/merged_all4 --render-seeds 6 \
        --out datasets/merged_all4_shape1

Why: a merged dataset holds whatever datasets/render held on the day it was
merged, and every photo once. To weigh the real photos up against the
renders, or to add one render batch and not another (seed 5's low views,
say), a list is enough: ultralytics takes a text file of image paths for
`train:` and finds each photo's labels beside it (.../images/... ->
.../labels/...). A render is 1280 pixels on its long side as the merged
copies are.

Writes <out>/pieces/train.txt, data.yaml (validation and test photos are
the base's) and README.txt (what is in the list). `train.py pieces --data
<out>/pieces/data.yaml --side-tone P` trains on it; without --side-tone
give train.py a data.yaml whose `path` is absolute (train.py makes one when
it copies the file for --side-tone).

A real photo is one whose file name does not start with `render__`; a
render batch is the files render_<seed, two digits>*.jpg of
datasets/render/pieces/images that have a label file.
"""

import argparse
import os
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", required=True, help="a folder written by build_dataset.py merge")
    ap.add_argument("--out", required=True)
    ap.add_argument("--real-times", type=int, default=1,
                    help="how often each real photo of the base is listed (renders once)")
    ap.add_argument("--render-seeds", type=int, nargs="*", default=[],
                    help="render batches to add from --render (the base's own renders stay)")
    ap.add_argument("--render", default=str(HERE / "datasets" / "render"),
                    help="the folder render_boards.py writes to")
    args = ap.parse_args()
    base = (Path(args.base) / "pieces").resolve()
    out = Path(args.out) / "pieces"
    out.mkdir(parents=True, exist_ok=True)

    lines, real, renders = [], 0, 0
    for img in sorted((base / "images" / "train").iterdir()):
        is_render = img.name.startswith("render__")
        real += not is_render
        renders += is_render
        lines += [str(img)] * (1 if is_render else args.real_times)
    added = {}
    images = Path(args.render).resolve() / "pieces" / "images"
    for seed in args.render_seeds:
        batch = [img for img in sorted(images.glob(f"render_{seed:03d}*.jpg"))
                 if (images.parent / "labels" / (img.stem + ".txt")).exists()]
        if not batch:
            ap.error(f"no labelled render of seed {seed} in {images}")
        if any(f"render__{img.name}" in line for img in batch[:1] for line in lines):
            ap.error(f"seed {seed} is in the base already")
        added[seed] = len(batch)
        lines += [str(img) for img in batch]
    (out / "train.txt").write_text("\n".join(lines) + "\n")

    spec = yaml.safe_load((base / "data.yaml").read_text())
    spec["path"] = os.path.relpath(base, out.resolve())
    spec["train"] = os.path.relpath(out.resolve() / "train.txt", base)
    (out / "data.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    text = (f"{args.base}'s training photos as a list (train.txt), made by train_list.py: "
            f"{real} real photos x {args.real_times}, its {renders} renders once"
            + "".join(f", {n} renders of seed {seed} from {os.path.relpath(args.render, HERE)}" for seed, n in added.items())
            + f"; {len(lines)} lines. Validation and test photos are the base's.\n")
    (out / "README.txt").write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
