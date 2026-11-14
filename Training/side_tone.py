"""A training augmentation for the piece model: the same set in another tone.

What the piece model gets wrong on a set it has not seen is the side: on
Fenit board B, whose "white" pieces are brown, it calls a share of them
black, because in the real photos it learned from a brown piece is nearly
always Black. Which side a piece is on is not a colour but a relation: the
lighter of the two sides is White.

The real photos have few sets, so here each photo is, with probability p,
shown in another tone: a colour cast (warm most of the time), a darker or
lighter tone curve and a little lift of the blacks, the same for every
pixel. A cream side then looks tan, orange or brown and the dark side
darker still: the order of the two sides stays, the colours do not, and the
labels need no change. `render_boards.py --palette relative` teaches the
same with rendered sets; this does it on real pieces.

It is added to the ultralytics dataset by `patch`, after its own colour
jitter and before the labels are packed, for training only. A run trained
with it is marked in its dataset file (`side_tone: p`) and is continued
with `train.py resume`, which patches again.
"""

import numpy as np

# Limits of the random tone, chosen by eye on contact sheets of ChessReD,
# chesscog and Fenit board A photos (side_tone.py --sheet): a cream piece
# goes as far as the brown of board B's "white" side and stays lighter than
# the black side beside it.
GAMMA = (0.75, 2.2)            # tone curve exponent: over 1 darkens the mid-tones
WARM = ((0.85, 1.0), (0.55, 0.88), (0.30, 0.70))   # red, green, blue gains of a warm cast
ANY = (0.55, 1.0)              # gains of a cast of any hue
WARM_SHARE = 0.65
LIFT = (0.0, 0.07)             # blacks raised by up to this much of full scale


def tone(rng):
    """A random tone: (gamma, (r, g, b) gains, lift)."""
    gamma = float(np.exp(rng.uniform(np.log(GAMMA[0]), np.log(GAMMA[1]))))
    if rng.random() < WARM_SHARE:
        gains = tuple(float(rng.uniform(lo, hi)) for lo, hi in WARM)
    else:
        gains = tuple(float(rng.uniform(*ANY)) for _ in range(3))
    return gamma, gains, float(rng.uniform(*LIFT))


def apply_tone(img_bgr, gamma, gains, lift):
    """The picture (uint8, BGR, as ultralytics holds it) in that tone. One
    lookup table a channel, so the order of any two pixel values of a channel
    is kept."""
    x = np.arange(256, dtype=np.float32) / 255.0
    out = np.empty_like(img_bgr)
    for channel, gain in zip((2, 1, 0), gains):      # gains are r, g, b
        table = np.clip((lift + (1.0 - lift) * x ** gamma) * gain, 0.0, 1.0)
        out[..., channel] = (table * 255.0 + 0.5).astype(np.uint8)[img_bgr[..., channel]]
    return out


class SideTone:
    """Dataset transform: with probability p, the picture in a random tone."""

    def __init__(self, p, seed=None):
        self.p = float(p)
        self.rng = np.random.default_rng(seed)

    def __call__(self, labels):
        if self.rng.random() < self.p:
            labels["img"] = apply_tone(labels["img"], *tone(self.rng))
        return labels


def patch(p):
    """Make every ultralytics training dataset built from now on in this
    process apply SideTone(p) just before the labels are packed. Validation
    pictures are left as they are."""
    from ultralytics.data.dataset import YOLODataset
    if getattr(YOLODataset, "_side_tone", None) is not None:
        return
    original = YOLODataset.build_transforms

    def build_transforms(self, hyp=None):
        transforms = original(self, hyp)
        if self.augment:
            transforms.insert(len(transforms.tolist()) - 1, SideTone(p))
        return transforms

    YOLODataset.build_transforms = build_transforms
    YOLODataset._side_tone = float(p)


def main():
    """Contact sheet: python side_tone.py --sheet out.jpg photo1.jpg photo2.jpg ..."""
    import argparse
    import cv2
    ap = argparse.ArgumentParser(description="rows of one photo in random tones, to judge the limits by eye")
    ap.add_argument("--sheet", required=True, help="JPEG to write")
    ap.add_argument("--tones", type=int, default=5, help="random tones a photo, after the photo as it is")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("photos", nargs="+")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    rows = []
    for path in args.photos:
        img = cv2.imread(path)
        if img is None:
            ap.error(f"cannot read {path}")
        scale = 300 / max(img.shape[:2])
        small = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)),
                           interpolation=cv2.INTER_AREA)
        tile = np.full((300, 300, 3), 114, np.uint8)
        cells = []
        for k in range(args.tones + 1):
            shown = small if k == 0 else apply_tone(small, *tone(rng))
            cell = tile.copy()
            cell[:shown.shape[0], :shown.shape[1]] = shown
            cells.append(cell)
        rows.append(np.hstack(cells))
    cv2.imwrite(args.sheet, np.vstack(rows))


if __name__ == "__main__":
    main()
