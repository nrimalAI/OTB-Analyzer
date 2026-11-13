"""Board corners in an order the picture fixes, and a1 as a class.

The corner labels are (a1, a8, h8, h1). A pose model trained on them has to
say where the corners are and which of them is a1 with the same four
numbers, and where it cannot tell a1 (an unfamiliar board, an endgame) it
answers with the average of the rotations it cannot tell apart: a
quadrilateral turned 45 degrees inside the board, or a patch in its middle.
That is what exp1-exp4 do on Fenit board B.

Here the two questions are separated. For training, the four corners are
turned (never mirrored) so that they start at the corner with the smallest
x + y in the model's input, the one nearest the top left, and go on in the
same direction as before; the class of the board says where a1 is in that
order (class c: a1 is corner number c). The location no longer depends on
knowing a1, and a wrong a1 is a board turned by a quarter or a half, not a
wrong board. Reading the model's answer, `a1_order` turns the corners back
into (a1, a8, h8, h1), and `export_coreml.py` does the same inside the
exported model, so the app sees the tensor it always saw.

The order is set after augmentation (a rotated or mosaic'd board is ordered
as it appears), by a transform added to the ultralytics dataset in `patch`.
A run trained this way is marked in its dataset file (`canonical_corners:
true`) and must be continued through `train.py resume`, which patches again.
"""

import numpy as np

CLASS_NAMES = {0: "a1_first", 1: "a1_second", 2: "a1_third", 3: "a1_fourth"}


def start_corner(xy):
    """Index of the corner an image-fixed order starts at: smallest x + y."""
    xy = np.asarray(xy, dtype=float)
    return int(np.argmin(xy[:, 0] + xy[:, 1]))


class CanonicalCorners:
    """Dataset transform: reorder each board's keypoints and set its class."""

    def __call__(self, labels):
        instances = labels.get("instances")
        if instances is None or instances.keypoints is None or len(instances) == 0:
            return labels
        h, w = labels["img"].shape[:2]
        sx, sy = (w, h) if instances.normalized else (1.0, 1.0)
        keypoints = instances.keypoints
        cls = labels["cls"]
        for i in range(len(keypoints)):
            s = start_corner(np.stack([keypoints[i, :, 0] * sx, keypoints[i, :, 1] * sy], axis=1))
            keypoints[i] = np.roll(keypoints[i], -s, axis=0)
            cls[i] = (-s) % 4          # where a1 (first on disk) is now
        return labels


def patch():
    """Make every ultralytics dataset built from now on in this process apply
    CanonicalCorners just before the labels are packed for the model."""
    from ultralytics.data.dataset import YOLODataset
    if getattr(YOLODataset, "_canonical_corners", False):
        return
    original = YOLODataset.build_transforms

    def build_transforms(self, hyp=None):
        transforms = original(self, hyp)
        if self.use_keypoints:
            transforms.insert(len(transforms.tolist()) - 1, CanonicalCorners())
        return transforms

    YOLODataset.build_transforms = build_transforms
    YOLODataset._canonical_corners = True


def is_canonical(names):
    """Whether a model's class names are those of a model trained this way."""
    return dict(names) == CLASS_NAMES


def a1_order(keypoints, cls, names):
    """The model's four corners as (a1, a8, h8, h1): unchanged for an older
    model, turned by the predicted class for one trained with this module."""
    keypoints = list(keypoints)
    if not is_canonical(names):
        return keypoints
    c = int(cls) % 4
    return keypoints[c:] + keypoints[:c]
