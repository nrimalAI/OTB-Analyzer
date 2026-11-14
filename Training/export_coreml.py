"""Export trained models to CoreML .mlpackage for the app.

    python export_coreml.py --corners runs/pose/corners/weights/best.pt \
                            --pieces  runs/detect/pieces/weights/best.pt \
                            --out ../Models

Exports FP16 and INT8 variants of each; run evaluate.py against both (Ultralytics
can load .mlpackage on macOS) and keep the smallest one whose accuracy holds.
Final names the app expects: Models/BoardCorners.mlpackage, Models/Pieces.mlpackage.

The two models are exported differently, because that is what the Swift code
in Packages/ChessVision reads (CoreMLBoardRecognizer.swift):
  corners  no NMS: one raw output of shape [1, 17, anchors] (box 4, score 1,
           4 keypoints x 3), which the app scans for the best-scoring anchor
  pieces   with NMS: a pipeline Vision turns into VNRecognizedObjectObservation,
           class labels like "white_pawn", IoU 0.7 / confidence 0.25 defaults,
           NMS across classes (one box per piece)
Both take a 640x640 RGB image named "image".

A corner model trained with --canonical-corners (canonical_corners.py) has
four classes and its corners in image order. Its export gets a last step,
A1Order, that turns each anchor's corners back into (a1, a8, h8, h1) by its
best class and keeps that class's score, so the app reads the same
[1, 17, anchors] tensor from it as from the older models.
"""

import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

from canonical_corners import is_canonical

VARIANTS = {"fp16": dict(quantize=16), "int8": dict(quantize=8)}
# agnostic_nms: the pieces model's NMS runs over all classes together, so a
# piece that scores for two classes comes back once (the more confident one),
# not as two boxes that then compete for squares.
MODEL_ARGS = {"BoardCorners": dict(nms=False),
              "Pieces": dict(nms=True, iou=0.7, agnostic_nms=True)}


class A1Order(torch.nn.Module):
    """[1, 4 + classes + 12, anchors] from a pose model whose class says where
    a1 is among its four corners -> [1, 17, anchors]: box, best class score,
    corners as (a1, a8, h8, h1)."""

    def __init__(self, model, classes=4):
        super().__init__()
        self.model = model
        self.classes = classes

    def forward(self, x):
        y = self.model(x)
        y = y[0] if isinstance(y, (list, tuple)) else y
        box, scores, kpts = y[:, :4], y[:, 4:4 + self.classes], y[:, 4 + self.classes:]
        score, where = scores.max(dim=1, keepdim=True)           # [1, 1, anchors]
        kpts = kpts.reshape(kpts.shape[0], 4, 3, kpts.shape[-1])
        ordered = torch.zeros_like(kpts)
        for c in range(self.classes):
            # Class c: a1 is corner c, so the corners start there.
            turned = torch.cat([kpts[:, c:], kpts[:, :c]], dim=1)
            ordered = ordered + turned * (where == c).unsqueeze(1).to(kpts.dtype)
        return torch.cat([box, score, ordered.reshape(kpts.shape[0], 12, kpts.shape[-1])], dim=1)


def plain_description(package):
    """Take the path of this machine out of a package's description.
    ultralytics writes "... model trained on <dataset file>" into it, and for
    a run with a dataset file of its own (train.py --kpt-sigma,
    --canonical-corners, --side-tone) that is an absolute path. It becomes
    the path below Training/, or the file's name. Only the description
    changes: the weights file is written back byte for byte. Returns whether
    anything was changed."""
    import re
    import tempfile

    import coremltools as ct
    package = Path(package)
    model = ct.models.MLModel(str(package), skip_model_load=True)
    spec = model.get_spec()
    here = str(Path(__file__).resolve().parent) + "/"

    def plain(match):
        path = match.group(0)
        return path[len(here):] if path.startswith(here) else Path(path).name

    def fix(part):
        changed = False
        text = part.description.metadata.shortDescription
        new = re.sub(r"/\S+", plain, text)
        if new != text:
            part.description.metadata.shortDescription = new
            changed = True
        if part.WhichOneof("Type") == "pipeline":
            for inner in part.pipeline.models:
                changed = fix(inner) or changed
        return changed

    if not fix(spec):
        return False
    with tempfile.TemporaryDirectory(dir=package.parent) as tmp:
        fresh = Path(tmp) / package.name
        ct.models.MLModel(spec, weights_dir=model.weights_dir, skip_model_load=True).save(str(fresh))
        shutil.rmtree(package)
        shutil.move(str(fresh), package)
    return True


def export(weights, out_dir, base_name, **overrides):
    for suffix, quant in VARIANTS.items():
        model = YOLO(weights)
        canonical = base_name == "BoardCorners" and is_canonical(model.names)
        if canonical:
            # ultralytics converts whatever module it is handed here. The
            # package then has one class again, and says so: ultralytics
            # reads the number of classes of a pose model from its names, and
            # evaluate.py must not turn the corners a second time.
            from ultralytics.utils.export import coreml
            convert = coreml.torch2coreml
            coreml.torch2coreml = lambda model, **kw: convert(
                model=A1Order(model),
                **{**kw, "metadata": {**(kw.get("metadata") or {}), "names": {0: "board"}}})
        try:
            path = model.export(format="coreml", imgsz=640,
                                **{**MODEL_ARGS[base_name], **overrides}, **quant)
        finally:
            if canonical:
                coreml.torch2coreml = convert
        dest = out_dir / f"{base_name}-{suffix}.mlpackage"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(path, dest)
        plain_description(dest)
        print(f"exported {dest}" + (" (corners turned to a1 order inside the model)" if canonical else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corners")
    ap.add_argument("--pieces")
    ap.add_argument("--plain-description", nargs="+", metavar="PACKAGE",
                    help="export nothing: take this machine's paths out of the "
                         "description of packages exported before 2026-11-14 evening")
    ap.add_argument("--out", default="../Models")
    ap.add_argument("--pieces-nms-per-class", action="store_true",
                    help="export the piece model as it was before 2026-11-13, with "
                         "NMS within each class (to reproduce or restore that model)")
    args = ap.parse_args()
    if args.plain_description:
        for package in args.plain_description:
            print(f"{package}: {'description rewritten' if plain_description(package) else 'nothing to change'}")
        return
    if not args.corners or not args.pieces:
        ap.error("--corners and --pieces are required")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    export(args.corners, out, "BoardCorners")
    export(args.pieces, out, "Pieces", agnostic_nms=not args.pieces_nms_per_class)
    print(
        "\nPick a variant per model (compare with evaluate.py), then rename to\n"
        f"  {out}/BoardCorners.mlpackage and {out}/Pieces.mlpackage\n"
        "and re-run `xcodegen generate` at the repo root."
    )


if __name__ == "__main__":
    main()
