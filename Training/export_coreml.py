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
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

VARIANTS = {"fp16": dict(quantize=16), "int8": dict(quantize=8)}
# agnostic_nms: the pieces model's NMS runs over all classes together, so a
# piece that scores for two classes comes back once (the more confident one),
# not as two boxes that then compete for squares.
MODEL_ARGS = {"BoardCorners": dict(nms=False),
              "Pieces": dict(nms=True, iou=0.7, agnostic_nms=True)}


def export(weights, out_dir, base_name, **overrides):
    for suffix, quant in VARIANTS.items():
        model = YOLO(weights)
        path = model.export(format="coreml", imgsz=640,
                            **{**MODEL_ARGS[base_name], **overrides}, **quant)
        dest = out_dir / f"{base_name}-{suffix}.mlpackage"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(path, dest)
        print(f"exported {dest}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corners", required=True)
    ap.add_argument("--pieces", required=True)
    ap.add_argument("--out", default="../Models")
    ap.add_argument("--pieces-nms-per-class", action="store_true",
                    help="export the piece model as it was before 2026-11-13, with "
                         "NMS within each class (to reproduce or restore that model)")
    args = ap.parse_args()

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
