"""Export trained models to CoreML .mlpackage for the app.

    python export_coreml.py --corners runs/pose/corners/weights/best.pt \
                            --pieces  runs/detect/pieces/weights/best.pt \
                            --out ../Models

Exports FP16 and INT8 variants of each; run evaluate.py against both (Ultralytics
can load .mlpackage on macOS) and keep the smallest one whose accuracy holds.
Final names the app expects: Models/BoardCorners.mlpackage, Models/Pieces.mlpackage.
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

VARIANTS = {"fp16": dict(half=True), "int8": dict(int8=True)}


def export(weights, out_dir, base_name):
    for suffix, quant in VARIANTS.items():
        model = YOLO(weights)
        path = model.export(format="coreml", imgsz=640, nms=True, **quant)
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
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    export(args.corners, out, "BoardCorners")
    export(args.pieces, out, "Pieces")
    print(
        "\nPick a variant per model (compare with evaluate.py), then rename to\n"
        f"  {out}/BoardCorners.mlpackage and {out}/Pieces.mlpackage\n"
        "and re-run `xcodegen generate` at the repo root."
    )


if __name__ == "__main__":
    main()
