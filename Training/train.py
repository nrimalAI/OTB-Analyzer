"""Train the two recognition models.

    python train.py corners --data datasets/yolo/corners/data.yaml
    python train.py pieces  --data datasets/yolo/pieces/data.yaml

Notes:
- Corner keypoints are semantic (a1 vs h1 etc.), so horizontal-flip
  augmentation is DISABLED for both models — a mirrored board is a different
  position, and a mirrored corner labeling is simply wrong.
- Perspective/rotation augmentation is dialed up instead: camera angle is the
  dominant nuisance variable for over-the-board photos.
"""

import argparse

from ultralytics import YOLO

COMMON = dict(
    imgsz=640,
    epochs=120,
    patience=25,
    batch=16,
    workers=4,  # keep dataloader RAM modest on 16 GB machines
    fliplr=0.0,      # semantic left/right — never mirror
    flipud=0.0,
    degrees=12,
    perspective=0.0008,
    translate=0.08,
    scale=0.35,
    hsv_h=0.02, hsv_s=0.5, hsv_v=0.45,
    mosaic=0.6,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["corners", "pieces"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default=None, help="base checkpoint override")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--device", default=None, help="e.g. mps, 0, cpu")
    ap.add_argument("--imgsz", type=int, default=None, help="input resolution override")
    ap.add_argument("--name", default=None, help="run name override")
    args = ap.parse_args()

    if args.task == "corners":
        base = args.model or "yolo11n-pose.pt"
    else:
        base = args.model or "yolo11n.pt"

    overrides = dict(COMMON)
    if args.epochs:
        overrides["epochs"] = args.epochs
    if args.batch:
        overrides["batch"] = args.batch
    if args.device:
        overrides["device"] = args.device
    if args.imgsz:
        overrides["imgsz"] = args.imgsz

    model = YOLO(base)
    model.train(data=args.data, name=args.name or args.task, **overrides)


if __name__ == "__main__":
    main()
