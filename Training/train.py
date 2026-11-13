"""Train the two recognition models.

    python train.py corners --data datasets/yolo/corners/data.yaml
    python train.py pieces  --data datasets/yolo/pieces/data.yaml
    python train.py resume  --model runs/pose/<name>/weights/last.pt   # continue a cut-off run

Notes:
- Corner keypoints are semantic (a1 vs h1 etc.), so horizontal-flip
  augmentation is DISABLED for both models — a mirrored board is a different
  position, and a mirrored corner labeling is simply wrong.
- Perspective/rotation augmentation is dialed up instead: camera angle is the
  dominant nuisance variable for over-the-board photos.
- Corner precision: for a model with 4 keypoints ultralytics scores a keypoint
  with a tolerance (OKS sigma) of 0.25, against 0.025-0.107 for human joints.
  With that, a corner one whole square off costs a loss of 0.03: nothing
  pushes the corners to be exact. --kpt-sigma sets a tighter value for the
  loss and for the validation metric. Too tight from a cold start gives no
  gradient at all (0.05 saturates beyond about two squares), so train with
  the default first and then continue from those weights with --kpt-sigma.
- --canonical-corners: see canonical_corners.py. The labels on disk stay as
  they are; the order and the class are set while training. Such a run is
  marked in its dataset file and is continued with `train.py resume`, never
  with a bare ultralytics resume, which would train on the labels as on disk.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).resolve().parent

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
    ap.add_argument("task", choices=["corners", "pieces", "resume"])
    ap.add_argument("--data", help="dataset file (not needed for resume)")
    ap.add_argument("--model", default=None, help="base checkpoint override")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None,
                    help="stop after this many epochs without improvement")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--device", default=None, help="e.g. mps, 0, cpu")
    ap.add_argument("--imgsz", type=int, default=None, help="input resolution override")
    ap.add_argument("--name", default=None, help="run name override")
    ap.add_argument("--kpt-sigma", type=float, default=None,
                    help="corners only: keypoint tolerance for the loss and the "
                         "metric (ultralytics' default for 4 keypoints is 0.25)")
    ap.add_argument("--canonical-corners", action="store_true",
                    help="corners only: corners in an order the picture fixes and "
                         "a1 as a class (canonical_corners.py)")
    args = ap.parse_args()

    if args.task == "resume":
        # Continue a cut-off run from its last.pt with the settings it had.
        import torch
        import yaml
        if not args.model:
            ap.error("resume needs --model <run>/weights/last.pt")
        saved = torch.load(args.model, map_location="cpu", weights_only=False).get("train_args", {})
        data_file = Path(str(saved.get("data", "")))
        if data_file.is_file() and (yaml.safe_load(data_file.read_text()) or {}).get("canonical_corners"):
            import canonical_corners
            canonical_corners.patch()
        extra = {"batch": args.batch} if args.batch else {}
        YOLO(args.model).train(resume=True, **extra)
        return
    if not args.data:
        ap.error("--data is required")
    if args.canonical_corners and args.task != "corners":
        ap.error("--canonical-corners is for the corners model")

    if args.task == "corners":
        base = args.model or "yolo11n-pose.pt"
    else:
        base = args.model or "yolo11n.pt"

    overrides = dict(COMMON)
    if args.epochs:
        overrides["epochs"] = args.epochs
    if args.patience is not None:
        overrides["patience"] = args.patience
    if args.batch:
        overrides["batch"] = args.batch
    if args.device:
        overrides["device"] = args.device
    if args.imgsz:
        overrides["imgsz"] = args.imgsz

    # Keep runs under Training/runs/ wherever the script is started from
    # (recent ultralytics would otherwise put them at the repo root).
    project = HERE / "runs" / ("pose" if args.task == "corners" else "detect")
    name = args.name or args.task

    data = args.data
    if args.kpt_sigma and args.task != "corners":
        ap.error("--kpt-sigma is for the corners model")
    if args.kpt_sigma or args.canonical_corners:
        # ultralytics reads the tolerance and the classes from the dataset
        # file, so train from a copy of it that carries the settings.
        import yaml
        spec = yaml.safe_load(Path(args.data).read_text())
        spec["path"] = str((Path(args.data).parent / spec.get("path", ".")).resolve())
        if args.kpt_sigma:
            spec["kpt_oks_sigmas"] = [args.kpt_sigma] * spec["kpt_shape"][0]
        if args.canonical_corners:
            import canonical_corners
            canonical_corners.patch()
            spec["names"] = dict(canonical_corners.CLASS_NAMES)
            spec["nc"] = 4
            spec["canonical_corners"] = True
        project.mkdir(parents=True, exist_ok=True)
        data = str(project / f"{name}_data.yaml")
        Path(data).write_text(yaml.safe_dump(spec, sort_keys=False))
    model = YOLO(base)
    model.train(data=data, project=str(project), name=name, **overrides)


if __name__ == "__main__":
    main()
