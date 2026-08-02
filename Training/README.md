# Training pipeline (Phase 1)

Trains the two CoreML models the app uses in Phase 2:

| Model | Task | Output |
|---|---|---|
| `board_corners` | YOLO pose — 1 class ("board"), 4 keypoints (a1, a8, h8, h1) | `BoardCorners.mlpackage` |
| `pieces` | YOLO detect — 12 classes (wK…wP, bK…bP) | `Pieces.mlpackage` |

The app assigns each detected piece to a square by mapping the **bottom-center
of its bounding box** through the corner homography — `chess_pipeline.py` is the
reference implementation the Swift port must match.

## Setup

```sh
cd Training
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## 1. Get data

**ChessReD2K** (2,078 real smartphone photos with piece bounding boxes + board
corners): download from https://data.4tu.nl/datasets/99b5c721-280b-450b-b058-b2900b69a90f
(`chessred2k.zip` + `annotations.json`), unzip into `datasets/chessred2k/`.

> ⚠️ **License: CC BY-NC-SA 4.0 (NonCommercial).** Use for prototyping and
> evaluation only. The model that ships in the App Store build must be
> retrained on permissively-licensed + self-captured data (Phase 3).

Optional supplements (CC BY 4.0): chess datasets on Roboflow Universe — export
in "YOLO" format and merge with `--extra-dataset`.

```sh
python prepare_data.py --annotations datasets/chessred2k/annotations.json \
                       --images datasets/chessred2k/images \
                       --out datasets/yolo
# If the schema doesn't match expectations, inspect it first:
python prepare_data.py --annotations .../annotations.json --inspect
```

## 2. Train

```sh
python train.py corners --data datasets/yolo/corners/data.yaml
python train.py pieces  --data datasets/yolo/pieces/data.yaml
```

Both default to YOLO11 nano at 640px. On a Mac this uses MPS; expect hours.
A cloud GPU (Colab, Lambda) is much faster — the scripts run unchanged.

## 3. Evaluate (product metrics, not just mAP)

```sh
python evaluate.py --corners-model runs/pose/corners/weights/best.pt \
                   --pieces-model runs/detect/pieces/weights/best.pt \
                   --annotations datasets/chessred2k/annotations.json \
                   --images datasets/chessred2k/images \
                   --split test
```

Reports per-square accuracy, **full-board exact-match rate**, and a per-class
confusion matrix. Target: ≥99.5% per-square before wiring into the app.
Evaluate on your own phone photos too — the gap vs ChessReD test is the real
generalization signal.

## 4. Export to CoreML

```sh
python export_coreml.py --corners runs/pose/corners/weights/best.pt \
                        --pieces  runs/detect/pieces/weights/best.pt \
                        --out ../Models
```

Exports FP16 and INT8 variants; compare their evaluate.py numbers before
choosing. Copy the winners to `Models/BoardCorners.mlpackage` and
`Models/Pieces.mlpackage`.

> **License note:** Ultralytics YOLO is AGPL-3. Fine for this GPL-3 app; a
> proprietary fork would need a different detector (see docs/PLAN.md).
