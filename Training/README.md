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

```sh
python prepare_data.py --annotations datasets/chessred2k/annotations.json \
                       --images datasets/chessred2k/images \
                       --out datasets/yolo
# If the schema doesn't match expectations, inspect it first:
python prepare_data.py --annotations .../annotations.json --inspect
```

### More than one dataset

ChessReD is one board and one piece set, so models trained on it alone do not
transfer well to other boards. `build_dataset.py` downloads the sources listed
in `sources.yaml` and merges them into `datasets/merged/`:

```sh
python build_dataset.py list              # sources, download sizes, what is on disk
python build_dataset.py fetch --dry-run   # what would be downloaded, nothing fetched
python build_dataset.py fetch             # ~15 GB; needs about twice that free while unpacking
python build_dataset.py merge             # everything on disk
python build_dataset.py merge --shippable # only data an App Store model may be trained on
```

| Source | Photos | Gives | License |
|---|---|---|---|
| `chessred2k` | 2,078 real | boxes + corners | CC BY-NC-SA 4.0 — not shippable |
| `chesscog` | 4,888 rendered | boxes + corners | CC BY 4.0 |
| `fenit` | 492 real | boxes + corners | CC BY 4.0 |
| `roboflow` | a few hundred per project, real | boxes only | per project, mostly CC BY 4.0 |
| `chessred_pseudo` | up to ~5,000 real | boxes only | CC BY-NC-SA 4.0 — not shippable |
| `own` | your photos | whatever you label | yours |

- **Roboflow** needs a free API key: `export ROBOFLOW_API_KEY=...` before
  `fetch`. Or download a project's "YOLOv8" export by hand and unzip it into
  `datasets/roboflow/<name>/`. Add projects to `sources.yaml` freely: Universe
  is full of forks, and the merge drops repeated photos (same original file
  name, or same image hash). An image with a label that is not a colored piece
  (e.g. a bare `bishop`) is dropped whole rather than left half-labelled.
- **chesscog** gives four unordered corner points; which one is a1 is recovered
  from the labelled piece squares. Checked against ChessReD2K, where the answer
  is known: 2,078 of 2,078 correct.
- **Fenit** (https://data.mendeley.com/datasets/wpp9xnbcp6) is the only real
  photo set besides ChessReD with board corners, and needs no account. Each
  photo is named after its position; the corner order is recovered from that.
  It has no official split, so 20% is held out as `test` and 10% as `val`
  (`test_fraction` / `val_fraction` in `sources.yaml`), by position. **The
  adapter has only been tested on stand-in files** built from ChessReD in
  Fenit's format — check a few label overlays after the first real download,
  and look at the `skipped` notes in the manifest. The held-out photos come
  from the same boards as the Fenit training photos, so they measure "boards
  like Fenit's", not a board never seen.
- **`chessred_pseudo`** (optional, 24.5 GB download) labels the ChessReD photos
  that have positions but no boxes — see `pseudo_label.py`. It only adds piece
  data for the same board, so it is the least valuable source.
- `datasets/merged/manifest.json` records what went in and what was skipped.

`evaluate.py` on ChessReD's test split is unchanged, so those numbers stay
comparable (baseline: 97.9% per-square, 73.2% boards exact). To score whole
boards on the held-out Fenit photos — the closest thing here to "a different
board" — use the `positions.json` the merge writes:

```sh
python evaluate.py --corners-model ... --pieces-model ... \
                   --positions datasets/merged/positions.json --source fenit --split test
```

Report both numbers for every new model.

## 2. Train

```sh
python train.py corners --data datasets/yolo/corners/data.yaml
python train.py pieces  --data datasets/yolo/pieces/data.yaml
# or, on the merged data:
python train.py corners --data datasets/merged/corners/data.yaml
python train.py pieces  --data datasets/merged/pieces/data.yaml
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

## Accuracy findings (2026-08, ChessReD2K val/test)

Operating point: single-pass corners @640, `pull=0.30` → **97.9% per-square /
73.2% exact on test** (val: 97.0% / 66.7%). Published end-to-end SOTA on this
dataset is 15.3% exact.

**Corner precision is the dominant residual error** — with ground-truth
corners the same pipeline reaches 99.6% / 86.1% (val). Attempts to close that
gap, all measured, all dead ends:

| Attempt | Result |
|---|---|
| Corner inference at 1280 (model trained @640) | Catastrophic — pose decode breaks off-resolution |
| Two-pass (crop board, re-run corners) | Worse — tight crops are out-of-distribution |
| Retrain corners @960 (batch 8, 8 h) | Keypoint head never converged (pose mAP 0.0 throughout; box fine) |
| cornerSubPix on the 4 outer corners | No change — ~50 px model error exceeds the saddle basin; outer corners abut border frames |
| Interior-grid saddle refinement + RANSAC H refit (win=31) | H genuinely improves (corner err 0.33→0.25 squares) but end-to-end only +0.6 pt exact — H is not the binding constraint mid-board |

Remaining levers for Phase 3 (untested): a dedicated higher-capacity corner
model (yolo11s-pose @640), heatmap-based keypoint architectures, or learning
per-piece anchor offsets. Do this together with the license-clean retrain on
own + CC-BY data rather than as a separate pass.
