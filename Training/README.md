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
python build_dataset.py merge --max-side 1280  # same, with small JPEG copies to train on
python build_dataset.py merge --shippable # only data an App Store model may be trained on
python preview_labels.py                  # draw the merged labels on a few photos per source
```

`fetch` was run end to end against the real servers in 2026-11 (ChessReD2K,
chesscog including the split `train` archive, Fenit, Roboflow). The 4TU and
OSF servers are slow (2-3 MB/s per file): fetching sources in separate shells
(`fetch chessred2k`, `fetch chesscog`, `fetch fenit`) takes about 40 minutes
instead of two hours.

`--max-side 1280` writes a JPEG copy of every photo larger than that (kept in
`datasets/_resized/`, so later merges are quick) and trains on the copies;
labels are normalised and do not change. `evaluate.py --positions` still
scores the original photos.

| Source | Photos | Gives | License |
|---|---|---|---|
| `chessred2k` | 2,078 real | boxes + corners | CC BY-NC-SA 4.0 — not shippable |
| `chesscog` | 4,888 rendered (1,500 of the 4,400 train renders used) | boxes + corners | CC BY 4.0 |
| `fenit` | 490 real (two boards) | boxes + corners | CC BY 4.0 |
| `roboflow` | 4,696 from 8 projects | boxes only | per project, mostly CC BY 4.0 |
| `chessred_pseudo` | up to ~5,000 real | boxes only | CC BY-NC-SA 4.0 — not shippable |
| `own` | your photos | whatever you label | yours |

- **Roboflow** needs a free API key: `export ROBOFLOW_API_KEY=...` before
  `fetch`. Or download a project's "YOLOv8" export by hand and unzip it into
  `datasets/roboflow/<name>/`. Only a project with a published version can be
  downloaded. Each project in `sources.yaml` is a different board or set of
  boards, which is what the piece model lacks; they give boxes only, so they
  do nothing for the corner model. Per project: `version` (pin one without
  flipped or rotated copies), `max_images` (a fixed random sample; default
  `max_per_project`), `only_splits`, `license`.

  The eight in use (2026-11; overlays of six photos from each looked at):

  | Project | Used | What it is |
  |---|---|---|
  | `roboflow-100/chess-pieces-mjzgj` | 275 | green vinyl board, tripod |
  | `multichess/chess-vision-ljby5` v1 | 1,500 of 40,261 | tournament video frames, 8+ boards, faces blacked out |
  | `chess-xezgz/chess-hndwj` v2 | 800 | dark-green/white vinyl on a wooden table |
  | `mars-lab-2/getchess` v1 | 600 of 7,231 | several wooden boards; about half are top-down boards pasted on backgrounds |
  | `m-s-teslaenthusiast/anotador-xadrez` | 503 | brown/beige board with an ornate border |
  | `tawan-afcpc/chess-f0mbu` | 500 | a rendered mahogany board, not photos |
  | `politechnika-warszawska/chess-bprbi` v4 | 261 | natural-wood folding board; valid+test only |
  | `captainred-aslrv/chess-vision-jmf56` v14 | 257 | product and web photos, nearly all different sets; license doubtful, marked not shippable |

  Photos are 640 px (stretched to a square by Roboflow) except hndwj (1280)
  and multichess (960x1080). The merge drops repeats: the same original file
  name with the same pieces (augmented copies, forks), or the same image hash
  with the same labels. An image with a label that is not a colored piece
  (e.g. a bare `bishop`) is dropped whole rather than left half-labelled.
  Dropped after a look: `logan-44qjh/chess-pieces-8slll` (pieces without
  boxes), `senku/chess-piece-vrw2r` (only flipped and rotated copies), and
  four of the original five (no published version, or empty).
- **chesscog** gives four unordered corner points; which one is a1 is recovered
  from the labelled piece squares. Checked against ChessReD2K, where the answer
  is known: 2,078 of 2,078 correct. On the real download all 4,888 images get
  an order and every box lands on its square. It is a single rendered board
  and piece set and would be 72% of the training images, so `max_train: 1500`
  in `sources.yaml` uses a fixed random sample of its train renders.
- **Fenit** (https://data.mendeley.com/datasets/wpp9xnbcp6) is the only real
  photo set besides ChessReD with board corners, and needs no account. Each
  photo with a position is named after it; the corner order is recovered from
  that. It has two boards. `holdout: B` in `sources.yaml` makes every board B
  photo the `test` split (100 with positions), a board and piece set the
  models never see, and splits board A 90/10 into train/val. Without that
  line 20% of positions from both boards are `test` and 10% `val`
  (`test_fraction` / `val_fraction`), which only measures "boards like
  Fenit's". Take the line out to train the final model on both boards.

  What the real download contains (checked 2026-11, all 490 photos looked at):
  - **Two boards, two piece sets.** Board A: cream / dark-brown roll-up board,
    light-wood and black pieces, 305 photos from three phones (names use `_`
    between rows). Board B: white / grey-brown board with an orange-brown
    edge, red-brown and black pieces, 186 photos from one phone (names use
    `-`, or are timestamps).
  - 386 photos show a position and are named after it. The names are right;
    the labels are not always: 18 photos have a piece with no box (their
    boxes are dropped), about 0.7% of boxes carry the wrong class, and 5
    photos have the board outline clicked on the image edge where the board
    runs out of frame (their corners are dropped).
  - 54 photos are close-ups of one piece on the floor (kept, boxes only) and
    56 are empty boards (not used: no pieces, and nothing to tell a1 from h8).
  - The board outline's keypoints are in image order (top-left first), not
    board order, so the order really does have to come from the position.
  - Fed through `chess_pipeline` with their own boxes and corners, board A
    comes out 91.8% exact and board B 78.5% (99.8% / 99.0% per-square). Most
    of the misses are label errors, not geometry: of 41 missed photos, 19 have
    a wrong class on a box, 9 an extra box, and 13 (10 of them board B, where
    the board often runs out of frame) a box that lands on the wrong square.
    A model can therefore score above these figures.

  `python preview_labels.py` draws the merged labels on a few photos per
  source; `python preview_labels.py --check` counts, for every photo, how many
  boxes land on the square the position says they should.
- **`chessred_pseudo`** (optional, 24.5 GB download) labels the ChessReD photos
  that have positions but no boxes — see `pseudo_label.py`. It only adds piece
  data for the same board, so it is the least valuable source.
- `datasets/merged/manifest.json` records what went in and what was skipped.

`evaluate.py` on ChessReD's test split is unchanged, so those numbers stay
comparable (baseline: 97.9% per-square, 73.2% boards exact). To score whole
boards on the held-out Fenit photos — board B, a board the models have not
seen — use the `positions.json` the merge writes:

```sh
python evaluate.py --corners-model ... --pieces-model ... \
                   --positions datasets/merged/positions.json --source fenit --split test
```

Board B is only 100 photos: the exact-match rate comes with a 95% interval
(about +-9 points mid-range), and it is also what setups are compared on, so
treat a difference of a few boards as noise. `corner_error.py` reports how
far off the predicted corners are and whether their order is right.

Report both numbers for every new model. `--group A` / `--group B` picks one
Fenit board whatever the split. To see which model is at fault, replace one
with the labels: `--gt-corners` (no corner model needed) or `--gt-boxes` (no
piece model needed); with both, no model runs and the score is what the
geometry alone allows.

## 2. Train

```sh
python train.py corners --data datasets/yolo/corners/data.yaml
python train.py pieces  --data datasets/yolo/pieces/data.yaml
# or, on the merged data:
python train.py corners --data datasets/merged/corners/data.yaml
python train.py pieces  --data datasets/merged/pieces/data.yaml
```

Both default to YOLO11 nano at 640px. On a Mac this uses MPS; expect hours.
Runs are written to `Training/runs/`. Measured on the Mac mini (M4, 16 GB),
yolo11s-pose, batch 16: 0.103 s per image on the full-size photos and 0.084 s
on the `--max-side 1280` copies (19% faster; the GPU is the limit, not
decoding), i.e. about 5 minutes per epoch for 3,200 training photos and about
20 hours for 120 epochs on the full uncapped set. `cache="disk"` is not usable:
it stores every photo at full size (8.5 GB for 330 ChessReD photos).
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

Export settings, fixed in the script because the Swift code depends on them:
both `format="coreml", imgsz=640`; corners `nms=False` (one raw output
`[1, 17, 8400]`: box 4, score 1, 4 keypoints x 3, in 640-px input pixels);
pieces `nms=True, iou=0.7` (a Vision object-detection pipeline with labels
like `white_pawn`). Checked 2026-11 with torch 2.8.0 / coremltools 9.0 by
re-exporting `baseline/*.pt`: same inputs and outputs as the models in
`Models/`, and 97.9% / 74.2% on ChessReD test for FP16 (INT8: 97.8% / 74.2%).

The app scales the photo to 640x640 with Vision's `.scaleFill`, which
stretches a 3:4 phone photo; `evaluate.py` letterboxes it instead. ChessReD
photos are square, so the two agreed until now. With non-square training
photos they no longer do: measure both before shipping a model.

> **License note:** Ultralytics YOLO is AGPL-3. Fine for this GPL-3 app; a
> proprietary fork would need a different detector (see docs/PLAN.md).

## Baseline on other boards (2026-11)

`baseline/*.pt` (yolo11n, ChessReD2K only) on every Fenit photo with a known
position; per-square / boards exact. An empty prediction already scores 79%
(A) and 73% (B) per-square, so anything below that is worse than nothing.

| | Board A (286) | Board B (100) |
|---|---|---|
| Both models | 72.1% / 0% | 58.0% / 0%, corners not found in 12 |
| Piece model + labelled corners | 92.2% / 1.5% | 84.2% / 0% |
| Corner model + labelled boxes | 72.8% / 0% | 59.9% / 0% |
| Labelled boxes + labelled corners | 99.8% / 91.8% | 99.0% / 78.5% |

**The corner model is what fails first on a new board**: with perfect boxes
it is still below the empty-board score, i.e. the corners it finds are wrong,
not just imprecise. The piece model also fails (it misses or misreads enough
pieces that almost no board is fully right), but far less badly.

**The anchor `pull` is not the problem.** With labelled boxes and corners,
boards exact by `pull`: ChessReD2K 100% for every value from 0.05 to 0.65;
chesscog 100% from 0.10 to 0.50; Fenit A 91.8% from 0.05 to 0.40; Fenit B
77.4-78.5% from 0.00 to 0.30, falling above 0.40. One fixed value serves all
four, so the default stays 0.30 and no angle-dependent pull is needed.

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
