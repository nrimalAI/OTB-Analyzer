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

## Retraining 2026-11: where it stands and how to carry on

Written so that a new session can continue from this file and
`.notes/training.md` (which says what is running right now).

**Setup on the Mac mini.** `Training/venv` is the system Python 3.9.6 with
torch 2.8.0, ultralytics 8.4.211, coremltools 9.0. Everything on disk is under
`Training/datasets/` (git-ignored): the four sources, `_resized/1280/` (the
cache of small training copies), `_dropped/` (a Roboflow project taken out),
and three merged sets, all built with `--max-side 1280`:

| Folder | Built with | Used by |
|---|---|---|
| `merged` | `merge --max-side 1280` before the extra Roboflow projects | both corner runs; its `positions.json` is the one to score with |
| `merged_v2` | `merge --max-side 1280 --out datasets/merged_v2` | `pieces_s_rf` |
| `merged_norf` | the same with `--sources chessred2k chesscog fenit` | `pieces_s_norf` |

Corner labels are identical in all three (Roboflow has none). `merge` refuses
to rebuild a folder a running job reads.

**Running jobs.** Start them detached and awake, so no session has to stay
open: `nohup caffeinate -i ./venv/bin/python -u train.py ... > runs/<name>.log 2>&1 &`.
`queue.sh` chains runs: it takes `runs/queue.txt` top to bottom, keeps
the running line in `runs/queue.current`, and when started again after a
power cut continues that run from its `last.pt` (start it with
`nohup bash queue.sh > /dev/null 2>&1 &`; the file's header has the rest).
Two runs at once only slow each other down on 16 GB.

**Scoring a run.** Always `best.pt` and `last.pt` (validation is 90% boards
the model knows, so `best.pt` is not chosen on what matters):

```sh
C=runs/pose/<name>/weights/last.pt;  P=runs/detect/<name>/weights/last.pt
POS="--positions datasets/merged/positions.json --source fenit"
python corner_error.py --corners-model $C --source fenit                      # boards A and B
python corner_error.py --corners-model $C --source chessred2k --split test
python evaluate.py --corners-model $C --pieces-model $P $POS --split test     # board B: the number
python evaluate.py --corners-model $C --pieces-model $P --split test \
    --annotations datasets/chessred2k/annotations.json --images datasets/chessred2k/images
python evaluate.py --pieces-model $P --gt-corners $POS --group B              # pieces alone
python evaluate.py --corners-model $C --gt-boxes  $POS --group B              # corners alone
```

Until a new corner model exists, score piece models with `--gt-corners`; until
a new piece model exists, corner models with `--gt-boxes`. `corner_error.py`
runs on the CPU and can be used next to a training job; `evaluate.py` uses the
GPU and slows it a little.

**Runs so far.**

| Run | What | Result |
|---|---|---|
| `corners_s_exp1` | yolo11s-pose, 640, 40 epochs, default tolerance | Finished. `last.pt` / `best.pt`, median corner error in squares: ChessReD test 0.32 / 0.34 (baseline 0.31), Fenit A 0.29 / 0.34, **board B 0.92 / 0.99**, board not found in 11 / 9 of 93, order right in 41 / 43. With labelled boxes: ChessReD test 78.4% / 75.8% exact, board B 2.1% / 3.2% exact (63-65% per-square, below an empty board) |
| `corners_s_exp3` | continues exp1 `last.pt`, `--kpt-sigma 0.1`, 20 epochs, on `merged_synth` (3,000 synthetic boards added); cut by a power failure after 5 epochs and continued | Finished (`best.pt` = `last.pt`). Median corner error: ChessReD test **0.22** (exp1 0.32), Fenit A 0.24, **board B 0.84** (order right 26 of 93). With labelled boxes: ChessReD test **95.8% exact** (exp1 78.4%), board B 3.2%, 23.7% with the best rotation. With the baseline piece model: ChessReD test 99.35% / **84.0% exact** (baseline pair 97.9% / 73.5%), board B 66.4% / 0% (baseline 58.0% / 0%) |
| `pieces_s_rf` | yolo11s, 30 epochs, `merged_v2` (8 Roboflow projects; rebuilt without the ChessReD copies, see below) | running since 21:34 on 2026-11-12 |
| `pieces_s_norf` | yolo11s, 30 epochs, `merged_norf` | queued |
| `corners_s_exp2` | continues exp1 `last.pt`, `--kpt-sigma 0.05`, 20 epochs | queued last; exp3 already shows what a tighter tolerance does |

**In `Models/` since 2026-11-13** (weights and history in `baseline/README.md`):
exp3's corners (FP16, 19 MB; INT8 is 9.5 MB and scores 86.9% where FP16
scores 87.6%) and the baseline piece weights exported with NMS across classes.
Scored through the `.mlpackage`s, letterboxed, per-square / boards exact:

| | ChessReD test (306) | Fenit board B (100) |
|---|---|---|
| Before | 97.86% / 74.2% (interval 69-79%) | 24.2% / 0%, board not found in 66 (58.0% / 0% with the `.pt` files) |
| Now | 99.47% / **87.6%** (83-91%) | 67.7% / 0%, board always found |

On board B no board is fully right yet with either: the gain there is in
squares, and what is missing is described below.

**What has been learned, and what to try next.**

- The corner model is the bottleneck, and only three boards have corner
  labels to train on (ChessReD, the chesscog render, Fenit A). Roboflow adds
  boards for the piece model only. More boards *with corners* is the lever
  most likely to matter: rendered boards (corners are exact by construction),
  or photos whose corners can be derived.
- exp1 learned the boards it was shown and nothing more: on ChessReD and
  Fenit A it matches the baseline (0.3 squares), on board B it is a square
  off and often draws its quadrilateral around a patch in the middle of the
  board. That happens at every piece count (median 0.95-1.36 squares from 0-7
  to 32+ pieces), so it is the unfamiliar board, not sparse positions. Board
  B is a plain brown-and-white board on a tiled floor: nothing about it is
  hard except that it is new. Hence exp3 on synthetic boards before exp2.
- Scores of each run are kept in `runs/scores/<run>_<best|last>.txt`.
- A tighter keypoint tolerance is what made corners precise on boards the
  model knows (exp3: ChessReD test 78% -> 96% exact with labelled boxes). It
  did nothing for a new board, and neither did 3,000 synthetic boards.
- **A classical detector finds the grid on boards no model has seen**
  (`grid_corners.py`: x-corners, a projective lattice fit, a self-check that
  returns nothing when unsure). Accepted results, median error in squares:
  board B 0.06 (90% of photos accepted), Fenit A 0.11 (92%), ChessReD test
  0.03 (97%), none more than a square off; its thresholds were tuned on those
  sets. With labelled boxes and the right rotation its corners give 77% of
  board B exact, the same as the labelled corners. Two uses: an app-side
  corner step (needs Swift, see the notes), and labels for training
  (`derive_corners.py` labels corners on the Roboflow photos, checks them
  against the piece boxes and takes a1 from square colour and pieces).
- **What the two app-side steps are worth**, measured with labelled boxes and
  exp3's corners (`evaluate.py --gt-boxes --grid-corners --a1-from-pieces 6`;
  boards exact / exact once turned 180 degrees):

  | | Board B (95) | ChessReD test (306) |
  |---|---|---|
  | exp3 corners as they are | 3% / 13% | 95.8% / - |
  | + classical grid fit, model corners where it rejects itself | 18% / 47% | 99.3% / 100% |
  | + a1 from the pieces (`chess_pipeline.orient_by_pieces`) | **63% / 77%** | 99.3% / 100% |

  The grid fit rejected itself on 8 of 95 board B photos and 20 of 306
  ChessReD photos. `--a1-from-pieces 6` adds 6 to the score of the corner
  model's own order: on ChessReD, where that order is right, a prior of 0
  loses 3 boards and 6 loses none; on board B the two give the same result.
  These numbers are an upper bound for any piece model.
- Which corner is a1 cannot be had from the corner model on a new board
  (right in 26-43 of 93 on B). From the pieces alone (true positions) the
  right rotation is picked in 97% of ChessReD and chesscog positions, 82% of
  Fenit A and 86% of board B, which are endgame-heavy.
- 60 of 98 corner labels on board B are clipped at the photo edge, so corner
  errors there are overstated by up to half a square. Exact-board scores do
  not use the corner labels.
- **`chess-vision-ljby5` contains copies of every ChessReD2K photo and 4,888
  chesscog renders** (`chessred2k_*`, `chesscog_*`); 254 were in `merged_v2`'s
  piece training data. `exclude_prefixes` in `sources.yaml` now drops them and
  `merged_v2` was rebuilt before any piece run used it.
- Validation pose mAP says little: at exp1 epoch 20 it was 0.87 (mAP50-95)
  with corners 0.6-0.8 squares off, because the default tolerance is loose
  (see `train.py`). Judge corner models with `corner_error.py`. About 0.25
  squares is what a piece on the edge of a square needs.
- On the unseen board the corner *order* is wrong about half the time (at
  epoch 20: right 43, turned 90 degrees 15, turned 180 degrees 34 of 93).
  Which side is White cannot be read off an unfamiliar set from four corners
  alone. `corner_error.py` prints the score with the best rotation, i.e. what
  location alone would give; if location gets good and order does not, the
  order should come from the pieces after detection, which is an app change
  (note it for the MacBook side; the app already lets the user flip the board).
- Scoring on the 1280-px copies instead of the original photos changes
  little (Fenit A 0.84 -> 0.67 squares, ChessReD the same), so the resize is
  not what limits precision.
- Not yet tried: all 4,400 chesscog renders for corners; `kpt_oks_sigmas`
  between 0.05 and 0.25; a longer first stage; yolo11m (about 2.5x slower).

**Things that are not in the repo.** The Roboflow API key (only needed to
fetch more Roboflow projects: `export ROBOFLOW_API_KEY=...`). Roboflow
Universe's website refuses pages from this network until about 05:50 UTC on
2026-11-13 (a search used up its quota); the API is unaffected.

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
