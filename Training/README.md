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
- Five more sources in `sources.yaml` are `test_only`: the second held-out
  test set. `merge` leaves them out of every training folder; see "Second
  held-out test set" below.

`evaluate.py` on ChessReD's test split is unchanged, so those numbers stay
comparable (baseline: 97.9% per-square, 73.5% boards exact). To score whole
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
pieces `nms=True, iou=0.7, agnostic_nms=True` (a Vision object-detection
pipeline with labels like `white_pawn`; NMS across classes since 2026-11-13,
`--pieces-nms-per-class` gives the earlier export). Checked 2026-11 with torch 2.8.0 / coremltools 9.0 by
re-exporting `baseline/*.pt`: same inputs and outputs as the models in
`Models/`, and 97.9% / 74.2% on ChessReD test for FP16 (INT8: 97.8% / 74.2%).

The app letterboxes the photo into 640x640 (grey 114, centred; `Letterbox.swift`,
since PR #3), as `evaluate.py` and Ultralytics do. Before that it stretched a
3:4 phone photo with Vision's `.scaleFill`; `evaluate.py --stretch` reproduces
that for comparison.

> **License note:** Ultralytics YOLO is AGPL-3. Fine for this GPL-3 app; a
> proprietary fork would need a different detector (see docs/PLAN.md).

## Retraining 2026-11: where it stands and how to carry on

Written so that a new session can continue from this file and
`.notes/training.md` (which says what is running right now).

**Setup on the Mac mini.** `Training/venv` is the system Python 3.9.6 with
torch 2.8.0, ultralytics 8.4.211, coremltools 9.0. Everything on disk is under
`Training/datasets/` (git-ignored): the four sources, `_resized/1280/` (the
cache of small training copies), `_dropped/` (a Roboflow project taken out),
and the merged sets, all built with `--max-side 1280`:

| Folder | Built with | Used by |
|---|---|---|
| `merged` | `merge --max-side 1280` before the extra Roboflow projects | `corners_s_exp1`, `corners_s_exp2`; its `positions.json` is the one to score with |
| `merged_synth` | the same after `synth_boards.py` made `datasets/synth` | `corners_s_exp3` |
| `merged_rf` | `--sources chessred2k chesscog fenit rf_corners` | `corners_s_exp4` |
| `merged_v2` | `merge --max-side 1280 --out datasets/merged_v2` | `pieces_s_rf` |
| `merged_norf` | the same with `--sources chessred2k chesscog fenit` | `pieces_s_norf` |

Corner labels of the labelled boards are the same in all of them; `merged_synth`
and `merged_rf` add corner photos to train only. `merge` refuses
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
# bash, not zsh: zsh passes $POS and $ONE as one word each (or write ${=POS})
C=runs/pose/<name>/weights/last.pt;  P=runs/detect/<name>/weights/last.pt
POS="--positions datasets/merged/positions.json --source fenit --device cpu"
ONE="--copies --one-piece-per-spot 0.7"    # wherever a piece model runs
python corner_error.py --corners-model $C --source fenit                      # boards A and B
python corner_error.py --corners-model $C --source chessred2k --split test
python evaluate.py --corners-model $C --pieces-model $P $POS $ONE --split test   # board B: the number
python evaluate.py --corners-model $C --pieces-model $P --split test \
    --annotations datasets/chessred2k/annotations.json --images datasets/chessred2k/images
python evaluate.py --pieces-model $P --gt-corners $POS $ONE --group B         # pieces alone
python evaluate.py --corners-model $C --gt-boxes  $POS --group B              # corners alone
```

For a piece model use `bash score.sh pieces <label> <weights>` instead of the
`--gt-corners` line above: it adds `--copies --one-piece-per-spot 0.7`, without
which the numbers are far too low (see "Piece scores" below).

Until a new corner model exists, score piece models with `--gt-corners`; until
a new piece model exists, corner models with `--gt-boxes`. `corner_error.py`
runs on the CPU; give `evaluate.py` `--device cpu` while a training run has
the GPU (the machine has 16 GB and swaps). The ChessReD line above scores the
full-size photos; for a piece model trained on the copies use `score.sh`.

**Runs so far.**

| Run | What | Result |
|---|---|---|
| `corners_s_exp1` | yolo11s-pose, 640, 40 epochs, default tolerance | Finished. `last.pt` / `best.pt`, median corner error in squares: ChessReD test 0.32 / 0.34 (baseline 0.31), Fenit A 0.29 / 0.34, **board B 0.92 / 0.99**, board not found in 11 / 9 of 93, order right in 41 / 43. With labelled boxes: ChessReD test 78.4% / 75.8% exact, board B 2.1% / 3.2% exact (63-65% per-square, below an empty board) |
| `corners_s_exp3` | continues exp1 `last.pt`, `--kpt-sigma 0.1`, 20 epochs, on `merged_synth` (3,000 synthetic boards added); cut by a power failure after 5 epochs and continued | Finished (`best.pt` = `last.pt`). Median corner error: ChessReD test **0.22** (exp1 0.32), Fenit A 0.24, **board B 0.84** (order right 26 of 93). With labelled boxes: ChessReD test **95.8% exact** (exp1 78.4%), board B 3.2%, 23.7% with the best rotation. With the baseline piece model: ChessReD test 99.35% / **84.0% exact** (baseline pair 97.9% / 73.5%), board B 66.4% / 0% (baseline 58.0% / 0%) |
| `corners_s_exp4` | continues exp3 `last.pt`, `--kpt-sigma 0.1`, 20 epochs, on `merged_rf` (4,946 Roboflow photos with derived corners added) | Finished. `last.pt` (better than `best.pt`): ChessReD test **0.19**, 97.1% exact with labelled boxes, but 4 of 306 turned 180 degrees (exp3: 2); Fenit A 0.15; **board B 0.91**, order right 29 of 93, 8.6% exact with labelled boxes. More boards with corners did not help on B: see "why learned corners fail" below. Through the app's code with the baseline pieces: ChessReD test 91.2% exact |
| `pieces_s_rf` | yolo11s, 30 epochs, `merged_v2` (8 Roboflow projects; rebuilt without the ChessReD copies, see below) | Killed at epoch 19 (exit 137; the machine was deep in swap); to be continued from epoch 18 with batch 12. At epoch 18 (`score.sh`, see "Piece scores" below): board B 91.3% / 1%, Fenit A 99.4% / 74%, ChessReD test 99.85% / 94.1% |
| `pieces_s_norf` | yolo11s, 30 epochs, `merged_norf` | Finished 2026-11-13. `last.pt` (`best.pt` is a little worse): board B 90.4% / 1%, Fenit A 99.6% / 86% (mostly photos it trained on), ChessReD test 99.90% / **96.1%** (baseline 94.4%) |
| `pieces_s_render1` | continues norf's `last.pt`, 12 epochs, `merged_render` (labelled boards + 4,000 renders) | Finished 2026-11-13 16:45. `last.pt`: board B 91.5% / 1%, Fenit A (trained on) 99.8% / 90%, ChessReD test 99.93% / **96.7%**. On board B the piece kind is much better (pawns right 88%, 65% before) and the side worse. Through the app it is the candidate described below: equal on board B, better on every part of the second test set |
| `pieces_s_render2` | continues render1's `last.pt`, 12 epochs, `merged_all` (every piece source incl. Roboflow + 8,000 renders, 15,438 photos) | running since 16:45 on 2026-11-13 |
| `pieces_s_crop1` | continues render1's `last.pt`, 12 epochs, `merged_all_crop` (`crop_boards.py`: the 7,370 photos of `merged_all` with a corner label, cut down to the board) | queued; score with `--crop-board` |
| `corners_s_canon1` | continues exp4's `last.pt`, `--kpt-sigma 0.1 --canonical-corners`, 12 epochs, `merged_rf_render` (as `merged_rf`, plus the renders whose corners are in the picture) | queued |
| `corners_s_exp2` | continues exp1 `last.pt`, `--kpt-sigma 0.05`, 20 epochs | dropped: exp3 and exp4 already show what a tighter tolerance does |

**A candidate that is not in `Models/`: the piece model `pieces_s_render1`**
(branch `model/pieces-render1`, commit 773e729, has it as
`Models/Pieces.mlpackage` with its weights; exports in `runs/export_render1/`,
with `main`'s corner model in `runs/app_check/models_render1/`). It was put in
and taken out again on 2026-11-13 because it does not meet the rule for
replacing a model as `CLAUDE.md` words it: on the held-out board B it changes
nothing, and on ChessReD test it is not shown better. Through `app_check`
(the decode-once binary, app defaults, colour step off; measured by the
app-side session, per-photo outputs in `runs/app_check/decode_once/`),
pair in `Models/` against candidate:

| | Pair in `Models/` | With `pieces_s_render1` |
|---|---|---|
| ChessReD test (306), boards exact | 279 | 285 |
| Fenit board B (100) | 84.6% of squares, 0 exact | 84.3%, 1 exact |
| Fenit A (286; render1 trained on these photos) | 91.4%, 16 exact | 95.3%, 185 exact |
| Second test set, all 1,619 | 75.5%, 2 exact | 85.4%, 31 exact |

What speaks for it is the second test set, where it is better on every part
(table in that section). Whether that is enough is the owner's call (the
notes, "Needs the owner"). An INT8 export of it is half the size (9.3 MB
against 18 MB); it was compared only with the old `app_check` (285 against
286 ChessReD boards) and the outputs were not kept.

**In `Models/` now (2026-11-13, since midday):** `main`'s corner model again, and
the baseline piece weights exported with NMS across classes. exp3's corners
were taken back out: with the app-side steps (a1 from the pieces, grid-fit
corners) they still turn three endgame photos of ChessReD game G033 by 180
degrees, one of them a fixture of the package's tests, and `main`'s corner
model does not.

**What a new corner model has to beat** (through `app_check` with the app's
defaults, i.e. grid fit on, a1 prior 6, colour step off, each photo decoded
once (the binary since pull request #9); tables in `docs/RECOGNITION.md`):
ChessReD test 279 of 306 boards exact, none turned; Fenit board B 84.6% of
squares, board not found in 4 of 100; Fenit A 91.4% of squares. Compare two
models only through the same build of `app_check`: before #9 it read
different pixels (280, 82.3%, 6 not found for the same pair). And the model
has to pass the fixture tests of `./Scripts/swift-test.sh`: the four fixtures
with the app's pipeline, and the corner model's own test with the grid fit
off, where a wrong a1 or a board that is not found shows (read
`EndToEndFixtureTests.swift` for what it asks; pull request #10 changes it
from a floor of 58 squares a photo to "each corner within one square of the
grid fit's, and 94% of squares over the four photos").

**The piece export with NMS across classes against `main`'s** (same weights;
both with `main`'s corner model and the app's defaults, photo by photo,
`runs/app_check/pair_*.txt`): ChessReD test 280 against 267 boards exact, 13
boards gained and none lost (sign test p = 0.0002), no photo with fewer
squares right; Fenit board B 82.3% against 78.5% of squares, more squares
right on 73 photos and fewer on 1, 0 boards exact with either; Fenit A 91.4%
against 89.1% of squares, 15 against 4 of 286 boards exact.

What exp3 scored while it was in `Models/`, for the record:

**In `Models/` from 2026-11-13 early to midday** (weights and history in `baseline/README.md`):
exp3's corners (FP16, 19 MB; INT8 is 9.5 MB and scores 86.9% where FP16
scores 87.6%) and the baseline piece weights exported with NMS across classes.
Scored through the `.mlpackage`s, letterboxed, per-square / boards exact:

| | ChessReD test (306) | Fenit board B (100) |
|---|---|---|
| Before | 97.86% / 74.2% (interval 69-79%) | 24.2% / 0%, board not found in 66 (58.0% / 0% with the `.pt` files) |
| With exp3 | 99.47% / **87.6%** (83-91%) | 67.7% / 0%, board always found |

On board B no board is fully right with either, and both are below what an
empty prediction scores there (73%). The gain is that the board is found: a
board that is not found counts as 64 wrong squares, and on the 34 photos
where the old exported model does find it, it scores about 71%, no worse than
the new one's 67.7% over all 100. So the new corner model reads known boards
much better and unknown boards no better; what is missing is described below.

**The app's own code, without Xcode.** `app_check/` is a small Swift
program that runs `Packages/ChessVision` (Vision does the resize, as on the
phone) over a list of photos and prints squares right and boards exact. It
builds with the Command Line Tools alone and gives the same result as the
package's fixture test, photo for photo:

```sh
python evaluate.py --dump-list runs/app_check/chessred_test.json --split test \
    --annotations datasets/chessred2k/annotations.json --images datasets/chessred2k/images
(cd app_check && swift build -c release)
app_check/.build/release/app_check --models ../Models --list runs/app_check/chessred_test.json
```

ChessReD test through it, boards exact of 306: `main`'s models 229 (74.8%);
exp3 corners + baseline pieces (what pull request #4 first held) 272 (88.9%); exp4 corners
279 (91.2%). exp3 and exp4 turn the last photos of game G033 by 90 or 180
degrees (G033_IMG099 to 102, 5-6 pieces left; `test_004005.jpg` in the
package's fixtures is G033_IMG101), and exp4 also G000_IMG102
(`test_000102.jpg`); `main`'s corner model gets all of them right. In those
positions the pieces point the wrong way (a white queen and pawn on ranks 7
and 8), so the new models seem to take a1 from the pieces where the old one
took it from the ChessReD board. That is why `swift test` failed while exp3
was in `Models/`, and why it was taken out again.

**Tools added on 2026-11-13** (each file's header says more):

- `train.py corners ... --canonical-corners` and `canonical_corners.py`: corner
  labels in an order the picture fixes, a1 as a class (see the first point
  below). `export_coreml.py` puts the corners back in a1 order inside the
  exported model; `evaluate.py` and `corner_error.py` read either kind.
- `train.py resume --model <run>/weights/last.pt`: continue a cut-off run.
  `queue.sh` resumes through it; a bare ultralytics resume would train a
  `--canonical-corners` run on the labels as they are on disk.
- `render_boards.py` (runs in Blender 5.x, `brew install --cask blender`):
  rendered positions with exact boxes, corners and position, a new piece set,
  board, floor, light and camera per image. `datasets/render` holds 4,000
  (`--seed 1`); source `render` in `sources.yaml`; merged into
  `datasets/merged_render` (pieces) and `datasets/merged_rf_render` (corners).
  In those 4,000 the sides of thick boards carry a warped copy of the board
  pattern (fixed in the script since; the labels are right).
- `score.sh pieces|corners <label> <weights>`: the standard scores on the CPU,
  into `runs/scores/`.
- `evaluate.py --device cpu`, `--dump-list` (the photo lists `app_check` takes),
  and with `--grid-corners` a board the corner model does not find is read by
  the grid fit alone, a1 from the pieces.

**What has been learned, and what to try next.**

- **Why learned corners fail on a new board.** Drawn on the photos, exp4's
  corners on board B are often a quadrilateral turned 45 degrees inside the
  board, or a small patch in its middle. The four keypoints are (a1, a8, h8,
  h1): where the model cannot tell which corner is a1 it predicts the average
  over the rotations it cannot tell apart, which is the middle of the edges
  or of the board. The location is learned together with a guess it cannot
  make. `--canonical-corners` (fde6818) separates the two: the corners are
  labelled in an order fixed by the image (from the corner with the smallest
  x + y, in the labels' direction) and "which of them is a1" is a class; the
  export puts them back in (a1, a8, h8, h1) order so the app sees the same
  `[1,17,8400]` tensor. Not yet shown to work: `corners_s_canon1` is the
  first run. Known weak spot: a board seen as a diamond, where two corners
  tie for the start, can be averaged over two orders again.
- **The grid fit's accept decision is touchy, and a simple cure did not
  help.** One grey level of noise on 3.5% of the pixels (what two JPEG
  decoders differ by) changes accept/reject on 10 of 306 ChessReD test photos
  and 9 of 286 Fenit A photos; the corners never differ by more than 0.1
  square when both are accepted. The cause found in the review of pull
  request #7: candidates that are not x-corners do not settle under
  `cornerSubPix`. Dropping the unsettled ones costs accepted photos on
  ChessReD (93% -> 89%) and is no more stable; putting them back at their
  whole-pixel start changes nothing measurable (5 of 306 and 10 of 286).
  `grid_corners.py` was left as it is; scripts and numbers in
  `runs/grid_stab/`.
- **Piece scores: how to measure them** (`score.sh pieces`, found the hard
  way on 2026-11-13). Two things made every new piece model look far worse
  than it is. (1) One box per piece: the model's own NMS is per class, so a
  pawn that also scores as a bishop comes back twice and the second box
  lands on a neighbouring square; the exported model's NMS runs across
  classes, and `--one-piece-per-spot 0.7` does the same in `evaluate.py`.
  (2) The downscale: ultralytics shrinks a 12 MP photo to 640 without
  smoothing. The baseline was trained on such photos; the new models are
  trained on the 1280-px copies and read the aliased originals badly
  (`pieces_s_norf` on ChessReD test: 62% of boards exact on the originals,
  96% on the copies). The app draws a photo into at most 1280 px before the
  models see it, so `--copies` is the closer test. With both, labelled
  corners, per-square / boards exact:

  | | Fenit board B (98) | Fenit A (270, mostly trained on) | ChessReD test (306) |
  |---|---|---|---|
  | baseline (yolo11n, ChessReD only) | 90.2% / 1% | 95.1% / 6% | 99.88% / 94.4% |
  | `pieces_s_norf` `last.pt` | 90.4% / 1% | 99.6% / 86% | 99.90% / 96.1% |
  | `pieces_s_rf` at epoch 18 | 91.3% / 1% | 99.4% / 74% | 99.85% / 94.1% |

  Earlier scores of piece models in this file and in `runs/scores/originals`
  and `runs/scores/per_class_nms` were made without one or both and are too
  low; the final word on a model is its export through `app_check`.
- **Rendered data teaches the piece kind on a set the model has not seen**
  (`pieces_s_render1`, 2026-11-13). On board B with labelled corners pawns
  are read right 88% of the time (65% before), bishops 90% (60%), rooks 92%
  (82%). It does not show in boards: through the app board B stays at 1 of
  100 fully right, because the same model calls board B's brown "white"
  pieces black (whether brown is White depends on the set) and because of
  what is still wrong, of 1,597 pieces: 104 pawns (34 read as knights, 34 as
  rooks, 18 as bishops, 16 missed), 47 of 196 kings read as queens, 14 queens
  as kings, 11 rooks as queens, nearly all glossy black pieces seen from
  above.
- **A step fitted to one board: `sides_from_colour`.** Splitting the pieces
  of a photo in two colour groups and calling the lighter one White, where
  that contradicts the model on more than 15% of the pieces, turns board B
  from 1 into 17 of 98 boards exact with labelled corners, and 15 of 100 end
  to end. The colour weight and the 15% were chosen on board B. On the
  second test set the step gains nothing and loses 12 points of squares on
  Y-LIChess (through the app: 78.0% -> 65.5%). It is off by default in the
  app and in `evaluate.py`; the 15 of 100 is not a result.
- Tried or queued for the remaining errors:
  - "Exactly one king a side" from every class score of each box (scratch
    prototype, not in the repo): board B 17 -> 21 of 98 with the colour step
    on, ChessReD unchanged.
  - The piece model looking at the board only (`board_crop`, `crop_boards.py`,
    `--crop-board`): `pieces_s_crop1` is queued. A model not trained on
    crops gains nothing from them (16 of 98 on board B with the colour step
    on, against 17 without the crop).
  - More renders and all real sources: `pieces_s_render2`, running.
- **The piece model is what stops board B.** With the labelled corners, one
  board B photo of 98 is read fully right by any of the three. Square by
  square (`pieces_s_norf`, copies): which squares are occupied 98.4% (all
  right on 61% of boards); with the piece type 91.7% (7%); with the colour
  too 90.4% (1%). So it is the type: a third of the pawns are read as
  bishops, knights, rooks or queens (568 of 868 right), bishops are 60%
  right, queens 83%, kings 73%, rooks 82%, knights 96%. The misread pieces
  are mostly seen from straight above (a pawn is a disc) and glossy black.
  The colour (brown "white" pieces) costs little. More pixels are not the
  answer: at 1280, or cropped to the board, the same model is worse. Roboflow
  photos (`pieces_s_rf`) add a point. What is missing is piece sets and
  viewpoints the training data does not have: hence the rendered data.

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

  **These are not held-out numbers.** The grid fit's thresholds were tuned on
  board B, Fenit A and ChessReD test, and the prior of 6 was chosen on
  ChessReD test: on a board none of this has seen, expect less, with a few
  percent of fits shifted by one square. They also use labelled boxes.
  The grid fit (run here on the whole photo, without the model's box, where it
  accepts slightly fewer than the 90% and 97% quoted above for the box prior)
  rejected itself on 8 of 95 board B photos and 20 of 306 ChessReD photos. `--a1-from-pieces 6` adds 6 to the score of the corner
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
- Known weakness of the piece data: Roboflow photos are split into train and
  val by file name, and `chess-vision-ljby5` is video, so neighbouring frames
  of one game are on both sides. Piece validation scores, and the choice of
  `best.pt`, are optimistic; the test scores (ChessReD test, board B) are not
  affected. A split by video would fix it and needs `merged_v2` rebuilt.
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
fetch more Roboflow projects: `export ROBOFLOW_API_KEY=...`).

## Second held-out test set (2026-11): `datasets/heldout2`

Fenit's board B is 100 photos of one board, and settings have been chosen on
it (the grid fit's thresholds, the a1 prior). This is a second test set that
nothing has been trained or tuned on: 1,619 real photos of 19 boards and
piece sets that are in no training source (not ChessReD's, not Fenit's, none
of the Roboflow projects), each with its full position, from five public
sources that need no account or key.

**The rule.** Never train on it, and never choose a threshold, a prior, an
epoch or a model on it. Score finished candidates and report the number. A
setting kept because it scored better here turns it into one more tuning set.
`build_dataset.py merge` enforces what it can: sources marked `test_only` in
`sources.yaml` are left out unless named, cannot be merged together with a
training source, every photo is split `test`, and they go into a folder of
their own, so the scripts that read `datasets/merged*/` never see them.

```sh
python build_dataset.py fetch chessq ylichess chesscog_real samryan chesscv_games   # 2.5 GB to download
python build_dataset.py merge --out datasets/heldout2 \
    --sources chessq ylichess chesscog_real samryan chesscv_games
python preview_labels.py --merged datasets/heldout2 --per-source 12   # look at the labels
./score_heldout2.sh <tag> <corners model> <pieces model> [evaluate.py arguments]
```

The download is 2.5 GB; with the archives kept it is 3.6 GB on disk.
`chesscv_games` needs python-chess (`pip install chess`; it is in
`requirements.txt`). `score_heldout2.sh` runs `evaluate.py` on each part on
the CPU and writes `runs/scores/heldout2_<tag>.txt`.

Outside `merge` the rule is not mechanical. `evaluate.py`, `corner_error.py`
and `grid_corners.py` read whatever `--positions` they are given, which is
what scoring needs, and `score_heldout2.sh` passes any `evaluate.py` setting
on. A test-only folder says so in its `manifest.json` (`"test_only": true`). The
rule is about what is done with the number: `--pull`, `--conf`, the a1 prior
and the grid fit's thresholds are chosen on other data. `render_boards.py
--positions` takes `train` rows only, so it never uses these positions.

| Source | Photos | Boards | Camera | Position from | Licence |
|---|---|---|---|---|---|
| `chessq` ([ChessQ Test](https://huggingface.co/datasets/joelseytre/chessq-test)) | 50 | 15 kinds of set, at most 5 photos each | phones, oblique, 768x1024 to 3072x4080 | `metadata.jsonl`, reviewed by hand by its author | CC BY-NC 4.0 |
| `ylichess` ([Y-LIChess](https://github.com/GibranBenitez/Y-LIChess), light-wood board) | 200: 102 game positions, 98 layouts | 1 folding wooden board | fixed webcam, high oblique, 1920x1080 | derived from its piece boxes and board outline; also gives boxes and corners | CC BY 4.0 |
| `chesscog_real` ([chesscog](https://github.com/georg-wolflein/chesscog) transfer set) | 29 | 1 black-bordered wooden board | iPhone, low, from a player's seat, 1200x900 | one JSON per photo | none stated |
| `samryan` ([samryan18/chess-dataset](https://github.com/samryan18/chess-dataset)) | 500 (476 positions) | 1 green vinyl mat on the floor | straight down, 6000x4000 | the file name | MIT |
| `chesscv_games` ([Rizo-R/chess-cv](https://github.com/Rizo-R/chess-cv)) | 840 (405 positions) | 1 folding brown and cream board | phone in the hand, oblique, 2048x1536 | 8 games replayed; photo N is the game after ply N | none stated |

What each is worth:

- **`chessq` is the one that answers "does it read a board it has never
  seen":** 15 different sets, phone photos, cluttered tables. It is only 50
  photos, so the interval on boards exact is wide (about +-13 points
  mid-range). The models published with it score 6% to 76% of boards exact.
- `ylichess`, `chesscog_real` and `chesscv_games` are one board each: many
  photos, one more unseen board apiece. Do not pool the five into one number;
  840 photos of one board would drown 50 photos of 15. Report per source.
- `chesscv_games` is 8 games photographed after every move from both sides:
  neighbouring photos differ by one piece, so it is far fewer than 840
  independent boards and its interval is too narrow as printed.
- `samryan` is taken from straight above, where a piece shows only its top.
  It measures that view, not a photo from a player's seat.

What to know about the labels:

- **a1 follows the pieces, not the print on the board.** In some ChessQ
  photos, in `chesscv_games`' `rev/` photos and in the second half of
  `ylichess` the
  position is set up against the coordinates printed on the board (White's
  pieces on the printed 7th and 8th ranks). The label is the position as a
  player reads it from the pieces. `evaluate.py` also prints "exact once
  turned 180 degrees", which does not depend on this.
- **`ylichess` has no position file.** Each photo has piece boxes and the
  outline of the playing area; the position is where the boxes stand
  (`load_ylichess`). A photo is left out unless every box lands on a square
  of its own and stays there for anchor pulls from 0.15 to 0.45 (4 of 204
  left out: a pawn on the line between two squares).
  Which corner is a1 is not recorded: the camera and board never move, photos
  numbered up to 100 have White on the far side (a1 top right, as the
  stickers on the board's frame say) and the rest are set up the other way
  round. That rule is all there is for a1, so the photos are in two groups:
  - `wood180` (102): the pieces themselves show which way the board is
    played (one king a side, six or more pawns, none on the first or last
    rank) and agree with the rule. Not all are legal positions: a few are
    shuffled 32-piece set-ups with the pawns on their home ranks.
  - `wood180-layout` (98): a1 is the rule's convention only. 96 are piece
    layouts made to cover the board (two knights; a king and queen; all 32
    pieces packed into four ranks). Two are king-and-pawn endings (photos 43
    and 144, `a1_unclear` in `sources.yaml`) where the rule's a1 has pawns
    that passed each other on two files, and the board turned 180 degrees is
    an ordinary ending: which way they were set up cannot be told.
    Score this group by "exact once turned 180 degrees".

  The split looks at the position alone, never at what a model or
  `white_side_score` makes of it.
- **Only one of Y-LIChess's four boards is used.** On the other three
  (`plastic_bw`, `wood_dark`, `mat_green`, 615 photos) the boxes are a
  detector's output, not labels: pieces without a box, kings labelled queen,
  two boxes on one piece. The four boards show the same positions, but the
  numbering differs between the labelled board and the others, and matching
  them through the noisy boxes was reliable for only 42 of 205 positions.
- `chesscv_games`: `orig/` is seen from White's side, `rev/` from Black's
  (the groups `white-side` and `black-side`).
- `chessq` is CC BY-NC and two sources state no licence: fine for scoring
  here, and nothing from this set is trained on or redistributed anyway.

Checked 2026-11-13: 18 photos compared with their position square by square
(ChessQ 4, `chesscv_games` 6 including the last position of four games from
both sides, `ylichess` 4, `samryan` 2, `chesscog_real` 2): all right. ChessQ's
two label fields agree on all 50 photos. No test photo is a copy of a
training photo: against the 16,358 images of `merged_v2`, `merged_rf` and
`merged_synth`, the closest 32x32 thumbnail has a correlation of 0.82 (a copy
gives 0.98 or more).

**Scores, 2026-11-13.** Per-square / boards exact; "empty" is what
predicting an empty board scores per square, so a figure below it is worse
than nothing. Scored with the `.pt` weights on the CPU, letterboxed
(`score_heldout2.sh`); the reports are kept in `scores/heldout2_2026-11-13/`.

- **Baseline**: `baseline/corners_yolo11n_chessred2k.pt` and
  `baseline/pieces_yolo11n_chessred2k.pt` as they are.
- **In `Models/` now** (since b546e6a): the same two weights, the piece model
  exported with NMS across classes (`--one-piece-per-spot 0.7` stands in for it).
- **exp3**: `baseline/corners_yolo11s_exp3.pt` with the same pieces: the pair
  that was in `Models/` for a few hours on 2026-11-13 and came out again
  because it turns the G033 endgames.

| Part | Photos | Empty | Baseline | In `Models/` now | exp3 corners | exp3 + `--a1-from-pieces 6` |
|---|---|---|---|---|---|---|
| `chessq` | 50 | 68.9% | 37.4% / 0%, board not found in 18 | 38.0% / 0%, board not found in 18 | 74.5% / 0% | 74.5% / 0% |
| `ylichess`, `wood180` | 102 | 57.5% | 0% / 0%, board not found in 102 | 0% / 0%, board not found in 102 | 69.9% / 0% | 65.1% / 0% |
| `ylichess`, `wood180-layout` | 98 | 89.7% | 0% / 0%, board not found in 98 | 0% / 0%, board not found in 98 | 87.8% / 1% | 86.6% / 1% |
| `chesscog_real` | 29 | 52.9% | 49.4% / 0% | 50.1% / 0% | 78.3% / 0% | 78.3% / 0% |
| `samryan` | 500 | 70.1% | 53.8% / 0%, board not found in 66 | 54.1% / 0%, board not found in 66 | 68.4% / 0% | 67.4% / 0% |
| `chesscv_games` | 840 | 59.4% | 53.0% / 0%, board not found in 10 | 54.2% / 0%, board not found in 10 | 85.4% / 0% | 85.4% / 0% |

**`pieces_s_render1` on this set, through the app's own code** (2026-11-13
evening; test only. `app_check` decoding each photo once, the app's defaults,
colour step off; the pair in `Models/` against the same corner model with the
candidate piece model; measured by the app-side session, per-photo outputs
in `runs/app_check/decode_once/`). Squares right; boards exact; photos with
no board found:

| Part | Photos | Pair in `Models/` | With `pieces_s_render1` | Photo by photo |
|---|---|---|---|---|
| `ylichess` | 200 | 70.1%; 1; 6 | 78.0%; 0; 6 | more squares right on 95, fewer on 97 |
| `chessq` | 50 | 64.6%; 1; 10 | 69.7%; 6; 10 | 25 against 12 (p = 0.05) |
| `chesscog_real` | 29 | 73.3%; 0; 0 | 92.7%; 0; 0 | 29 against 0 |
| `samryan` | 500 | 70.4%; 0; 16 | 82.9%; 16; 16 | 359 against 80 |
| `chesscv_games` | 840 | 80.5%; 0; 12 | 89.3%; 9; 12 | not computed (file names repeat) |

Better in squares on every part, by 5 to 19 points; on Y-LIChess as many
photos lose squares as gain them. Boards fully right: 31 of 1,619 against 2,
so about 2% where the target is 85%. (Pooled, which this section warns
against because `chesscv_games` is half the photos: 85.4% against 75.5%.)

The same comparison in Python (`.pt` weights, `main`'s corner model,
`--one-piece-per-spot 0.7 --grid-corners --a1-from-pieces 6`), **on the
full-size photos**: `heldout2` has no 1280-px copies, so `--copies` did
nothing there and ultralytics shrank the photos without smoothing, which
understates a model trained on copies (`runs/scores/heldout2_steps_*.txt`;
a re-score with `--max-side 1280` is in `runs/scores/heldout2_1280_*.txt`).
Squares right / boards exact:

| Part | Photos | Baseline pieces | `pieces_s_render1` | render1 + `--sides-from-colour` |
|---|---|---|---|---|
| `chessq` | 50 | 66.8% / 2% | 70.6% / 8% | 68.6% / 8% |
| `ylichess`, `wood180` | 102 | 57.1% / 0% | 75.2% / 0% | **51.9%** / 0% |
| `ylichess`, `wood180-layout` | 98 | 83.1% / 1% | 75.4% / 0% | 74.3% / 0% |
| `chesscog_real` | 29 | 78.3% / 0% | 93.7% / 0% | 93.7% / 0% |
| `samryan` | 500 | 64.6% / 0% | 79.5% / 2% | 79.2% / 2% |
| `chesscv_games` | 840 | 81.5% / 0% | 89.1% / 0.5% | 89.1% / 0.5% |

Two things to take from it. **`sides_from_colour` does not hold up**: it
gains nothing here and loses 23 points on the Y-LIChess game positions (the
colour-step files were made before the reference computed Lab by the
formula; the difference is two pieces in 23,523). Its colour weight and 15%
threshold were chosen on board B, where it turns 1 exact board into 17; that
gain was fitted, not found. And on the Y-LIChess layouts the candidate is
below the baseline and both are below what an empty board scores there
(89.7%): with labelled corners nearly all the pieces it sees on empty
squares are pieces standing beside the board, which `board_square` takes in
from up to half a square outside (see `EDGE_MARGIN` in `chess_pipeline.py`).

**The app-side steps on this set, through the app's own code, 2026-11-13 midday** (measured with
`app_check` before it decoded each photo once; with the later binary the two
totals are 46.2% and 75.5%, 340 and 44 photos without a board. `app_check`, the exported models
in `Models/` at b546e6a: `main`'s corner model and the piece export with NMS
across classes; measured by the app-side session, nothing chosen on it).
"Steps" are the two app-side steps, corners from the grid fit (used alone
when the model finds no board) and a1 from the pieces with prior 6. Squares
right, photos with no board found, boards exact:

| Part | Photos | Steps off | Both steps |
|---|---|---|---|
| `chessq` | 50 | 28.3%, 26, 0 | 64.6%, 10, 1 |
| `ylichess` | 200 | 0.0%, 200, 0 | 69.7%, 7, 1 (`wood180` 55.3%, `wood180-layout` 84.7%) |
| `chesscog_real` | 29 | 32.9%, 10, 0 | 73.3%, 0, 0 |
| `samryan` | 500 | 53.6%, 80, 0 | 70.3%, 19, 0 |
| `chesscv_games` | 840 | 53.5%, 32, 0 | 80.5%, 12, 0 |
| all | 1,619 | 45.8%, 348, 0 | 75.4%, 48, 2 |

With `main`'s corner model the board is not found at all on a fifth of these
photos (all of Y-LIChess) and the fit alone recovers most of them. Boards
exact stay at 2 of 1,619: that is the piece model. To redo: a photo ->
placement JSON per source from `datasets/heldout2/positions.json`, then
`app_check --models ../Models --list L --a1-prior 6 --grid-fit on` and
`--a1-prior off --grid-fit off`.

- **None of the four pairs in the table reads one game position fully right
  on any of the 19 boards** (with exp3's corners one layout photo of 98
  comes out right; with both app-side steps, below, one ChessQ photo of 50).
  Counting a board turned 180 degrees as right changes nothing. The target
  is 85% of boards exact.
- **The corner model now in `Models/` does not find the board in 294 of the
  1,619 photos**, among them every photo of the Y-LIChess board, and is below
  an empty board on every part. exp3's corners find the board in all 1,619.
  NMS across classes is worth up to about one point per square.
- **Which model fails**, on the one source that has boxes and corners
  (`ylichess`; `ylichess_parts.txt` among the reports). On the 102 game
  positions exp3's corners with the labelled boxes give 99.6% / **97.1%
  exact** (interval 92-99%): on this board that corner model is precise and
  has a1 right. The piece model with the labelled corners gives 70.1% / 0%.
  The piece model is what fails, as on board B. On the 98 layouts the
  corners with labelled boxes give 32% exact, 45% counting a turn: a1 cannot
  be told there.
- `--a1-from-pieces 6` (what the app does since pull request #5) does not
  help exp3 on any part and costs 5 points per square on `wood180`: with
  pieces misread, it turns boards the corner model had the right way round.
  This is a measurement, not a reason to pick another prior on these photos.
- `samryan` (straight down) and the layouts score below an empty board with
  every pair.
- **Both app-side steps on the pair in `Models/` now** (`--grid-corners
  --a1-from-pieces 6`, in Python; the four smaller parts, `now_steps.txt`).
  The grid fit's thresholds and the prior were chosen on board B, Fenit A
  and ChessReD, so this is their first score on boards they have not seen:

  | Part | Photos | Without the steps | With both | Board not found | Fit gave no result |
  |---|---|---|---|---|---|
  | `chessq` | 50 | 38.0% / 0% | 66.8% / 2% | 18 -> 8 | 15 |
  | `ylichess`, `wood180` | 102 | 0% / 0% | 57.1% / 0% | 102 -> 7 | 7 |
  | `ylichess`, `wood180-layout` | 98 | 0% / 0% | 83.1% / 1% | 98 -> 4 | 4 |
  | `chesscog_real` | 29 | 50.1% / 0% | 78.3% / 0% | 0 | 0 |

  The fit finds most boards the corner model misses (on the Y-LIChess board
  it works alone in 189 of 200 photos) and on `chesscog_real` it gives the
  same squares as exp3's corners. One ChessQ photo of 50 comes out fully
  right. The squares still wrong are the piece model's.

**Looked at and not taken** (2026-11-13; searched: Hugging Face, GitHub,
OSF, Zenodo, Mendeley Data, 4TU, figshare, Dataverse, papers on board
recognition):

- CVChess "Multiviews3" (352 iPhone photos, one green vinyl board, four
  viewpoints; images in a Google Drive folder, corrected labels in
  `JSeytre/chessqueries`): usable, a sixth source if one is wanted. Left out
  for now: no licence, the published labels are one ply ahead (the corrected
  copy fixes it), the photos carry GPS tags, and 352 single-file downloads.
- OpenChessRobot (4TU, CC BY 4.0, 160 photos): positions are coverage
  layouts with 63 pieces and several kings; one fixed robot camera.
- `OliverButlerSU/Realtime-Chess-Digitizer` (about 1,300 webcam frames of 20
  games): near top-down, heavily compressed, labels are move lists with an
  undocumented rotation code; also on Kaggle.
- LiveChess2FEN `TestImages.zip`: 5 photos with a position.
- `dopaul/chess-pieces-dominique` (Hugging Face): boxes and a board polygon,
  no position, 164 photos of which 71 are the start position; a Roboflow export.
- `Deinigu/TFG-Diego`: 130 video frames with boxes, no corners, no position.
- `youyexie/Dataset-Of-Chessboards`, `neural-chessboard`, `jayhack/CVChess`,
  `jialinding/ChessVision`: no positions (or piece type without colour).
- Pre-rectified boards (`felixfabricius/robot-chess-commentator-squares`,
  `roni-hershko/chess_data`): the board is already cut out and warped.
- Copies of data already in use: `ehcalabres/chess-rfdetr-coco`,
  `acapitani/chesspiece-detection-yolo`, `smallchess/knight-vision-datasets`,
  `tranmh/pgnize` (ChessReD); `Enzo2806/RecogniChess` (the samryan photos);
  figshare "FENiT Data" (Fenit); several copies of Joseph Nelson's set.
- Kaggle and Roboflow Universe need an account or a key; IEEE DataPort a login.

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
