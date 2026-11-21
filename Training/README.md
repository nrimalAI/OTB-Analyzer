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

**"The pair in `Models/`"** in the passages below written before
2026-11-16 means `pieces_s_tone1` + `corners_s_canon1` (from pull request
#19); since pull request #32 the piece model is `pieces_s_shape1`. What a
new model has to beat now is under "What a new model has to beat".

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

**The training process leaks memory on Apple's GPU.** Its footprint (nearly
all of it compressed and swapped out, so `ps` shows under 1 GB; look with
`top -o mem`) grows with every batch: `pieces_s_render2` held 85 GB after
four hours, swap had grown to 36 GB on disk, epochs took 50 minutes instead
of 26, and a run left alone ends with exit 137 (`pieces_s_rf`). On 15,000
photos it grows by 30 GB within one epoch. `queue.sh` therefore stops a run
that holds more than `MEM_LIMIT_GB` (30) once its next epoch is saved and
continues it in a new process, on such a dataset after every epoch
(`queue.log` says when); a run the system kills for memory is continued up
to three times. A run started by hand has no such guard: start runs through the
queue.

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
python evaluate.py --corners-model $C --pieces-model $P --split test --device cpu \
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
| `pieces_s_render2` | continues render1's `last.pt`, 12 epochs, `merged_all` (every piece source incl. Roboflow + 8,000 renders, 15,438 photos) | Finished 2026-11-13 23:27. `last.pt`, labelled corners: ChessReD test 99.96% / **98.4%** (301 of 306), Fenit A (trained on) 99.7% / 88%, board B 90.9% / 0%. On board B the kinds are better again (pawns right 95.5%, knights all; kings still read as queens 56 times of 196) and what is wrong is mostly the side: 243 of 868 brown "white" pawns read as black. The candidate described below |
| `pieces_s_crop1` | continues render1's `last.pt`, 12 epochs, `merged_all_crop` (`crop_boards.py`: the 7,370 photos of `merged_all` with a corner label, cut down to the board) | Finished 2026-11-14 02:20. Scored with `--crop-board`, against `pieces_s_render2` (same start, same 12 epochs, whole photos). Labelled corners (`scores/pieces_s_crop1_last.txt`): board B 92.5% / 2 of 98 exact against 90.9% / 0; ChessReD test 99.97% / 98.4% against 99.96% / 98.4%; Fenit A (trained on) 94% exact against 88%. End to end in Python with `main`'s corner model, grid fit, a1 prior 6, one king a side (`scores/endtoend_crop1_vs_render2.txt`): board B 84.3% / 2 of 100 against 82.9% / 0; ChessReD test 298 against 297 of 306. A small gain on board B and none on ChessReD: not by itself worth the change in the app. The side is still what is wrong on board B (P read as p 145 times, was 243) |
| `pieces_s_render3` | continues render2's `last.pt`, 10 epochs, `merged_all3` (as `merged_all`, with 4,000 more renders whose two sides are two shades of one list, the lighter one White, and more views from above: 19,438 photos) | Finished 2026-11-14 08:17. `last.pt`, labelled corners (`scores/pieces_s_render3_last.txt`): **board B 94.7% / 8 of 98 exact** (render2 90.9% / 0), Fenit A (trained on) 99.8% / 92%, ChessReD test 99.94% / 96.7% (296 of 306; render2 301). On board B, P read as p 132 times (render2 243); during the run it moved between 111 and 175 from one epoch to the next. **In `Models/`**: see below |
| `pieces_s_tone1` | continues render3's `last.pt`, 8 epochs, `merged_all4` (as `merged_all3`, with 8,000 more renders of the same kind, `--seed 4`: 20,000 renders in all, 27,437 photos), `--side-tone 0.35` (`side_tone.py`: a third of the training photos shown in another tone, so that a cream side looks tan or brown beside a darker one) | Finished 2026-11-14 17:09. `last.pt`, labelled corners (`scores/pieces_s_tone1_last.txt`): **board B 96.8% / 20 of 98 exact** (render3 94.7% / 8), P read as p 39 times (render3 132; after epoch 3 it was 72); Fenit A (trained on) 99.7% / 89%; ChessReD test 99.95% / 97.7% (299 of 306; render3 296). Two changes in one run (8,000 more renders, the tone): which of the two did how much is not known. Mosaic was off for the whole run, as for render3 (8 epochs, `close_mosaic` left at 10). **In `Models/`**: see below |
| `pieces_s_tone2` | continues tone1's `last.pt`, 8 epochs, `merged_all4`, `--side-tone 0.6` | Finished 2026-11-15 00:04. **Not taken: better on board B, clearly worse on one board of the second test set.** Labelled corners (`scores/pieces_s_tone2_last.txt`): board B 97.2% / 26 of 98 exact (tone1 96.8% / 20), P read as p 16 times (39); ChessReD test 299 of 306 (the same); Fenit A 91% (89%). Through the app with the corner model in `Models/` (`runs/app_check/dark/`, `dark_h2/`; the build of c8afd31, tone1 against tone2): board B 94.48% / 16 -> 94.88% / **27** boards exact (more squares right on 40 photos, fewer on 14; two five-piece endgames that tone1 read upright, at 62 and 61 squares, are read half a turn off at 54); ChessReD test 295 -> 294; Fenit A 233 -> 230; all 115 tests pass. Then the second test set, as a test: 96.38% -> 96.34% of squares, **459 -> 407 boards exact** (108 gained, 160 lost). The whole loss is on one part, the chess-cv games (267 -> 210; 840 photos of eight games on one board and one piece set), and it is the kind of a piece, not its side: kings read as queens, queens and knights as rooks and kings; six of the eight games lose boards. The other four parts are level or better (Y-LIChess 55 -> 54, ChessQ 14 -> 14, samryan 115 -> 115, chesscog 8 -> 14; squares up on each), and averaged by part instead of by photo tone2 is ahead (boards exact 27.6% -> 30.3%). So it is not shown better on unseen boards and is clearly worse on one of them; it stays out. Why is not known: the tone share (0.6 against 0.35) and eight more epochs changed together. It passed the rule in `CLAUDE.md` (board B and ChessReD test); what kept it out is the second test set, which was thereby used to choose (see that section) |
| `pieces_s_low1` | continues **tone1's** `last.pt`, 8 epochs, `merged_all5` (as `merged_all4`, with 6,000 more renders, six in ten of them seen from 12 to 32 degrees above the board: 26,000 renders), `--side-tone 0.35` | Finished 2026-11-15 07:59 (a first start at 00:05 from tone2 was stopped when tone2 failed the second test set). **Not taken: better on board B, worse on Fenit board A and on one board of the second test set.** Labelled corners (`scores/pieces_s_low1_last.txt`): board B 97.0% / 22 of 98 exact (tone1 96.8% / 20), P read as p 22 times (39); ChessReD test 301 of 306 (299); Fenit A 99.66% / 84% (99.74% / 89%). Through the app with the corner model in `Models/` (`runs/app_check/low1/` against `window/now_*`, one build): board B 94.48% / 16 -> 95.08% / **24** boards exact (more squares right on 31 photos, fewer on 15, sign test p = 0.03; side wrong on 61 -> 37 squares, kinds as before); ChessReD test 295 -> 296, as 3:4 photos 282 -> 279 (neither beyond noise); **Fenit A 234 -> 212 of 286** (fewer squares right on 31 photos, more on 9, p = 0.0007): bishops read as queens (3 -> 14 squares), rooks as queens, more pawns on empty squares. Then the second test set, as a test (`runs/app_check/low1_h2/` against `window_h2/now_*`): 97.00% -> 96.91% of squares, **464 -> 440 boards exact**. By part: Y-LIChess 60 -> 43 of 200 (fewer squares right on 70 photos, more on 19; pawns read as bishops and rooks, 35 -> 88 squares), samryan 115 -> 102 of 500 and the chess-cv games 267 -> 266 of 840 (squares level on both), ChessQ 14 -> 14, chesscog 8 -> 15 of 29 (more right on 13 photos, fewer on none). It passed the rule in `CLAUDE.md`; what speaks against it is a board it was trained on (Fenit A, a development set) and one unseen board. What the two have in common is a shorter piece read as a taller one, which is what a low camera does to a piece; that the low views are the cause is a guess, the run also had 6,000 more renders and eight more epochs |
| `pieces_s_crop2` | continues tone1's `last.pt`, 8 epochs, `merged_all5_crop` (the 17,013 photos of `merged_all5` with a corner label, cut down to the board), `--side-tone 0.35` | Finished 2026-11-15 11:45. Scored with `--crop-board`, against tone1 without it. Labelled corners (`scores/pieces_s_crop2_last.txt`): board B 97.1% / 24 of 98 exact (tone1 96.8% / 20), P read as p 10 times (39); Fenit A 99.80% / 92% (99.74% / 89%); ChessReD test 302 of 306 (299). End to end in Python with the corner model in `Models/` (`scores/crop2_endtoend.txt` against `runs/dumps/tone1.txt`): board B 94.28% / 19 -> 94.72% / 24 boards exact; Fenit A 231 -> 236 of 286; ChessReD test 297 -> 299. The first candidate that is ahead on all three development sets, by a little on each. With the side rule and the queen rule both on in Python, board B is 38 with crop2 against 45 with tone1; the queen rule's Python gain is not the app's (see its entry), so only the app will tell how the two compare with the rules on. It needs the app to give the piece model the board only (`board_crop`, not ported). **Not taken as it stands: on the second test set it trades boards.** In Python at the app's photo size, as a test, tone1 -> crop2, both with canon1's corners (`runs/heldout2_crop2.sh`, `scores/heldout2_1280_{tone1,crop2}_canon1.txt`), boards exact: samryan 94 -> 153 of 500 and the chess-cv games 46 -> 84 of 840 (both photographed from straight above; squares 96.20% -> 97.51% and 95.99% -> 96.58%), ChessQ 12 -> 16 of 50 (squares 93.84% -> 93.41%); the Y-LIChess games 41 -> 32 of 102 (98.28% -> 97.90%), its layouts 22 -> 20 of 98, chesscog 13 -> 8 of 29 (98.60% -> 97.63%). By photo 228 -> 313 of 1,619; averaged by part 26.0% -> 25.3%. Two parts gain much and two lose, as with tone2 and low1, so the port is not asked for. The two that gain are the two the Python reads furthest below the app with tone1 (samryan 94 against 115 through the app, chess-cv 46 against 267), so how much of that gain the app would see is not known: this is a statement about the Python's reading. A first read on the full-size photos (`heldout2_crop2_canon1_spot07_base.txt`) is in the list of uses of that set. Two things differ from tone1 besides the crop: its photos include the low-view renders (low1 lost on Y-LIChess too), and a cropped set has only the photos with a corner label: `merged_all5_crop` has not one of the 4,243 Roboflow training photos (and 13,841 of the 25,997 renders). The crop itself is not judged by this run: one with the Roboflow photos in it (corners for 4,946 of them were derived for the corner model, `datasets/merged_rf/corners`, under other file names) and without the low views would be |
| `pieces_s_real1` | continues tone1's `last.pt`, 6 epochs, `merged_all4_real3` (`train_list.py --real-times 3`: `merged_all4` with every photo that is not one of our renders listed three times in its `train.txt`: 7,439 x 3 + 19,998 renders = 42,315 a round, real photos 53% instead of 27%), `--side-tone 0.35` | Finished 2026-11-15 20:05. **Not taken: board B no better, ChessReD test worse.** Labelled corners (`scores/pieces_s_real1_last.txt`): board B 97.07% / 19 of 98 exact (tone1 96.78% / 20), P read as p 20 times (39); Fenit A 92% (89%); ChessReD test 295 of 306 (299). Through the app against tone1 + canon1, the side of single pieces on in both (`runs/app_check/pieces_s_real1/` against `now30/`): board B 94.94% -> 95.42% of squares, 31 -> 31 boards exact (8 gained, 8 lost; more squares right on 29 photos, fewer on 17, p = 0.1); **ChessReD test 295 -> 287** (2 gained, 10 lost, p = 0.04); Fenit A 234 -> 239 (not beyond noise); as 3:4 photos ChessReD test 282 -> 282. With the side rule off this model reads board B at 24 boards, so the rule is worth 7 with it (15 with tone1). Not read on the second test set. Why: tone2, low1 and the renders before them each traded one real board against another, and low1 lost kinds on Fenit A, a board it is trained on, as the renders grew to 78% of the photos. Whether the share of real photos is what does it is the question; no low views in this one. The answer: the share of real photos is not it. This is the fourth run that continues tone1 and trades boards (tone2, low1, crop2, real1), on four different changes of data; what they share is how they continue: ultralytics' own choice of optimiser and rate for a fresh start (AdamW at 0.000625, three epochs of warm-up). Whether that alone moves a model by this much has not been measured; `pieces_s_shape1` continues at a sixth of the rate instead (`train.py --lr0`) |
| `pieces_s_shape1` | continues tone1's `last.pt`, 8 epochs, `merged_all4_shape1` (`train_list.py --render-seeds 6`: `merged_all4`'s training photos and the 8,000 renders of `--seed 6`, whose piece sets are of the wider range of `render_boards.py --shapes varied`; 35,437 a round; no low views), `--side-tone 0.35` | Why: board B's set is a heavy wooden Staunton whose king is hardly taller than its queen, and kings read as queens, rooks as queens and pawns as taller pieces were what was left there; every render before it had a king with a large cross, clearly the tallest piece. Run with `--lr0 0.0001` (AdamW at a sixth of the rate the runs above continued at, one epoch of warm-up). Finished 2026-11-16 04:22 (8 epochs, a new process after every epoch for memory; with 8 epochs and ultralytics' `close_mosaic` 10 it had no mosaic at all, as tone1's other 8-epoch continuations). **Taken: in `Models/` since pull request #32.** Labelled corners (`scores/pieces_s_shape1_last.txt`, tone1 in brackets): board B 27 of 98 exact (20), Fenit A 249 of 270 (239), ChessReD test 298 of 306 (299); epochs 1 to 7 on board B 17, 27, 26, 24, 21, 28, 29. Through the app against tone1 + canon1 (`runs/app_check/pieces_s_shape1/`): board B 94.94% -> 95.67% of squares, **31 -> 43 boards exact** (14 gained, 2 lost, sign test p = 0.004; the same as 3:4 photos), ChessReD test 295 -> 297 (3 gained, 1 lost), Fenit A 234 -> 230 (4 gained, 8 lost, p = 0.39, within noise); the side of single pieces still helps it (board B 34 -> 43). Second test set through the app, as a test (look 12, `runs/app_check/shape1_h2/`): **464 -> 640 of 1,619 boards exact**, 97.00% -> 97.41% of squares, no part worse: chess-cv games 267 -> 412, chesscog real 8 -> 23, samryan 115 -> 127, ChessQ 14 -> 17, Y-LIChess 60 -> 61. What did it is not separated (the renders and the gentle rate changed together); the per-epoch scores sit close to tone1 on the development sets, while the unseen piece sets gain much, which points at the renders' wider range of piece shapes |
| `pieces_s_shape2` | continues shape1's `last.pt`, 8 epochs, `merged_all4_shape2` (`train_list.py --render-seeds 6 7`: merged_all4 + 16,000 varied-shape renders, 43,437 a round), `--side-tone 0.35 --lr0 0.0001` | Finished 2026-11-16 18:08. **Not taken: no different from shape1.** Labelled corners (`scores/pieces_s_shape2_last.txt`, shape1 in brackets): board B 28 of 98 (27), Fenit A 241 of 270 (249), ChessReD test 297 (298); per epoch board B 26, 32, 19, ... (noisy). Through the app against shape1 (`runs/app_check/pieces_s_shape2/`): board B 43 -> 42 (8 gained, 9 lost), ChessReD test 297 -> 295, Fenit A 230 -> 230; as 3:4 photos 43 -> 42 and 285 -> 284. More renders of the same kind on top of shape1 move nothing at this rate |
| `pieces_s_shape3` | shape2 continued on seeds 6, 7 and 9 (`merged_all4_shape3`, 51,437 a round) | Stopped after 50 minutes (2026-11-16 18:57): shape2 showed the lever saturated. The seed-9 renders stay for later runs |
| `pieces_s_960a` | continues shape1's `last.pt`, 4 epochs, `merged_all4_shape3`, **`--imgsz 960 --batch 8`**, `--side-tone 0.35 --lr0 0.0002` | **Stopped after one hour (2026-11-16 19:58):** at 960 px an epoch of 51,437 photos is 6,430 batches (about 3 h), and the process reached 30 GB 40 minutes in; `queue.sh` restarts it only after an epoch, so swap would have filled the disk before the first. Run again as `pieces_s_960b`. Why: on unseen boards almost every wrong square is the kind of piece, and both the cropped-board run and the piece classifier showed that more pixels per piece help with kinds (tone1 read at 960 without training for it gained nothing). Same model size; about 2.25 times the computation per photo on the phone. Scored and exported at 960 by `runs/after_run_960.sh` |
| `pieces_s_960b` | as 960a on `merged_all4_960sub` (every real photo of `merged_all4_shape3`, 7,439, and 9,000 of its renders at random: 16,439, about 2,050 batches an epoch at batch 8), 6 epochs | Finished 2026-11-17 01:58. **Not taken: better or level everywhere, but not on board B beyond noise.** Labelled corners at 960 (`scores/pieces_s_960b_last.txt`, shape1 in brackets): board B 33 of 98 (27), Fenit A 259 of 270 (249), ChessReD test 301 (298); per epoch board B 34, 35, 36, 28, ... Through the app against shape1 (`runs/app_check/pieces_s_960b/`): board B 43 -> 45 (10 gained, 8 lost), Fenit A 230 -> 240 (12, 2; p = 0.013), ChessReD test 297 -> 299; as 3:4 photos board B 43 -> 45, ChessReD test 285 -> 296 (17, 6; p = 0.035). Continued as `pieces_s_960c`. **At 960 px keep an epoch near 2,000 batches**, or the memory leak outruns the restart after each epoch |
| `pieces_s_960c` | continues 960b's `last.pt`, 4 epochs at 960, `merged_all4_960sub2` (the real photos and 9,000 other renders), `--lr0 0.0001` | Finished 2026-11-17 05:59. **Not taken: board B through the app still within noise.** Labelled corners at 960: **board B 40 of 98** (the best of any piece model; shape1 27), Fenit A 253 of 270 (249), ChessReD test 302 (298). Through the app against shape1 (`runs/app_check/pieces_s_960c/`): board B 43 -> 47 (14 gained, 10 lost), Fenit A 230 -> 232, ChessReD test 297 -> 299; as 3:4 photos ChessReD test **285 -> 298** (17, 4; p = 0.007). On board B through the app it makes fewer kind errors than shape1 (60 squares against 86) but reads more boards half a turn round (7 against 4; exact once turned 54 against 47): there a1 on sparse endgames, not the piece model, holds it back. With the piece classifier (development only, `runs/app_check/kinds960c/`): board B 52, Fenit A 245, ChessReD test 298, about what the classifier gives with shape1 (53, 247, 299): the two fix the same errors. Continued as `pieces_s_960d` |
| `pieces_s_960d` | continues 960c, 4 epochs at 960, `merged_all4_960sub3` (real photos + 9,000 renders not in sub/sub2), `--lr0 0.0001` | Finished 2026-11-17 10:05. **Not taken: board B through the app within noise again.** Labelled corners: board B 39 of 98, Fenit A 254, ChessReD test 303. Through the app against shape1 (`runs/app_check/pieces_s_960d/`): board B 43 -> 49 (14 gained, 8 lost, p = 0.29; squares more right on 28 photos, fewer on 16, p = 0.096), Fenit A 230 -> 235, ChessReD test 297 -> 300; as 3:4 photos ChessReD test **285 -> 299** (17, 3; p = 0.003). Board B through the app across the 960 runs: 45, 47, 49. Its 6 board B photos read half a turn round have every piece right: endgames with White's king and pawns advanced (e.g. `8/6K1/7P/4kp2/8/6P1/8/8`), where the half-turned board looks as plausible to `white_side_score` and the square colours cannot tell (a half turn keeps a1 dark); the flip button mends them. Continued as `pieces_s_960e` |
| `pieces_s_960e` | continues 960d, 4 epochs at 960, `merged_all4_960sub4` (real photos + 9,000 renders in none of sub, sub2, sub3), `--lr0 0.0001` | Finished 2026-11-17 14:57. **Not taken; went back.** Labelled corners: board B 38 of 98, Fenit A 252, ChessReD test 301. Through the app against shape1: board B 43 -> 45 (15 gained, 13 lost), Fenit A 230 -> 228, ChessReD test 299; as 3:4 photos ChessReD test 285 -> 299. **The 960 line peaked at 960d** (board B through the app 45, 47, 49, 45) |
| `pieces_s_960f` | 960e continued on the last 7,998 unused renders | Stopped after one epoch (2026-11-17 16:00): 960e went back, the line had peaked |
| `pieces_m_a` | **YOLO11m** detect from the public COCO weights (`yolo11m.pt`), 640 px, 15 epochs, batch 8, `merged_all4_960sub` (every real photo + 9,000 renders), `--side-tone 0.35`, ultralytics' own rate | **Stopped after 7 of 15 epochs (2026-11-18 02:15).** Labelled corners (board B of 98, Fenit A of 270, ChessReD test of 306; shape1 27, 249, 298): epoch 2 8, 107, 240; epoch 4 17, 26, 275; epoch 7 16, 44, 289. On Fenit A, a board it trains on, it misses the light pieces (White pawns read as empty squares 199 times, kings 92): a cold start from COCO on 16k photos a round is far from the many stages behind shape1, and would not catch up in 8 epochs. A larger model would have to start from a model that already reads boards (none of that size exists here). Started 2026-11-17 18:00, about 55 min an epoch; it leaks memory faster (disk at 23 GB free at each epoch's end). Why: the 960 runs and the classifier both mend the kind of piece and their gains overlap; a larger model at 640 (about the computation of YOLO11s at 960) is the lever not tried. Scored by `runs/after_run.sh` |
| `pieces_s_rf1` | continues **shape1's** `last.pt`, 8 epochs, `--lr0 0.0001`, `--side-tone 0.35`, on `merged_rf6_shape1` (shape1's list with its 4,243 Roboflow photos replaced by `merged_rf6`: 7,775 from 28 projects, 20 of them fetched 2026-11-19 with the owner's key; 38,969 a round) | Finished 2026-11-20 14:58. Through the app with PieceKinds in place against `main` (`runs/app_check/pieces_s_rf1/`): **Roboflow development set (324 photos, 6 boards never trained on) 140 -> 152 boards exact (20 gained, 8 lost, p = 0.036), squares 96.61% -> 97.04% (more right on 78 photos, fewer on 28, p = 1e-6)**; most of it on one board (cf8lk 27 -> 40); classifier off 114 -> 129. Board B 53 -> 55 (7/5, noise), ChessReD test 299 -> 296 (0/3, p = 0.25), Fenit A 247 -> 243 (0/4, p = 0.12; squares -0.04 points, interval -0.09 to -0.01). Epoch 3: dev 140, B 49. Held until `pieces_s_rf2` (all 40 projects) is read; the second test set not read. |
| `pieces_s_rf2` | as rf1, on `merged_rf7_shape1` (all 40 Roboflow projects: 11,383 Roboflow photos; 42,577 a round) | Finished 2026-11-21 01:20. Through the app against `main` (`runs/app_check/pieces_s_rf2/`): **dev set 140 -> 159 (32 gained, 13 lost, p = 0.0066), squares p = 3e-7; classifier off 114 -> 139**; again mostly board cf8lk (27 -> 51); against rf1 152 -> 159 (18/11). Board B 53 -> 52 (6/7), ChessReD test 299 -> 299 (1/1), Fenit A 247 -> 243 (0/4, p = 0.12). Epoch 3: dev 160, B 52. **Vetoed by the second test set (look 17): 817 -> 791**; Y-LIChess 26 gained / 7 lost, chess-cv games 81/55, ChessQ 5/1, chesscog real 5/0, but **samryan 2 gained / 82 lost, squares -5 points: white pieces missed** (813 white pawns that main read). Cause found: two of the new projects leave many pieces unlabelled (`runs/unlabelled_check.py`: main's model finds unlabelled pieces at about half the labelled count in chess-pieces-paoho and chess-piece-detection-qskmb, a top-down green mat; under 3% elsewhere). Both dropped; `pieces_s_rf3` retrains without them. |
| `pieces_s_rf3` | as rf2 without the two projects with unlabelled pieces (`merged_rf7b_shape1`, 42,054 a round) | Finished 2026-11-21 ~12:05. Through the app against `main`: **dev set 140 -> 160 (28 gained, 8 lost, p = 0.0012)**, board B 53 -> 49 (5/9), ChessReD 299 -> 297 (1/3), Fenit A 247 -> 244 (0/3). **Vetoed by the second test set (look 18): 817 -> 772**; samryan now level (32/43) but chess-cv games 47 gained / 92 lost (black bishops read as knights 31 times, queens as rooks). |
| `soup_rf2_rf3` | weights of rf2 and rf3 averaged (`runs/soup.py`; both fine-tuned from shape1) | Dev set 140 -> 160 (28/8, p = 0.0012; squares p = 2.5e-8), board B 53 -> 55 (8/6), ChessReD 299 (1/1), Fenit A 247 -> 242 (0/5). **Vetoed by the second test set (look 19): 817 -> 745** (samryan 13/57, chess-cv 45/83). Soup of shape1 and rf3: dev 142, B 54, ChessReD 296 (not read further). |
| `cls_s_crops3` | the piece **classifier** cls_s_crops1 continued 4 epochs (lr0 3e-4) on `datasets/crops3` (742,748 crops from `merged_rf7b_shape1`: the new Roboflow boards in), exported as crops1 was (ultralytics CoreML, 128 px, FP16); used as PieceKinds with main's piece and corner models | Finished 2026-11-21 14:41. Through the app against `main` (`runs/app_check/cls_s_crops3/`): dev set 140 -> 145 (12 gained, 7 lost, p = 0.36; squares more right on 55 photos, fewer on 27, p = 0.003), board B 53 -> 49 (4/8), ChessReD test and Fenit A unchanged. Not taken (boards within noise, board B down); the second test set not read. |
| `corners_s_canon1` | continues exp4's `last.pt`, `--kpt-sigma 0.1 --canonical-corners`, 12 epochs, `merged_rf_render` (as `merged_rf`, plus the renders whose corners are in the picture) | Finished 2026-11-14 11:12. `last.pt` (`scores/corners_s_canon1_last.txt`), median corner error in squares: **board B 0.27** (exp4 0.91; within half a square 83% against 32%), ChessReD test **0.14** (exp4 0.19, baseline 0.31), Fenit A 0.11. Which corner is a1 it gets right on ChessReD (301 of 306) and wrong on board B (11 of the 90 boards it finds there; the app takes a1 from the pieces). The first learned corner model that passes the package's tests (103 of 103, the G033 endgame fixture read right with the grid fit off). **In `Models/`**: see below |
| `corners_s_exp2` | continues exp1 `last.pt`, `--kpt-sigma 0.05`, 20 epochs | dropped: exp3 and exp4 already show what a tighter tolerance does |

**In `Models/` since 2026-11-14 evening: `pieces_s_tone1` and `corners_s_canon1`**
(FP16 exports, 18 MB and 19 MB; `runs/export_tone1/Pieces-fp16.mlpackage`,
`runs/export_canon1/BoardCorners-fp16.mlpackage`; weights
`baseline/pieces_yolo11s_tone1.pt`, `baseline/corners_yolo11s_canon1.pt`).
Against the pair they replace (`pieces_s_render3` with the first corner
model), through `app_check` with the app's defaults (`runs/app_check/tone1/`,
`final/render3_*.txt`, `final_h2/`; `compare_app_check.py`). The two new
columns are from the build of 2026-11-14 09:02; the first column is from
the build before it, which reads board B and ChessReD test the same photo
for photo (`runs/app_check/after16/`). Squares right; boards exact; photos with no board
found:

| | Pair they replace | `pieces_s_tone1`, old corners | `pieces_s_tone1` + `corners_s_canon1` |
|---|---|---|---|
| Fenit board B (100; held out) | 88.55%; 6; 4 | 91.19%; 15; 4 | **94.36%; 16; 0** |
| ChessReD test (306) | 99.68%; 288; 0 | 99.67%; 291; 0 | 99.80%; 295; 0 |
| Fenit A (286; trained on) | 95.32%; 183; 0 | 95.41%; 197; 0 | 97.79%; 231; 0 |
| Second test set: `ylichess` (200) | 85.73%; 45; 6 | 85.83%; 45; 6 | 88.16%; 44; 0 |
| `chessq` (50) | 71.66%; 14; 10 | 71.28%; 12; 10 | 93.03%; 14; 0 |
| `chesscog_real` (29) | 98.44%; 10; 0 | 98.17%; 8; 0 | 98.17%; 8; 0 |
| `samryan` (500) | 84.85%; 58; 16 | 85.10%; 80; 16 | 95.50%; 106; 0 |
| `chesscv_games` (840) | 91.97%; 171; 13 | 92.42%; 242; 13 | 97.99%; 267; 0 |
| Second test set, all 1,619 | 88.49%; 298; 45 | 88.80%; 387; 45 | **95.86%; 439; 0** |

Photo by photo, the new pair against the pair it replaces:

- **Board B:** more squares right on 57 photos, fewer on 14 (sign test
  p = 0.0000003); 5.8 points of squares more (95% bootstrap interval 3.2 to
  9.1); 13 boards gained, 3 lost (p = 0.02).
- **ChessReD test:** 12 boards gained, 5 lost (p = 0.14); +0.12 points of
  squares (-0.17 to +0.45). Not worse.
- **Second test set** (a test, nothing chosen on it): 7.4 points more (6.4
  to 8.4); 232 boards gained, 91 lost.

Each model's share on board B: the piece model alone, with the old corners,
gains 2.6 points (1.7 to 3.7; more right on 52 photos, fewer on 15; 12
boards gained, 3 lost), which meets the rule by itself; the corner model on
top of it gains 3.2 more, all of it on 5 photos, 4 of them boards the old
corner model did not find, and makes no photo worse. That is 5 photos
against none (sign test p = 0.06): by itself not shown beyond noise, as it
was not with render3's pieces (8 against 6). (The bootstrap gives 0.6 to
6.4 points, but with no photo worse it cannot reach zero and is no test
here.) Of the four boards found, one is read fully and three badly (30, 40
and 49 of 64 squares). On the
second test set the corner model is the larger part (88.49% -> 95.44% with
render3's pieces) and the piece model adds 0.4 points of squares and 109
boards (330 -> 439). So what speaks for the corner model is the pair's
result on board B, which the rule asks for, and its own on the second test
set, which the rule does not name: with this change that set is no longer
one that nothing was chosen on.

**What the corner model costs.** It reads two ChessReD test photos of game
G033 (`G033_IMG100`, `G033_IMG102`: endgames of two kings and four pawns)
turned by 180 degrees, 52 of 64 squares, which the first corner model reads
right; they are two of the five boards lost on ChessReD test. In such a
position the pieces cannot say which way the board lies, and the first
corner model knew ChessReD's own board better (which corner is a1: 301 of
306 with canon1). The package's fixture of that game, `test_004005.jpg`, is
read right as it is and with the grid fit off, and turned when the test
redraws it as a portrait photo (52 of 64; that test asks for 90% of squares
over the four fixtures and gets 95.3%). The fixture is `G033_IMG101`, the
frame between the two: of three neighbouring frames of that endgame it reads
one right as a square photo and none as a 3:4 photo. The weight given to
the model's own a1 does not change it (the same two photos at
`--a1-prior` 0, 3, 6 and 12). A turned board is one tap in the app. exp3
and exp4 turned the fixture itself and were not put in.

As 3:4 photos, the shape the app's camera gives (`app_check --portrait`,
`runs/app_check/portrait/`): ChessReD test 282 boards exact with the new
pair, 281 with the new piece model and the old corners, 275 with the pair
it replaces; the new pair turns two of the 306 (`G033_IMG101`,
`G033_IMG102`), the old corner model none; board B 94.33%, 91.16% and
88.56% of squares. So the turn stays at two photos of that one endgame.

On the second test set, boards that are right once turned by 180 degrees:
12 with the new pair, 18 with the new piece model and the old corners (4
with the pair it replaces, whose pieces make fewer boards right either
way). Two boards the pair it replaces read fully (`wood_0033`, `wood_0036`
of Y-LIChess) are turned by the new corner model; 15 photos lose 5 squares
or more, 254 gain as much.

With labelled corners (`evaluate.py`, which prints the interval), board B:
`pieces_s_tone1` 96.8% of squares and 20 of 98 boards exact (14%-29%);
render3 94.7% and 8 (4%-15%). On Fenit A, which both trained on, it reads
fewer boards fully than render3 did (88.5% against 92.2%); on ChessReD test
more (299 against 296).

**Board B is not a blind test** (see the same heading under render3 below):
the tone augmentation was thought of from what goes wrong on board B, the
run was scored on it after epochs 1 and 3, and `--side-tone`'s limits were
set by eye on other photos. For the piece model the second test set,
which it was not chosen on, agrees (with the old corners: 387 boards exact
against 298).

**The mark for the user** (confidence under 0.85, not changed): wrong
pieces marked on board B 57% (147 of 259; render3 54% of 351), ChessReD
test 50% (12 of 24), Fenit A 23%; right pieces marked on board B 15%; on
the second test set 49% of wrong pieces and 8.2% of right ones.

**What the app-side steps are worth with this pair** (app-side session, its
build of the same source; per-photo outputs `runs/app_check/tone1_switches/`;
tables in `docs/RECOGNITION.md`, "What the app reads with the third pair").
Much less than before, and on two sets they cost:

- *a1 from the pieces.* Board B needs it: 76.5% of squares without, 94.3%
  with (the corner model's own a1 is wrong there). On Fenit A, where the
  corner model has a1 right by itself, the pieces overrule it: 254 boards
  exact without the step, 230 with (16 turned by a half, 8 by a quarter).
  The head start given to the model's own a1 (`--a1-prior`, 6 in the app)
  at 6 / 12 / 20 / 40 / pieces off: Fenit A 231 / 243 / 251 / 255 / 255
  boards; board B 94.36 / 94.36 / 93.83 / 92.20 / 76.50% of squares;
  ChessReD test 295 at every value from 6 up (0 / 3 give 292 / 294, and
  Fenit A 214 / 223). So 12 costs nothing on these three sets and gains 12
  boards on Fenit A, **and it stays at 6**: tried on the second test set as
  a test (app-side session, 2026-11-14; `docs/RECOGNITION.md` has the
  account), 12 gives 95.71% of squares against 95.86% and 440 boards exact
  against 439. What it does there is turn boards, not misread pieces: 24
  photos upright at 6 are turned at 12 (23 of them by a quarter, which the
  app's flip button does not undo), 5 the other way. The photos that change
  have few pieces (35 of 36 have nine or fewer), and on them the corner
  model's own a1 is wrong on 31 and right on 5; on Fenit A, a board it was
  trained on, it is right on 12 of 13. The 25 worse photos are not 25
  independent ones (runs of neighbouring positions from two sources; by
  run about 8 worse and 3 better), so the sign test over photos overstates
  it; the decision needs no p-value: no gain on unseen boards. 6 is the
  incumbent, not shown to be the best value: the development sets cannot
  place it any more (board B reads the same from 0 to 12; the other two are
  boards the corner model knows). Choosing again needs an unseen
  development board with sparse positions, not another look at the second
  test set. No value mends the two G033 endgames: they score better turned
  at any head start.
- *The grid fit.* It mends board B (+131 squares on 11 photos, 2 lost) and
  changes one Fenit A photo for the worse. On the second test set it
  changes nothing on three parts, mends ChessQ, and on Y-LIChess 54 photos
  gain 196 squares and 21 lose 680: on 18 of them the fit accepts a grid
  that is not the board, mostly one rank off. The development sets have no
  such photo; the rule that came of it ("The corner model picks the grid
  fit's window", below) was tested there on windows moved on purpose.
- Quarter turns were a real part of what was left: with the app's defaults
  of that build 26 boards of the second test set were exact once turned by
  a quarter (12 by a half), and 8 against 16 on Fenit A. That is what the
  next paragraph is about; with it they are 0.

**a1 is a dark square (2026-11-14; in the reference and, since pull request #22, in the app, on by default).**
Quarter turns are the larger part of what is left on unseen boards, and no
head start mends them. The square colours do, without any model: on a board
set up as the rules say a1 is dark, so of the four ways the board can lie
two are left, half a turn apart, and the pieces choose between those two
only. `grid_corners.dark_corner_rotations` reads the two rotations from the
64 cells (it was written to label corners, `derive_corners.py`; where the
colours do not split cleanly it says nothing and all four stay);
`chess_pipeline.orient_by_pieces(..., turns=)` takes them;
`evaluate.py --a1-dark` switches it on. End to end in Python with the pair
in `Models/` (grid fit, head start 6, one king a side), without -> with.
Squares right; boards exact; exact or half a turn away:

| | Without | With `--a1-dark` |
|---|---|---|
| Fenit board B (100) | 94.16%; 19; 23 | 94.28%; 19; 23 |
| Fenit A (286) | 97.83%; 229; 245 | 98.14%; 231; 253 |
| ChessReD test (306) | 99.83%; 297; 299 | the same |
| Second test set, as a test: `chessq` (50) | 93.44%; 12; 13 | 93.84%; 12; 14 |
| `ylichess` games (102) | 90.23%; 35; 35 | 90.18%; 35; 35 |
| `ylichess` layouts (98) | 88.38%; 13; 19 | 90.53%; 22; 32 |
| `chesscog_real` (29) | 98.60%; 13; 13 | the same |
| `samryan` (500) | 95.28%; 86; 93 | 96.20%; 94; 102 |
| `chesscv_games` (840) | 95.99%; 46; 46 | the same |

(`runs/scores/a1_dark_pair.txt`, `heldout2_pair_tone1_canon1*.txt`. These
are Python figures: they differ a little from the app's, and the chess-cv
photos are read here without their EXIF turn, which is why so few of them
are exact; both columns are read the same way.) It was chosen on the three
development sets, where no set loses (through the app three Fenit A photos
lose 5 squares), and then run once on the second test set: better on three
parts, the same on two, three squares worse on one. That quarter turns were
worth going after was seen on the second test set first (the 26 boards
above); the rule itself is the law of the game with the thresholds
`derive_corners.py` already had. The corners as the model gave them had a light a1 on 34 of board B's
photos, 107 of samryan's and 40 of the Y-LIChess layouts'. Two things it
gets wrong. A board set up with a light square on a1 is read a quarter turn
off, every time, and the app's flip button does not undo a quarter; none of
these sets has one. And where the grid is found one square off, the colours
are the other way round and the rule forces a quarter turn on a reading
that was wrong already: that is 8 of the 13 photos of the second test set
that read worse with it (Y-LIChess full boards, 21-25 squares before, 19-24
after). On those boards the rule makes a quarter turn where there was none:
16 Y-LIChess full boards that were read the right way round and one rank
off are a quarter turn off as well with it (app-side session's count; the
squares hardly move, 357 before and 358 after, so the scores do not show
it). "No quarter turns" holds for boards that are exact once turned.
Counted by where the pieces land, readings a quarter turn off go from 56 to
19 on the second test set (Y-LIChess 20 -> 18, samryan 32 -> 0, ChessQ 4 ->
1), from 10 to 1 on Fenit A and from 2 to 0 on board B. The cure for the
rest is not in this rule but in the grid fit's choice of window (below).

Through the app (app-side session's port, `CoreMLBoardRecognizer(a1Dark:)`,
`app_check --a1-dark on|off`; `docs/RECOGNITION.md`, "What a dark a1 is
worth"), the pair in `Models/`, off -> on: ChessReD test the same (295
boards exact); board B 94.36% -> 94.48% of squares, 16 exact both; Fenit A
231 -> 233 boards exact, 247 -> 255 exact or half a turn away; second test
set, once, as a test: 95.86% -> 96.38% of squares, 439 -> 459 boards exact,
boards a quarter turn off 26 -> 0, no fully right board lost. Swift and
Python give the same answer on all 2,311 photos.

With the pieces no longer able to turn a board by a quarter, the head start
for the model's own a1 only trades half turns, and stays at 6. In Python with `--a1-dark`, head start 0 / 6 /
12 / 20 / 40 (`runs/scores/a1_prior_dark.txt`, `a1_dark_pair.txt`): Fenit
A, a board the corner model knows, 228 / 231 / 245 / 250 / 253 boards
exact; board B, one it does not, 94.28 / 94.28 / 94.08 / 93.70 / 93.06% of
squares, with 14 / 14 / 15 / 17 / 21 boards read half a turn off; ChessReD
test the same at every value. More trust in the model's a1 mends half
turns on a known board and makes them on an unknown one, and the unknown
one is what the app is for.

**The corner model picks the grid fit's window (2026-11-15; in the
reference and, since pull request #25, in the app, on by default).** The classical fit finds the lines of the
grid more exactly than a model does, but it has to choose which 8x8 squares
of the grid are the board, and a border that looks like a rank of squares
makes it take a window one rank off: 16 to 18 photos of the Y-LIChess
board, which the dark-a1 rule then turns by a quarter as well. The corner
model is less exact and seldom a whole square off. So the fit keeps the
lines and the model picks the window: `chess_pipeline.window_by_model`
looks at the model's four corners in the fit's own board frame and, when
all four lie within `WINDOW_NEAR` (0.75) squares of the same shift by one
square, moves the fit's window by it; otherwise nothing changes.
`evaluate.py --fit-window` switches it on; `--shift-fit FILES,RANKS` moves
the fit's window on purpose, to test it.

The development sets have no photo with a window off, so it was measured
there in two ways (Python, the pair in `Models/`, `--a1-dark`;
`runs/scores/fit_window_075.txt` at the tolerance it has;
`fit_window.txt` is the same at 0.5, with the damage a moved window does
without the rule). Left alone, the rule never fires on board B, Fenit A or
ChessReD test (692 photos, 659 of them with both a fit and model corners):
no cost. With the window moved one rank on purpose, squares right without
the rule -> with it: Fenit A 68.2% -> 98.1% (277 of 278 windows moved
back; 98.1% with no shift), ChessReD test 53.0% -> 99.5% (287 of 289),
board B 63.2% -> 84.8% (60 of 92; the corner model has not seen that
board). At a tolerance of 0.5 board B's count is 46 and at 0.95 it is 66,
and no value fires on a fit left alone: nothing on these sets tells 0.75
from 0.95, and 0.75 is a middle value, not a measured best. A window is
moved back exactly when all four of the model's corners lie within the
tolerance of the fit's own, so the count is the same whichever way the
window was moved, and "squares right with the rule" is computed from the
model's agreement with a right fit, not measured on a real fault: a real
one comes from a border, and the window next to it is an extrapolation.
The only real faults seen are the Y-LIChess photos below and one Fenit A
photo through the app.

Then once on the second test set, as a test
(`runs/scores/heldout2_pair_tone1_canon1_a1dark_window.txt` against
`..._a1dark.txt`): it fires 14 times, all on Y-LIChess. Games 90.18% ->
98.28% of squares, 35 -> 41 of 102 boards exact; layouts 90.53% -> 90.80%
(one window moved); ChessQ, chesscog, samryan and chess-cv unchanged, not
one window moved. What it cannot mend is a photo where the corner model is
a rank off too. What it gets wrong: a window that was right is moved when
the model's corners lie, on average, more than half a square off along a
rank or file and each within 0.75 of the whole square, so every corner at
least a quarter of a square off the same way. None was seen, but the
margin is thin: on a fit left alone the largest mean offset of the model's
corners is 0.72 squares on board B and 0.70 on ChessReD test in Python,
and only the corners' disagreement with each other keeps those windows
where they are (through the app, on the 2,000 photos left alone where both
found the board, the corner least far off the common way is never more
than 0.14 squares off it).

Through the app (app-side session's port, `app_check --fit-window on|off`;
`docs/RECOGNITION.md`, "What the model's choice of window is worth"), the
pair in `Models/`, off -> on: ChessReD test and board B unchanged, no
window moved; Fenit A 233 -> 234 boards exact (one endgame whose window
the app's fit had a file off, 55 -> 64 squares); second test set, once, as
a test: 96.38% -> 97.00% of squares, 459 -> 464 boards exact, 18 windows
moved, all on Y-LIChess (89.80% -> 94.81%; 17 of the 18 gain 665 squares;
one, `wood_0202`, ends on the right squares half a turn round and scores 0
where it had 24), and readings a quarter turn off go from 19 to 1. Given
the same corners, the Swift and the Python make the same shift on all
2,019 photos where both found the board; end to end the Python moves 14
windows on the second test set and the app 18, because their corners
differ a little.

The gain on Y-LIChess is not a blind test of the rule: the fault was found
on those very photos, and all the windows moved are theirs. What is blind
is that nothing else on the second test set changed. The tolerance came
from board B. On the second test set 0.5 and 0.95 move the same windows as
0.75 (computed afterwards; nothing was chosen by it); through the app
on board B windows moved on purpose come back on 68 / 80 / 88 of 95 at
0.5 / 0.75 / 0.95, so a wider tolerance is an open choice for the
development sets.

**The side of single pieces (2026-11-15; in the reference, `chess_pipeline.odd_sides`
and `evaluate.py --odd-sides`; in the app and on by default there since
pull request #30; `evaluate.py` leaves it off unless asked).** Through
the app the pair in `Models/` reads 30 of board B's 100 photos with one
square wrong and 16 with two, and of those 62 squares 28 are a white piece
read as black (20 of them pawns). The model is right about most pieces of
such a photo, so its own answers say what colour each side has there: the
median colour of the pieces it calls White and of those it calls Black. A
piece called Black whose colour is twice as near White's as the other Black
pieces' is made White. Three guards: the two colours must be three spreads
apart and White's the lighter; a piece with another standing in front of the
patch its colour is measured on is left out (without that, three pieces of
ChessReD test are changed in Python, and five while the rule still worked
both ways: all wrongly, all behind a piece of the other side); and it works
one way only, Black to White, because a light piece in a
shadow is as dark as a black one and little makes a dark piece light.

End to end in Python with the pair in `Models/` (`--one-piece-per-spot 0.7
--grid-corners --fit-window --a1-from-pieces 6 --a1-dark --one-king`,
`runs/dumps/tone1.txt` -> `runs/scores/odd_sides_tone1.txt`): **board B
94.28% -> 94.98% of squares, 19 -> 36 boards exact**, 48 pieces changed on
38 photos, every one of them rightly; Fenit A (231 of 286) and ChessReD test
(297 of 306): not one piece changed.

What it does to pieces that are right was measured where the sides are
known: the 9,178 real photos of `merged_all5` with their labelled boxes
(eleven sources; `side_rule_check.py`, which writes
`runs/dumps/truth_all5.jsonl`). Changing sides both ways, as the rule first
did (that form is no longer in the code),
it changed a right piece on 301 photos (3.3%), and four in five of the
408 pieces are light pieces made Black: chesscog's light wood in the shadow
of its spot lamps (229), then the tournament photos of
`chess-vision-ljby5`, where hands cover pieces. One way, it is 65 photos
(0.7%): 48 of ljby5's 1,500 and 17 of the other 7,678 (0.2%). Of one
white piece a photo relabelled Black it puts 76% back. Stricter numbers
change fewer right pieces and mend fewer wrong ones: `nearer` 3 takes the
65 photos to 42 and board B's 36 boards to 30; `apart` 6, to 55 and 34;
both, to 37 and 28. The numbers stay where board B had them. Round
them board B is flat only a little way (`runs/dumps/tone1_fenit_B.jsonl`,
36 boards at the numbers as they are): `nearer` 1.5 to 2 gives 36 and 2.5
gives 35; `chroma` 3 to 5 gives 36, 2.5 gives 35 and 2 gives 33; `apart` 2
to 4 gives 36 and 1.5 gives 37; `cover` 0.1 to 0.2 gives 36 and 0.3 to 0.5
gives 37.

Then the second test set, as a test, in Python with the same flags for both
(`runs/scores/heldout2_pair_tone1_canon1_spot07_base.txt` against
`..._oddsides_oneway.txt`): 53 pieces changed on 50 photos (47 on samryan, 5
on the Y-LIChess layouts, 1 on ChessQ; none on the Y-LIChess games, chesscog
or the chess-cv games); boards exact the same on every part (14, 24, 21, 13,
33 and 43), samryan 15 squares more, the rest not a square different (the
changes on Y-LIChess and ChessQ are pieces standing beside the board). So on
nineteen boards it has not seen it mends next to nothing and breaks
nothing: the gain is board B's, the board whose fault it was written for.
The two-way rule, run there first, changed 71 pieces for 14 squares more
and one board fewer; that it was no better than a coin there is what sent
me to the training photos, where the shadow showed. These Python runs
left out `--max-side 1280` by mistake: they read the full-size photos, which
ultralytics shrinks without smoothing. That is why samryan reads so low in
them (6.6% of boards; 18.8% with the flag, `heldout2_1280_tone1_canon1.txt`;
23% through the app). The chess-cv games read low in Python with the flag
or without (5.1% and 5.5% of boards, against 32% through the app), which
the flag does not explain: the Python reads those photos without their
EXIF turn (said where a1 is a dark square, above), and whether that is all
of the gap has not been measured. Like is compared with like, but on
poorer readings than the app's; what counts is the next paragraph.

**Through the app** (`app_check` built from b3e18dd, pull request #28 in;
the pair in `Models/`, `--odd-sides off` -> `on`, photo by photo;
`runs/app_check/odd/`, `odd_h2/`; off reads every list as the build before
did): **board B 94.48% -> 94.94% of squares, 16 -> 31 boards exact**, more
squares right on 32 photos and fewer on 1 (sign test p = 8e-9), 15 boards
gained and none lost. The one that loses is a five-piece endgame whose pawn
is mended and which a1 from the pieces then reads half a turn round (63 ->
54 squares, exact once turned). ChessReD test (295) and Fenit A (234): no
reading differs. Second test set, as a test, by part: not one reading
differs on any of the 1,619 photos (readings were compared there, pieces
were not counted: one made White beside the board would not show). On board
B the app changes 41 pieces where the Python changes 48. So the rule is worth
fifteen boards on board B and nothing, either way, on the nineteen boards
of the second test set: on by default in the app since pull request #30.

What it costs, found by the app-side session when it switched it on
(`docs/RECOGNITION.md`, "What the side of single pieces is worth"): drawn
as 3:4 photos, ChessReD test loses one square on one photo (G000_IMG064, 63
-> 62; 282 boards exact either way; board B as 3:4 gains as above, 16 ->
31). A black rook stands behind a white knight; as the photo is taken the
piece model finds the knight, the rook counts as hidden and is left alone;
drawn as 3:4 it does not find the knight, nothing stands in front of the
rook for the rule, and the rook, whose patch shows the knight, is made
White. So the guard depends on the piece model finding the piece in front.
The second test set served this rule as a gate (items 7 and 9 of its list
of uses): the run through the app tests for harm only. And the rule mends
a fault of this piece model: it is to be measured again whenever the piece
model changes.

**A side without a king takes its queen (2026-11-15; in the reference,
`one_king_a_side(..., queen_for_king=True)` and `evaluate.py
--queen-for-king`; in the app since pull request #29, off by default in
both, and to stay off: see the end of this entry).** With the sides
mended (the paragraphs above, Python), board B has 36 boards right, 21 one
square off and 13 two off, and of those 47 squares 15 are the black king
read as a queen and 2 the white one. `one_king_a_side` leaves them: it
crowns a piece only if the model gave it a "king" score of 0.02, and these
the model is sure are queens. A game position has both kings, and a king is
seldom not found at all, so a side left without one takes its queen; of two
queens, the one the model is less sure of (tried on board B's dump: 13
photos gain a square and 4 lose one; with the taller box, measured in
squares at its foot, 13 and 7).

End to end in Python, on top of the side rule
(`runs/scores/odd_sides_tone1.txt` -> `queen_for_king_tone1.txt`): **board
B 36 -> 45 boards exact** (squares 94.98% -> 94.97%: where it crowns the
wrong one of two queens a board one square off becomes two off); ChessReD
test unchanged (297); Fenit A 231 -> 229. Fenit A's losses are what the
one-king rule loses already: boards set up without a king, where a real
queen is crowned. Such a position is not one the engine can be asked about.

Then the second test set, as a test, in Python, the side rule on in both
(`..._spot07_oddsides_oneway.txt` against `..._oddsides_queenking.txt`): no
part worse. Boards exact: samryan 33 -> 42 of 500 (93.35% -> 93.45% of
squares), the chess-cv games 43 -> 66 of 840 (95.92% -> 96.11%), the
Y-LIChess games 24 -> 25, its layouts 21, ChessQ 14 and chesscog 13
unchanged; 148 -> 181 in all. The two that gain are the ones photographed
from straight above, where a king and a queen look most alike. These runs
too left out `--max-side 1280` (see above), so they are of readings poorer
than the app's.

**Through the app it does next to nothing, and stays off.** Measured by the
app-side session on the development lists (pull request #29; the pair in
`Models/`, one build; neither / queen for king / odd sides / both): board B
16 / 17 / 31 / 32 boards exact; Fenit A 234 -> 230 with it (boards set up
without two kings); ChessReD test unchanged. Why the Python gain (36 ->
45) does not carry over has not been measured. The two differ in the
scores a piece has for other kinds (`evaluate.py` makes them from
overlapping boxes, the app has the model's own twelve scores for the one
box) and in the boxes themselves. What has been counted is this. On board
B through the app 22 kings are read as queens; 11 of them are on readings
that already have two "kings" (on 8 the model itself called another piece a
king, on 3 the one-king rule crowned one), which the rule leaves alone; and
where the switch acts (10 photos) the queen it crowns is the king on 3, a
real queen on 2 and no queen of that side at all on 5.

**The king step with the app's scores: tried, nothing taken.** With every
piece's twelve scores written out (`app_check --pieces`,
`runs/app_check/now30/pieces_*.json`) and what follows the pieces replayed
in Python (`swift_reference.replay_pieces`, which gives the app's reading
on all 692 development photos; `runs/dumps/king_replay.py`): on board B the
model reads 28 kings as queens, and 27 of them have some "king" score, half
of them under the 0.02 the rule asks for. Crowning a queen that has any
king score before anything else crowns the king on 13 of the 21 kingless
sides in reach instead of 6, and gives board B 31 -> 33 boards exact (more
squares right on 5 photos, fewer on 3: a crowned king can turn a1) and
Fenit A 234 -> 232; no floor at all for any piece, the same on board B and
230 on Fenit A. Two boards for two: the step is left as it is.

**A larger input for the piece model: probed, not trained (2026-11-15).**
On board B's photos a piece is 20 to 28 pixels tall in the model's 640
pixel input. `evaluate.py --pieces-imgsz` and `export_coreml.py
--pieces-imgsz` give the same weights a larger input, and the app takes
such a model as it is (it reads the input's side from the model). With
`pieces_s_tone1`, which was trained at 640: labelled corners in Python
(`scores/pieces_s_tone1_imgsz{960,1280}.txt`), 640 / 960 / 1280: Fenit A
89% / 91% / 78% of boards exact, ChessReD test 97.7% / 98.4% / 93.1%, board
B 20 / 12 / 1 of 98 (at 960 the sides go: P read as p 53 times against 39,
K as k 24 against 5; of the kinds, k as q 11 against 23 and r as q 17
against 7). Through the app at 960 (`runs/app_check/hi/` against `now30/`):
board B 31 -> 14 boards exact, Fenit A 234 -> 221, ChessReD test 295 ->
293. So a model that never saw pieces that large gains nothing from them
through the app; what a model trained at 960 would do is not known. It
would cost about two and a quarter times the training time and as much on
the phone.

**What is left on board B through the app with tone1 + canon1** (with the side of single
pieces on, `runs/app_check/now30/now_fenit_B.txt`): 31 boards are right, 22
are one square off, 10 two, and 18 are eight or more off (14 of them
photos of one endgame, read half a turn round and with a side or a kind
wrong as well). The one square: a black pawn on an empty square 6 times, r
as q 3, k as q 3, r as b 2, the rest once each. Before the side of single
pieces (`runs/app_check/tone1/tone1_canon1_fenit_B.txt`) it was 16, 30 and
16, and the one square was P read as p 8 times. The phantom pawn is the head of a tall piece seen as a pawn of
its own: its box lies inside the king's or queen's, with too little overlap
for the model's own NMS to drop it. **Tried and not taken:** dropping a box
of which more than a share lies inside the larger box of a more confident
detection. It takes real pawns that stand in front of or behind a taller
piece with it: with labelled corners, at a share of 0.9 / 0.8 / 0.7 / 0.6,
ChessReD test falls from 97.7% of boards exact to 91.5 / 88.2 / 83.0 /
75.5%, and board B gains nothing (20.4% -> 20.4 / 19.4 / 18.4 / 14.3%;
`runs/scores/drop_inside_tone1.txt`). Boxes cannot tell the two apart; the
model has to.

**Also tried and not taken: no more of a kind on a side than a set has.**
Through the app the pair in `Models/` reads a second queen, a third knight
or the like on 32 of board B's 100 boards (9 of Fenit A's 286, 5 of
ChessReD test's 306), and no position of these sets has one. A rule after
the two-kings rule that renames the surplus pieces to the kind the model
scored next best (kings never) can only act where the model gave a second
score, and for most of them it gives none: it is sure of its second queen.
With labelled corners and `pieces_s_tone1` it changes 4 boards of board B
(96.83% -> 96.88% of squares, no board gained), none of Fenit A and 2 of
ChessReD test (299 -> 301 boards exact; `runs/scores/set_limits_tone1.txt`).
Too little for a rule that reads a promoted piece wrong; not in the
reference.

The goal is still far: 16% of boards fully right on board B and 28% on the
second test set (27% before a1 on a dark square), where the target is 85%. What is wrong on board B now,
with labelled corners: P read as p 39 times, kings read as queens 23 times,
Q read as q 15.

Trained on: pieces ChessReD2K (CC BY-NC-SA 4.0), chesscog, Fenit board A,
the Roboflow projects in `sources.yaml`, 20,000 renders; corners ChessReD2K,
chesscog, Fenit board A, Roboflow photos with derived corners, renders.
Because of ChessReD both are for personal builds only.

**The piece model before it: `pieces_s_render3`** (in `Models/` from pull
request #15 on 2026-11-14 morning until the evening; weights
`baseline/pieces_yolo11s_render3.pt`). "The pair in `Models/`" in the next
paragraphs, down to "The first pair", is render3 with the first corner
model.

(FP16 export, 18 MB, `runs/export_render3/`; weights
`baseline/pieces_yolo11s_render3.pt`; with the first corner model, which stayed then).
It is the first model that meets the rule for replacing one as `CLAUDE.md`
words it: better than the pair it replaces on the held-out board B by more
than the noise, and not worse on ChessReD test. Through `app_check` with the
app's defaults (grid fit, a1 prior 6, one king a side, rim tolerance 0.25,
each photo decoded once), one build of `app_check` for the pair it
replaces, render2 and render3 (`runs/app_check/final/`, `final_h2/`). The
render1 column on the second test set is from the build before
(`runs/app_check/exif_full_decode/`, `one_king/`), whose sources differ in
nothing that changes a score: the pair it replaces gives the same figures
with both (76.08% and 3, part by part). Squares right; boards exact:

| | Pair it replaces | `pieces_s_render1` | `pieces_s_render2` | `pieces_s_render3` |
|---|---|---|---|---|
| Fenit board B (100; held out) | 84.67%; 1 | 84.33%; 1 | 82.98%; 0 | **88.55%; 6** |
| ChessReD test (306) | 99.54%; 280 | 99.64%; 285 | 99.70%; 293 | 99.68%; 288 |
| Fenit A (286; the three candidates trained on these photos) | 91.69%; 32 | 95.41%; 185 | 95.35%; 183 | 95.32%; 183 |
| Second test set: `ylichess` (200) | 72.5%; 2 | 85.0%; 23 | 85.5%; 47 | 85.7%; 45 |
| `chessq` (50) | 64.6%; 1 | 69.8%; 6 | 71.6%; 12 | 71.7%; 14 |
| `chesscog_real` (29) | 73.7%; 0 | 93.6%; 0 | 97.6%; 8 | 98.4%; 10 |
| `samryan` (500) | 71.1%; 0 | 83.3%; 18 | 84.2%; 29 | 84.9%; 58 |
| `chesscv_games` (840) | 80.65%; 0 | 89.72%; 21 | 91.5%; 100 | 92.0%; 171 |
| Second test set, all 1,619 | 76.08%; 3 | 86.62%; 68 | 87.98%; 196 | **88.49%; 298** |

Photo by photo, render3 against the pair it replaces (same corner model, so
the same photos have a board):

- **Board B:** more squares right on 62 photos, fewer on 23 (sign test
  p = 0.00003); 3.9 points of squares more (95% bootstrap interval 2.1 to
  5.8); 6 boards gained, 1 lost. The count of exact boards alone, 6 against
  1 of 100, is not beyond noise (p = 0.125); the squares are. So "beyond
  the noise" rests on the paired test of squares, not on the interval
  `evaluate.py` prints for exact boards: with labelled corners that is 8 of
  98 (4%-15%) against 1 of 98 (0%-6%) for the model it replaces (94.7%
  against 90.2% of squares; `scores/pieces_baseline.txt`), and the two
  intervals overlap.
- **ChessReD test:** 10 boards gained, 2 lost (p = 0.04); more squares right
  on 17 photos, fewer on 2. Not worse; better.
- **Second test set** (a test, nothing chosen on it): 12.4 points of squares
  more (interval 11.9 to 13.0), 296 boards gained, 1 lost.
- Against render2: board B more squares right on 63 photos, fewer on 5;
  ChessReD test 2 boards gained, 7 lost (p = 0.18, within noise).

**Board B is no longer a blind test.** The renders that taught this model
the side of a piece were designed from what went wrong on board B, the run
was scored on it twice while it trained, and render1, render2, crop1 and
render3 were all compared on it; the paired test also treats 100 photos of
one board as independent. What stands against a result fitted to board B:
the gain there (3.9 points) is several times the swing from one epoch to
the next (about 1 point), and the second test set, which nothing was chosen
on, agrees: against render2 with the same build, 0.5 points of squares more
(interval 0.4 to 0.6), 171 boards gained and 69 lost.

On the second test set about 18% of boards are fully right, and 6% on
board B, where the target is 85%. What is still wrong on board B: the
side of its brown "white" pieces (P read as p 132 times of 868 with labelled
corners), kings read as queens, and the corners (88.5% of squares through
the app against 94.7% with labelled corners). `pieces_s_tone1` (20,000
renders and the tone augmentation) and `corners_s_canon1` were the next
attempts at each, and are what is in `Models/` now.

Trained on ChessReD2K (CC BY-NC-SA 4.0), chesscog, Fenit board A, the
Roboflow projects in `sources.yaml` and 12,000 renders of `render_boards.py`:
because of ChessReD it is for personal builds only.

**It is surer of itself when it is wrong.** The app marks a piece whose
confidence is under 0.85 for the user to check, a threshold set with the
model before. Wrong pieces that get the mark, model before against render3
(`runs/app_check/final/`): board B 77% (412 of 538) against 54% (189 of
351), so 126 wrong pieces went unmarked before and 162 do now; ChessReD
test 53% against 38% (30 and 25 unmarked); Fenit A 63% against 22% (382 and
342 unmarked). Right pieces marked fall too: board B 40% against 22%. The
threshold is the owner's to set (`App/Sources/AppModel.swift`);
`app_check --flag-below` shows what other values would mark. On the second
test set (app-side session, same build): 40% of wrong pieces and 8.8% of
right ones marked, 2.7 squares a photo (first pair 79%, 48%, 11.3); at 0.7
20% and 3.3%, at 0.5 7.8% and 1.3%.

**What the app-side steps are worth with it** (app-side session, the build
of ae9dd12; tables in `docs/RECOGNITION.md`, "What the app read with the
second pair";
per-photo outputs `runs/app_check/render3_switches/`). The rule for the
kings off: ChessReD test 288 boards exact (the same), board B 88.45% and 5
(on: 88.55% and 6), Fenit A 184 (on: 183; both losses are positions without
two kings), second test set 88.31% and 218 (on: 88.49% and 298): the rule
is worth 80 boards there. Grid fit, a1 from the pieces and the rule all
off: ChessReD test 246, board B 46.88% and no board found in 36 of 100,
second test set 46.69% and 0 exact.

Two things about how it was trained, for the next run. Mosaic was off for
the whole run: `train.py` leaves ultralytics' `close_mosaic` at 10 epochs
and the run had 10 (render2, with 12, had it for two). And about 2% of the
relative-palette renders have a "White" side darker than L* 40 against a
near-black one, before exposure: two dark sides, a little label noise.

**`corners_s_canon1` with render3's pieces, before it went in** (FP16 export
`runs/export_canon1/BoardCorners-fp16.mlpackage`, the corners turned to a1
order inside the model so the app sees the tensor it always saw; with
`pieces_s_render3` in `runs/app_check/models_canon1/`; weights
`runs/pose/corners_s_canon1/weights/last.pt`). With render3's pieces its
gain on board B, which is what the rule in `CLAUDE.md` names, was within
the noise, so it waited for a piece model. Through `app_check` with the app's defaults against the pair in
`Models/` (same piece model, same build; `runs/app_check/canon1/last_*.txt`,
`final_h2/canon1_*.txt`; `compare_app_check.py`). Squares right; boards
exact; photos with no board found:

| | render3 + first corner model | render3 + `corners_s_canon1` | Photo by photo |
|---|---|---|---|
| Fenit board B (100; held out) | 88.55%; 6; 4 | 90.41%; 6; 0 | more squares right on 8 photos, fewer on 6; +1.9 points, interval -1.0 to +5.3: **within noise** |
| ChessReD test (306) | 99.68%; 288; 0 | 99.79%; 291; 0 | 5 boards gained, 2 lost: not worse |
| Fenit A (286; trained on) | 95.32%; 183; 0 | 97.83%; 217; 0 | 34 boards gained, none lost |
| Second test set: `ylichess` (200) | 85.73%; 45; 6 | 87.96%; 42; 0 | +2.2 points (0.3 to 4.5) |
| `chessq` (50) | 71.66%; 14; 10 | 93.81%; 16; 0 | +22.2 points; more right on 15 photos, fewer on none |
| `chesscog_real` (29) | 98.44%; 10; 0 | 98.44%; 10; 0 | no photo differs |
| `samryan` (500) | 84.85%; 58; 16 | 95.03%; 76; 0 | +10.2 points; 135 photos against 14 |
| `chesscv_games` (840) | 91.97%; 171; 13 | 97.46%; 186; 0 | +5.5 points; 96 photos against none |
| Second test set, all 1,619 | 88.49%; 298; 45 | **95.44%; 330; 0** | +7.0 points (6.0 to 7.9); 255 photos against 22; 37 boards gained, 5 lost |

On boards nobody trained or tuned on it finds every board (45 were not
found) and gains 7 points of squares; on board B the old corner model with
the grid fit was already close, and the 4 boards it did not find are the
whole gain. It went in with `pieces_s_tone1`, which carries the pair past
board B (above). Two things seen on the way: the weight of the model's own a1
(`--a1-prior`) hardly matters with it (board B 91.0% / 90.7% / 90.4% /
89.9% and ChessReD test 288 / 290 / 291 / 291 boards for 0 / 3 / 6 / 12), so
6 stays; and with the grid fit off the pair is worse on board B (87.75%
after 7 epochs), so the fit stays too. In the package's tests the fixtures
read as portrait photos give 95.3% of squares with it (100% with the first
corner model), inside what that test asks.

**"The first pair"** below is what was in `Models/` from midday on
2026-11-13 until pull request #15: `main`'s corner model and the baseline
piece weights exported with NMS across classes. Everything in this file
measured "with the pair in `Models/`" before 2026-11-14 was measured with
it. "`main`'s corner model" in this file means the first corner model
(`baseline/corners_yolo11n_chessred2k.pt`), which was on `main` until
`corners_s_canon1` replaced it on 2026-11-14.

**The earlier candidate `pieces_s_render2`** (exports in
`runs/export_render2/`, weights `runs/detect/pieces_s_render2/weights/last.pt`)
was not put in: on board B it gained nothing (more squares right on 35
photos, fewer on 42), because it calls that set's brown "white" pieces
black. Its columns are in the table above.

**The earlier candidate: `pieces_s_render1`**

(branch `model/pieces-render1`, commit 773e729, has it as
`Models/Pieces.mlpackage` with its weights; exports in `runs/export_render1/`,
with `main`'s corner model in `runs/app_check/models_render1/`). It was put in
and taken out again on 2026-11-13 because it does not meet the rule for
replacing a model as `CLAUDE.md` words it: on the held-out board B it changes
nothing, and on ChessReD test it is not shown better. Through `app_check`
(the decode-once binary, app defaults, colour step off; measured by the
app-side session, per-photo outputs in `runs/app_check/decode_once/`),
the first pair against the candidate:

| | First pair | With `pieces_s_render1` |
|---|---|---|
| ChessReD test (306), boards exact | 279 | 285 |
| Fenit board B (100) | 84.6% of squares, 0 exact | 84.3%, 1 exact |
| Fenit A (286; render1 trained on these photos) | 91.4%, 16 exact | 95.3%, 185 exact |
| Second test set, all 1,619 | 75.5%, 2 exact | 85.4%, 31 exact |
| Second test set, with `EDGE_MARGIN` 0.25 (pull request #11) | 75.9%, 3 exact | 86.3%, 45 exact |
| Second test set, with the two-kings rule as well (pull request #12) | 76.1%, 3 exact | 86.6%, 63 exact |
| The same with the chess-cv photos read from one full decode (pull request #14; what the branch does now) | 76.08%, 3 exact | 86.62%, 68 exact |

(The first four rows are with the rim tolerance of half a square; with a
quarter the three development sets do not move, and on the second test set
Y-LIChess goes from 78.0% and 0 boards exact to 84.8% and 14 with the
candidate. Per-photo outputs: `runs/app_check/edge_margin_025/`. Every
second-test-set row but the last read the 840 chess-cv photos, which carry
an EXIF orientation, through a thumbnail path; the last row is the one to
quote: 76.08% and 3 boards exact against 86.62% and 68.)

What speaks for it is the second test set, where it is better on every part
(table in that section). Whether that is enough is the owner's call (the
notes, "Needs the owner"). An INT8 export of it is half the size (9.3 MB
against 18 MB); it was compared only with the old `app_check` (285 against
286 ChessReD boards) and the outputs were not kept.

**The first corner model** (`main`'s, the baseline corner weights) was in
`Models/` until 2026-11-14 evening, apart from a few hours on 2026-11-13.
exp3's corners were taken back out then: with the app-side steps (a1 from
the pieces, grid-fit corners) they still turn three endgame photos of
ChessReD game G033 by 180 degrees, one of them a fixture of the package's
tests, and the first corner model does not. Until pull request #15 the piece
model beside it was the baseline piece weights exported with NMS across
classes.

**A development set of unseen boards (2026-11-20).** With the owner's Roboflow key, 42 more Roboflow projects were picked from a Universe search (481 chess projects, 119 with colour+type classes; a contact sheet of each looked at): 32 go into training (`sources.yaml`, `roboflow`), 10 boards are kept out (`roboflow_dev`, test_only). `make_rf_dev.py` derives a position for each kept-out photo it can pin down (classical grid found, every piece on it, all four outer rows occupied, colours and pieces agree on a1) and writes `datasets/roboflow_dev/positions.json`: **324 photos on 6 boards** (two boards are side views the grid fit cannot read, one fails the a1 checks, one labels only the board). No photo of these boards is in a training project (16x16 average hash, exact). It is easier than real use (only photos the grid reads) and its labels are Roboflow's, so compare models on it pair by pair (`compare_app_check.py`), not by the rate. `main` through the app: **140 of 324 exact (43%), 96.61% of squares; classifier off 114**. Per board: `runs/rfdev_by_board.py`. The owner chose this as what models are chosen on (option b); the second test set stays a veto.

**What a new model has to beat since pull request #33 (2026-11-18): `pieces_s_shape1` with `corners_s_canon1` and the piece classifier `cls_s_crops1`** (`Models/PieceKinds`), through `app_check` built from `app/piece-kinds` or later: board B 96.05% and 53 of 100, ChessReD test 299 of 306, Fenit A 247 of 286, second test set 817 of 1,619 (`runs/app_check/kinds09/`, `kinds_h2/`), as 3:4 photos board B 53, ChessReD test 290; models folder `runs/app_check/models_shape1_kinds/`. A new piece model is judged with the classifier in place (and once without, `--piece-kinds off`).

**What the pair before #33 had to beat, since pull request #32 (2026-11-16): `pieces_s_shape1` with `corners_s_canon1`**, through the same `app_check` build (13:29 on 2026-11-15, defaults as below): ChessReD test 297 of 306 boards exact (99.82% of squares), Fenit board B 95.67% and 43 of 100, Fenit A 98.09% and 230 of 286, second test set 97.41% and 640 of 1,619 (`runs/app_check/shape1_h2/shape1_*.txt`), packages in `runs/app_check/models_pieces_s_shape1/`; the pair before it, read with that build, in `runs/app_check/shape1_h2/now_*.txt` and `now30/`. The paragraph below is the pair before it.

**What the pair before had to beat** (through `app_check` built from 00fd9b1
or later, with the app's defaults: grid fit on, its window where the corner
model has the board, a1 on a dark square, head start 6, the side of single
pieces, one king a side, the whole-board colour step and the queen for a
king off, each photo decoded once; the pair in `Models/`,
`runs/app_check/now30/now_*.txt`, and `window_h2/now_*.txt` for the second
test set, where the side of single pieces changes no reading): ChessReD
test 295 of 306 boards exact, two more once turned by 180 degrees; Fenit
board B 94.94% of squares, 31 boards exact (35 once turned), every board
found; Fenit A 98.14% of squares, 234 of 286 exact; second test set 97.00%
and 464 of 1,619; as 3:4 photos (`now30/now_*_portrait.txt`) ChessReD
test 282, board B 94.91% and 31. Before the side of single pieces (builds
up to b3e18dd with it off, `runs/app_check/window/`): board B 94.48% and
16, the rest the same. The side of single pieces mends a fault of the
piece model in `Models/`: with another piece model, run it off and on
(`app_check --odd-sides off|on`) before counting on it. Outputs
in `runs/app_check/dark*/` and `tone1/` are from earlier builds: re-run the
pair in `Models/` with the build a candidate is run with. Compare two models only through the same
build of `app_check`: every app-side step moved these numbers a little, and
before pull request #9 it read different pixels. And the model has to pass
the fixture tests of `./Scripts/swift-test.sh`: the four fixtures with the
app's pipeline, and the corner model's own test with the grid fit off, where
a wrong a1 or a board that is not found shows (each corner within one square
of the grid fit's, and 94% of squares over the four photos).

**Fenit A counts too (since `pieces_s_low1`, 2026-11-15).** Two candidates
in a row passed the rule on board B and ChessReD test and read other boards
worse. Fenit A is a board the models are trained on, so it may be used to
choose, and low1's loss there (22 boards, the same kind of error it then
made on an unseen board) showed before the second test set was read. A
candidate that reads Fenit A worse beyond noise is not read on the second
test set at all: it is out.

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
  board, floor, light and camera per image. `datasets/render` holds 34,000:
  `--seed 1` and `--seed 2` (4,000 each) with `--palette split`, the first
  renders' two lists of side colours, `--seed 3` (4,000) and `--seed 4`
  (8,000) with `--palette relative --from-above 0.4`, where the two sides
  are two shades of one list and the lighter is White, and `--seed 5`
  (6,000) with `--low-views 0.6 --from-above 0.15` as well: six in ten seen
  from 12 to 32 degrees above the board, where tall pieces stand in front
  of the ones behind them. `--seed 6` (8,000, 2026-11-15)
  with `--shapes varied --from-above 0.4`: piece sets from a wider range
  than the first five batches' (`--shapes classic`, the default: every king
  with a large cross and clearly the tallest piece, slim feet): heavier and
  slimmer feet, a king from about its queen's height up (a knob finial can leave it a little shorter), its finial a
  cross, a small cross, a ball or a knob, the queen's coronet with balls,
  spikes or plain, taller rooks, bigger pawn heads. The classic draws are
  untouched: `--seed 4 --from-above 0.4 --count 1` gives `render_00400000`'s
  labels again. `relative` is the
  default now, so give `--palette split` to add to seeds 1 and 2; the
  lines of `positions.jsonl` say which it was from seed 5 on (seeds 1 to 4
  were made before that was written down). With `--low-views` no picture is
  taken from between 32 and 40 degrees. Source
  `render` in `sources.yaml`; merged into `datasets/merged_render` (seed 1,
  pieces), `merged_rf_render` (corners), `merged_all` (seeds 1-2),
  `merged_all3` (1-3), `merged_all4` (1-4), `merged_all5` (1-5). In the first 4,000 the sides of
  thick boards carry a warped copy of the board pattern (fixed in the script
  since; the labels are right).
- `side_tone.py`, `train.py pieces --side-tone P`: an augmentation for the
  side of a piece. With probability P a training photo gets a colour cast
  (warm two times in three), another tone curve and a little lift of the
  blacks, the same for every pixel: the lighter side stays the lighter one,
  its colour changes. `python side_tone.py --sheet out.jpg <photos>` shows
  what it does. `--close-mosaic N` sets how many last epochs go without
  mosaic (ultralytics' 10 switches it off for a whole run of 10 epochs).
- `score.sh pieces|corners <label> <weights>`: the standard scores on the CPU,
  into `runs/scores/`.
- `compare_app_check.py <pair to beat>.txt <candidate>.txt`: two
  `app_check --verbose` outputs of the same list of photos, photo by photo:
  on how many photos the candidate has more squares right and on how many
  fewer (sign test), the difference with a bootstrap interval, boards gained
  and lost. This is the test "better by more than the noise" is checked
  with; the figures for `pieces_s_render3` above come from it.
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
  - Exactly two kings, from every kind the model scored for each piece:
    `one_king_a_side`, in the repo and on by default in the app since pull
    request #12 (numbers in the second test set's section).
  - The piece model looking at the board only (`board_crop`, `crop_boards.py`,
    `--crop-board`): `pieces_s_crop1`, a small gain on board B and none on
    ChessReD test (the runs table). A model not trained on crops gains
    nothing from them (16 of 98 on board B with the colour step on, against
    17 without the crop).
  - More renders and all real sources: `pieces_s_render2` (ChessReD test
    better, board B not), then `pieces_s_render3` with renders whose side
    colours are relative, then `pieces_s_tone1` with more of them and the
    tone augmentation: the model in `Models/`.
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

### What is wrong on unseen boards now, and a piece classifier (2026-11-15 night)

Through the app with tone1 + canon1, the pair before pull request #32 (`runs/app_check/window_h2/now_*.txt`,
`now30/now_fenit_B.txt`), wrong squares by kind of error (look 11 at the second
test set in the list under "Second held-out test set"):

| Set | Boards exact | Wrong squares | Side wrong | Kind wrong | Missing | Extra | Most common |
|---|---|---|---|---|---|---|---|
| chess-cv games (840) | 267 | 1,082 | 0 | 1,078 | 1 | 3 | k->q 175, b->n 155, q->r 124, N->B 80 |
| samryan (500) | 115 | 1,120 | 8 | 833 | 137 | 142 | b->r 90, R->B 90, k->q 85, B->P 84 |
| Y-LIChess (200) | 60 | 664 | 3 | 184 | 214 | 263 | pieces a square off (P/p, K/k) |
| ChessQ (50) | 14 | 210 | 4 | 115 | 41 | 50 | P->B 34, p->b 15 |
| chesscog real (29) | 8 | 34 | 0 | 34 | 0 | 0 | p->b 24 |
| board B (100) | 31 | 324 | 22 | 103 | 101 | 98 | pawns a square off, k->q 16, r->q 16 |

So on unseen piece sets the side is now nearly always right, and what is left is
the kind of piece (and on Y-LIChess and board B, tall pieces landing a square
off). About 98% of squares are right; 85% of boards exact needs about 99.75%.

**Test-time augmentation does not help** (`evaluate.py --tta`, ultralytics'
flips and two smaller scales; `--tta-flip`, the photo and its mirror image with
the scores of matched boxes averaged). tone1, labelled corners, board B / Fenit A
/ ChessReD test boards exact: alone 20 of 98 / 239 / 299; `--tta` 24 / 229 / 296;
`--tta-flip` 22 / 238 / 299. Not worth reading every photo twice on a phone.

**A piece classifier as a second stage: in the app since pull request #33 (2026-11-18, the owner's decision), better on every development set, worse on one part of the second test set (samryan).** The detector reads the
photo at 640 px, where a piece is 20 to 50 px tall. `make_piece_crops.py` cuts
every labelled piece of `merged_all4_shape1` out of the 1280-px copies (a
square of 1.2 times the box's longer side, a second copy shifted and scaled by
up to 12%, 30% of the render photos kept): `datasets/crops1`, 556,292 crops
to train on, 18,667 from the val photos. `train.py classify` trains a
YOLO11s-cls on them at 128 px (`cls_s_crops1`, 6 epochs, 98.3% top-1 on the
val crops; weights `runs/detect/cls_s_crops1/`, CoreML export
`runs/export_cls_s_crops1/cls_s_crops1.mlpackage`, 10 MB FP16, same top
class as the `.pt` on 40 of 40 crops). Training it on the CPU while a piece
run had the GPU slowed that run fourfold: classifier runs go through the queue.

- **How it is used.** A crop cannot tell the side: on board B, whose
  "white" pieces are brown, the classifier alone calls them black (P->p 356
  times, `--classifier-mode replace`: 0 of 98 boards exact). So only the kind
  is taken from it: its two sides' probabilities summed, times the piece
  model's (its confidence for its own kind, the rest shared evenly), the side
  the piece model's (`kind-product`, reference
  `chess_pipeline.kinds_from_classifier`; the product's shares are also the
  kind scores one king a side gets). A piece the piece model gives 0.9 or
  more is left alone (`CLASSIFIER_SURE`; see below).
- **Development sets** (with `pieces_s_shape1`): labelled corners in Python,
  board B 27 -> 34 of 98, Fenit A 249 -> 262 of 270, ChessReD test 298 ->
  301. Through the app (branch `app/piece-kinds`, `app_check --piece-kinds`,
  `runs/app_check/kinds09/` against `pieces_s_shape1/`): **board B 43 -> 53
  of 100**, Fenit A 230 -> 247, ChessReD test 297 -> 299; as 3:4 photos board
  B 43 -> 53, ChessReD test 285 -> 290.
- **Second test set (looks 13 and 14, replayed on the app's pieces with
  `runs/dumps/cls_replay.py`; vetoes only).** Without the 0.9 rule: 640 ->
  831 boards exact, chess-cv games 412 -> 584, Y-LIChess 61 -> 131, ChessQ
  17 -> 18, chesscog real 23 -> 23, but **samryan 127 -> 75** (26 gained,
  78 lost). The likely cause, views from straight above, was then looked for
  on 600 new renders of an unused seed (`datasets/render_dev8`, 486 from 70
  to 90 degrees): there the classifier gains (from above 157 -> 187 boards
  exact, 34 gained, 4 lost; from the side 20 -> 22). Leaving pieces of 0.9 or
  more alone was then chosen on the development sets alone, where it gains
  as much as the classifier everywhere (replayed: board B 54, Fenit A 246,
  ChessReD test 299) and changes fewer pieces. With it, the second test set:
  640 -> 798, chess-cv 412 -> 563, Y-LIChess 61 -> 117, ChessQ 17 -> 18,
  chesscog real 23 -> 21, **samryan 127 -> 99** (32 gained, 60 lost; its
  squares 96.54% -> 96.78%). A part read worse, so under the rule kept then (the second test set may
  only veto) it was held back and put to the owner, who chose to ship it
  (2026-11-18; next bullet).
- **`evaluate.py --classifier`:** the default mode is `kind-product` (the
  rule the app uses); the older modes (`replace`, `product`, `confident`,
  `kind`) now cut their crops as the reference does (the photo drawn at most
  1,280 px), so their figures above can differ a little when re-run.
- **A second classifier, `cls_s_crops2` (2026-11-17), is no better.** `cls_s_crops1` continued 4 epochs (`--lr0 0.0003`) on `datasets/crops2`: 703,232 crops cut from `merged_all4_shape3` (render seeds 1-4, 6, 7, 9; 30% of the render photos), 98.3% top-1 on the val crops as the first. Replayed on the app's pieces with shape1, 0.9 rule, against `cls_s_crops1`: board B 50 against 54, Fenit A 247 against 246, ChessReD test 299 for both. More crops of the same kind add nothing; the second test set was not read for it.
- **Shipped on the owner's word (2026-11-18: "yes, ship it").** Through the app with the models in `Models/` (`app_check` built from `app/piece-kinds`; `runs/app_check/kinds09/`, `kinds_h2/` against `pieces_s_shape1/`, `shape1_h2/`): board B 43 -> 53 (10 gained, 0 lost, p = 0.002), Fenit A 230 -> 247, ChessReD test 297 -> 299; as 3:4 photos board B 43 -> 53, ChessReD test 285 -> 290. Second test set (look 15, a confirmation for the pull request): **640 -> 817 of 1,619** (264 gained, 87 lost), Y-LIChess 61 -> 117, chess-cv games 412 -> 556, ChessQ 17 -> 18, chesscog real 23 -> 22, **samryan 127 -> 104** (35 gained, 58 lost). Not timed on a phone: up to 32 crops a photo through a 128-px YOLO11s-cls.
- **What the user is shown.** The "check this piece" mark (confidence under
  0.85) stays the piece model's own score for the kind taken: where the
  classifier changes a kind, that is the piece model's (usually low) score
  for the new kind, so the piece is marked. (A first version showed the
  classifier's share, near 1 for most pieces, and marked 22% of board B's
  wrong pieces instead of 54%.) With it: board B 76 of 152 wrong pieces
  marked (50%; before the classifier 98 of 182, 54%), 3.1 squares a photo;
  samryan 686 of 854 (80%; before 750 of 968, 77%).
- **Averaging the classifier over each crop and its mirror image** (2026-11-18; replayed on the app's pieces, 0.9 rule): board B 51 against 54 without it, Fenit A 247 against 246, ChessReD test 299 for both. Not worth twice the classifier's time.
- **Where it is:** Swift `Packages/ChessVision/Sources/ChessVision/PieceKinds.swift`
  and the recognizer's `kindModel:` / `pieceKinds:` on branch
  `app/piece-kinds` (tests `PieceKindsTests` against `swift_reference.py
  kinds-cases`); the package is `Models/PieceKinds.mlpackage` (the app loads it where the
  bundle has it; without it the app reads as before). Swift and Python pick another
  kind for 11 of 1,693 board B pieces (resampling of the crops; close calls
  between queen and king, rook and bishop).

**The head start for the corner model's own a1, again with a1 on a dark square (2026-11-17).** The earlier test (head start 12: free on the development sets, nothing on the second test set, where it turned sparse boards by a quarter) was before "a1 is a dark square", which now leaves only two ways round. Replayed on the app's pieces with shape1 (`swift_reference.replay_pieces`, `runs/app_check/cls/pieces_shape1_*.json`), head start 6 / 9 / 12 / 20 / 40: board B 43 / 43 / 43 / 43 / 42 boards exact (squares fewer on 3 or 4 photos from 9 up), Fenit A 230 / 236 / 243 / 250 / 253, ChessReD test 297 throughout. Fenit A is a board the corner model was trained on, so it knows a1 there; on the unseen board it gains nothing. Left at 6: no development set can place it.

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

**Where it has been used to choose all the same (2026-11-14 and 15).** Each
time to keep something out, never to pick a value; each is a look that
makes the set a little less of a test, and they are written down here so
that the count is known:

1. A head start of 12 for the corner model's own a1 was left out because it
   gained nothing here (the development sets had it free).
2. `corners_s_canon1` went in on the strength of this set as much as of
   board B, where its own share was not beyond noise.
3. `pieces_s_tone2` passed the rule in `CLAUDE.md` (better on board B, not
   worse on ChessReD test) and was left out because it loses 57 boards on
   the chess-cv games here; `pieces_s_low1` was then started from tone1
   instead of tone2.
4. That quarter turns were worth a rule was seen here first.
5. That the grid fit takes a window one rank off was seen here (Y-LIChess),
   and the rule against it (the corner model picks the window) exists
   because of it; its tolerance was set on board B.
6. `pieces_s_low1` passed the rule in `CLAUDE.md` too and was read here by
   part before any pull request. It had already lost 22 boards on Fenit
   board A, a development set, which alone would have kept it out; here it
   loses 17 boards on Y-LIChess and 24 in all, and that settled it.
7. The side rule for single pieces (`odd_sides`) was run here twice in
   Python: first changing sides both ways, where it gained nothing, and,
   after it was made one-way on the training photos' labelled sides, again.
   The first run is why the wrong changes were looked for; they were looked
   for and counted on training photos, not here.
8. The queen taken for a missing king (`queen_for_king`) was run here once
   in Python, after it was fixed on the development sets; nothing about it
   was chosen here.
9. The side rule was then read here through the app, by part, before it
   was asked to be on by default: no reading differs.
10. `pieces_s_crop2`, ahead on all three development sets, was read here
   twice in Python before its app-side port was asked for, and the port
   was not asked for. First on the full-size photos, by the same mistake
   as above (`heldout2_crop2_canon1_spot07_base.txt` against
   `heldout2_pair_tone1_canon1_spot07_base.txt`), then with `--max-side
   1280`. Both say two parts gain much (samryan, the chess-cv games) and
   chesscog and the Y-LIChess layouts lose; the Y-LIChess games gain in the
   first (24 -> 29 boards) and lose in the second (41 -> 32).
11. The kinds of error left on unseen boards (2026-11-15, 23:00) were
   counted here, from the app outputs already on disk
   (`runs/app_check/window_h2/now_*.txt`; no new reading): almost all
   wrong squares are the kind of piece. That is why a piece classifier as
   a second stage was tried next (README, "What is wrong on unseen boards
   now"). Nothing was set on it, but the idea came from here.
12. `pieces_s_shape1` was read here through the app, by part, after it had
   passed the development sets (board B 31 -> 43, ChessReD test and Fenit
   A not worse) and the package's tests, before its pull request: 464 ->
   640 boards exact, no part worse. A veto only; nothing was chosen here.
13. The piece classifier (`kind-product`, with `pieces_s_shape1`) was
   replayed here on the app's pieces after it had gained on every
   development set: 640 -> 831 boards exact, samryan 127 -> 75. Vetoed.
   That samryan lost is why views from above were then looked into, on
   renders, which showed nothing (README, "A piece classifier").
14. The same with pieces of 0.9 or more left alone, a rule chosen on the
   development sets after look 13: 640 -> 798, samryan 127 -> 99. Vetoed.
   That look 13 led to a second try at all is the test set steering the
   work; the rule itself was not set on it.
15. The classifier with the 0.9 rule through the app itself, for the pull
   request that ships it on the owner's decision (2026-11-18): 640 -> 817,
   samryan 127 -> 104. Nothing was chosen on it.
16. The same again after the review of that pull request changed the crop
   (edges repeated before scaling) and what the user is shown: every count
   the same (640 -> 817, samryan 127 -> 104). A confirmation only.
17. `pieces_s_rf2` (all 40 new Roboflow projects), the candidate chosen on
   the Roboflow development set, read once as a veto: 817 -> 791. Vetoed
   (samryan 2 gained, 82 lost: white pieces missed). The cause was looked for
   in the training data, not on this set: two projects with unlabelled pieces.
18. `pieces_s_rf3` (rf2 without those projects): 817 -> 772. Vetoed: samryan
   level now, chess-cv games 47 gained / 92 lost (bishops read as knights).
19. `soup_rf2_rf3` (their weights averaged): 817 -> 745. Vetoed.

**What looks 17-19 show (2026-11-21).** Three piece models trained with the new
Roboflow projects each beat `main` on the Roboflow development set beyond noise
(140 -> 159 / 160 / 160) and each lost on the second test set, on whichever
board their particular confusions hit (samryan's white pieces, chess-cv's
bishops). The development set is Roboflow photos too: it rewards fitting that
kind of photo, and does not stand in for the second test set's sources. No more
looks are spent on piece-detector variants of this data.

What this shows is that board B, one board of 100 photos, no longer tells
whether a model reads unseen boards better, and that there is nothing else
to choose on. The remedy is an unseen development set, not more looks at
this one: either new boards (the Roboflow key would fetch some; renders of
a style kept out of training are a proxy), or, if the owner allows it, this
set split by board into a development half and a test half.
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
- **First pair** (in `Models/` from b546e6a until pull request #15): the same
  two weights, the piece model exported with NMS across classes
  (`--one-piece-per-spot 0.7` stands in for it).
- **exp3**: `baseline/corners_yolo11s_exp3.pt` with the same pieces: the pair
  that was in `Models/` for a few hours on 2026-11-13 and came out again
  because it turns the G033 endgames.

| Part | Photos | Empty | Baseline | First pair | exp3 corners | exp3 + `--a1-from-pieces 6` |
|---|---|---|---|---|---|---|
| `chessq` | 50 | 68.9% | 37.4% / 0%, board not found in 18 | 38.0% / 0%, board not found in 18 | 74.5% / 0% | 74.5% / 0% |
| `ylichess`, `wood180` | 102 | 57.5% | 0% / 0%, board not found in 102 | 0% / 0%, board not found in 102 | 69.9% / 0% | 65.1% / 0% |
| `ylichess`, `wood180-layout` | 98 | 89.7% | 0% / 0%, board not found in 98 | 0% / 0%, board not found in 98 | 87.8% / 1% | 86.6% / 1% |
| `chesscog_real` | 29 | 52.9% | 49.4% / 0% | 50.1% / 0% | 78.3% / 0% | 78.3% / 0% |
| `samryan` | 500 | 70.1% | 53.8% / 0%, board not found in 66 | 54.1% / 0%, board not found in 66 | 68.4% / 0% | 67.4% / 0% |
| `chesscv_games` | 840 | 59.4% | 53.0% / 0%, board not found in 10 | 54.2% / 0%, board not found in 10 | 85.4% / 0% | 85.4% / 0% |

**`pieces_s_render1` on this set, through the app's own code** (2026-11-13
evening; test only. `app_check` at pull request #10, i.e. before the rim fix
and the two-kings rule, and with the chess-cv photos still read through a
thumbnail path (pull request #14 reads them from one full decode: with
today's defaults the chess-cv row is 80.65%; 0 and 89.72%; 21, no board in
13). The app's defaults of that build, colour step off; the first pair against the same corner model with the
candidate piece model; measured by the app-side session, per-photo outputs
in `runs/app_check/decode_once/`). Squares right; boards exact; photos with
no board found:

| Part | Photos | First pair | With `pieces_s_render1` | Photo by photo |
|---|---|---|---|---|
| `ylichess` | 200 | 70.1%; 1; 6 | 78.0%; 0; 6 | more squares right on 95, fewer on 97 |
| `chessq` | 50 | 64.6%; 1; 10 | 69.7%; 6; 10 | 25 against 12 (p = 0.05) |
| `chesscog_real` | 29 | 73.3%; 0; 0 | 92.7%; 0; 0 | 29 against 0 |
| `samryan` | 500 | 70.4%; 0; 16 | 82.9%; 16; 16 | 359 against 80 |
| `chesscv_games` | 840 | 80.5%; 0; 12 | 89.3%; 9; 12 | not computed (file names repeat) |

Better in squares on every part, by 5 to 19 points; on Y-LIChess as many
photos lose squares as gain them. Boards fully right: 31 of 1,619 against 2,
so about 2% where the target is 85%. (Pooled, which this section warns
against because `chesscv_games` is half the photos: 85.4% against 75.5%;
with today's app defaults and the full decode 86.62% and 68 boards against
76.08% and 3.)

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
squares are pieces standing beside the board, which `board_square` took in
from up to half a square outside. That is `EDGE_MARGIN` in
`chess_pipeline.py`, and since f75d1f8 it is a quarter square (next
paragraph).

**Pieces beside the board (`EDGE_MARGIN`, 2026-11-13).** `board_square` puts
a piece whose foot comes out just over the edge back on the nearest rim
square. With half a square of tolerance, captured and spare pieces standing
next to the board landed on the rim: on Y-LIChess, with labelled corners and
`pieces_s_render1`, 231 of 258 pieces seen on empty squares in the game
positions and 364 of 385 in the layouts (on Fenit B 4 of 36, on ChessReD
test 0 of 7). The value is 0.25 now, chosen on ChessReD test, Fenit A and
Fenit B, where it gives the same boards as 0.5 with labelled corners; end to
end, measured on ChessReD test and Fenit B only, it is one square of 26,000
fewer; 0.1 starts to lose real rim pieces
(`runs/edge_margin.txt`, `runs/edge_margin_e2e.txt`). On this set, test
only, `pieces_s_render1` in Python with `--max-side 1280` and the app-side
steps, 0.5 -> 0.25: Y-LIChess game positions 77.9% -> 81.8% of squares and
0 -> 10 of 102 boards exact; layouts 76.2% -> 84.9%, 0 -> 11 of 98; ChessQ
72.0% -> 72.3%; chesscog 93.7% -> 93.9%; samryan 84.1% -> 84.3%; chess-cv
89.6% -> 89.6% (`runs/scores/heldout2_1280_*.txt` against
`heldout2_1280_m025_*.txt`). Through the app with the first pair
(app-side session): Y-LIChess 70.1% -> 72.4%, the other parts and the three
development sets within 0.6 points. The same constant is used when labels
are built (`derive_corners.check_pieces`, `build_dataset.py`'s corner
orientation): a dataset rebuilt now can keep or drop a few other photos
than the folders on disk, which were built with 0.5.

**Exactly two kings (`one_king_a_side`, `evaluate.py --one-king`,
2026-11-13).** With fewer than two kings on the board, a side that has
pieces and no king gets its best king candidate; with more than two, a
side's extra kings become their next kind; a board with two kings is left
alone. Development sets (`pieces_s_render1`, labelled corners): Fenit B
91.45% -> 91.52% of squares, Fenit A 244 -> 246 of 270 boards exact,
ChessReD test unchanged. On this set, test only, same settings as the
paragraph above with `EDGE_MARGIN` 0.25, without -> with the rule: Y-LIChess
game positions 81.8% -> 82.1% of squares and 10 -> 15 of 102 boards exact;
layouts 84.9% -> 85.1%; chesscog 93.9% -> 94.4%; samryan 84.25% -> 84.29%;
chess-cv 89.6% -> 89.8% and 11 -> 13 boards exact; ChessQ 72.25% -> 72.22%
(one square fewer, the same 8 boards exact)
(`runs/scores/heldout2_1280_m025_king_render1.txt`). Small and nearly
always in the right direction. The app has it since pull request #12, on by
default, with the model's own score for every label of a box (Vision's
label confidence times the observation's confidence; scores under 0.02 left
out). Through the app, off -> on: the first pair ChessReD test 279 ->
280 boards exact, Fenit B 0 -> 1, Fenit A 16 -> 32 of 286, this set 76.00%
-> 76.08% of squares (3 exact both); with `pieces_s_render1` this set
86.40% -> 86.62% and 49 -> 68 boards exact (the chess-cv photos read from
one full decode, pull request #14: `runs/app_check/exif_full_decode/`; the
files in `runs/app_check/one_king/` read them through a thumbnail path and
give 75.98% -> 76.07% and 86.34% -> 86.56%, 45 -> 63).
Where it loses: a board set up without a king on a side is not a game
position (13 of Fenit A's 286, 50 of Y-LIChess's 200), and there the rule
crowns the best candidate; the three boards that were exact without it and
are not with it are such. It can also move a1, since a1 is taken from the
pieces after it.

**The app-side steps on this set, through the app's own code, 2026-11-13 midday** (measured with
`app_check` before it decoded each photo once; with the later binary the two
totals are 46.2% and 75.5%, 340 and 44 photos without a board, still with
the chess-cv photos through a thumbnail path. `app_check`, the exported models
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
- **The first corner model (in `Models/` when this was measured, without
  the grid fit) does not find the board in 294 of the 1,619 photos**, among them every photo of the Y-LIChess board, and is below
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
- **Both app-side steps on the first pair** (`--grid-corners
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
