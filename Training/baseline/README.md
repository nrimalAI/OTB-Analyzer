# Weights of every model that reached `Models/`

PyTorch checkpoints of the models the app shipped with before the
multi-dataset retrain and of every model that has replaced them since, so any
machine can score them and any replacement can be undone. Which export of
which file is in `Models/` now is in the table at the end.

| File | Model | Trained on |
|---|---|---|
| `corners_yolo11n_chessred2k.pt` | YOLO11n-pose, 640 px, 4 keypoints (a1, a8, h8, h1) | ChessReD2K train |
| `pieces_yolo11n_chessred2k.pt` | YOLO11n detect, 640 px, 12 classes | ChessReD2K train |
| `pieces_yolo11s_render3.pt` | YOLO11s detect, 640 px, 12 classes (run `pieces_s_render3`; the saved settings carry no local paths) | ChessReD2K, chesscog, Fenit board A, Roboflow projects, 12,000 renders |
| `pieces_yolo11s_tone1.pt` | YOLO11s detect, 640 px, 12 classes (run `pieces_s_tone1`; no local paths) | as render3, with 20,000 renders and `--side-tone 0.35` |
| `pieces_yolo11s_shape1.pt` | YOLO11s detect, 640 px, 12 classes (run `pieces_s_shape1`: tone1 continued 8 epochs with AdamW at 0.0001; no local paths) | as tone1, with 8,000 more renders of more varied piece shapes (`render_boards.py --shapes varied`, seed 6) and `--side-tone 0.35` |
| `corners_yolo11s_canon1.pt` | YOLO11s-pose, 640 px, 4 keypoints in an order the picture fixes, a1 as one of 4 classes (run `corners_s_canon1`, `canonical_corners.py`; no local paths). Score and export it with this repo's `evaluate.py`, `corner_error.py` and `export_coreml.py`, which turn the corners back to (a1, a8, h8, h1) | ChessReD2K, chesscog, Fenit board A, Roboflow photos with derived corners, renders |

ChessReD test split with the first two (the baseline pair): 97.92% per-square, 73.5% boards exact
(225 of 306; an earlier run of the same files gave 224).

Score them on other data to get the number a new model has to beat, e.g. the
held-out Fenit photos:

```sh
python evaluate.py --corners-model baseline/corners_yolo11n_chessred2k.pt \
                   --pieces-model baseline/pieces_yolo11n_chessred2k.pt \
                   --positions datasets/merged/positions.json --source fenit --split test
```

Trained on ChessReD (CC BY-NC-SA 4.0), so these are for evaluation and personal
builds, not for an App Store release.

## Models that replaced the baseline

Kept so that any step can be undone: export the files of the row to go
back to with `export_coreml.py`, pick the variant the row names, and put the
results in `Models/`. The first row's piece model needs
`--pieces-nms-per-class`; without it the export has NMS across classes.

| Date | `Models/BoardCorners` | `Models/Pieces` | Trained on | Scores through the `.mlpackage`s (per-square / boards exact) |
|---|---|---|---|---|
| before 2026-11 | `corners_yolo11n_chessred2k.pt` | `pieces_yolo11n_chessred2k.pt`, NMS per class | ChessReD2K | ChessReD test 97.86% / 74.2%; Fenit board B 24.2% / 0%: the board is not found in 66 of 100 photos, which count as all wrong; on the 34 found about 71% (the `.pt` files find 88 and give 58.0% / 0%) |
| 2026-11-13 | `corners_yolo11s_exp3.pt` (run `corners_s_exp3`, FP16) | `pieces_yolo11n_chessred2k.pt` again, exported with NMS across classes (FP16) | corners: ChessReD2K, chesscog, Fenit board A, 3,000 synthetic boards; pieces: ChessReD2K | ChessReD test 99.47% / 87.6%; Fenit board B 67.7% / 0%, board found in all 100 |
| 2026-11-13, later | `corners_yolo11n_chessred2k.pt` again (the file from `main`) | unchanged: `pieces_yolo11n_chessred2k.pt` with NMS across classes (FP16) | ChessReD2K | Through the app's code (`app_check`, a1 from the pieces), ChessReD test boards exact: 238 of 306 (77.8%; with `main`'s piece export 229), Fenit board B 46.6% of squares, board not found in 35 of 100 |
| 2026-11-14 | unchanged: `corners_yolo11n_chessred2k.pt` | `pieces_yolo11s_render3.pt` (run `pieces_s_render3`, YOLO11s detect, 640 px, 12 classes; FP16, NMS across classes) | pieces: ChessReD2K, chesscog, Fenit board A, the Roboflow projects in `sources.yaml`, 12,000 renders of `render_boards.py` | Through the app's code (`app_check`, the app's defaults): ChessReD test 99.68% / 288 of 306 (the row above, same binary: 99.54% / 280); Fenit board B 88.55% / 6 of 100 (84.67% / 1), board not found in 4; second test set 88.49% / 298 of 1,619 (76.08% / 3) |
| 2026-11-14, evening | `corners_yolo11s_canon1.pt` (run `corners_s_canon1`, YOLO11s-pose, 640 px, trained with `--canonical-corners`; FP16; `export_coreml.py` turns the corners to a1 order inside the model, so the output is the tensor the app always read) | `pieces_yolo11s_tone1.pt` (run `pieces_s_tone1`, YOLO11s detect; FP16, NMS across classes) | corners: ChessReD2K, chesscog, Fenit board A, Roboflow photos with derived corners, renders; pieces: ChessReD2K, chesscog, Fenit board A, the Roboflow projects in `sources.yaml`, 20,000 renders, with `--side-tone 0.35` | Through the app's code (`app_check`, the app's defaults; the row above with the same build in brackets): ChessReD test 99.80% / 295 of 306 (99.68% / 288), two G033 endgames turned by 180 degrees; Fenit board B 94.36% / 16 of 100 (88.55% / 6), every board found (4 not); second test set 95.86% / 439 of 1,619 (88.49% / 298) |
| 2026-11-16 | unchanged: `corners_yolo11s_canon1.pt` | `pieces_yolo11s_shape1.pt` (run `pieces_s_shape1`, YOLO11s detect; FP16, NMS across classes) | pieces: as the row above, with 8,000 more renders of more varied piece shapes (28,000 renders in all) | Through the app's code (`app_check` built 2026-11-15 13:29, the app's defaults incl. the side of single pieces; the row above with the same build in brackets): ChessReD test 99.82% / 297 of 306 (99.80% / 295); Fenit board B 95.67% / 43 of 100 (94.94% / 31); Fenit board A 98.09% / 230 of 286 (98.14% / 234); second test set 97.41% / 640 of 1,619 (97.00% / 464), no part worse |

The second row never reached `main`: exp3's corners turn three endgame photos
of ChessReD game G033 by 180 degrees (one is the package's fixture
`test_004005.jpg`), which `main`'s corner model gets right, so the tests fail
with it once the corners are precise (grid fit). The third row's corner
model stayed in `Models/` until one read that fixture right
(`corners_s_canon1`, fifth row: right as the square photo it is and with the
grid fit off, turned when the test redraws it as a 3:4 photo, and it turns
the frames before and after it, `G033_IMG100` and `G033_IMG102`); the fifth
row's corner model and the sixth row's piece model are what is in `Models/` now. With the app's defaults
as they are since pull request #12 (grid fit, a1 prior 6, one king a side;
each photo decoded once) it reads 280 of 306 ChessReD test boards exactly and
84.67% of board B's squares, board not found in 4 of 100. exp3's weights are
not kept here, since it never reached `main`: they are
`runs/pose/corners_s_exp3/weights/last.pt` on the Mac mini, and
`Training/baseline/corners_yolo11s_exp3.pt` in this branch's history up to
the commit that removed it.

The first and third rows count the same corner model's misses on board B
differently (66 and 35 of 100) because they were measured differently: the
first by ultralytics running the exported package, the third by the app's
own code (`app_check`), which is the number that describes the app. Why
ultralytics finds fewer boards with the same package was not looked into.

All rows include ChessReD (CC BY-NC-SA 4.0): personal builds only.
