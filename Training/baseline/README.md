# Weights of every model that reached `Models/`

PyTorch checkpoints of the models the app shipped with before the
multi-dataset retrain and of every model that has replaced them since, so any
machine can score them and any replacement can be undone. Which export of
which file is in `Models/` now is in the table at the end.

| File | Model | Trained on |
|---|---|---|
| `corners_yolo11n_chessred2k.pt` | YOLO11n-pose, 640 px, 4 keypoints (a1, a8, h8, h1) | ChessReD2K train |
| `pieces_yolo11n_chessred2k.pt` | YOLO11n detect, 640 px, 12 classes | ChessReD2K train |

ChessReD test split with these two: 97.92% per-square, 73.5% boards exact
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

The second row never reached `main`: exp3's corners turn three endgame photos
of ChessReD game G033 by 180 degrees (one is the package's fixture
`test_004005.jpg`), which `main`'s corner model gets right, so the tests fail
with it once the corners are precise (grid fit). The third row is what is in
`Models/` until a corner model passes that fixture. With the app's defaults
(grid fit on, a1 prior 6) it reads 280 of 306 ChessReD test boards exactly and
82.3% of board B's squares, board not found in 6 of 100. exp3's weights are
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
