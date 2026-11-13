# Weights of every model that reached `Models/`

The models the app shipped with before the multi-dataset retrain, as PyTorch
checkpoints, so any machine can score them. `Models/*.mlpackage` are the CoreML
exports of these same two files.

| File | Model | Trained on |
|---|---|---|
| `corners_yolo11n_chessred2k.pt` | YOLO11n-pose, 640 px, 4 keypoints (a1, a8, h8, h1) | ChessReD2K train |
| `pieces_yolo11n_chessred2k.pt` | YOLO11n detect, 640 px, 12 classes | ChessReD2K train |

ChessReD test split: 97.89% per-square, 73.20% boards exact.

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

Kept so that any step can be undone: export the row above the one to undo
with `export_coreml.py` and put the result back in `Models/`.

| Date | `Models/BoardCorners` | `Models/Pieces` | Trained on | Scores through the `.mlpackage`s (per-square / boards exact) |
|---|---|---|---|---|
| before 2026-11 | `corners_yolo11n_chessred2k.pt` | `pieces_yolo11n_chessred2k.pt`, NMS per class | ChessReD2K | ChessReD test 97.86% / 74.2%; Fenit board B 24.2% / 0% (board not found in 66 of 100; the `.pt` files give 58.0% / 0%) |
| 2026-11-13 | `corners_yolo11s_exp3.pt` (run `corners_s_exp3`, FP16) | `pieces_yolo11n_chessred2k.pt` again, exported with NMS across classes (FP16) | corners: ChessReD2K, chesscog, Fenit board A, 3,000 synthetic boards; pieces: ChessReD2K | ChessReD test 99.47% / 87.6%; Fenit board B 67.7% / 0% |

Both rows include ChessReD (CC BY-NC-SA 4.0): personal builds only.
