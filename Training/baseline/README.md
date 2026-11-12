# Baseline weights

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
