# Reports behind the scores of the second held-out test set, 2026-11-13

`evaluate.py` reports, one per part, as written by `score_heldout2.sh`
(`.pt` weights on the CPU). The table and what it means are in
`Training/README.md`, "Second held-out test set".

| File | Corner model | Piece model | Other arguments |
|---|---|---|---|
| `baseline.txt` | `baseline/corners_yolo11n_chessred2k.pt` | `baseline/pieces_yolo11n_chessred2k.pt` | |
| `now.txt` | the same | the same | `--one-piece-per-spot 0.7` |
| `now_steps.txt` | the same | the same | `--one-piece-per-spot 0.7 --grid-corners --a1-from-pieces 6`; four parts |
| `exp3.txt` | `baseline/corners_yolo11s_exp3.pt` | the same | `--one-piece-per-spot 0.7` |
| `exp3_a1.txt` | the same | the same | `--one-piece-per-spot 0.7 --a1-from-pieces 6` |
| `ylichess_parts.txt` | exp3's, or the labelled corners | the same with `--one-piece-per-spot 0.7`, or the labelled boxes | `--gt-corners` / `--gt-boxes` |
