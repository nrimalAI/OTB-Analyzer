# Reports behind the scores of the second held-out test set, 2026-11-13

`evaluate.py` reports, one per part (`.pt` weights on the CPU). The table and
what it means are in `Training/README.md`, "Second held-out test set".

`baseline.txt`, `now.txt`, `exp3.txt` and `exp3_a1.txt` are written by
`score_heldout2.sh <tag> <corners> <pieces> <other arguments>`. The other two
come from `evaluate.py` run by hand, once per heading in the file, with
`--positions datasets/heldout2/positions.json --split test --device cpu` and:

- `now_steps.txt`: the models and arguments in the table, `--source <part>`
  (and `--group` for the two Y-LIChess groups), four parts only.
- `ylichess_parts.txt`: `--source ylichess --group wood180` or
  `wood180-layout`, and per heading: `--gt-corners --pieces-model <pieces>
  --one-piece-per-spot 0.7`; `--gt-boxes --corners-model <exp3 corners>`;
  `--gt-boxes --gt-corners`.

| File | Corner model | Piece model | Other arguments |
|---|---|---|---|
| `baseline.txt` | `baseline/corners_yolo11n_chessred2k.pt` | `baseline/pieces_yolo11n_chessred2k.pt` | |
| `now.txt` | the same | the same | `--one-piece-per-spot 0.7` |
| `now_steps.txt` | the same | the same | `--one-piece-per-spot 0.7 --grid-corners --a1-from-pieces 6`; four parts |
| `exp3.txt` | `baseline/corners_yolo11s_exp3.pt` | the same | `--one-piece-per-spot 0.7` |
| `exp3_a1.txt` | the same | the same | `--one-piece-per-spot 0.7 --a1-from-pieces 6` |
| `ylichess_parts.txt` | exp3's, or the labelled corners | the same with `--one-piece-per-spot 0.7`, or the labelled boxes | `--gt-corners` / `--gt-boxes` |
