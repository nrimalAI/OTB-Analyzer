"""A development set with known positions from Roboflow boards kept out of training.

    python make_rf_dev.py            # -> datasets/roboflow_dev/positions.json (+ report)

The `roboflow_dev` source in sources.yaml lists ten Roboflow projects (boards)
that no training merge takes (test_only). They have piece boxes but no
position and no corners. For each photo (the same ones `load_roboflow` would
take: de-duplicated, capped at the source's max_per_project) this runs
derive_corners.derive, the checks used for the Roboflow corner labels: the
classical grid must be found, every piece must land on it, all four outer
rows must hold a piece (so the grid cannot be one square off), and the
square colours and the pieces must agree on where a1 is. The position is then
the labelled boxes put on that grid exactly as evaluate.py puts detections
(bottom centre, pull 0.30). A photo two labelled pieces share a square in is
dropped (a label is wrong or the grid is).

What it is and is not: a set of unseen boards to choose models on, so that
the second held-out test set can stay a test. Its photos are those the
classical grid reads, so it is easier than real use; its labels are
Roboflow's, so a few positions will be wrong the same way for every model.
Compare models on it with the paired tests (compare_app_check.py), not by
the absolute rate.

Rows have the layout of build_dataset.py's positions.json (source
roboflow_dev, split test, group = project), so evaluate.py --positions and
app_check read it as they read the other test sets.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import yaml

from build_dataset import load_roboflow
from chess_pipeline import PIECE_CLASSES, board_square, board_to_fen_placement, homography
import derive_corners as dc

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--out", default="datasets/roboflow_dev/positions.json")
    args = ap.parse_args()

    cfg = yaml.safe_load((HERE / "sources.yaml").read_text())["sources"]["roboflow_dev"]
    root = HERE / cfg["dir"]
    skipped = Counter()
    seen = {"names": set(), "hashes": defaultdict(list)}
    samples = list(load_roboflow("roboflow_dev", cfg, root, skipped, seen))
    projects = sorted(p.name for p in root.iterdir() if p.is_dir())
    rules = SimpleNamespace(pinned=4, margin=0.5, min_pieces=6)
    jobs = []
    for s in samples:
        pieces = [(PIECE_CLASSES[c], (cx, cy, w, h)) for c, cx, cy, w, h in s.boxes]
        jobs.append((s.image, pieces, rules, {
            "normalized": True, "stem": s.stem, "project": dc.project_of(s, projects)}))
    results = dc._run(jobs, args.procs)

    rows, why = [], defaultdict(Counter)
    for (image, pieces, _, extra), r in zip(jobs, results):
        if r["stage"] != 3:
            why[extra["project"]][r["status"]] += 1
            continue
        w, h = r["size"]
        H = homography(r["corners"])
        board, clash = {}, False
        for cls, (cx, cy, bw, bh) in pieces:
            sq = board_square(H, (cx * w, (cy + bh / 2) * h), dc.PULL)
            if sq is None or sq in board:
                clash = True
                break
            board[sq] = (cls, 1.0)
        if clash:
            why[extra["project"]]["two pieces on one square"] += 1
            continue
        why[extra["project"]]["kept"] += 1
        rows.append({"id": extra["stem"], "image": str(image), "original": str(image),
                     "source": "roboflow_dev", "split": "test",
                     "placement": board_to_fen_placement(board), "group": extra["project"],
                     "corners": r["corners"]})
    out = HERE / args.out
    out.write_text(json.dumps(rows, indent=1))
    print(f"{len(samples)} photos, {len(rows)} with a position -> {args.out}")
    for p in projects:
        c = why[p]
        print(f"  {p:40} kept {c['kept']:4d} of {sum(c.values()):4d}   "
              + ", ".join(f"{k}: {v}" for k, v in c.most_common() if k != "kept"))
    for k, v in skipped.items():
        print(f"  load_roboflow skipped {v}: {k}")


if __name__ == "__main__":
    main()
