"""Writes the cases that the Swift tests replay, so that Packages/ChessVision
is checked against this folder's Python and not against a second opinion.

    python swift_reference.py orientation

orientation: random boards seen in perspective with their corners given in a
random rotation; what chess_pipeline.orient_by_pieces makes of them. Written
to Packages/ChessVision/Tests/ChessVisionTests/Fixtures/orientation_cases.json
and replayed by BoardOrientationTests.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np

from chess_pipeline import (
    Detection, FEN_LETTER, assign_squares, board_to_fen_placement,
    fen_placement_to_board, homography, orient_by_pieces, project, white_side_score,
)

HERE = Path(__file__).resolve().parent
FIXTURES = HERE.parent / "Packages/ChessVision/Tests/ChessVisionTests/Fixtures"
CLASS_OF = {letter: name for name, letter in FEN_LETTER.items()}

PLACEMENTS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R",
    "R7/5pk1/8/8/8/3p4/1p1r2P1/6K1",
    "7Q/8/1k3p2/4p3/8/p3K3/8/8",          # the pieces point the wrong way
    "8/4k3/1p4R1/p5Kp/4r2P/1P4P1/P7/8",
    "8/8/8/8/8/8/8/K6k",
    "4k3/8/8/8/3Q4/8/8/4K3",
    "2r3k1/pp3ppp/2n1b3/3p4/3P4/2N1PN2/PP3PPP/2R3K1",
    "8/P7/8/8/8/8/7p/k6K",
]


def orientation_cases(count, seed):
    rng = random.Random(seed)
    cases = []
    while len(cases) < count:
        # a board seen in perspective: a square, squeezed towards the far side,
        # turned by any angle, with each corner nudged
        size = rng.uniform(500, 1500)
        far = rng.uniform(0.55, 1.0)
        quad = np.array([[-0.5, 0.5], [-0.5 * far, -0.5 * rng.uniform(0.5, 1.0)],
                         [0.5 * far, -0.5 * rng.uniform(0.5, 1.0)], [0.5, 0.5]]) * size
        angle = rng.uniform(0, 2 * np.pi)
        turn = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        quad = quad @ turn.T + np.array([rng.uniform(800, 2200), rng.uniform(800, 2200)])
        quad += np.array([[rng.uniform(-15, 15), rng.uniform(-15, 15)] for _ in range(4)])
        true_corners = [tuple(float(v) for v in p) for p in quad]     # a1 a8 h8 h1
        to_image = np.linalg.inv(homography(true_corners))

        detections = []
        confidences = rng.sample(range(300, 1000), 64)
        board = fen_placement_to_board(rng.choice(PLACEMENTS))
        for (file, rank), letter in board.items():
            if rng.random() < 0.08:
                continue                          # a missed piece
            if rng.random() < 0.05:
                letter = rng.choice("KQRBNPkqrbnp")   # a misread one
            x, y = project(to_image, (file + 0.5 + rng.uniform(-0.2, 0.2),
                                      rank + 0.5 + rng.uniform(-0.2, 0.2)))
            w, h = size * rng.uniform(0.04, 0.07), size * rng.uniform(0.08, 0.2)
            y += 0.02 * size                     # the box bottom is the front of the base
            detections.append(Detection(CLASS_OF[letter], confidences.pop() / 1000,
                                        (x - w / 2, y - h, x + w / 2, y)))
        rng.shuffle(detections)

        given_turn = rng.randrange(4)             # the corner model's idea of a1
        given = true_corners[given_turn:] + true_corners[:given_turn]
        pull = rng.choice([0.0, 0.3, 0.3, 0.3])
        prior = rng.choice([0.0, 6.0, 6.0, 2.5])
        scores = []
        for k in range(4):
            turned = given[k:] + given[:k]
            scores.append(white_side_score(assign_squares(homography(turned), detections, pull))
                          + (prior if k == 0 else 0.0))
        # A score that hangs on a piece sitting exactly on a line between two
        # squares would make the case a test of rounding, not of the port.
        if any(abs(project(homography(given), d.anchor)[i] % 1 - 0.5) > 0.499
               for d in detections for i in (0, 1)):
            continue
        oriented = orient_by_pieces(given, detections, pull, prior)
        chosen = next(k for k in range(4) if given[k:] + given[:k] == oriented)
        assert scores[chosen] == max(scores) and chosen == scores.index(max(scores))
        placement = board_to_fen_placement(assign_squares(homography(oriented), detections, pull))
        cases.append(dict(
            corners=[list(p) for p in given], pull=pull, prior=prior,
            detections=[dict(cls=d.cls, confidence=d.confidence, box=list(d.bbox))
                        for d in detections],
            scores=scores, turn=chosen, placement=placement))
    return cases


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=["orientation"])
    ap.add_argument("--count", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20261113)
    args = ap.parse_args()
    if args.what == "orientation":
        cases = orientation_cases(args.count, args.seed)
        out = FIXTURES / "orientation_cases.json"
        out.write_text(json.dumps(cases, separators=(",", ":")) + "\n")
        turns = [c["turn"] for c in cases]
        print(f"{len(cases)} cases written to {out}; "
              f"rotations chosen: {[turns.count(k) for k in range(4)]}")


if __name__ == "__main__":
    main()
