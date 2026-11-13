"""Writes the cases that the Swift tests replay, so that Packages/ChessVision
is checked against this folder's Python and not against a second opinion.

    python swift_reference.py orientation
    python swift_reference.py grid-cases
    python swift_reference.py grid --list photos.json --out python_grid.json
    python swift_reference.py compare --python python_grid.json --swift swift_grid.json

orientation: random boards seen in perspective with their corners given in a
random rotation, some pieces missed, misread or found twice, some boxes beside
the board; what chess_pipeline.orient_by_pieces makes of them. Written
to Packages/ChessVision/Tests/ChessVisionTests/Fixtures/orientation_cases.json
and replayed by BoardOrientationTests.

grid-cases: what grid_corners.py and the OpenCV calls it is built on make of
small inputs, for GridCornersTests: three drawn boards (written next to the
JSON as grid_board_*.png), the four fixture photos, and a patch of image for
the filters. Written to Fixtures/grid_cases.json.

grid: what grid_corners.find_grid makes of each photo of a list (a JSON object
whose keys are photo paths, e.g. one written by `evaluate.py --dump-list`):
the corners before the validity checks, the confidence and the numbers the
checks look at. `app_check --grid` writes the same for the Swift port, and
`swift_reference.py compare` puts the two side by side.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np

from chess_pipeline import (
    Detection, FEN_LETTER, assign_detections, assign_squares, board_square,
    board_to_fen_placement, fen_placement_to_board, homography, orient_by_pieces,
    project, pulled_board_point, white_side_score,
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
    counts = {"boxes": 0, "moved to a neighbour": 0, "dropped": 0, "outside": 0}
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
        confidences = rng.sample(range(300, 1000), 90)

        def box(letter, file_x, rank_y):
            x, y = project(to_image, (file_x, rank_y))
            w, h = size * rng.uniform(0.04, 0.07), size * rng.uniform(0.08, 0.2)
            y += 0.02 * size                     # the box bottom is the front of the base
            return Detection(CLASS_OF[letter], confidences.pop() / 1000,
                             (x - w / 2, y - h, x + w / 2, y))

        board = fen_placement_to_board(rng.choice(PLACEMENTS))
        for (file, rank), letter in board.items():
            if rng.random() < 0.08:
                continue                          # a missed piece
            if rng.random() < 0.05:
                letter = rng.choice("KQRBNPkqrbnp")   # a misread one
            detections.append(box(letter, file + 0.5 + rng.uniform(-0.2, 0.2),
                                  rank + 0.5 + rng.uniform(-0.2, 0.2)))
            if rng.random() < 0.12:
                # the same piece found twice, perhaps as another class: the
                # less confident box is moved to a free neighbour or dropped
                detections.append(box(rng.choice([letter, rng.choice("KQRBNPkqrbnp")]),
                                      file + 0.5 + rng.uniform(-0.3, 0.3),
                                      rank + 0.5 + rng.uniform(-0.3, 0.3)))
        for _ in range(rng.choice([0, 0, 1, 2])):
            # something beside the board: within half a square it is pulled
            # onto the edge square, further out it is left out
            along, out = rng.uniform(0.2, 7.8), rng.choice([-1, 1]) * rng.uniform(0.1, 0.9)
            out = out if out < 0 else 8 + out
            detections.append(box(rng.choice("KQRBNPkqrbnp"),
                                  *((along, out) if rng.random() < 0.5 else (out, along))))
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
        # A piece sitting exactly on a line between two squares, or exactly
        # half a square outside the board, would make the case a test of
        # rounding, not of the port. (Turning the corners turns the board
        # frame, so one rotation stands for all four.)
        def on_a_line(d):
            return any(abs(b % 1 - 0.5) > 0.499 or abs(b + 0.5) < 0.001 or abs(b - 8.5) < 0.001
                       for b in pulled_board_point(homography(given), d.anchor, pull))
        if any(on_a_line(d) for d in detections):
            continue
        oriented = orient_by_pieces(given, detections, pull, prior)
        chosen = next(k for k in range(4) if given[k:] + given[:k] == oriented)
        assert scores[chosen] == max(scores) and chosen == scores.index(max(scores))
        H = homography(oriented)
        placed = assign_detections(H, detections, pull)
        placement = board_to_fen_placement(assign_squares(H, detections, pull))
        outside = sum(board_square(H, d.anchor, pull) is None for d in detections)
        moved = sum(board_square(H, d.anchor, pull) != sq for sq, d in placed.items())
        counts["outside"] += outside
        counts["moved to a neighbour"] += moved
        counts["dropped"] += len(detections) - outside - len(placed)
        counts["boxes"] += len(detections)
        cases.append(dict(
            corners=[list(p) for p in given], pull=pull, prior=prior,
            detections=[dict(cls=d.cls, confidence=d.confidence, box=list(d.bbox))
                        for d in detections],
            scores=scores, turn=chosen, placement=placement))
    return cases, counts


def drawn_board(seed, pieces, board=True, size=(800, 600)):
    """A chessboard drawn in perspective on a table, with a frame and, if
    asked, discs for pieces. Returns (BGR image, the grid's outer corners)."""
    import cv2
    rng = np.random.default_rng(seed)
    px, frame = 64, 22
    side = 8 * px + 2 * frame
    flat = np.zeros((side, side, 3), np.uint8)
    flat[:] = (40, 70, 110)                                   # the frame
    light, dark = (208, 236, 235), (86, 149, 119)
    for i in range(8):
        for j in range(8):
            flat[frame + j * px:frame + (j + 1) * px, frame + i * px:frame + (i + 1) * px] = \
                light if (i + j) % 2 == 0 else dark
    if pieces:
        for i in range(8):
            for j in range(8):
                if rng.random() < 0.35:
                    centre = (frame + i * px + px // 2, frame + j * px + px // 2)
                    colour = (245, 245, 240) if rng.random() < 0.5 else (35, 35, 40)
                    cv2.circle(flat, centre, int(px * rng.uniform(0.22, 0.34)), colour, -1, cv2.LINE_AA)
    w, h = size
    far = rng.uniform(0.62, 0.9)
    quad = np.array([[-0.5 * far, -0.5 * rng.uniform(0.7, 1.0)], [0.5 * far, -0.5 * rng.uniform(0.7, 1.0)],
                     [0.5, 0.5], [-0.5, 0.5]]) * 0.78 * h
    angle = rng.uniform(-0.6, 0.6)
    turn = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    quad = quad @ turn.T + np.array([w / 2 + rng.uniform(-30, 30), h / 2 + rng.uniform(-20, 20)])
    src = np.array([[0, 0], [side, 0], [side, side], [0, side]], np.float32)
    M = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
    table = np.zeros((h, w, 3), np.uint8)
    table[:] = (150, 160, 165)
    table = (table * np.linspace(0.85, 1.1, w)[None, :, None]).clip(0, 255).astype(np.uint8)
    img = table
    if board:
        img = cv2.warpPerspective(flat, M, (w, h), dst=table.copy(), flags=cv2.INTER_AREA,
                                  borderMode=cv2.BORDER_TRANSPARENT)
    else:
        for _ in range(25):                                   # clutter, no board
            a, b = rng.integers(0, w, 2), rng.integers(0, h, 2)
            cv2.rectangle(img, (int(a.min()), int(b.min())), (int(a.max()), int(b.max())),
                          tuple(int(v) for v in rng.integers(0, 255, 3)), -1)
    img = cv2.GaussianBlur(img, (0, 0), 0.9)
    img = (img.astype(np.float32) + rng.normal(0, 2.0, img.shape)).clip(0, 255).astype(np.uint8)
    grid = np.array([[frame, frame], [side - frame, frame], [side - frame, side - frame],
                     [frame, side - frame]], np.float32)             # clockwise in the image
    corners = cv2.perspectiveTransform(grid[None], M)[0]
    return img, corners.tolist()


def grid_cases():
    import cv2
    import grid_corners as g
    cv2.setNumThreads(1)
    out = {}

    def fit(image, box=None):
        _, conf, info = g.find_grid(image, prior_box=box, return_info=True)
        return dict(raw=info["raw_corners"], confidence=conf, stage=info["stage"],
                    **{"pass": info.get("pass", 1)},
                    values={k: float(v) for k, v in info.items()
                            if isinstance(v, (int, float, bool))})

    # 1. drawn boards: the same pixels for Python and Swift (PNG is lossless)
    boards = {}
    for name, seed, pieces, board in (("plain", 1, False, True), ("pieces", 2, True, True),
                                      ("clutter", 3, False, False)):
        img, corners = drawn_board(seed, pieces, board)
        file = f"grid_board_{name}.png"
        cv2.imwrite(str(FIXTURES / file), img)
        img = cv2.imread(str(FIXTURES / file))
        boards[name] = dict(file=file, truth=corners if board else None, fit=fit(img))
        if board:
            xs, ys = [c[0] for c in corners], [c[1] for c in corners]
            box = [min(xs), min(ys), max(xs), max(ys)]
            boards[name]["box"] = box
            boards[name]["fit_in_box"] = fit(img, tuple(box))
    out["boards"] = boards

    # 2. the fixture photos (JPEG: the two decoders differ by a grey level here and there)
    out["photos"] = {name: fit(cv2.imread(str(FIXTURES / name)))
                     for name in sorted(json.loads((FIXTURES / "expected.json").read_text()))}

    # 3. the image operations, on a patch of the first drawn board
    img = cv2.imread(str(FIXTURES / "grid_board_pieces.png"))
    gray, s, _ = g._prepare(img, None)
    assert s == 1.0
    patch = gray[250:298, 330:394].copy()                   # 48 rows, 64 columns
    blur2 = cv2.GaussianBlur(patch, (0, 0), 2.0)
    blur1 = cv2.GaussianBlur(patch, (0, 0), 1.0)
    ixx = cv2.Sobel(blur2, cv2.CV_32F, 2, 0, ksize=3)
    iyy = cv2.Sobel(blur2, cv2.CV_32F, 0, 2, ksize=3)
    ixy = cv2.Sobel(blur2, cv2.CV_32F, 1, 1, ksize=3)
    sad = ixy * ixy - ixx * iyy
    sad[sad < 0] = 0
    M = np.array([[0.9, 0.2, -3.0], [-0.15, 1.1, 2.0], [0.0004, 0.0009, 1.0]])
    warp = cv2.warpPerspective(patch, M, (40, 30), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
    pts = g.x_corners(gray)
    start = np.rint(pts[:12]).astype(np.float32) + np.float32(1.0)   # a pixel off, to be pulled back
    g1 = cv2.GaussianBlur(gray, (0, 0), 1.0)
    refined = start.copy()
    cv2.cornerSubPix(g1, refined, (4, 4), (-1, -1),
                     (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
    ring = [[float(v) for v in g._ring_test(g1, np.rint(pts[:6]), r)[0]] for r in (4, 7)]
    out["ops"] = dict(
        width=64, height=48, patch=patch.astype(int).ravel().tolist(),
        gray_sum=float(gray.astype(np.float64).sum()), gray_at=[float(gray[y, x]) for y, x in ((0, 0), (300, 400), (599, 799))],
        blur1=blur1.ravel().tolist(), blur2=blur2.ravel().tolist(), saddle=sad.ravel().tolist(),
        dilated=cv2.dilate(sad, np.ones((9, 9), np.uint8)).ravel().tolist(),
        warp_matrix=M.ravel().tolist(), warp=warp.ravel().tolist(),
        x_corners=pts.tolist(), subpix_start=start.tolist(), subpix=refined.tolist(),
        ring_points=np.rint(pts[:6]).tolist(), ring_response=ring)

    # 4. the shrink to the working size (cv2.resize INTER_AREA, then grey), on
    # a picture both sides can compute: 1500x1100, each channel a formula
    yy, xx = np.mgrid[0:1100, 0:1500]
    formula = np.stack([(xx + yy * 13) % 256, (xx * 5 + yy * 11) % 256, (xx * 7 + yy * 3) % 256],
                       axis=2).astype(np.uint8)                       # B, G, R
    small, scale, _ = g._prepare(formula, None)
    out["resize"] = dict(
        width=int(small.shape[1]), height=int(small.shape[0]), scale=scale,
        step_x=37, step_y=41, samples=small[::41, ::37].astype(int).ravel().tolist())
    boxed, scale, origin = g._prepare(formula, (200.5, 100.25, 1400.0, 1090.0))
    out["resize_in_box"] = dict(
        box=[200.5, 100.25, 1400.0, 1090.0], width=int(boxed.shape[1]), height=int(boxed.shape[0]),
        scale=scale, origin=[int(origin[0]), int(origin[1])],
        step_x=37, step_y=41, samples=boxed[::41, ::37].astype(int).ravel().tolist())

    # (a shrink by exactly two takes a shortcut in OpenCV that rounds differently)
    yy, xx = np.mgrid[0:1300, 0:2000]
    formula = np.stack([(xx + yy * 13) % 256, (xx * 5 + yy * 11) % 256, (xx * 7 + yy * 3) % 256],
                       axis=2).astype(np.uint8)
    halved, scale, _ = g._prepare(formula, None)
    out["resize_by_two"] = dict(
        width=int(halved.shape[1]), height=int(halved.shape[0]), scale=scale,
        step_x=37, step_y=41, samples=halved[::41, ::37].astype(int).ravel().tolist())

    # 5. homographies from points
    rng = np.random.default_rng(7)
    true = np.array([[0.031, 0.004, -9.2], [-0.006, 0.027, -3.1], [0.00004, 0.00011, 1.0]])
    src = rng.uniform(100, 900, (40, 2)).astype(np.float32)
    proj = np.c_[src, np.ones(40)] @ true.T
    dst = np.rint(proj[:, :2] / proj[:, 2:3] + rng.normal(0, 0.03, (40, 2))).astype(np.float32)
    H, _ = cv2.findHomography(src, dst, 0)
    quad = np.array([[10, 20], [300, 40], [280, 260], [30, 240]], np.float32)
    unit = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    sample = rng.normal(50, 20, 37)
    out["math"] = dict(
        src=src.tolist(), dst=dst.tolist(), least_squares=H.ravel().tolist(),
        quad=quad.tolist(), four_point=cv2.getPerspectiveTransform(quad, unit).ravel().tolist(),
        sample=sample.tolist(), median=float(np.median(sample)),
        percentiles={str(q): float(np.percentile(sample, q)) for q in (8, 20, 80, 92)})

    path = FIXTURES / "grid_cases.json"
    path.write_text(json.dumps(out, separators=(",", ":")) + "\n")
    for name, b in boards.items():
        print(f"{name}: {b['fit']['stage']} pass {b['fit']['pass']} confidence {b['fit']['confidence']:.3f}"
              + (f"; in its box {b['fit_in_box']['confidence']:.3f}" if "fit_in_box" in b else ""))
    for name, p in out["photos"].items():
        print(f"{name}: confidence {p['confidence']:.3f}")
    print(f"written to {path} ({path.stat().st_size // 1024} kB)")


def grid_reference(list_path, out_path, limit):
    import cv2
    from grid_corners import find_grid
    cv2.setNumThreads(1)
    photos = sorted(json.loads(Path(list_path).read_text()))
    if limit:
        photos = photos[:limit]
    out = {}
    for n, photo in enumerate(photos):
        path = photo if photo.startswith("/") else str(Path(list_path).parent / photo)
        _, conf, info = find_grid(cv2.imread(path), return_info=True)
        raw = info.pop("raw_corners")
        stage = info.pop("stage")
        out[photo] = dict(
            raw=raw, confidence=conf, stage=stage, **{"pass": info.pop("pass", 1)},
            values={k: float(v) for k, v in info.items() if isinstance(v, (int, float, bool))})
        if (n + 1) % 25 == 0:
            print(f"{n + 1} of {len(photos)}", flush=True)
    Path(out_path).write_text(json.dumps(out, indent=0, sort_keys=True))
    print(f"{len(out)} photos written to {out_path}")


def compare(python_path, swift_path):
    """Python's and Swift's grid fits of the same photos, side by side."""
    py = json.loads(Path(python_path).read_text())
    sw = json.loads(Path(swift_path).read_text())
    photos = sorted(set(py) & set(sw))
    both = only_py = only_sw = neither = 0
    worst = []
    for photo in photos:
        a, b = py[photo], sw[photo]
        ok_a, ok_b = a["confidence"] > 0, b["confidence"] > 0
        both += ok_a and ok_b
        only_py += ok_a and not ok_b
        only_sw += ok_b and not ok_a
        neither += not ok_a and not ok_b
        if ok_a != ok_b:
            print(f"  accepted by {'Python' if ok_a else 'Swift'} only: {Path(photo).name}  "
                  f"python {a['stage']} pass {a['pass']}, swift {b['stage']} pass {b['pass']}")
        if ok_a and ok_b:
            ca, cb = np.array(a["raw"]), np.array(b["raw"])
            # the two may start at different corners: best of the 4 rotations,
            # in squares (the side of a square from the board's area)
            square = np.sqrt(a["values"]["area"]) / 8
            d = min(np.linalg.norm(ca - np.roll(cb, k, axis=0), axis=1).max() for k in range(4))
            worst.append((d / square, d, Path(photo).name))
    print(f"{len(photos)} photos: accepted by both {both}, by Python only {only_py}, "
          f"by Swift only {only_sw}, by neither {neither}")
    if worst:
        e = np.array([w[0] for w in worst])
        print(f"corners where both accept, Swift against Python, in squares: "
              f"median {np.median(e):.4f}, p95 {np.percentile(e, 95):.4f}, max {e.max():.4f}")
        for sq, px, name in sorted(worst, reverse=True)[:5]:
            print(f"  {name}: {sq:.4f} squares ({px:.2f} px)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=["orientation", "grid-cases", "grid", "compare"])
    ap.add_argument("--list", help="grid: JSON object keyed by photo path")
    ap.add_argument("--out", help="grid: where to write the reference")
    ap.add_argument("--limit", type=int, default=0, help="grid: only the first N photos")
    ap.add_argument("--python", help="compare: file written by `grid`")
    ap.add_argument("--swift", help="compare: file written by `app_check --grid`")
    ap.add_argument("--count", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20261113)
    args = ap.parse_args()
    if args.what == "orientation":
        cases, counts = orientation_cases(args.count, args.seed)
        out = FIXTURES / "orientation_cases.json"
        out.write_text(json.dumps(cases, separators=(",", ":")) + "\n")
        turns = [c["turn"] for c in cases]
        print(f"{len(cases)} cases written to {out}; "
              f"rotations chosen: {[turns.count(k) for k in range(4)]}; {counts}")
    elif args.what == "grid-cases":
        grid_cases()
    elif args.what == "grid":
        grid_reference(args.list, args.out, args.limit)
    else:
        compare(args.python, args.swift)


if __name__ == "__main__":
    main()
