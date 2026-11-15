"""Writes the cases that the Swift tests replay, so that Packages/ChessVision
is checked against this folder's Python and not against a second opinion.

    python swift_reference.py orientation
    python swift_reference.py grid-cases
    python swift_reference.py colour-cases
    python swift_reference.py king-cases
    python swift_reference.py dark-cases
    python swift_reference.py window-cases
    python swift_reference.py odd-cases
    python swift_reference.py grid --list photos.json --out python_grid.json
    python swift_reference.py compare --python python_grid.json --swift swift_grid.json
    python swift_reference.py colours --swift swift_colours.json [--list photos.json]
    python swift_reference.py dark --swift swift_dark.json [--list photos.json]
    python swift_reference.py window --swift swift_window.json

orientation: random boards seen in perspective with their corners given in a
random rotation, some pieces missed, misread or found twice, some boxes beside
the board; what chess_pipeline.orient_by_pieces makes of them. Written
to Packages/ChessVision/Tests/ChessVisionTests/Fixtures/orientation_cases.json
and replayed by BoardOrientationTests.

grid-cases: what grid_corners.py and the OpenCV calls it is built on make of
small inputs, for GridCornersTests: six drawn pictures (written next to the
JSON as grid_board_*.png), the four fixture photos, sets of x-corners for the
lattice fit, and a patch of image for the filters. Written to Fixtures/grid_cases.json.

colour-cases: what chess_pipeline.sides_from_colour and its parts make of
small inputs, for PieceSidesTests: colours in CIE Lab (evaluate.srgb_to_lab),
the patch of a box, points split in two groups, detections with their colours
(the model wrong about the side of none, a few or all of the pieces), and one
drawn picture of brown and black pieces that a model calls all black (written
next to the JSON as colour_pieces.png). Written to Fixtures/colour_cases.json.

king-cases: what chess_pipeline.one_king_a_side makes of detections and the
scores the model gave each of them for every kind: boards with no, one, two
and more kings, on one side or both, scores on the floor and just under it,
equal scores, kinds without a score; and of each of them with
queen_for_king (a side without a king and without a piece the model gave a
"king" score takes its queen: one queen, several, equally sure ones, none). Written to Fixtures/king_cases.json and
replayed by OneKingASideTests. These cases are the whole contract: the Python
gets its scores from overlapping boxes (evaluate.kind_scores_of) and the app
from the class scores of the box itself, so the two cannot be compared on
photos piece for piece.

dark-cases: a1 is a dark square. What grid_corners.dark_corner_rotations
makes of chequered pictures that both sides draw from a few numbers (each of
its three thresholds from both sides: 40 cells seen and 39, on a board turned
in the picture; contrast; how cleanly the colours split; a cell exactly at
the middle grey), of the drawn boards of grid-cases and of the four fixture
photos, each with rings of corners in every rotation and a square off; and
which rotation chess_pipeline.orient_by_pieces chooses for the cases of
orientation_cases.json when only two of the four are allowed (turns=).
Written to Fixtures/dark_cases.json, replayed by GridSquareColoursTests.
Needs grid_cases.json and orientation_cases.json as they are in Fixtures.

window-cases: the corner model picks the grid fit's window. What
chess_pipeline.window_by_model makes of a fit's corners and a model's: the
model's corners near the fit's, a square on in each of the eight directions,
two squares on, one corner unlike the others, each threshold from both sides
(the mean just under and over half a square, a corner just inside and outside
the tolerance), the model's corners a rotation on, other tolerances; and where
a window lies when it is moved on purpose (evaluate.py --shift-fit). Written
to Fixtures/window_cases.json, replayed by GridFitWindowTests.

odd-cases: the side of single pieces. What chess_pipeline.odd_sides and
patch_hidden make of detections with their colours: boards where the model
calls a few light pieces Black, each of the rule's conditions from both
sides (twice as near; three spreads apart, with the spread on its floor and
above it; White the lighter side, and equally light; two pieces a side; a
fifth of the patch covered, a foot lower and level), pieces without a
colour, a box of no width, random boards with boxes that overlap, and the
drawn picture of colour-cases with three of its brown pieces called Black.
Written to Fixtures/odd_cases.json, replayed by OddSidesTests. Needs
colour_pieces.png as colour-cases writes it.

grid: what grid_corners.find_grid makes of each photo of a list (a JSON object
whose keys are photo paths, e.g. one written by `evaluate.py --dump-list`):
the corners before the validity checks, the confidence and the numbers the
checks look at. `app_check --grid` writes the same for the Swift port, and
`swift_reference.py compare` puts the two side by side.

colours: the colour step on photos, Swift against Python on the same boxes.
`app_check --colours` writes what the app's piece model finds in each photo
of a list, the colour Swift measures for each piece and the sides it ends up
with; this reads that file, measures the colours and decides the sides in
Python (evaluate.piece_colours, chess_pipeline.sides_from_colour) and counts
where the two differ. The same for the side of single pieces
(chess_pipeline.odd_sides, patch_hidden), from the Python's own colours and
from the Swift's. `--list`: the list given to app_check, when its photo
paths are relative.

dark: a1 is a dark square, on photos, Swift against Python on the same
corners. `app_check --dark` writes, for each photo of a list, the corners the
app ends with before a1 is checked (the grid fit's or the model's) and what
the colours of the squares say of them; this reads that file, asks
grid_corners.dark_corner_rotations about the same corners and counts where
the two differ.

window: the fit's window, on photos, Swift against Python on the same
corners. `app_check --window` writes, for each photo of a list where the
corner model and the grid fit both find the board, the two sets of corners
and what the app makes of them (with `--shift-fit FILES,RANKS`: after the
fit's window was moved on purpose); this reads that file, asks
chess_pipeline.window_by_model about the same corners and counts where the
two differ, and how many windows were moved, and moved back.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np

from chess_pipeline import (
    Detection, EDGE_MARGIN, FEN_LETTER, KING_FLOOR, KINDS, PIECE_CLASSES, assign_detections, assign_squares,
    board_square, board_to_fen_placement, colour_patch, fen_placement_to_board, homography,
    odd_sides, one_king_a_side, orient_by_pieces, colour_sides, patch_hidden, project, pulled_board_point,
    sides_from_colour, two_groups, white_side_score, window_by_model, WINDOW_NEAR,
    CHROMA_WEIGHT, COLOUR_PATCH, SIDE_APART, SIDE_HIDDEN, SIDE_NEARER, SIDE_SPREAD_FLOOR,
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
    counts = {"boxes": 0, "moved to a neighbour": 0, "dropped": 0, "outside": 0,
              "taken back onto the rim": 0}
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
            # something beside the board: within EDGE_MARGIN of a square it
            # is taken back onto the rim square, further out it is left out
            along, out = rng.uniform(0.2, 7.8), rng.choice([-1, 1]) * rng.uniform(0.05, 0.6)
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
        # EDGE_MARGIN outside the board, would make the case a test of
        # rounding, not of the port. (Turning the corners turns the board
        # frame, so one rotation stands for all four.)
        def on_a_line(d):
            return any(abs(b % 1 - 0.5) > 0.499 or abs(b + EDGE_MARGIN) < 0.001
                       or abs(b - 8 - EDGE_MARGIN) < 0.001
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
        counts["taken back onto the rim"] += sum(
            board_square(H, d.anchor, pull) is not None
            and not all(0 <= b < 8 for b in pulled_board_point(H, d.anchor, pull))
            for d in detections)
        counts["moved to a neighbour"] += moved
        counts["dropped"] += len(detections) - outside - len(placed)
        counts["boxes"] += len(detections)
        cases.append(dict(
            corners=[list(p) for p in given], pull=pull, prior=prior,
            detections=[dict(cls=d.cls, confidence=d.confidence, box=list(d.bbox))
                        for d in detections],
            scores=scores, turn=chosen, placement=placement))
    return cases, counts


def drawn_board(seed, pieces, board=True, size=(800, 600), cells=8, offset=(0, 0), strays=False):
    """A chessboard drawn in perspective on a table, with a frame and, if
    asked, discs for pieces. `cells`: squares a side (10 is not a chessboard).
    `offset`: moves the board, e.g. partly out of the picture. `strays`: other
    chequered things on the table. Returns (BGR image, the grid's corners)."""
    import cv2
    rng = np.random.default_rng(seed)
    px, frame = 64 * 8 // cells, 22
    side = cells * px + 2 * frame
    flat = np.zeros((side, side, 3), np.uint8)
    flat[:] = (40, 70, 110)                                   # the frame
    light, dark = (208, 236, 235), (86, 149, 119)
    for i in range(cells):
        for j in range(cells):
            flat[frame + j * px:frame + (j + 1) * px, frame + i * px:frame + (i + 1) * px] = \
                light if (i + j) % 2 == 0 else dark
    if pieces:
        for i in range(cells):
            for j in range(cells):
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
    quad = quad + np.array(offset)
    src = np.array([[0, 0], [side, 0], [side, side], [0, side]], np.float32)
    M = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
    table = np.zeros((h, w, 3), np.uint8)
    table[:] = (150, 160, 165)
    table = (table * np.linspace(0.85, 1.1, w)[None, :, None]).clip(0, 255).astype(np.uint8)
    img = table
    if strays:
        # a chequered strip and a small chequered mat beside the board
        for x0, y0, n, m, c in ((20, 30, 9, 2, 26), (610, 430, 4, 4, 34)):
            for i in range(n):
                for j in range(m):
                    cv2.rectangle(img, (x0 + i * c, y0 + j * c), (x0 + (i + 1) * c - 1, y0 + (j + 1) * c - 1),
                                  (230, 230, 230) if (i + j) % 2 else (40, 40, 40), -1)
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
    for name, seed, pieces, board, more in (
            ("plain", 1, False, True, {}), ("pieces", 2, True, True, {}),
            ("clutter", 3, False, False, {}),
            ("cut", 4, True, True, dict(offset=(175, 0))),      # part of the board outside the picture
            ("ten", 5, False, True, dict(cells=10)),            # ten squares a side: no chessboard
            ("busy", 6, True, True, dict(strays=True))):        # other chequered things beside it
        img, corners = drawn_board(seed, pieces, board, **more)
        file = f"grid_board_{name}.png"
        cv2.imwrite(str(FIXTURES / file), img)
        img = cv2.imread(str(FIXTURES / file))
        is_chessboard = board and more.get("cells", 8) == 8
        boards[name] = dict(file=file, truth=corners if is_chessboard else None, fit=fit(img))
        if is_chessboard:
            xs, ys = [c[0] for c in corners], [c[1] for c in corners]
            box = [min(xs), min(ys), max(xs), max(ys)]
            boards[name]["box"] = box
            boards[name]["fit_in_box"] = fit(img, tuple(box))
    out["boards"] = boards

    # 2. the fixture photos (JPEG: the two decoders differ by a grey level here and there)
    photos = sorted(json.loads((FIXTURES / "expected.json").read_text()))
    out["photos"] = {name: fit(cv2.imread(str(FIXTURES / name))) for name in photos}

    # 2b. the lattice fit on its own: x-corners in, homography and nodes out.
    # The x-corners of the photos (as Python finds them), and made-up sets with
    # points that are not x-corners: square centres (pieces), edge midpoints,
    # strays.
    def lattice(points):
        points = np.asarray(points, np.float32)
        found = g.fit_lattice(points)
        if found is None:
            return dict(points=points.tolist(), fit=None)
        H, ok, r = found
        return dict(points=points.tolist(),
                    fit=dict(h=(H / H[2, 2]).ravel().tolist(), inlier=ok.astype(int).tolist(),
                             node=r.tolist()))

    sets = {}
    for name in photos:
        gray, _, _ = g._prepare(cv2.imread(str(FIXTURES / name)), None)
        sets[name] = lattice(g.x_corners(gray))
    rng = np.random.default_rng(11)
    to_image = np.array([[52.0, 9, 310], [-6, 47, 220], [0.004, 0.011, 1]])

    def seen(ij):
        p = np.c_[np.asarray(ij, float), np.ones(len(ij))] @ to_image.T
        return p[:, :2] / p[:, 2:3] + rng.normal(0, 0.15, (len(ij), 2))

    corners = [(i, j) for i in range(7) for j in range(7)]
    centres = [(i + 0.5, j + 0.5) for i in range(6) for j in range(6)]
    sets["centres"] = lattice(np.r_[seen(corners), seen(centres)])
    sets["a few centres"] = lattice(np.r_[seen(corners), seen(centres[::5])])
    sets["midpoints"] = lattice(np.r_[seen(corners), seen([(i + 0.5, j) for i in range(6) for j in (0, 3, 6)])])
    sets["strays"] = lattice(np.r_[seen(corners), rng.uniform(50, 900, (25, 2))])
    sets["two rows"] = lattice(seen([(i, j) for i in range(7) for j in range(2)]))
    sets["too few"] = lattice(seen(corners[:8]))
    out["lattices"] = sets

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

    # (a shrink by a whole number takes a shortcut in OpenCV that rounds
    # differently, and an odd side leaves a last row of cut-off blocks)
    out["resize_whole"] = []
    for w, h in ((2000, 1300), (2000, 1503), (1001, 3000), (6000, 1205)):
        yy, xx = np.mgrid[0:h, 0:w]
        formula = np.stack([(xx + yy * 13) % 256, (xx * 5 + yy * 11) % 256, (xx * 7 + yy * 3) % 256],
                           axis=2).astype(np.uint8)
        small, scale, _ = g._prepare(formula, None)
        last_row = small[-1, ::7].astype(int).tolist()
        last_column = small[::7, -1].astype(int).tolist()
        out["resize_whole"].append(dict(
            source=[w, h], width=int(small.shape[1]), height=int(small.shape[0]), scale=scale,
            step_x=37, step_y=41, samples=small[::41, ::37].astype(int).ravel().tolist(),
            last_row=last_row, last_column=last_column))

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
    for name, case in sets.items():
        found = case["fit"]
        print(f"lattice {name}: {len(case['points'])} points, "
              + ("none" if found is None else f"{sum(found['inlier'])} inliers"))
    print(f"written to {path} ({path.stat().st_size // 1024} kB)")


def drawn_pieces(seed, size=(480, 400)):
    """A chequered table top under uneven light with upright pieces of two
    colours on it: light brown ("White") and near black. Returns (BGR image,
    [(side, bbox)]), the boxes round each piece."""
    import cv2
    rng = np.random.default_rng(seed)
    w, h = size
    img = np.zeros((h, w, 3), np.uint8)
    cell = 60
    for j in range(0, h, cell):
        for i in range(0, w, cell):
            img[j:j + cell, i:i + cell] = (150, 200, 225) if (i + j) // cell % 2 == 0 else (60, 95, 140)
    pieces = []
    for j in range(3):
        for i in range(7):
            if rng.random() < 0.15:
                continue
            side = "white" if rng.random() < 0.5 else "black"
            cx = 40 + i * 62 + int(rng.integers(-8, 9))
            foot = 110 + j * 125 + int(rng.integers(-8, 9))
            bw, bh = int(rng.integers(26, 40)), int(rng.integers(60, 100))
            shade = rng.uniform(0.8, 1.15)
            base = (95, 140, 190) if side == "white" else (38, 36, 40)
            colour = tuple(int(min(255, v * shade)) for v in base)
            cv2.ellipse(img, (cx, foot - bh // 3), (bw // 2, bh // 3), 0, 0, 360, colour, -1, cv2.LINE_AA)
            cv2.rectangle(img, (cx - bw // 4, foot - bh + bw // 3), (cx + bw // 4, foot - bh // 3), colour, -1)
            cv2.circle(img, (cx, foot - bh + bw // 3), bw // 3, colour, -1, cv2.LINE_AA)
            pieces.append((side, (cx - bw / 2 - 1.5, foot - bh - 1.25, cx + bw / 2 + 1.5, foot + 1.75)))
    light = np.linspace(0.6, 1.15, w)[None, :, None] * np.linspace(1.05, 0.85, h)[:, None, None]
    img = (img * light).clip(0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), 0.9)
    img = (img.astype(np.float32) + rng.normal(0, 2.0, img.shape)).clip(0, 255).astype(np.uint8)
    return img, pieces


def colour_cases(seed):
    import cv2
    from evaluate import piece_colours, srgb_to_lab
    cv2.setNumThreads(1)
    rng = random.Random(seed)
    out = {}

    # 1. Colours: sRGB bytes -> CIE Lab.
    colours = [(0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (1, 1, 1),
               (10, 10, 10), (11, 11, 11), (23, 24, 25), (128, 128, 128)]
    colours += [tuple(rng.randrange(256) for _ in range(3)) for _ in range(150)]
    colours += [tuple(rng.randrange(40) for _ in range(3)) for _ in range(40)]   # dark: the linear parts
    lab = srgb_to_lab(np.array([(b, g, r) for r, g, b in colours], np.uint8))
    out["lab"] = [dict(rgb=list(c), lab=[float(v) for v in l]) for c, l in zip(colours, lab)]

    # 2. The patch of a box, also for boxes that stick out of the photo or
    # lie beside it, and boxes too small to hold a pixel.
    patches = []
    for k in range(120):
        width, height = rng.choice([(640, 480), (3024, 4032), (50, 40)])
        w, h = rng.uniform(0, 0.2) * width, rng.uniform(0, 0.4) * height
        if k % 10 == 0:
            w = h = rng.choice([0.0, 0.3, 1.0])
        x, y = rng.uniform(-0.3, 1.2) * width, rng.uniform(-0.3, 1.2) * height
        box = (x, y, x + w, y + h)
        patches.append(dict(box=list(box), width=width, height=height,
                            patch=colour_patch(box, width, height)))
    out["patches"] = patches

    # 3. Points split in two groups: clouds, points on a grid (ties: a point
    # as near to one mean as to the other; several pairs equally far apart),
    # the same point twice, too few points, all points the same.
    groups = []
    for k in range(60):
        n = rng.choice([0, 1, 2, 2, 3, 5, 8, 16, 32])
        kind = k % 4
        if kind == 0:      # two clouds, sometimes overlapping
            gap = rng.choice([0.5, 2.0, 10.0, 40.0])
            pts = [[rng.gauss(gap * (i % 2), 3), rng.gauss(0, 12), rng.gauss(-gap * (i % 2), 12)]
                   for i in range(n)]
        elif kind == 1:    # whole numbers
            pts = [[float(rng.randrange(-2, 3)) for _ in range(3)] for _ in range(n)]
        elif kind == 2:    # one cloud
            pts = [[rng.uniform(0, 100), rng.uniform(-80, 80), rng.uniform(-80, 80)] for _ in range(n)]
        else:              # all the same
            pts = [[50.0, 8.0, -12.0]] * n
        split = two_groups(pts)
        groups.append(dict(points=pts, group=None if split is None else split[0],
                           centres=None if split is None else split[1].tolist()))
    out["groups"] = groups

    # 4. Detections with their colours. The pieces are of two colours; the
    # model has the side of some share of them wrong.
    sides = []
    tally = {"changed": 0, "kept, contradicted a little": 0, "kept, agreed": 0, "no split": 0}

    def case(n, wrong, gap, missing):
        light = (rng.uniform(55, 85), rng.uniform(-5, 15), rng.uniform(5, 35))
        dark = (light[0] - gap, rng.uniform(-3, 6), rng.uniform(-6, 10))
        truth = [rng.choice(["white", "black"]) for _ in range(n)]
        wrong_ones = set(rng.sample(range(n), min(n, wrong)))
        detections, cols = [], []
        for k in range(n):
            said = truth[k] if k not in wrong_ones else ("black" if truth[k] == "white" else "white")
            kind = rng.choice(["king", "queen", "rook", "bishop", "knight", "pawn"])
            x, y = rng.uniform(0, 2800), rng.uniform(0, 3800)
            detections.append(Detection(f"{said}_{kind}", rng.randrange(250, 1000) / 1000,
                                        (x, y, x + rng.uniform(60, 200), y + rng.uniform(120, 420))))
            centre = light if truth[k] == "white" else dark
            cols.append(None if rng.random() < missing
                        else tuple(c + rng.gauss(0, s) for c, s in zip(centre, (6, 1.5, 3))))
        return record(detections, cols, bool(wrong_ones))

    def record(detections, cols, contradicted=True):
        result = sides_from_colour(detections, cols)
        changed = any(a.cls != b.cls for a, b in zip(detections, result))
        by_colour = colour_sides(cols)
        if by_colour is None:
            tally["no split"] += 1
        elif changed:
            tally["changed"] += 1
        elif contradicted:
            tally["kept, contradicted a little"] += 1
        else:
            tally["kept, agreed"] += 1
        sides.append(dict(
            detections=[dict(cls=d.cls, confidence=d.confidence, box=list(d.bbox)) for d in detections],
            colours=[None if c is None else list(c) for c in cols],
            by_colour=by_colour,
            sides=[d.cls for d in result]))
        return changed

    for _ in range(70):
        n = rng.choice([0, 1, 2, 3, 6, 12, 20, 20, 32])
        case(n, wrong=rng.choice([0, 0, 1, 2, 3, 4, n // 2, n]),
             gap=rng.choice([3.0, 15.0, 30.0, 45.0]), missing=rng.choice([0.0, 0.0, 0.1, 0.6]))
    # On the line: 3 of 20 is 15%, which is not more than 15%; 4 of 20 is.
    for wrong in (3, 4, 3, 4):
        assert case(20, wrong=wrong, gap=45.0, missing=0.0) == (wrong == 4)

    def pieces(said, cols):
        return [Detection(f"{side}_pawn", 0.9 - 0.01 * k, (100.0 * k, 50.0, 100.0 * k + 60, 170.0))
                for k, side in enumerate(said)], cols
    cream, ebony = (80.0, 2.0, 20.0), (20.0, 1.0, 2.0)
    # The share is of the pieces that have a colour: 20 with a colour, the
    # model wrong about 4 of them, and 7 more without. 4 of 20 is over 15%
    # (4 of 27 would not be), and the 7 keep the model's side.
    truth = ["white", "black"] * 10
    said = ["black", "white"] * 2 + truth[4:] + ["white"] * 7
    assert record(*pieces(said, [cream if t == "white" else ebony for t in truth] + [None] * 7))
    assert [c.split("_")[0] for c in sides[-1]["sides"]] == truth + ["white"] * 7
    # Two groups equally light, told apart by colour alone: the group of the
    # first of the two points farthest apart is White.
    grey, brown = (45.0, 0.0, 0.0), (45.0, 14.0, 28.0)
    for cols in ([grey, brown, grey, brown], [brown, grey, brown, grey]):
        assert record(*pieces(["black"] * 4, cols))
        assert [c.split("_")[0] for c in sides[-1]["sides"]] == ["white", "black", "white", "black"]
    out["sides"] = sides

    # 5. A drawn picture: brown and black pieces, and a model that calls
    # every one of them black.
    img, pieces = drawn_pieces(seed)
    name = "colour_pieces.png"
    cv2.imwrite(str(FIXTURES / name), img)
    kinds = ["king", "queen", "rook", "bishop", "knight", "pawn"]
    detections = [Detection(f"black_{kinds[k % 6]}", 0.9 - 0.01 * k, bbox)
                  for k, (_, bbox) in enumerate(pieces)]
    # one box beside the picture: no colour, its side stays as the model said
    detections.append(Detection("black_pawn", 0.4, (500.0, 100.0, 540.0, 180.0)))
    cols = piece_colours(FIXTURES / name, detections)
    result = sides_from_colour(detections, cols)
    assert [d.cls.split("_")[0] for d in result[:-1]] == [side for side, _ in pieces]
    assert cols[-1] is None and result[-1].cls == "black_pawn"
    out["picture"] = dict(
        file=name,
        detections=[dict(cls=d.cls, confidence=d.confidence, box=list(d.bbox)) for d in detections],
        colours=[None if c is None else list(c) for c in cols],
        sides=[d.cls for d in result])

    # 6. A photo: the colours of boxes spread over a fixture (a JPEG: the two
    # decoders differ by a grey level here and there, the medians hardly).
    photo = "test_000000.jpg"
    height, width = cv2.imread(str(FIXTURES / photo)).shape[:2]
    boxes = []
    for _ in range(24):
        x, y = rng.uniform(0.15, 0.75) * width, rng.uniform(0.15, 0.7) * height
        boxes.append(Detection("white_pawn", 0.9, (x, y, x + rng.uniform(0.03, 0.07) * width,
                                                   y + rng.uniform(0.06, 0.14) * height)))
    out["photo"] = dict(file=photo, boxes=[list(d.bbox) for d in boxes],
                        colours=[list(c) for c in piece_colours(FIXTURES / photo, boxes)])

    path = FIXTURES / "colour_cases.json"
    path.write_text(json.dumps(out, separators=(",", ":")) + "\n")
    print(f"written to {path} ({path.stat().st_size // 1024} kB) and {name}: "
          f"{len(out['lab'])} colours, {len(patches)} patches "
          f"({sum(p['patch'] is None for p in patches)} empty), {len(groups)} sets of points "
          f"({sum(g['group'] is None for g in groups)} without a split), "
          f"{len(sides)} sets of detections {tally}, "
          f"{len(pieces)} drawn pieces ({sum(s == 'white' for s, _ in pieces)} brown)")


def odd_cases(seed):
    """chess_pipeline.odd_sides on detections and colours built from a few numbers."""
    import cv2
    from evaluate import piece_colours
    rng = random.Random(seed)
    kinds = ["king", "queen", "rook", "bishop", "knight", "pawn"]
    cases, tally = [], {"changed": 0, "pieces changed": 0, "hidden": 0}

    class OnTheEdge(Exception):
        """A case that hangs on the last bits of the arithmetic."""

    def far(condition):
        if not condition:
            raise OnTheEdge

    def margins(detections, colours, chroma, nearer, apart, cover):
        # No case may hang on the last bits of the arithmetic: every
        # comparison the rule makes is this far from a tie, or exactly on it
        # where the case is there for the tie. And every box must come back
        # the same from a corner and a size, which is how the Swift holds it
        # (x1 + (x2 - x1) is not always x2): two feet exactly level here
        # must be level there.
        for k, det in enumerate(detections):
            x1, y1, x2, y2 = det.bbox
            far(x1 + (x2 - x1) == x2 and y1 + (y2 - y1) == y2)
            w, h = x2 - x1, y2 - y1
            l, r, t, b = COLOUR_PATCH
            px1, py1, px2, py2 = x1 + l * w, y1 + t * h, x1 + r * w, y1 + b * h
            area = (px2 - px1) * (py2 - py1)
            for j, other in enumerate(detections):
                if j == k or area <= 0:
                    continue
                far(other.bbox[3] == y2 or abs(other.bbox[3] - y2) > 1e-6)
                iw = min(px2, other.bbox[2]) - max(px1, other.bbox[0])
                ih = min(py2, other.bbox[3]) - max(py1, other.bbox[1])
                if iw > 0 and ih > 0:
                    far(abs(iw * ih - cover * area) > 1e-6 * area)
        idx = [k for k, c in enumerate(colours) if c is not None and not patch_hidden(detections, k, cover)]
        point = {k: np.array([colours[k][0], chroma * colours[k][1], chroma * colours[k][2]]) for k in idx}
        side = {k: detections[k].cls.split("_", 1)[0] for k in idx}
        groups = {s: [k for k in idx if side[k] == s] for s in ("white", "black")}
        if min(len(groups["white"]), len(groups["black"])) < 2:
            return
        centre = {s: np.median([point[k] for k in groups[s]], axis=0) for s in groups}
        gap = centre["white"][0] - centre["black"][0]
        far(gap == 0 or abs(gap) > 1e-6)
        if gap <= 0:
            return
        spread = float(np.median([np.linalg.norm(point[k] - centre[side[k]]) for k in idx]))
        far(abs(spread - SIDE_SPREAD_FLOOR) > 1e-6)
        spread = max(SIDE_SPREAD_FLOOR, spread)
        apart_by = np.linalg.norm(centre["white"] - centre["black"]) - apart * spread
        far(abs(apart_by) > 1e-6)
        if apart_by < 0:
            return
        for k in groups["black"]:
            own = np.median([point[j] for j in groups["black"] if j != k], axis=0)
            far(abs(nearer * np.linalg.norm(point[k] - centre["white"]) - np.linalg.norm(point[k] - own)) > 1e-6)

    def record(name, detections, colours, chroma=CHROMA_WEIGHT, nearer=SIDE_NEARER, apart=SIDE_APART,
               cover=SIDE_HIDDEN):
        margins(detections, colours, chroma, nearer, apart, cover)
        result = odd_sides(detections, colours, chroma=chroma, nearer=nearer, apart=apart, cover=cover)
        hidden = [patch_hidden(detections, k, cover) for k in range(len(detections))]
        changed = [k for k, (a, b) in enumerate(zip(detections, result)) if a.cls != b.cls]
        assert all(detections[k].cls.startswith("black_") and result[k].cls.startswith("white_")
                   and not hidden[k] and colours[k] is not None for k in changed)
        assert all((a.confidence, a.bbox, a.cls.split("_")[1]) == (b.confidence, b.bbox, b.cls.split("_")[1])
                   for a, b in zip(detections, result))
        tally["changed"] += bool(changed)
        tally["pieces changed"] += len(changed)
        tally["hidden"] += sum(hidden)
        cases.append(dict(
            name=name, chroma=chroma, nearer=nearer, apart=apart, cover=cover,
            detections=[dict(cls=d.cls, confidence=d.confidence, box=list(d.bbox)) for d in detections],
            colours=[None if c is None else list(c) for c in colours], hidden=hidden,
            sides=[d.cls for d in result]))
        return changed

    def row(said, colours, boxes=None):
        # Pieces side by side, none in front of another unless boxes are given.
        boxes = boxes or [(100.0 * k, 100.0, 100.0 * k + 60, 220.0) for k in range(len(said))]
        return [Detection(f"{side}_{kinds[k % 6]}", 0.9 - 0.01 * k, box)
                for k, (side, box) in enumerate(zip(said, boxes))], list(colours)

    def greys(light, dark, n=6, step=1.0):
        offsets = [step * v for v in (-2, -1, 0, 0, 1, 2)][:n]
        return [(light + o, 0.0, 0.0) for o in offsets], [(dark + o, 0.0, 0.0) for o in offsets]

    # 1. The rule at work: six light and six dark pieces read rightly, and
    # one, two, three light ones the model calls Black.
    whites, blacks = greys(70.0, 20.0)
    for odd in (0, 1, 2, 3):
        said = ["white"] * 6 + ["black"] * 6 + ["black"] * odd
        changed = record(f"{odd} light pieces called Black", *row(said, whites + blacks + [(69.0, 0.0, 0.0)] * odd))
        assert changed == list(range(12, 12 + odd))
    # A piece is held against the OTHER pieces called Black, not against all
    # of them: of two, the one at L* 56 against the one at 20 (with itself
    # counted, Black's colour would be 38 and it would stay).
    assert record("two called Black, one of them light", *row(
        ["white"] * 2 + ["black"] * 2, [(70.0, 0.0, 0.0)] * 2 + [(20.0, 0.0, 0.0), (56.0, 0.0, 0.0)])) == [3]
    # ... and against the others as the model called them, whatever the
    # rule makes of them: of four called Black, at 10, 93, 93 and 75, the
    # two at 93 are White's; the one at 75 has 93 for the others' colour
    # and stays (with the two already taken away it would not).
    assert record("three light pieces of four called Black", *row(
        ["white"] * 6 + ["black"] * 4,
        [(95.0, 0.0, 0.0)] * 6 + [(10.0, 0.0, 0.0), (93.0, 0.0, 0.0), (93.0, 0.0, 0.0), (75.0, 0.0, 0.0)])) == [7, 8]
    # A side's colour is the median of each number, not the mean: one piece
    # called White far off in a* does not pull White's colour towards a
    # piece called Black that is as far off.
    assert record("one piece called White far off in a*", *row(
        ["white"] * 5 + ["black"] * 7,
        [(70.0, 0.0, 0.0)] * 4 + [(70.0, 6.0, 0.0)] + [(40.0, 0.0, 0.0)] * 6 + [(70.0, 5.0, 0.0)])) == []
    # One way only: a dark piece called White stays White.
    assert record("a dark piece called White", *row(["white"] * 7 + ["black"] * 6,
                                                   whites + [(21.0, 0.0, 0.0)] + blacks)) == []
    # 2. Twice as near: a piece called Black at L* = t is 70 - t from White
    # and t - 20 from the other Black pieces; twice as near from 53.33 on.
    for t, moves in ((53.3, False), (53.4, True), (45.0, False), (60.0, True)):
        changed = record(f"called Black at L* {t}", *row(["white"] * 6 + ["black"] * 7, whites + blacks + [(t, 0.0, 0.0)]))
        assert changed == ([12] if moves else [])
    # ... and with other numbers for "nearer".
    for nearer, moves in ((1.5, True), (3.0, False)):
        changed = record(f"called Black at L* 52, nearer {nearer}",
                         *row(["white"] * 6 + ["black"] * 7, whites + blacks + [(52.0, 0.0, 0.0)]), nearer=nearer)
        assert changed == ([12] if moves else [])
    # 3. Three spreads apart. The spread on its floor of 3: the two sides 8.9
    # and 9.1 apart. The spread above the floor (pieces 5 from their side's
    # colour): 14.9 and 15.1 apart. The piece called Black has White's colour.
    for gap, moves in ((8.9, False), (9.1, True)):
        whites2, blacks2 = greys(40.0 + gap, 40.0, step=0.0)
        changed = record(f"the sides {gap} apart, the spread on its floor",
                         *row(["white"] * 6 + ["black"] * 7, whites2 + blacks2 + [(40.0 + gap, 0.0, 0.0)]))
        assert changed == ([12] if moves else [])
    for gap, moves in ((14.9, False), (15.1, True)):
        # each piece 5 from its side's colour, by its colour and not its lightness
        whites2 = [(40.0 + gap, 5.0 * s / CHROMA_WEIGHT, 0.0) for s in (-1, 1, -1, 1, -1, 1)]
        blacks2 = [(40.0, 5.0 * s / CHROMA_WEIGHT, 0.0) for s in (-1, 1, -1, 1, -1, 1)]
        whites2 += [(40.0 + gap, 0.0, 0.0)]
        changed = record(f"the sides {gap} apart, a spread of 5",
                         *row(["white"] * 7 + ["black"] * 7, whites2 + blacks2 + [(40.0 + gap, 0.0, 0.0)]))
        assert changed == ([13] if moves else [])
    for apart, moves in ((2.0, True), (4.0, False)):
        whites2, blacks2 = greys(50.0, 40.0, step=0.0)
        changed = record(f"the sides 10 apart, apart {apart}",
                         *row(["white"] * 6 + ["black"] * 7, whites2 + blacks2 + [(50.0, 0.0, 0.0)]), apart=apart)
        assert changed == ([12] if moves else [])
    # 4. White must be the lighter side: the model's sides the wrong way
    # round, and two sides equally light that differ in colour only.
    assert record("the pieces called White are the darker", *row(
        ["black"] * 6 + ["white"] * 6 + ["black"], whites + blacks + [(21.0, 0.0, 0.0)])) == []
    level_w = [(40.0, 10.0, 0.0)] * 6
    level_b = [(40.0, -10.0, 0.0)] * 6
    assert record("the two sides equally light", *row(
        ["white"] * 6 + ["black"] * 7, level_w + level_b + [(40.0, 10.0, 0.0)])) == []
    assert record("White lighter by a hair", *row(
        ["white"] * 6 + ["black"] * 7, [(40.001, 10.0, 0.0)] * 6 + level_b + [(40.001, 10.0, 0.0)])) == [12]
    # The colour counts four times the lightness: brown and black equally
    # dark but for a little.
    brown, ebony = (32.0, 2.0, 4.0), (28.0, 0.0, 0.0)
    assert record("brown pieces hardly lighter than the black", *row(
        ["white"] * 5 + ["black"] * 6, [brown] * 5 + [ebony] * 5 + [brown])) == [10]
    assert record("the same with the colour counting once", *row(
        ["white"] * 5 + ["black"] * 6, [brown] * 5 + [ebony] * 5 + [brown]), chroma=1.0) == []
    # 5. Two pieces a side, with a colour.
    light, dark = (70.0, 0.0, 0.0), (20.0, 0.0, 0.0)
    assert record("one piece called White", *row(["white"] + ["black"] * 4, [light] + [dark] * 3 + [light])) == []
    assert record("two pieces called White", *row(["white"] * 2 + ["black"] * 4, [light] * 2 + [dark] * 3 + [light])) == [5]
    assert record("one piece called Black", *row(["white"] * 4 + ["black"], [light] * 4 + [light])) == []
    assert record("two pieces called Black", *row(["white"] * 4 + ["black"] * 2, [light] * 4 + [dark, light])) == [5]
    assert record("two pieces called Black, both light", *row(["white"] * 4 + ["black"] * 2, [light] * 6)) == []
    assert record("of three called White two have no colour", *row(
        ["white"] * 3 + ["black"] * 4, [light, None, None] + [dark] * 3 + [light])) == []
    assert record("the piece called Black has no colour", *row(
        ["white"] * 3 + ["black"] * 4, [light] * 3 + [dark] * 3 + [None])) == []
    assert record("no piece at all", [], []) == []
    # 6. A piece in front. The odd piece's box is (100, 100)-(160, 220): its
    # patch is 24 wide and 78 high. A box in front (its foot lower) covers
    # a strip of the patch's right side.
    def scene(front, odd_colour=light, said="black"):
        boxes = [(100.0, 100.0, 160.0, 220.0), front]
        boxes += [(300.0 + 100 * k, 100.0, 360.0 + 100 * k, 220.0) for k in range(6)]
        return row([said, "black"] + ["white"] * 3 + ["black"] * 3, [odd_colour, dark] + [light] * 3 + [dark] * 3, boxes)
    for share, moves in ((0.19, True), (0.21, False), (0.5, False)):
        front = (142.0 - share * 24.0, 120.0, 200.0, 260.0)
        assert record(f"{share} of the patch behind a piece in front", *scene(front)) == ([0] if moves else [])
        assert cases[-1]["hidden"][0] == (not moves)
    # The same half covered by a box whose foot is higher, and one whose
    # foot is level: neither stands in front.
    assert record("half the patch behind a box whose foot is higher", *scene((130.0, 120.0, 200.0, 215.0))) == [0]
    assert record("half the patch behind a box whose foot is level", *scene((130.0, 100.0, 200.0, 220.0))) == [0]
    for cover, moves in ((0.4, True), (0.3, False)):
        assert record(f"a third of the patch covered, cover {cover}",
                      *scene((142.0 - 24.0 / 3, 120.0, 200.0, 260.0)), cover=cover) == ([0] if moves else [])
    # A piece behind another is not counted either: of the two pieces
    # called White that would make a side, one is hidden.
    boxes = [(100.0, 100.0, 160.0, 220.0), (120.0, 120.0, 200.0, 260.0), (300.0, 100.0, 360.0, 220.0)]
    boxes += [(500.0 + 100 * k, 100.0, 560.0 + 100 * k, 220.0) for k in range(4)]
    assert record("one of two pieces called White is hidden", *row(
        ["white", "black", "white"] + ["black"] * 4, [light, dark, light] + [dark] * 3 + [light], boxes)) == []
    assert record("the same with the box in front moved away", *row(
        ["white", "black", "white"] + ["black"] * 4, [light, dark, light] + [dark] * 3 + [light],
        [boxes[0], (200.0, 120.0, 280.0, 260.0)] + boxes[2:])) == [6]
    # A box of no width has a pixel of colour and no patch to be seen on.
    flat = [(100.0 * k, 100.0, 100.0 * k + 60, 220.0) for k in range(6)] + [(700.0, 100.0, 700.0, 220.0)]
    assert record("a box of no width", *row(["white"] * 3 + ["black"] * 4, [light] * 3 + [dark] * 3 + [light], flat)) == []
    assert cases[-1]["hidden"][6]

    # 7. Random boards: pieces on a grid of places with boxes that reach
    # over their neighbours, two colours with some scatter, some light
    # pieces called Black, some dark ones called White, some in shadow,
    # some without a colour.
    made = len(cases)
    fired = on_the_edge = 0
    while len(cases) < made + 60:
        n = rng.randint(4, 24)
        places = rng.sample([(i, j) for i in range(8) for j in range(8)], n)
        white_at = (rng.uniform(45, 85), rng.uniform(-3, 12), rng.uniform(-5, 30))
        black_at = (rng.uniform(10, 45), rng.uniform(-3, 5), rng.uniform(-5, 8))
        noise = rng.choice([0.5, 2.0, 5.0, 9.0])
        detections, colours = [], []
        for k, (i, j) in enumerate(places):
            truth = rng.choice(["white", "black"])
            at = white_at if truth == "white" else black_at
            colour = tuple(round(v + rng.gauss(0, noise), 3) for v in at)
            if truth == "white" and rng.random() < 0.1:
                colour = (round(colour[0] - rng.uniform(10, 35), 3), colour[1], colour[2])     # in shadow
            said = truth
            if rng.random() < 0.15:
                said = "black" if truth == "white" else "white"
            cx, foot = 100 + 100 * i + rng.uniform(-15, 15), 180 + 95 * j + rng.uniform(-15, 15)
            w, h = rng.uniform(45, 95), rng.uniform(90, 190)
            detections.append(Detection(f"{said}_{rng.choice(kinds)}", round(rng.uniform(0.3, 0.99), 3),
                                        tuple(round(v, 2) for v in (cx - w / 2, foot - h, cx + w / 2, foot))))
            colours.append(None if rng.random() < 0.05 else colour)
        try:
            fired += bool(record(f"random {len(cases) - made}", detections, colours))
        except OnTheEdge:
            on_the_edge += 1                      # another one
    assert 15 <= fired <= 45, fired               # the random boards show both outcomes

    # 8. The drawn picture of colour-cases, with three of its brown pieces
    # called Black and the rest read rightly.
    name = "colour_pieces.png"
    img, pieces = drawn_pieces(seed)
    assert np.array_equal(cv2.imread(str(FIXTURES / name)), img), "run colour-cases first, with the same seed"
    brown_ones = [k for k, (side, _) in enumerate(pieces) if side == "white"]
    wrong = brown_ones[1::3][:3]
    detections = [Detection(f"{'black' if k in wrong else side}_{kinds[k % 6]}", 0.9 - 0.01 * k, bbox)
                  for k, (side, bbox) in enumerate(pieces)]
    detections.append(Detection("black_pawn", 0.4, (500.0, 100.0, 540.0, 180.0)))    # beside the picture
    cols = piece_colours(FIXTURES / name, detections)
    assert len(wrong) == 3 and record("the drawn picture", detections, cols) == wrong
    picture = dict(cases.pop(), file=name)

    out = dict(nearer=SIDE_NEARER, apart=SIDE_APART, spread_floor=SIDE_SPREAD_FLOOR, hidden=SIDE_HIDDEN,
               chroma=CHROMA_WEIGHT, cases=cases, picture=picture)
    path = FIXTURES / "odd_cases.json"
    path.write_text(json.dumps(out, separators=(",", ":")) + "\n")
    print(f"{len(cases)} sets of detections and the drawn picture: a piece changed in {tally['changed']}, "
          f"{tally['pieces changed']} pieces in all; {tally['hidden']} pieces behind another "
          f"({sum(sum(c['hidden']) for c in cases[made:])} of them on the random boards); "
          f"{on_the_edge} random boards refused for hanging on the last bits")
    print(f"written to {path} ({path.stat().st_size // 1024} kB)")


def king_cases(seed):
    rng = random.Random(seed)
    cases = []
    tally = {"two kings, left alone": 0, "a king given": 0, "fewer than two kings, none given": 0,
             "kings taken away": 0, "other": 0, "a queen crowned": 0}

    def piece(side, kind, scores, k, confidence=None):
        # (a number is always drawn, so that the boards before a new case
        # stay what they were)
        drawn = rng.randrange(250, 1000) / 1000
        return Detection(f"{side}_{kind}", drawn if confidence is None else confidence,
                         (60.0 * k, 40.0, 60.0 * k + 50, 150.0)), scores

    def record(pieces):
        detections = [d for d, _ in pieces]
        scores = [s for _, s in pieces]
        result = one_king_a_side(detections, scores)
        assert all(a.confidence == b.confidence and a.bbox == b.bbox
                   and a.cls.split("_")[0] == b.cls.split("_")[0] for a, b in zip(detections, result))
        kings = sum(d.cls.endswith("_king") for d in detections)
        changed = [(a.cls, b.cls) for a, b in zip(detections, result) if a.cls != b.cls]
        if kings == 2:
            assert not changed
            tally["two kings, left alone"] += 1
        elif kings < 2 and changed:
            assert all(b.endswith("_king") for _, b in changed)
            tally["a king given"] += 1
        elif kings < 2:
            tally["fewer than two kings, none given"] += 1
        elif changed:
            assert all(a.endswith("_king") and not b.endswith("_king") for a, b in changed)
            tally["kings taken away"] += 1
        else:
            tally["other"] += 1
        # The same with a side without a king taking its queen: where it
        # differs, queens became kings and nothing else, on a board with
        # fewer than two kings.
        crowned = one_king_a_side(detections, scores, queen_for_king=True)
        differs = [(a.cls, b.cls) for a, b in zip(result, crowned) if a.cls != b.cls]
        assert all(a.endswith("_queen") and b == a.replace("_queen", "_king") for a, b in differs)
        assert not differs or kings < 2
        assert all(a.confidence == b.confidence and a.bbox == b.bbox for a, b in zip(detections, crowned))
        tally["a queen crowned"] += bool(differs)
        cases.append(dict(
            detections=[dict(cls=d.cls, confidence=d.confidence, scores=s) for d, s in pieces],
            out=[d.cls for d in result], out_queen=[d.cls for d in crowned]))
        return [d.cls for d in result]

    def crowned(pieces):
        """What the case just recorded comes to with queen_for_king."""
        record(pieces)
        return cases[-1]["out_queen"]

    def scores_for(kind):
        """What a model might score a piece of this kind: its own kind high,
        a few others low, some kinds not at all."""
        scores = {kind: rng.randrange(250, 1000) / 1000}
        for other in KINDS:
            if other != kind and rng.random() < 0.45:
                scores[other] = rng.choice([0.0, 0.004, 0.019, 0.02, 0.021, 0.05, 0.1, 0.3,
                                            round(rng.uniform(0, 0.6), 3)])
        return scores

    # 1. Random boards: 0 to 4 kings, on either side, any number of pieces.
    for n in range(260):
        count = rng.choice([0, 1, 2, 3, 5, 8, 16, 24, 32])
        kings = rng.choice([0, 0, 1, 1, 2, 3, 3, 4])
        pieces = []
        for k in range(count):
            side = rng.choice(["white", "black"]) if n % 5 else "white"
            kind = "king" if k < kings else rng.choice(KINDS[1:])
            pieces.append(piece(side, kind, scores_for(kind), k))
        rng.shuffle(pieces)
        record(pieces)

    # 2. By hand, each sentence of the rule.
    def p(cls, k=0, confidence=None, **scores):
        side, kind = cls.split("_")
        return piece(side, kind, scores, k, confidence)
    # No pieces; one side only.
    assert record([]) == []
    assert record([p("white_queen", king=0.3)]) == ["white_king"]
    # Two kings: nothing, also when both are on one side.
    assert record([p("white_king", king=0.9), p("black_king", king=0.9), p("black_queen", king=0.8)]) \
        == ["white_king", "black_king", "black_queen"]
    assert record([p("white_king", king=0.9), p("white_king", king=0.2, queen=0.6),
                   p("black_queen", king=0.5)]) == ["white_king", "white_king", "black_queen"]
    # One king: the other side gets one, the side that has one is left alone.
    assert record([p("white_king", king=0.9), p("white_queen", king=0.5), p("black_queen", king=0.05),
                   p("black_rook", king=0.3)]) == ["white_king", "white_queen", "black_queen", "black_king"]
    # No king: each side its own best; one side may find none.
    assert record([p("white_queen", king=0.1), p("white_rook", king=0.2), p("black_pawn", king=0.019),
                   p("black_queen")]) == ["white_queen", "white_king", "black_pawn", "black_queen"]
    # On the floor counts, under it does not.
    assert record([p("white_queen", king=KING_FLOOR)]) == ["white_king"]
    assert record([p("white_queen", king=KING_FLOOR - 1e-9)]) == ["white_queen"]
    # Equal king scores: the earlier piece.
    assert record([p("black_rook", king=0.4), p("black_queen", king=0.4), p("black_pawn", king=0.39)]) \
        == ["black_king", "black_queen", "black_pawn"]
    # More than two kings: a side with several keeps its best, by the king
    # score and not by the confidence of the box; a side with one is left.
    assert record([p("white_king", king=0.5, queen=0.3), p("white_king", king=0.7),
                   p("white_king", king=0.6, rook=0.2, queen=0.1), p("black_king", king=0.3)]) \
        == ["white_queen", "white_king", "white_rook", "black_king"]
    # Equal king scores among several kings: the earlier one stays.
    assert record([p("white_king", king=0.6, bishop=0.1), p("white_king", king=0.6, knight=0.2),
                   p("black_king", king=0.9)]) == ["white_king", "white_knight", "black_king"]
    # The other kind: the best score; equal scores in the order queen, rook,
    # bishop, knight, pawn; no other score, a queen.
    assert record([p("black_king", king=0.9), p("black_king", king=0.5, pawn=0.2, knight=0.2),
                   p("black_king", king=0.4, rook=0.3, queen=0.3), p("black_king", king=0.3),
                   p("black_king", king=0.2, pawn=0.01)]) \
        == ["black_king", "black_knight", "black_queen", "black_queen", "black_pawn"]
    # Three kings on one side and none on the other: the other side gets none
    # (it has more than two kings on the board, not fewer).
    assert record([p("white_king", king=0.9), p("white_king", king=0.8, queen=0.1),
                   p("white_king", king=0.7), p("black_queen", king=0.6)]) \
        == ["white_king", "white_queen", "white_queen", "black_queen"]
    # Four kings, two a side.
    assert record([p("white_king", king=0.3, pawn=0.1), p("black_king", king=0.3, rook=0.2),
                   p("white_king", king=0.8), p("black_king", king=0.2, bishop=0.1)]) \
        == ["white_pawn", "black_king", "white_king", "black_bishop"]
    # A king that the model gave no king score (scores from elsewhere) loses to one that has one.
    assert record([p("white_king"), p("white_king", king=0.1), p("black_king", king=0.5)]) \
        == ["white_queen", "white_king", "black_king"]

    # 3. queen_for_king: a side without a king, and without a piece the
    # model gave a "king" score of the floor, takes its queen.
    # One queen. Without the switch nothing (record() checks that too).
    assert crowned([p("white_king", king=0.9), p("black_queen"), p("black_rook", king=0.01)]) \
        == ["white_king", "black_king", "black_rook"]
    assert cases[-1]["out"] == ["white_king", "black_queen", "black_rook"]
    # A piece with a "king" score on the floor comes first: the queen stays.
    assert crowned([p("white_king", king=0.9), p("black_queen"), p("black_rook", king=KING_FLOOR)]) \
        == ["white_king", "black_queen", "black_king"]
    assert crowned([p("white_king", king=0.9), p("black_queen"), p("black_rook", king=KING_FLOOR - 1e-9)]) \
        == ["white_king", "black_king", "black_rook"]
    # Several queens: the one the model is least sure of; equally sure, the
    # first; the confidence counts, not the "queen" score.
    assert crowned([p("white_queen", confidence=0.9, queen=0.2), p("white_queen", confidence=0.4, queen=0.9),
                    p("white_queen", confidence=0.6), p("white_pawn", confidence=0.3)]) \
        == ["white_queen", "white_king", "white_queen", "white_pawn"]
    assert crowned([p("white_queen", confidence=0.7), p("white_queen", confidence=0.5),
                    p("white_queen", confidence=0.5)]) == ["white_queen", "white_king", "white_queen"]
    # No queen: nothing, as without the switch.
    assert crowned([p("white_rook", confidence=0.2), p("white_pawn", king=0.019)]) == ["white_rook", "white_pawn"]
    # Neither side has a king: each takes its own queen, and only its own.
    assert crowned([p("white_queen"), p("black_queen"), p("black_queen", confidence=0.26), p("white_rook")]) \
        == ["white_king", "black_queen", "black_king", "white_rook"]
    assert crowned([p("white_rook"), p("black_queen")]) == ["white_rook", "black_king"]
    # Two kings, both on one side: left alone, the other side's queen too.
    assert crowned([p("white_king", king=0.9), p("white_king", king=0.8), p("black_queen")]) \
        == ["white_king", "white_king", "black_queen"]
    # Three kings on one side: the board has more than two, the other side
    # gets none, its queen stays.
    assert crowned([p("white_king", king=0.9), p("white_king", king=0.8, queen=0.1),
                    p("white_king", king=0.7), p("black_queen")]) \
        == ["white_king", "white_queen", "white_queen", "black_queen"]
    # A side that has its king is left alone whatever its queens.
    assert crowned([p("white_king", king=0.9), p("white_queen")]) == ["white_king", "white_queen"]
    # Random boards with no king or one, queens on them, and "king" scores
    # under the floor, on it, or none.
    for n in range(40):
        pieces = []
        for k in range(rng.choice([2, 4, 8, 16, 24])):
            side = rng.choice(["white", "black"])
            kind = "king" if k == 0 and n % 2 else rng.choice(["queen", "queen", "rook", "bishop", "knight", "pawn"])
            scores = scores_for(kind)
            if kind != "king" and rng.random() < 0.9:
                scores.pop("king", None)
                if rng.random() < 0.3:
                    scores["king"] = rng.choice([0.0, 0.004, 0.019])
            pieces.append(piece(side, kind, scores, k, rng.choice([None, None, 0.5, 0.75])))
        rng.shuffle(pieces)
        record(pieces)

    path = FIXTURES / "king_cases.json"
    path.write_text(json.dumps(cases, separators=(",", ":")) + "\n")
    print(f"{len(cases)} cases written to {path} ({path.stat().st_size // 1024} kB): {tally}")


def checker(spec):
    """The chequered picture of a dark-cases entry, as grey levels: 8x8 cells
    of `cell` pixels from `origin`, the cells whose column and row add up to
    an even number at `even`, the others at `odd`, the ones listed in
    `swapped` the other way round, `background` around them. The Swift test
    draws the same from the same numbers."""
    img = np.full((spec["height"], spec["width"]), spec["background"], np.uint8)
    ox, oy = spec["origin"]
    c = spec["cell"]
    swapped = {tuple(t) for t in spec["swapped"]}
    other = {(i, j): level for i, j, level in spec.get("levels", [])}
    for j in range(8):
        for i in range(8):
            even = ((i + j) % 2 == 0) != ((i, j) in swapped)
            x0, y0 = max(ox + i * c, 0), max(oy + j * c, 0)
            img[y0:max(oy + (j + 1) * c, 0), x0:max(ox + (i + 1) * c, 0)] = \
                other.get((i, j), spec["even"] if even else spec["odd"])
    return img


def turned_checker(spec):
    """A chequered board turned in the picture, so that the picture's edge
    cuts across its cells one at a time: the pixel (x, y) lies on the board
    at u = ((x - cx) cos + (y - cy) sin) / cell + 4,
    v = (-(x - cx) sin + (y - cy) cos) / cell + 4, in the cell (floor u,
    floor v) when both are within 0..8. The Swift test computes the same."""
    cx, cy = spec["centre"]
    c, s, cell = spec["cos"], spec["sin"], spec["cell"]
    yy, xx = np.mgrid[0:spec["height"], 0:spec["width"]].astype(np.float64)
    u = ((xx - cx) * c + (yy - cy) * s) / cell + 4
    v = (-(xx - cx) * s + (yy - cy) * c) / cell + 4
    inside = (u >= 0) & (u < 8) & (v >= 0) & (v < 8)
    even = (np.floor(u) + np.floor(v)) % 2 == 0
    img = np.where(even, spec["even"], spec["odd"])
    return np.where(inside, img, spec["background"]).astype(np.uint8)


def dark_cases():
    import cv2
    from grid_corners import dark_corner_rotations
    cv2.setNumThreads(1)

    def answer(image_bgr, ring):
        turns, strength = dark_corner_rotations(image_bgr, ring)
        return dict(ring=[[float(x), float(y)] for x, y in ring],
                    rotations=None if turns is None else list(turns), strength=float(strength))

    def turned(ring, k):
        return list(ring[k:]) + list(ring[:k])

    # 1. chequered pictures drawn from numbers: the thresholds, from both sides
    def spec(name, even=60, odd=200, width=400, height=400, origin=(40, 40), swapped=(), turn=0,
             background=128, levels=()):
        ox, oy = origin
        ring = [(ox, oy), (ox, oy + 320), (ox + 320, oy + 320), (ox + 320, oy)]
        s = dict(name=name, width=width, height=height, cell=40, origin=list(origin), even=even,
                 odd=odd, background=background, swapped=[list(t) for t in swapped],
                 levels=[list(t) for t in levels])
        grey = checker(s)
        s.update(answer(cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR), turned(ring, turn)))
        return s

    evens = [(i, j) for j in range(8) for i in range(8) if (i + j) % 2 == 0]
    odds = [(i, j) for j in range(8) for i in range(8) if (i + j) % 2 == 1]
    checkers = [
        spec("the first corner on a dark cell"),
        spec("the first corner on a light cell", even=200, odd=60),
        spec("a dark first corner, the ring a corner on", turn=1),
        spec("a dark first corner, the ring two corners on", turn=2),
        spec("a dark first corner, the ring three corners on", turn=3),
        spec("15 grey levels apart: enough", even=100, odd=115),
        spec("14 grey levels apart: too few", even=100, odd=114),
        spec("three columns off the picture: 40 cells seen, enough", width=240),
        spec("four columns off the picture: 32 cells seen, too few", width=200),
        spec("three rows and a bit off the picture, the ring two corners on", height=235, turn=2),
        spec("a quarter of each colour the other way: split 0.5, enough",
             swapped=evens[:8] + odds[:8]),
        spec("one cell more the other way: split under 0.5", swapped=evens[:9] + odds[:8]),
        spec("the same with the first corner light", even=200, odd=60, swapped=evens[:9] + odds[:8]),
        spec("a quarter the other way, the first corner light", even=200, odd=60,
             swapped=evens[:8] + odds[:8], turn=3),
        spec("nearly all cells one colour", swapped=odds[:27]),
        spec("one colour all over", even=90, odd=90),
        # Cells exactly at the middle between the two colours are not light:
        # with them light the split would be under 0.5.
        spec("nine cells exactly at the middle grey: not light", swapped=odds[:8],
             levels=[(i, j, 130) for i, j in evens[:9]]),
    ]
    want = {"15 grey levels apart: enough": [0, 2], "14 grey levels apart: too few": None,
            "three columns off the picture: 40 cells seen, enough": [0, 2],
            "four columns off the picture: 32 cells seen, too few": None,
            "a quarter of each colour the other way: split 0.5, enough": [0, 2],
            "one cell more the other way: split under 0.5": None,
            "a quarter the other way, the first corner light": [0, 2]}
    for c in checkers:
        if c["name"] in want:
            assert c["rotations"] == want[c["name"]], (c["name"], c["rotations"], c["strength"])
    assert next(c for c in checkers if c["name"].startswith("a quarter of each"))["strength"] == 0.5
    assert checkers[-1]["rotations"] == [0, 2] and checkers[-1]["strength"] == 0.75

    # 1b. the same board turned in the picture, the picture cut narrower one
    # pixel at a time: the edge then takes cells one by one, and the widest
    # picture with 39 cells seen and the narrowest with 40 are kept (cut along
    # the board, as above, cells go eight at a time)
    from grid_corners import _cell_values

    def turned_spec(width):
        s = dict(width=width, height=560, cell=40, centre=[280.0, 280.0], cos=0.96, sin=0.28,
                 even=60, odd=200, background=128)
        cx, cy = s["centre"]
        ring = [(cx + ((U - 4) * s["cos"] - (V - 4) * s["sin"]) * s["cell"],
                 cy + ((U - 4) * s["sin"] + (V - 4) * s["cos"]) * s["cell"])
                for U, V in ((0, 0), (0, 8), (8, 8), (8, 0))]
        grey = turned_checker(s)
        v = _cell_values(grey.astype(np.float32), homography(ring), 0, 0, 8, 8)
        s["seen"] = int((~np.isnan(v).all(axis=2)).sum())
        s.update(answer(cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR), ring))
        return s

    by_seen = {}
    for width in range(560, 200, -1):
        s = turned_spec(width)
        by_seen.setdefault(s["seen"], s)             # the widest picture for each count
        if s["seen"] < 39:
            break
    narrowest_40 = [turned_spec(w) for w in range(by_seen[39]["width"] + 1, by_seen[39]["width"] + 3)]
    turned_boards = [by_seen[64], next(s for s in narrowest_40 if s["seen"] == 40), by_seen[39]]
    for s in turned_boards:
        s["name"] = f"turned in the picture, {s['width']} wide: {s['seen']} cells seen"
    assert [s["seen"] for s in turned_boards] == [64, 40, 39]
    assert turned_boards[0]["rotations"] == [0, 2] and turned_boards[1]["rotations"] == [0, 2]
    assert turned_boards[2]["rotations"] is None and turned_boards[2]["strength"] == 0.0

    # 2. the drawn boards of grid-cases (PNG: the same pixels on both sides)
    recorded = json.loads((FIXTURES / "grid_cases.json").read_text())
    boards = []
    for name, seed, pieces, more in (("plain", 1, False, {}), ("pieces", 2, True, {}),
                                     ("cut", 4, True, dict(offset=(175, 0))),
                                     ("ten", 5, False, dict(cells=10)),
                                     ("busy", 6, True, dict(strays=True))):
        file = f"grid_board_{name}.png"
        _, corners = drawn_board(seed, pieces, True, **more)
        if recorded["boards"][name]["truth"] is not None:
            assert np.allclose(corners, recorded["boards"][name]["truth"])
        img = cv2.imread(str(FIXTURES / file))
        to_image = np.linalg.inv(homography(corners))
        rings = [(f"turned by {k}", turned(corners, k)) for k in range(4)]
        # a ring one square along (the colours swapped, one line of cells off
        # the board) and one half a square off (every cell half of each colour)
        for label, dx, dy in (("one square along", 1, 0), ("one square up", 0, 1),
                              ("half a square off", 0.5, 0.5)):
            rings.append((label, [project(to_image, (x + dx, y + dy))
                                  for x, y in ((0, 0), (0, 8), (8, 8), (8, 0))]))
        for label, ring in rings:
            boards.append(dict(file=file, label=label, **answer(img, ring)))
    img = cv2.imread(str(FIXTURES / "grid_board_clutter.png"))
    boards.append(dict(file="grid_board_clutter.png", label="no board",
                       **answer(img, [(150, 100), (150, 500), (650, 500), (650, 100)])))

    # 3. the fixture photos, on the corners the Python's grid fit gives them
    # (JPEG: the two decoders differ by a grey level here and there)
    photos = []
    for name, fit in sorted(recorded["photos"].items()):
        img = cv2.imread(str(FIXTURES / name))
        for k in range(4):
            photos.append(dict(file=name, label=f"turned by {k}", **answer(img, turned(fit["raw"], k))))

    # 4. a1 from the pieces among two rotations only
    cases = json.loads((FIXTURES / "orientation_cases.json").read_text())
    chosen = {"0,2": [], "1,3": []}
    for case in cases:
        given = [tuple(p) for p in case["corners"]]
        detections = [Detection(d["cls"], d["confidence"], tuple(d["box"])) for d in case["detections"]]
        for key, turns in (("0,2", (0, 2)), ("1,3", (1, 3))):
            oriented = orient_by_pieces(given, detections, case["pull"], case["prior"], turns=turns)
            k = next(k for k in range(4) if given[k:] + given[:k] == oriented)
            assert k in turns and case["scores"][k] == max(case["scores"][t] for t in turns)
            chosen[key].append(k)

    out = dict(checkers=checkers, turned=turned_boards, boards=boards, photos=photos,
               orientation=chosen)
    path = FIXTURES / "dark_cases.json"
    path.write_text(json.dumps(out, separators=(",", ":")) + "\n")
    for group in ("checkers", "turned", "boards", "photos"):
        for c in out[group]:
            print(f"{c.get('name') or c['file'] + ', ' + c['label']}: {c['rotations']} "
                  f"strength {c['strength']:.4f}")
    free = [c["turn"] for c in cases]
    print(f"orientation, {len(cases)} cases: the free choice was allowed in "
          f"{sum(a == f for a, f in zip(chosen['0,2'], free))} with (0, 2), "
          f"{sum(a == f for a, f in zip(chosen['1,3'], free))} with (1, 3)")
    print(f"written to {path} ({path.stat().st_size // 1024} kB)")


def compare_dark(swift_path, list_path):
    """a1 is a dark square, of the same photos and corners in Python and in Swift."""
    import cv2
    from grid_corners import dark_corner_rotations
    cv2.setNumThreads(1)
    sw = json.loads(Path(swift_path).read_text())
    base = Path(list_path).resolve().parent if list_path else Path.cwd()
    photos = same = unclear_py = unclear_sw = moved_py = moved_sw = 0
    off = []
    for photo in sorted(sw):
        found = sw[photo]
        if found is None:
            continue                              # no board in the app
        src = Path(photo) if Path(photo).is_absolute() else base / photo
        turns, strength = dark_corner_rotations(cv2.imread(str(src)), found["corners"])
        mine = None if turns is None else list(turns)
        photos += 1
        same += mine == found["rotations"]
        unclear_py += mine is None
        unclear_sw += found["rotations"] is None
        moved_py += mine is not None and 0 not in mine
        moved_sw += found["rotations"] is not None and 0 not in found["rotations"]
        off.append(abs(strength - found["strength"]))
        if mine != found["rotations"]:
            print(f"  {Path(photo).name}: Python {mine} strength {strength:.3f}, "
                  f"Swift {found['rotations']} strength {found['strength']:.3f}")
    off = np.array(off) if off else np.zeros(1)
    print(f"{photos} photos with a board in the app: the same answer on {same}; "
          f"colours unclear: Python {unclear_py}, Swift {unclear_sw}; "
          f"a light a1 as the corners came: Python {moved_py}, Swift {moved_sw}")
    print(f"strength, Swift against Python: median {np.median(off):.4f}, "
          f"p95 {np.percentile(off, 95):.4f}, max {off.max():.4f}")


FRAME = ((0.0, 0.0), (0.0, 8.0), (8.0, 8.0), (8.0, 0.0))

# Fits: a board seen from above, one seen at a slant, one seen from low
# down, and one whose corners start elsewhere and run the other way round.
WINDOW_RINGS = {
    "above": [(200.0, 700.0), (200.0, 100.0), (800.0, 100.0), (800.0, 700.0)],
    "slant": [(150.0, 820.0), (260.0, 240.0), (790.0, 215.0), (930.0, 790.0)],
    "low": [(90.0, 640.0), (330.0, 410.0), (720.0, 405.0), (955.0, 630.0)],
    "other way": [(790.0, 215.0), (260.0, 240.0), (150.0, 820.0), (930.0, 790.0)],
}


def moved_window(ring, files, ranks):
    """The window `files` and `ranks` squares on, as evaluate.py --shift-fit moves it."""
    back = np.linalg.inv(homography(ring))
    return [list(map(float, project(back, (x + files, y + ranks)))) for x, y in FRAME]


def window_cases(seed):
    """chess_pipeline.window_by_model on fits and models built from a few numbers."""
    def model_of(ring, shift, noise):
        # The model's corners: the fit's window moved by `shift`, each corner
        # then `noise` squares off, all in the fit's board frame.
        back = np.linalg.inv(homography(ring))
        return [list(map(float, project(back, (x + shift[0] + ex, y + shift[1] + ey))))
                for (x, y), (ex, ey) in zip(FRAME, noise)]

    still = [(0.0, 0.0)] * 4
    eight = [(f, r) for f in (-1, 0, 1) for r in (-1, 0, 1) if (f, r) != (0, 0)]
    made = [(f"a square on {shift}", shift, [(0.2, -0.1), (-0.15, 0.25), (0.1, 0.1), (-0.2, -0.2)], WINDOW_NEAR)
            for shift in eight]
    made += [
        ("the fit's own corners", (0, 0), still, WINDOW_NEAR),
        ("near the fit's", (0, 0), [(0.3, -0.2), (-0.25, 0.3), (0.2, 0.1), (-0.3, -0.3)], WINDOW_NEAR),
        # The mean decides the shift: just under and just over half a square.
        ("mean just under half a file", (0.4999, 0), still, WINDOW_NEAR),
        ("mean just over half a file", (0.5001, 0), still, WINDOW_NEAR),
        ("mean just under half a rank back", (0, -0.4999), still, WINDOW_NEAR),
        ("mean just over half a rank back", (0, -0.5001), still, WINDOW_NEAR),
        # More than one square: never.
        ("two ranks on", (0, 2), still, WINDOW_NEAR),
        ("two files back", (-2, 0), still, WINDOW_NEAR),
        ("a file on and two ranks on", (1, 2), still, WINDOW_NEAR),
        ("just under a rank and a half", (0, 1.4999), still, WINDOW_NEAR),
        ("just over a rank and a half", (0, 1.5001), still, WINDOW_NEAR),
        # One corner just inside and just outside the tolerance, on each axis
        # and to each side (the other three pull the mean back a little).
        ("a corner just inside, files", (0, 1), [(0.7499, 0)] + still[:3], WINDOW_NEAR),
        ("a corner just outside, files", (0, 1), [(0.7501, 0)] + still[:3], WINDOW_NEAR),
        ("a corner just inside, files back", (0, -1), still[:2] + [(-0.7499, 0), (0, 0)], WINDOW_NEAR),
        ("a corner just outside, files back", (0, -1), still[:2] + [(-0.7501, 0), (0, 0)], WINDOW_NEAR),
        ("a corner just inside, ranks", (1, 0), still[:3] + [(0, -0.7499)], WINDOW_NEAR),
        ("a corner just outside, ranks", (1, 0), still[:3] + [(0, -0.7501)], WINDOW_NEAR),
        ("a corner just inside, ranks up", (-1, 0), [(0, 0.7499)] + still[:3], WINDOW_NEAR),
        ("a corner just outside, ranks up", (-1, 0), [(0, 0.7501)] + still[:3], WINDOW_NEAR),
        # Three corners a rank on and one where the fit has it.
        ("one corner stays", (0, 1), [(0, 0), (0, 0), (0, -1), (0, 0)], WINDOW_NEAR),
        # Other tolerances.
        ("0.6 off at 0.5", (0, 1), [(0.6, 0), (-0.6, 0), (0, 0), (0, 0)], 0.5),
        ("0.6 off at 0.75", (0, 1), [(0.6, 0), (-0.6, 0), (0, 0), (0, 0)], WINDOW_NEAR),
        ("0.9 off at 0.75", (0, 1), [(0.9, 0), (-0.9, 0), (0, 0), (0, 0)], WINDOW_NEAR),
        ("0.9 off at 0.95", (0, 1), [(0.9, 0), (-0.9, 0), (0, 0), (0, 0)], 0.95),
    ]
    cases = []
    for ring_name, ring in WINDOW_RINGS.items():
        for name, shift, noise, near in made:
            cases.append((f"{ring_name}: {name}", ring, model_of(ring, shift, noise), near))
        # The model's corners in another rotation than the fit's: the caller
        # is to put them in the same order, and nothing is moved if not.
        for turn in (1, 2, 3):
            model = model_of(ring, (0, 1), still)
            cases.append((f"{ring_name}: a rank on, {turn} corners round", ring, model[turn:] + model[:turn],
                          WINDOW_NEAR))
    rng = random.Random(seed)
    for k in range(80):
        ring_name = rng.choice(list(WINDOW_RINGS))
        ring = [(x + rng.uniform(-30, 30), y + rng.uniform(-30, 30)) for x, y in WINDOW_RINGS[ring_name]]
        shift = rng.choice(eight + [(0, 0), (0, 0), (0, 2), (-2, 1)])
        spread = rng.choice([0.1, 0.3, 0.6, 0.74, 0.9, 1.2])
        noise = [(rng.uniform(-spread, spread), rng.uniform(-spread, spread)) for _ in range(4)]
        cases.append((f"random {k}: {ring_name} {shift} spread {spread}", ring, model_of(ring, shift, noise),
                      rng.choice([WINDOW_NEAR, WINDOW_NEAR, 0.5, 0.95])))

    out, moved_by = [], {}
    for name, ring, model, near in cases:
        # No case may hang on the last bits of the arithmetic.
        H = homography(ring)
        offsets = np.array([np.subtract(project(H, c), at) for c, at in zip(model, FRAME)])
        mean = offsets.mean(axis=0)
        assert np.abs(np.abs(mean - np.floor(mean)) - 0.5).min() > 2e-5, name
        shift = np.round(mean)
        if shift.any() and np.abs(shift).max() <= 1:
            assert np.abs(np.abs(offsets - shift) - near).min() > 2e-5, name
        corners, made_shift = window_by_model(ring, model, near=near)
        moved_by[made_shift] = moved_by.get(made_shift, 0) + 1
        out.append({"name": name, "fit": [list(map(float, c)) for c in ring], "model": model, "near": near,
                    "corners": [list(c) for c in corners], "shift": list(made_shift)})
    shifts = [{"ring": name, "files": f, "ranks": r, "corners": moved_window(ring, f, r)}
              for name, ring in WINDOW_RINGS.items() for f, r in eight + [(0, 0), (0, 2), (-3, 1)]]
    path = FIXTURES / "window_cases.json"
    path.write_text(json.dumps({"near": WINDOW_NEAR, "cases": out, "shifts": shifts},
                               separators=(",", ":")) + "\n")
    print(f"{len(out)} cases: left alone {moved_by.get((0, 0), 0)}, moved "
          f"{ {k: v for k, v in sorted(moved_by.items()) if k != (0, 0)} }; {len(shifts)} windows moved on purpose")
    print(f"written to {path} ({path.stat().st_size // 1024} kB)")


def compare_window(swift_path):
    """The fit's window, of the same corners in Python and in Swift."""
    sw = json.loads(Path(swift_path).read_text())
    purpose = tuple(sw["shift_fit"])
    photos = same = moved_py = moved_sw = back_py = back_sw = 0
    off, off_given = [], []
    for photo in sorted(sw["photos"]):
        found = sw["photos"][photo]
        if found is None:
            continue                              # the model or the fit found no board
        given = moved_window(found["fit"], *purpose) if any(purpose) else found["fit"]
        corners, shift = window_by_model(given, found["model"])
        photos += 1
        off_given.append(np.abs(np.subtract(given, found["given"])).max())
        off.append(np.abs(np.subtract(corners, found["corners"])).max())
        same += list(shift) == found["shift"]
        moved_py += shift != (0, 0)
        moved_sw += found["shift"] != [0, 0]
        back_py += any(purpose) and shift == (-purpose[0], -purpose[1])
        back_sw += any(purpose) and found["shift"] == [-purpose[0], -purpose[1]]
        if list(shift) != found["shift"]:
            print(f"  {Path(photo).name}: Python {shift}, Swift {found['shift']}")
        elif shift != (0, 0) and not any(purpose):
            print(f"  {Path(photo).name}: moved by {shift}")
    off = np.array(off) if off else np.zeros(1)
    off_given = np.array(off_given) if off_given else np.zeros(1)
    print(f"{len(sw['photos'])} photos, the model and the fit both found the board on {photos}"
          + (f", the fit's window moved by {purpose} on purpose" if any(purpose) else "")
          + f": the same shift on {same}; windows moved: Python {moved_py}, Swift {moved_sw}"
          + (f"; moved back: Python {back_py}, Swift {back_sw}" if any(purpose) else ""))
    print(f"corners, Swift against Python, in pixels: max {off.max():.2e}"
          + (f"; the window moved on purpose: max {off_given.max():.2e}" if any(purpose) else ""))


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


def compare_colours(swift_path, list_path):
    """The colour step of the same photos and boxes in Python and in Swift."""
    from evaluate import piece_colours
    sw = json.loads(Path(swift_path).read_text())
    base = Path(list_path).resolve().parent if list_path else Path.cwd()
    photos = pieces = no_colour_differs = side_differs = final_differs = 0
    fired_py = fired_sw = photos_side_differs = photos_final_differs = split_differs = 0
    # The side of single pieces: from the Python's colours, and from the Swift's.
    odd = dict(py=0, sw=0, photos_py=0, photos_sw=0, differ=0, photos_differ=0, differ_on_swift_colours=0,
               hidden_differ=0, hidden=0)
    has_odd = True
    off = []
    for photo in sorted(sw):
        found = sw[photo]
        src = Path(photo) if Path(photo).is_absolute() else base / photo
        detections = [Detection(d["cls"], d["confidence"], tuple(d["box"])) for d in found]
        cols = piece_colours(src, detections)
        final = [d.cls for d in sides_from_colour(detections, cols)]
        split = colour_sides(cols)
        by_colour = split or [None] * len(detections)
        photos += 1
        pieces += len(detections)
        sides_here = finals_here = 0
        split_differs += (split is None) != all(d["colour_side"] is None for d in found)
        for d, c, side, cls in zip(found, cols, by_colour, final):
            if (c is None) != (d["colour"] is None):
                no_colour_differs += 1
            elif c is not None:
                off.append(max(abs(a - b) for a, b in zip(c, d["colour"])))
            sides_here += side != d["colour_side"]
            finals_here += cls != d["final"]
        side_differs += sides_here
        final_differs += finals_here
        photos_side_differs += sides_here > 0
        photos_final_differs += finals_here > 0
        fired_py += any(cls != d["cls"] for cls, d in zip(final, found))
        fired_sw += any(d["final"] != d["cls"] for d in found)
        if found and "odd" not in found[0]:
            has_odd = False                       # a file written before app_check knew the step
        elif found:
            mine = [d.cls for d in odd_sides(detections, cols)]
            on_theirs = [d.cls for d in odd_sides(
                detections, [None if d["colour"] is None else tuple(d["colour"]) for d in found])]
            theirs = [d["odd"] for d in found]
            hidden = [patch_hidden(detections, k) for k in range(len(detections))]
            n_py = sum(a != d.cls for a, d in zip(mine, detections))
            n_sw = sum(a != d.cls for a, d in zip(theirs, detections))
            differ = sum(a != b for a, b in zip(mine, theirs))
            odd["py"] += n_py
            odd["sw"] += n_sw
            odd["photos_py"] += n_py > 0
            odd["photos_sw"] += n_sw > 0
            odd["differ"] += differ
            odd["photos_differ"] += differ > 0
            odd["differ_on_swift_colours"] += sum(a != b for a, b in zip(on_theirs, theirs))
            odd["hidden"] += sum(hidden)
            odd["hidden_differ"] += sum(a != d["hidden"] for a, d in zip(hidden, found))
            if differ:
                print(f"  {Path(photo).name}: the side of single pieces differs for {differ} of "
                      f"{len(detections)} pieces (Python changed {n_py}, Swift {n_sw})")
        if sides_here or finals_here:
            print(f"  {Path(photo).name}: {len(detections)} pieces, side by colour differs for "
                  f"{sides_here}, final class for {finals_here}")
    off = np.array(off) if off else np.zeros(1)
    print(f"{photos} photos, {pieces} pieces found by the app's piece model")
    print(f"colours, Swift against Python, largest of L*, a*, b*: median {np.median(off):.3f}, "
          f"p95 {np.percentile(off, 95):.3f}, max {off.max():.3f}; "
          f"a colour in one and none in the other: {no_colour_differs}")
    print(f"side by colour alone differs: {side_differs} pieces on {photos_side_differs} photos "
          f"(split in one and none in the other: {split_differs} photos)")
    print(f"the step changed a side: Python on {fired_py} photos, Swift on {fired_sw}; "
          f"final classes differ: {final_differs} pieces on {photos_final_differs} photos")
    if has_odd:
        print(f"the side of single pieces: Python made {odd['py']} pieces White on {odd['photos_py']} photos, "
              f"Swift {odd['sw']} on {odd['photos_sw']}; classes differ: {odd['differ']} pieces on "
              f"{odd['photos_differ']} photos (from the Swift's own colours: {odd['differ_on_swift_colours']}); "
              f"pieces behind another: {odd['hidden']}, told differently: {odd['hidden_differ']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=["orientation", "grid-cases", "colour-cases", "king-cases", "dark-cases",
                                     "window-cases", "odd-cases", "grid", "compare", "colours", "dark", "window"])
    ap.add_argument("--list", help="grid: JSON object keyed by photo path")
    ap.add_argument("--out", help="grid: where to write the reference")
    ap.add_argument("--limit", type=int, default=0, help="grid: only the first N photos")
    ap.add_argument("--python", help="compare: file written by `grid`")
    ap.add_argument("--swift", help="compare: file written by `app_check --grid`; "
                                    "colours: file written by `app_check --colours`; "
                                    "dark: file written by `app_check --dark`; "
                                    "window: file written by `app_check --window`")
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
    elif args.what == "colour-cases":
        colour_cases(args.seed)
    elif args.what == "king-cases":
        king_cases(args.seed)
    elif args.what == "dark-cases":
        dark_cases()
    elif args.what == "window-cases":
        window_cases(args.seed)
    elif args.what == "odd-cases":
        odd_cases(args.seed)
    elif args.what == "dark":
        compare_dark(args.swift, args.list)
    elif args.what == "window":
        compare_window(args.swift)
    elif args.what == "grid":
        grid_reference(args.list, args.out, args.limit)
    elif args.what == "colours":
        compare_colours(args.swift, args.list)
    else:
        compare(args.python, args.swift)


if __name__ == "__main__":
    main()
