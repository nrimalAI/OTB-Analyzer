"""Geometry shared by evaluation and (as reference) the Swift app.

The recognition pipeline:
  1. corner model -> 4 board corners in image coordinates (a1, a8, h8, h1)
  2. homography H mapping image points onto the unit board [0,8]x[0,8]
     (file axis = a->h, rank axis = 1->8)
  3. each detected piece's bounding-box *bottom-center* -> H -> (file, rank)
  4. conflict resolution: one piece per square, highest confidence wins;
     the loser is demoted to its next-nearest square if empty, else dropped.

Optional steps around it:
  - sides_from_colour (evaluate.py --sides-from-colour): White and Black from
    the colours of the pieces in the photo, where the model's own answer does
    not fit them (before step 3). OFF by default everywhere: it was fitted
    to one board and loses on boards nobody tuned on (see its docstring);
  - orient_by_pieces (--a1-from-pieces): which corner is a1, from where the
    pieces stand;
  - board_crop (--crop-board): the piece model looks at the board only.

The Swift port in ChessVision (Phase 2) must match this file's behavior;
evaluate.py is the executable spec.
"""

from dataclasses import dataclass

import numpy as np

PIECE_CLASSES = [
    "white_king", "white_queen", "white_rook", "white_bishop", "white_knight", "white_pawn",
    "black_king", "black_queen", "black_rook", "black_bishop", "black_knight", "black_pawn",
]

FEN_LETTER = {
    "white_king": "K", "white_queen": "Q", "white_rook": "R",
    "white_bishop": "B", "white_knight": "N", "white_pawn": "P",
    "black_king": "k", "black_queen": "q", "black_rook": "r",
    "black_bishop": "b", "black_knight": "n", "black_pawn": "p",
}


@dataclass
class Detection:
    cls: str            # one of PIECE_CLASSES
    confidence: float
    bbox: tuple         # (x1, y1, x2, y2) image coords

    @property
    def anchor(self):
        """Bottom-center of the box — the piece's contact point with the board."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, y2)


def homography(corners):
    """H mapping image points -> board coords, given corners as a dict or
    4x2 array in (a1, a8, h8, h1) order.

    Board frame: x = file in [0,8] (a=0..h=8), y = rank in [0,8] (1=0..8=8).
    a1 -> (0,0), a8 -> (0,8), h8 -> (8,8), h1 -> (8,0).
    """
    src = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    dst = np.array([[0, 0], [0, 8], [8, 8], [8, 0]], dtype=np.float64)

    # Direct linear transform (4-point).
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    A = np.asarray(A)
    _, _, vt = np.linalg.svd(A)
    H = vt[-1].reshape(3, 3)
    return H / H[2, 2]


def project(H, point):
    x, y = point
    p = H @ np.array([x, y, 1.0])
    return p[0] / p[2], p[1] / p[2]


def pulled_board_point(H, point, pull=0.30):
    """Project an anchor into board coords with camera-aware correction.

    A piece's bbox bottom-center is the *front edge* of its base — displaced
    from the base center toward the camera. In image space that displacement
    is straight down, so the correction is a step along image-"up" (-y),
    mapped through the homography and applied in board coordinates. This is
    correct regardless of which side of the board the camera is on.
    """
    x, y = point
    bx, by = project(H, (x, y))
    ux, uy = project(H, (x, y - 1.0))  # one pixel up in image space
    dx, dy = ux - bx, uy - by
    norm = np.hypot(dx, dy)
    if norm > 1e-9:
        bx += pull * dx / norm
        by += pull * dy / norm
    return bx, by


# How far outside the board, in squares, a piece's (pulled) foot may be and
# still count as standing on the nearest rim square. It was 0.5 until
# 2026-11-13; captured pieces standing beside the board then landed on the
# rim (on the Y-LIChess photos nearly every piece seen on an empty square).
# On ChessReD test, Fenit A and Fenit B 0.25 gives the same boards as 0.5
# with labelled corners, and one square of 26,000 fewer end to end; at 0.1
# real rim pieces start to drop out. Chosen on those three sets.
EDGE_MARGIN = 0.25


def board_square(H, point, pull=0.30):
    """Map an image point to a 0-indexed (file, rank) square, or None."""
    bx, by = pulled_board_point(H, point, pull)
    f, r = int(np.floor(bx)), int(np.floor(by))
    if 0 <= f < 8 and 0 <= r < 8:
        return f, r
    # A piece on a rim square can come out just over the edge (corners a
    # little off, a piece leaning out): those are taken back in. Not further
    # than EDGE_MARGIN, or pieces standing beside the board land on the rim.
    m = EDGE_MARGIN
    if -m <= bx < 8 + m and -m <= by < 8 + m:
        return min(max(f, 0), 7), min(max(r, 0), 7)
    return None


def assign_squares(H, detections, pull=0.30):
    """Detections -> {(file, rank): (cls, confidence)} with conflict resolution."""
    return {sq: (det.cls, det.confidence)
            for sq, det in assign_detections(H, detections, pull).items()}


def assign_detections(H, detections, pull=0.30):
    """Like assign_squares, but keeps the winning Detection for each square."""
    board = {}
    for det in sorted(detections, key=lambda d: -d.confidence):
        sq = board_square(H, det.anchor, pull)
        if sq is None:
            continue
        if sq not in board:
            board[sq] = det
            continue
        # Occupied by a higher-confidence piece: demote to nearest free neighbor.
        bx, by = pulled_board_point(H, det.anchor, pull)
        candidates = []
        for df in (-1, 0, 1):
            for dr in (-1, 0, 1):
                if df == 0 and dr == 0:
                    continue
                nf, nr = sq[0] + df, sq[1] + dr
                if 0 <= nf < 8 and 0 <= nr < 8 and (nf, nr) not in board:
                    dist = (bx - (nf + 0.5)) ** 2 + (by - (nr + 0.5)) ** 2
                    candidates.append((dist, (nf, nr)))
        if candidates:
            candidates.sort()
            board[candidates[0][1]] = det
        # else: dropped
    return board


# --------------------------------------------------------------------------
# The part of the photo the piece model looks at

# Margins around the board's corners, as fractions of the width and height of
# the box round the four corners (left and right, top, bottom). Pieces stand
# up from the board, so the far side needs room for the back rank's heads.
BOARD_CROP_MARGINS = (0.12, 0.35, 0.08)


def board_crop(corners, width, height, margins=BOARD_CROP_MARGINS):
    """Pixel rectangle (x0, y0, x1, y1), end exclusive, round the board with
    room for the pieces, clipped to the photo. The piece model then sees the
    board at 640 px instead of the whole photo, so a piece is larger by the
    share of the photo that is not board; None if nothing is left."""
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    w, h = max(xs) - min(xs), max(ys) - min(ys)
    side, top, bottom = margins
    x0 = max(int(min(xs) - side * w), 0)
    x1 = min(int(max(xs) + side * w) + 1, width)
    y0 = max(int(min(ys) - top * h), 0)
    y1 = min(int(max(ys) + bottom * h) + 1, height)
    return (x0, y0, x1, y1) if x1 - x0 >= 8 and y1 - y0 >= 8 else None


# --------------------------------------------------------------------------
# Which side a piece belongs to, from the colours of the pieces in the photo

# Where in a piece's box its colour is measured: the middle, as fractions of
# the box (left, right, top, bottom). The body of the piece, not the square
# behind its head or the board in front of its foot.
COLOUR_PATCH = (0.3, 0.7, 0.2, 0.85)
# How much more a* and b* (the colour) count than L* (the lightness) when the
# pieces are split in two: a brown piece and a black one under a lamp can be
# equally light, but only one of them is brown.
CHROMA_WEIGHT = 4.0
# The model's own answer stands unless the colours contradict it for more
# than this share of the pieces.
SIDE_DISAGREEMENT = 0.15


def colour_patch(bbox, width, height):
    """Pixel rectangle (x0, y0, x1, y1), end exclusive, of the part of a
    piece's box whose colour is measured; None when it is empty."""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    l, r, t, b = COLOUR_PATCH
    xa, xb = int(x1 + l * w), int(x1 + r * w) + 1
    ya, yb = int(y1 + t * h), int(y1 + b * h) + 1
    xa, ya, xb, yb = max(xa, 0), max(ya, 0), min(xb, width), min(yb, height)
    return (xa, ya, xb, yb) if xb > xa and yb > ya else None


def two_groups(points):
    """Split points (rows of numbers) in two by 2-means, started from the two
    points farthest apart (the first such pair). Returns a list of 0/1, the
    group of each point, and the two group means; None when there are fewer
    than two points or they are all the same."""
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n < 2:
        return None
    d = ((pts[:, None] - pts[None]) ** 2).sum(axis=2)
    i, j = divmod(int(np.argmax(d)), n)
    if d[i, j] <= 0:
        return None
    centres = np.stack([pts[i], pts[j]])
    group = None
    for _ in range(50):
        dist = ((pts[:, None] - centres[None]) ** 2).sum(axis=2)
        new = [0 if dist[k, 0] <= dist[k, 1] else 1 for k in range(n)]
        if new == group:
            break
        group = new
        for g in (0, 1):
            members = [pts[k] for k in range(n) if group[k] == g]
            if members:
                centres[g] = np.mean(members, axis=0)
    return group, centres


def colour_sides(colours, chroma=CHROMA_WEIGHT):
    """The side of each piece by colour alone: "white" or "black" for every
    piece that has a colour (None for one that has none), or None when the
    pieces cannot be split in two. `colours` holds the median (L*, a*, b*) of
    each piece's colour_patch (CIE Lab, L* 0-100), or None for an empty
    patch. The pieces are split in two groups by two_groups on
    (L*, chroma a*, chroma b*); the lighter group is White."""
    idx = [k for k, c in enumerate(colours) if c is not None]
    if len(idx) < 2:
        return None
    split = two_groups([(colours[k][0], chroma * colours[k][1], chroma * colours[k][2])
                        for k in idx])
    if split is None:
        return None
    group, centres = split
    if len(set(group)) < 2:
        return None
    white = 0 if centres[0][0] >= centres[1][0] else 1
    sides = [None] * len(colours)
    for k, g in zip(idx, group):
        sides[k] = "white" if g == white else "black"
    return sides


def sides_from_colour(detections, colours, chroma=CHROMA_WEIGHT,
                      disagreement=SIDE_DISAGREEMENT):
    """The detections with White and Black decided by the pieces' colours
    (colour_sides) where the model's own answer does not fit them.

    A piece set the model has not seen (brown "white" pieces, say) is still
    two groups of one colour each, which the model's class scores alone do
    not use. But the split can be wrong for a piece or two (a shadow, a
    glint), so on boards the model reads well it would cost more than it
    gains. It is therefore used only when it contradicts the model for more
    than `disagreement` of the pieces that have a colour: then every such
    piece takes its group's side. A piece keeps its kind and confidence
    either way.

    NOT TO BE SWITCHED ON AS IT IS. The colour weight and the 15% were
    chosen on Fenit board B (brown "white" pieces), where the step turns 1
    fully right board into 17 of 98. On the second test set, which nobody
    tunes on, it gains nothing and the Y-LIChess game positions fall from
    75% to 52% of squares (Training/README.md, "Second held-out test set").
    By colour alone every piece of a photo gets the right side on only 85%
    of the training photos, so this split cannot overrule a model. There is
    also no smallest distance between the two groups: four white pieces and
    a model that is right come out two and two. Kept because the app has a
    port of it (off by default) and a better rule would start from here."""
    sides = colour_sides(colours, chroma)
    if sides is None:
        return list(detections)
    idx = [k for k, s_ in enumerate(sides) if s_ is not None]
    differ = sum(1 for k in idx if not detections[k].cls.startswith(sides[k]))
    if differ <= disagreement * len(idx):
        return list(detections)
    out = list(detections)
    for k in idx:
        det = detections[k]
        out[k] = Detection(sides[k] + "_" + det.cls.split("_", 1)[1], det.confidence, det.bbox)
    return out


def white_side_score(board):
    """How much a board {(file, rank): (cls, conf)} looks like one seen with
    White at ranks 1-2: White's pieces count for being near rank 1, Black's
    for being near rank 8, pawns three times and kings one and a half times
    as much as the rest, and a pawn on the first or last rank (impossible)
    costs 20. Positive when the orientation is plausible, negative when the
    board is upside down, near zero when it is turned by 90 degrees."""
    score = 0.0
    for (_, rank), (cls, _) in board.items():
        letter = FEN_LETTER[cls]
        weight = {"p": 3.0, "k": 1.5}.get(letter.lower(), 1.0)
        score += weight * ((3.5 - rank) if letter.isupper() else (rank - 3.5))
        if letter.lower() == "p" and rank in (0, 7):
            score -= 20.0
    return score


def orient_by_pieces(corners, detections, pull=0.30, prior=0.0):
    """The corners (a1, a8, h8, h1) turned so that the pieces say which one is
    a1: of the four rotations, the one whose board has the highest
    white_side_score. `prior` is added to the score of the corners as given,
    i.e. how much to trust the corner model's own choice (0: not at all)."""
    corners = [tuple(c) for c in corners]
    best, best_score = corners, None
    for turn in range(4):
        turned = corners[turn:] + corners[:turn]
        score = white_side_score(assign_squares(homography(turned), detections, pull))
        if turn == 0:
            score += prior
        if best_score is None or score > best_score:
            best, best_score = turned, score
    return best


def board_to_fen_placement(board):
    """{(file, rank): (cls, conf)} -> FEN placement field."""
    rows = []
    for rank in range(7, -1, -1):
        row, empty = "", 0
        for file in range(8):
            entry = board.get((file, rank))
            if entry is None:
                empty += 1
            else:
                if empty:
                    row += str(empty)
                    empty = 0
                row += FEN_LETTER[entry[0]]
        if empty:
            row += str(empty)
        rows.append(row)
    return "/".join(rows)


def fen_placement_to_board(placement):
    """FEN placement -> {(file, rank): fen_letter}."""
    board = {}
    for row_index, row in enumerate(placement.split("/")):
        rank = 7 - row_index
        file = 0
        for ch in row:
            if ch.isdigit():
                file += int(ch)
            else:
                board[(file, rank)] = ch
                file += 1
    return board
