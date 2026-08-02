"""Geometry shared by evaluation and (as reference) the Swift app.

The recognition pipeline:
  1. corner model -> 4 board corners in image coordinates (a1, a8, h8, h1)
  2. homography H mapping image points onto the unit board [0,8]x[0,8]
     (file axis = a->h, rank axis = 1->8)
  3. each detected piece's bounding-box *bottom-center* -> H -> (file, rank)
  4. conflict resolution: one piece per square, highest confidence wins;
     the loser is demoted to its next-nearest square if empty, else dropped.

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


def board_square(H, point, pull=0.30):
    """Map an image point to a 0-indexed (file, rank) square, or None.

    `pull` shifts the anchor toward the board center along the rank axis
    before binning: a piece's contact point sits ON its square, but tall
    pieces photographed at an angle have their box bottom slightly in front
    of the square center; pulling by a fraction of a square compensates.
    """
    bx, by = project(H, point)
    by = by + pull  # toward higher ranks (away from the camera at low ranks)
    f, r = int(np.floor(bx)), int(np.floor(by))
    if 0 <= f < 8 and 0 <= r < 8:
        return f, r
    # Clamp near-miss anchors (piece leaning over the board edge).
    if -0.5 <= bx < 8.5 and -0.5 <= by < 8.5:
        return min(max(f, 0), 7), min(max(r, 0), 7)
    return None


def assign_squares(H, detections, pull=0.30):
    """Detections -> {(file, rank): (cls, confidence)} with conflict resolution."""
    board = {}
    for det in sorted(detections, key=lambda d: -d.confidence):
        sq = board_square(H, det.anchor, pull)
        if sq is None:
            continue
        if sq not in board:
            board[sq] = (det.cls, det.confidence)
            continue
        # Occupied by a higher-confidence piece: demote to nearest free neighbor.
        bx, by = project(H, det.anchor)
        by += pull
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
            board[candidates[0][1]] = (det.cls, det.confidence)
        # else: dropped
    return board


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
