"""Make synthetic photos of chess boards with exact corner labels.

    python synth_boards.py sprites            # cut piece sprites out of chesscog + Fenit
    python synth_boards.py make --count 3000  # -> datasets/synth/corners/{images,labels}
    python synth_boards.py make --count 12 --preview   # draw the labels on them

Why: the corner model has only three boards to learn from (ChessReD, the
chesscog render, Fenit board A), and on a fourth board its corners are more
than a square off. Corner labels are scarce in public data, but a board is a
flat pattern: draw one in any colours, look at it through a random camera, and
the corners are known exactly.

Each image is a ground plane (wood, tiles, cloth, plain) seen by a random
pinhole camera, a board lying on it (random square colours, border, frame,
coordinates, a1 always a dark square), and pieces from a real game position
standing on their squares. Pieces are flat cut-outs recoloured for each side,
scaled by the camera: they are there to hide parts of the board and to say
which side is White, as in real photos, not to train the piece model. Then
lighting, glare, blur, noise and JPEG artefacts.

The output is in the `native` layout build_dataset.py merges (source `synth`
in sources.yaml), corners only.
"""

import argparse
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np

from chess_pipeline import FEN_LETTER, PIECE_CLASSES, fen_placement_to_board

HERE = Path(__file__).resolve().parent
SPRITES = HERE / "datasets" / "synth" / "sprites"
KIND_OF = {FEN_LETTER[c].lower(): c.split("_")[1] for c in PIECE_CLASSES}
# Piece height in squares, for scaling a cut-out to the camera.
HEIGHT = {"pawn": 0.95, "rook": 1.1, "knight": 1.25, "bishop": 1.45, "queen": 1.65, "king": 1.85}


# --------------------------------------------------------------------------
# Sprites

def cut_out(img, box):
    """An isolated piece as (crop, alpha) via GrabCut seeded with its box, or
    None when the result does not look like one object."""
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    w, h = x2 - x1, y2 - y1
    if w < 24 or h < 40:
        return None
    pad = int(0.25 * max(w, h))
    X1, Y1 = max(x1 - pad, 0), max(y1 - pad, 0)
    X2, Y2 = min(x2 + pad, img.shape[1]), min(y2 + pad, img.shape[0])
    crop = img[Y1:Y2, X1:X2]
    mask = np.zeros(crop.shape[:2], np.uint8)
    rect = (x1 - X1, y1 - Y1, w, h)
    try:
        cv2.grabCut(crop, mask, rect, np.zeros((1, 65)), np.zeros((1, 65)), 4,
                    cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    fg = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg)
    if n < 2:
        return None
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    fg = (labels == biggest).astype(np.uint8)
    piece = fg[y1 - Y1:y2 - Y1, x1 - X1:x2 - X1]
    share = piece.mean()
    # A piece fills roughly a third to three quarters of its box and touches
    # the box's top and bottom.
    if not 0.25 < share < 0.85 or piece[:4].sum() == 0 or piece[-4:].sum() == 0:
        return None
    alpha = cv2.GaussianBlur(piece.astype(np.float32), (0, 0), 1.0)
    return img[y1:y2, x1:x2].copy(), (alpha * 255).astype(np.uint8)


def isolated(boxes, i, margin=4):
    a = boxes[i]
    for j, b in enumerate(boxes):
        if j != i and not (a[2] + margin < b[0] or b[2] + margin < a[0]
                           or a[3] + margin < b[1] or b[3] + margin < a[1]):
            return False
    return True


def make_sprites(args):
    """Cut pieces that stand clear of their neighbours out of chesscog renders
    and Fenit board A photos (both CC BY 4.0)."""
    rng = random.Random(0)
    per_kind = {k: 0 for k in HEIGHT}
    SPRITES.mkdir(parents=True, exist_ok=True)

    def save(kind, crop, alpha, source):
        n = per_kind[kind]
        cv2.imwrite(str(SPRITES / f"{kind}_{source}_{n:04d}.png"),
                    np.dstack([crop, alpha]))
        per_kind[kind] += 1

    files = sorted((HERE / "datasets/chesscog/train").glob("*.json"))
    rng.shuffle(files)
    for js in files:
        if min(per_kind.values()) >= args.per_kind:
            break
        meta = json.loads(js.read_text())
        img = cv2.imread(str(js.with_suffix(".png")))
        boxes = [(p["box"][0], p["box"][1], p["box"][0] + p["box"][2], p["box"][1] + p["box"][3])
                 for p in meta["pieces"]]
        for i, p in enumerate(meta["pieces"]):
            kind = KIND_OF[p["piece"].lower()]
            if per_kind[kind] >= args.per_kind or not isolated(boxes, i):
                continue
            cut = cut_out(img, boxes[i])
            if cut:
                save(kind, *cut, "chesscog")

    target = {k: v + args.per_kind // 2 for k, v in per_kind.items()}
    labels = sorted((HERE / "datasets/merged/pieces/labels/train").glob("fenit__*.txt"))
    rng.shuffle(labels)
    for label in labels:
        if all(per_kind[k] >= target[k] for k in per_kind):
            break
        image = next(iter((label.parents[2] / "images" / "train").glob(label.stem + ".*")), None)
        img = cv2.imread(str(image)) if image else None
        if img is None:
            continue
        h, w = img.shape[:2]
        rows = [line.split() for line in label.read_text().splitlines()]
        boxes = [((float(r[1]) - float(r[3]) / 2) * w, (float(r[2]) - float(r[4]) / 2) * h,
                  (float(r[1]) + float(r[3]) / 2) * w, (float(r[2]) + float(r[4]) / 2) * h)
                 for r in rows]
        for i, r in enumerate(rows):
            kind = PIECE_CLASSES[int(r[0])].split("_")[1]
            if per_kind[kind] >= target[kind] or not isolated(boxes, i):
                continue
            cut = cut_out(img, boxes[i])
            if cut:
                save(kind, *cut, "fenit")
    print("sprites:", per_kind, "->", SPRITES)


def load_sprites():
    sprites = {k: [] for k in HEIGHT}
    for path in sorted(SPRITES.glob("*.png")):
        rgba = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.shape[2] != 4:
            continue
        gray = cv2.cvtColor(rgba[..., :3], cv2.COLOR_BGR2GRAY).astype(np.float32)
        alpha = rgba[..., 3].astype(np.float32) / 255
        inside = gray[alpha > 0.5]
        if inside.size < 200:
            continue
        # Keep the shading only; each side gets its own colour when pasted.
        shade = np.clip(gray / (np.median(inside) + 1e-6), 0.25, 2.2)
        sprites[path.name.split("_")[0]].append((shade, alpha))
    return sprites


# --------------------------------------------------------------------------
# Textures

def noise(rng, size, scale):
    """Smooth noise in [0, 1], `scale` = blob size in pixels."""
    small = rng.random((max(size[0] // scale, 2), max(size[1] // scale, 2))).astype(np.float32)
    return cv2.resize(small, (size[1], size[0]), interpolation=cv2.INTER_CUBIC).clip(0, 1)


def wood(rng, size, colour, strength=0.18):
    grain = cv2.resize(rng.random((size[0] // 3 + 2, 6)).astype(np.float32),
                       (size[1], size[0]), interpolation=cv2.INTER_CUBIC)
    grain = 0.6 * grain + 0.4 * noise(rng, size, 48)
    if rng.random() < 0.5:
        grain = grain.T.copy() if size[0] == size[1] else grain
    tex = np.array(colour, np.float32)[None, None] * (1 + strength * (grain[..., None] - 0.5) * 2)
    return tex.clip(0, 255)


def random_colour(rng, value, saturation):
    hsv = np.uint8([[[rng.integers(0, 180), int(255 * saturation), int(255 * value)]]])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].astype(np.float32)


def board_colours(rng):
    """(light square, dark square), BGR. Mostly the families real boards come
    in, sometimes anything."""
    family = rng.choice(["wood", "green", "black", "brown", "grey", "blue", "red", "any"],
                        p=[0.22, 0.16, 0.14, 0.16, 0.08, 0.06, 0.06, 0.12])
    j = lambda c, s=14: np.clip(np.array(c, np.float32) + rng.normal(0, s, 3), 0, 255)
    if family == "wood":
        return j((150, 200, 228)), j((50, 85, 130), 20)
    if family == "green":
        return j((215, 232, 236)), j((75, 120, 60), 22)
    if family == "black":
        return j((235, 236, 236), 8), j((35, 35, 38), 12)
    if family == "brown":
        return j((205, 228, 240)), j((45, 55, 85), 18)
    if family == "grey":
        return j((235, 235, 232), 8), j((95, 98, 105), 18)
    if family == "blue":
        return j((235, 232, 225)), j((140, 95, 55), 22)
    if family == "red":
        return j((200, 225, 240)), j((45, 45, 150), 22)
    light = random_colour(rng, rng.uniform(0.72, 0.98), rng.uniform(0.0, 0.3))
    dark = random_colour(rng, rng.uniform(0.15, 0.55), rng.uniform(0.1, 0.8))
    return light, dark


def board_texture(rng, px=96):
    """Top-down board: (image, margin in squares). Square (file f, rank r) is
    at x = margin + f, y = margin + 7 - r squares; a1 is dark."""
    light, dark = board_colours(rng)
    margin = float(rng.choice([0.0, rng.uniform(0.08, 0.35), rng.uniform(0.35, 1.2)],
                              p=[0.2, 0.4, 0.4]))
    size = int(round((8 + 2 * margin) * px))
    m = int(round(margin * px))
    grainy = rng.random() < 0.45
    border_colour = [light, dark, random_colour(rng, rng.uniform(0.15, 0.9), rng.uniform(0, 0.6)),
                     np.array((70, 110, 160), np.float32)][rng.integers(0, 4)]
    img = wood(rng, (size, size), border_colour, 0.15 if grainy else 0.03)
    for f in range(8):
        for r in range(8):
            colour = dark if (f + r) % 2 == 0 else light  # a1 = (0, 0) is dark
            colour = np.clip(colour + rng.normal(0, 3, 3), 0, 255)
            x, y = m + f * px, m + (7 - r) * px
            img[y:y + px, x:x + px] = (wood(rng, (px, px), colour, 0.12)
                                       if grainy else colour[None, None])
    line = tuple(float(c) for c in (dark if rng.random() < 0.7 else light))
    if m > 6 and rng.random() < 0.6:  # a line round the squares, as on vinyl boards
        gap = int(rng.uniform(0.05, 0.4) * m)
        cv2.rectangle(img, (m - gap, m - gap), (m + 8 * px + gap, m + 8 * px + gap), line,
                      max(1, int(rng.uniform(0.02, 0.08) * px)))
    if m > 0.3 * px and rng.random() < 0.6:  # coordinates
        scale = m / px * 1.6
        for i in range(8):
            for text, org in ((("abcdefgh"[i]), (m + i * px + px // 3, size - m // 4)),
                              (str(8 - i), (m // 4, m + i * px + int(0.65 * px)))):
                cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, line,
                            max(1, int(2 * scale)), cv2.LINE_AA)
    if rng.random() < 0.3:  # a fold down the middle
        cv2.line(img, (0, size // 2), (size, size // 2), (30, 30, 30), 1)
    return img.clip(0, 255).astype(np.float32), margin, px


def ground_texture(rng, size=1024):
    kind = rng.choice(["wood", "tiles", "plain", "cloth"], p=[0.35, 0.25, 0.2, 0.2])
    # Floors and tables are mostly greys, creams and browns.
    base = (random_colour(rng, rng.uniform(0.25, 0.97), rng.uniform(0.0, 0.12))
            if rng.random() < 0.55 else
            np.clip(np.array((70, 120, 170), np.float32) * rng.uniform(0.5, 1.4)
                    + rng.normal(0, 15, 3), 0, 255) if rng.random() < 0.7 else
            random_colour(rng, rng.uniform(0.25, 0.9), rng.uniform(0.1, 0.5)))
    if kind == "wood":
        tex = wood(rng, (size, size), base, 0.25)
        for x in range(0, size, int(rng.integers(120, 400))):
            cv2.line(tex, (x, 0), (x, size), tuple(float(c) * 0.6 for c in base), 2)
    elif kind == "tiles":
        tex = np.tile(base[None, None], (size, size, 1)) * (0.94 + 0.12 * noise(rng, (size, size), 96)[..., None])
        step = size // int(rng.integers(2, 6))
        grout = tuple(float(c) * rng.uniform(0.5, 0.9) for c in base)
        for k in range(0, size, step):
            cv2.line(tex, (k, 0), (k, size), grout, int(rng.integers(2, 8)))
            cv2.line(tex, (0, k), (size, k), grout, int(rng.integers(2, 8)))
    elif kind == "cloth":
        tex = np.tile(base[None, None], (size, size, 1)) * (0.9 + 0.2 * rng.random((size, size, 1)).astype(np.float32))
        tex = cv2.GaussianBlur(tex, (0, 0), 1.2)
    else:
        tex = np.tile(base[None, None], (size, size, 1)) * (0.92 + 0.16 * noise(rng, (size, size), 200)[..., None])
    return np.clip(tex, 0, 255).astype(np.float32), float(rng.uniform(4, 14))  # squares per tile


# --------------------------------------------------------------------------
# Camera

class Camera:
    """Pinhole camera looking at a board whose squares are the unit cells of
    [0, 8] x [0, 8] on the plane z = 0 (a1 at the origin, files along x)."""

    def __init__(self, rng, width, height):
        self.w, self.h = width, height
        for _ in range(200):
            elevation = math.radians(rng.uniform(22, 80))
            azimuth = rng.uniform(0, 2 * math.pi)
            distance = rng.uniform(9, 26)
            target = np.array([4 + rng.normal(0, 1.2), 4 + rng.normal(0, 1.2), 0.0])
            self.c = target + distance * np.array([
                math.cos(elevation) * math.cos(azimuth),
                math.cos(elevation) * math.sin(azimuth), math.sin(elevation)])
            f = target - self.c
            f /= np.linalg.norm(f)
            r = np.cross(f, [0, 0, 1.0])
            r /= np.linalg.norm(r)
            u = np.cross(r, f)
            roll = math.radians(rng.normal(0, 4))
            self.r = math.cos(roll) * r + math.sin(roll) * u
            self.u = math.cos(roll) * u - math.sin(roll) * r
            self.f = f
            # Focal length and image centre: the board fills a random share of
            # the frame, somewhere in it.
            self.focal, self.cx, self.cy = 1.0, 0.0, 0.0
            pts = self.project(np.array([[0, 0, 0], [0, 8, 0], [8, 8, 0], [8, 0, 0.0]]))
            span = pts.max(0) - pts.min(0)
            fill = rng.uniform(0.45, 0.97)
            self.focal = fill * min(width / span[0], height / span[1])
            pts = pts * self.focal
            lo, hi = pts.min(0), pts.max(0)
            pad = 0.01 * min(width, height)
            room_x, room_y = width - (hi[0] - lo[0]) - 2 * pad, height - (hi[1] - lo[1]) - 2 * pad
            if room_x < 0 or room_y < 0:
                continue
            self.cx = pad - lo[0] + rng.uniform(0, room_x)
            self.cy = pad - lo[1] + rng.uniform(0, room_y)
            return
        raise RuntimeError("no camera found")

    def project(self, points):
        d = np.atleast_2d(points) - self.c
        z = d @ self.f
        return np.stack([self.focal * (d @ self.r) / z + self.cx,
                         -self.focal * (d @ self.u) / z + self.cy], axis=1)

    def ground(self, scale):
        """World (x, y) on the plane z = 0 seen by every pixel of the image
        rendered `scale` times larger; NaN above the horizon."""
        xs = (np.arange(self.w * scale, dtype=np.float32) + 0.5) / scale
        ys = (np.arange(self.h * scale, dtype=np.float32) + 0.5) / scale
        px, py = np.meshgrid((xs - self.cx) / self.focal, (ys - self.cy) / self.focal)
        ray = (self.f[None, None] + px[..., None] * self.r[None, None]
               - py[..., None] * self.u[None, None])
        with np.errstate(divide="ignore", invalid="ignore"):
            t = -self.c[2] / ray[..., 2]
        t[(t <= 0) | (t > 400)] = np.nan
        return self.c[0] + t * ray[..., 0], self.c[1] + t * ray[..., 1]


# --------------------------------------------------------------------------
# One image

def side_colours(rng):
    """Piece colours for White and Black: White is always clearly the lighter,
    but not always white (natural wood, red-brown sets exist)."""
    white = [np.array((225, 235, 240)), np.array((150, 205, 235)), np.array((95, 150, 200)),
             np.array((60, 95, 165))][rng.choice(4, p=[0.4, 0.3, 0.2, 0.1])]
    black = [np.array((28, 28, 30)), np.array((35, 50, 75)), np.array((20, 25, 45))][rng.integers(0, 3)]
    return (np.clip(white + rng.normal(0, 10, 3), 0, 255).astype(np.float32),
            np.clip(black + rng.normal(0, 6, 3), 0, 255).astype(np.float32))


def paste(img, shade, alpha, colour, base, height_px):
    """Paste a recoloured cut-out with its bottom centre at `base`."""
    scale = height_px / shade.shape[0]
    w, h = max(int(shade.shape[1] * scale), 2), max(int(shade.shape[0] * scale), 2)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    shade = cv2.resize(shade, (w, h), interpolation=interp)
    alpha = cv2.resize(alpha, (w, h), interpolation=interp)
    x0, y0 = int(round(base[0] - w / 2)), int(round(base[1] - h))
    X0, Y0, X1, Y1 = max(x0, 0), max(y0, 0), min(x0 + w, img.shape[1]), min(y0 + h, img.shape[0])
    if X0 >= X1 or Y0 >= Y1:
        return
    s = shade[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0, None]
    a = alpha[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0, None]
    img[Y0:Y1, X0:X1] = img[Y0:Y1, X0:X1] * (1 - a) + np.clip(colour[None, None] * s, 0, 255) * a


def render(rng, sprites, placements):
    width, height = [(960, 1280), (1280, 960), (1080, 1080)][rng.choice(3, p=[0.55, 0.25, 0.2])]
    cam = Camera(rng, width, height)
    ss = 2  # render the planes at twice the size, then shrink: no jagged squares
    gx, gy = cam.ground(ss)
    sky = np.isnan(gx)
    gx, gy = np.nan_to_num(gx), np.nan_to_num(gy)

    ground, tile = ground_texture(rng)
    n = ground.shape[0]
    angle = rng.uniform(0, math.pi)
    rx = (gx * math.cos(angle) - gy * math.sin(angle)) / tile * n
    ry = (gx * math.sin(angle) + gy * math.cos(angle)) / tile * n
    img = cv2.remap(ground, np.mod(rx, n).astype(np.float32), np.mod(ry, n).astype(np.float32),
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    wall = random_colour(rng, rng.uniform(0.2, 0.95), rng.uniform(0, 0.15))
    img[sky] = wall

    board, margin, px = board_texture(rng)
    bx = (gx + margin) * px
    by = (8 - gy + margin) * px
    on_board = (~sky) & (bx >= 0) & (by >= 0) & (bx < board.shape[1]) & (by < board.shape[0])
    warped = cv2.remap(board, bx.astype(np.float32), by.astype(np.float32), cv2.INTER_LINEAR)
    # The board's own shadow on the ground, then the board.
    if margin >= 0 and rng.random() < 0.7:
        off = rng.uniform(0.05, 0.3) * np.array([math.cos(angle), math.sin(angle)])
        sx, sy = (gx + margin + off[0]) * px, (8 - gy + margin + off[1]) * px
        shadow = (~sky) & (sx >= 0) & (sy >= 0) & (sx < board.shape[1]) & (sy < board.shape[0])
        img[shadow & ~on_board] *= rng.uniform(0.45, 0.8)
    img[on_board] = warped[on_board]
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    # Pieces, far ones first.
    placement = placements[rng.integers(0, len(placements))]
    squares = list(fen_placement_to_board(placement).items())
    if rng.random() < 0.08:
        squares = []
    elif rng.random() < 0.3:
        rng.shuffle(squares)
        squares = squares[:rng.integers(2, max(3, len(squares)))]
    white, black = side_colours(rng)
    light = rng.uniform(0, 2 * math.pi)
    todo = []
    for (f, r), letter in squares:
        centre = np.array([f + 0.5 + rng.normal(0, 0.07), r + 0.5 + rng.normal(0, 0.07), 0.0])
        todo.append((np.linalg.norm(centre - cam.c), centre, letter))
    for _, centre, letter in sorted(todo, key=lambda t: -t[0]):
        kind = KIND_OF[letter.lower()]
        if not sprites[kind]:
            continue
        shade, alpha = sprites[kind][rng.integers(0, len(sprites[kind]))]
        tall = HEIGHT[kind] * rng.uniform(0.9, 1.1)
        base, top = cam.project(centre)[0], cam.project(centre + [0, 0, tall])[0]
        edge = cam.project(centre + [0.3, 0, 0])[0]
        radius = max(np.linalg.norm(edge - base), 2.0)
        size = np.linalg.norm(top - base) + 0.9 * radius
        if size < 6:
            continue
        # A soft shadow beside the piece, the same way for all of them.
        shadow = img.copy()
        centre_s = (int(base[0] + 0.7 * radius * math.cos(light)),
                    int(base[1] + 0.35 * radius * math.sin(light)))
        cv2.ellipse(shadow, centre_s, (int(1.3 * radius), int(0.6 * radius)), 0, 0, 360, (0, 0, 0), -1)
        img = cv2.addWeighted(img, 0.72, shadow, 0.28, 0)
        paste(img, shade, alpha, white if letter.isupper() else black,
              (base[0], base[1] + 0.45 * radius), size)

    corners = cam.project(np.array([[0, 0, 0], [0, 8, 0], [8, 8, 0], [8, 0, 0.0]]))
    return finish(rng, img), corners, placement


def finish(rng, img):
    h, w = img.shape[:2]
    # Uneven light, a colour cast, glare, then the camera: blur, noise, JPEG.
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    a = rng.uniform(0, 2 * math.pi)
    ramp = ((xx / w - 0.5) * math.cos(a) + (yy / h - 0.5) * math.sin(a)) * rng.uniform(0, 0.45)
    img = img * (rng.uniform(0.6, 1.05) + ramp[..., None]) * rng.normal(1.0, 0.04, 3)[None, None]
    for _ in range(rng.choice(3, p=[0.5, 0.35, 0.15])):
        cx, cy, s = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(0.04, 0.16) * w
        img += (rng.uniform(30, 110) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s)))[..., None]
    if rng.random() < 0.5:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.4, 1.8))
    img = img + rng.normal(0, rng.uniform(0, 7), img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(55, 96))])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


# --------------------------------------------------------------------------

def make(args):
    sprites = load_sprites()
    if min(len(v) for v in sprites.values()) == 0:
        raise SystemExit("no sprites: run `python synth_boards.py sprites` first")
    placements = sorted({json.loads(p.read_text())["fen"].split()[0]
                         for p in (HERE / "datasets/chesscog/train").glob("*.json")})
    out = HERE / args.out / "corners"
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    for i in range(args.start, args.start + args.count):
        rng = np.random.default_rng(args.seed * 1_000_003 + i)
        img, corners, _ = render(rng, sprites, placements)
        h, w = img.shape[:2]
        xs, ys = corners[:, 0] / w, corners[:, 1] / h
        if xs.min() < 0 or ys.min() < 0 or xs.max() > 1 or ys.max() > 1:
            continue
        pad = 0.03
        x1, x2 = max(xs.min() - pad, 0), min(xs.max() + pad, 1)
        y1, y2 = max(ys.min() - pad, 0), min(ys.max() + pad, 1)
        kpts = " ".join(f"{x:.6f} {y:.6f} 2" for x, y in zip(xs, ys))
        stem = f"synth_{args.seed}_{i:06d}"
        (out / "labels" / f"{stem}.txt").write_text(
            f"0 {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f} {kpts}")
        if args.preview:
            for name, (x, y) in zip(("a1", "a8", "h8", "h1"), corners):
                cv2.circle(img, (int(x), int(y)), 9, (0, 0, 255), -1)
                cv2.putText(img, name, (int(x) + 10, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            1.1, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.imwrite(str(out / "images" / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"wrote {len(list((out / 'images').glob('*.jpg')))} images under {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sprites")
    s.add_argument("--per-kind", type=int, default=60, help="cut-outs per piece type from chesscog")
    m = sub.add_parser("make")
    m.add_argument("--count", type=int, default=3000)
    m.add_argument("--start", type=int, default=0, help="first image number (to add more later)")
    m.add_argument("--seed", type=int, default=1)
    m.add_argument("--out", default="datasets/synth")
    m.add_argument("--preview", action="store_true", help="draw the corner labels on the images")
    args = ap.parse_args()
    {"sprites": make_sprites, "make": make}[args.cmd](args)


if __name__ == "__main__":
    main()
