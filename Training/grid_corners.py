"""Classical (no learning) detector for the four outer corners of the 8x8 grid.

    python grid_corners.py --source fenit --group B              # box prior from the pose model
    python grid_corners.py --source fenit --group B --prior none # no model at all
    python grid_corners.py --source chessred2k --split test --save-fails 8

    from grid_corners import find_grid
    corners, confidence = find_grid(image_bgr, prior_box=(x1, y1, x2, y2))

How it works:
  1. x-corners: saddle points of the smoothed image (negative Hessian
     determinant), kept only if a ring around them shows the 4-sector
     light/dark/light/dark pattern, then refined to sub-pixel.
  2. lattice: every small parallelogram of neighbouring x-corners proposes a
     homography image -> integer lattice; the one that puts most x-corners on
     integers wins and is refitted on its inliers.
  3. which 8x8: the photo is warped onto the lattice, each cell gets a grey
     level, and the 8x8 window whose cells alternate best (and whose
     surroundings do not) is chosen among those that contain all inliers.
  4. refinement: edge points are searched along all 18 grid lines of that
     window (the outer ones included) and the homography is refitted to them.
  5. validity: enough x-corners, cells alternate, the pattern stops outside,
     the choice of window is not a near tie, the outer lines have edge
     support, and (with a prior box) the result fills that box.
find_grid returns (None, 0.0) when any of these fail.

Corners come back clockwise as seen in the image (the order a1, a8, h8, h1
has when white sits at the bottom); which of them is a1 is not known.
CPU only.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "runs" / "grid"
BOARD_CORNERS = np.array([[0, 0], [0, 8], [8, 8], [8, 0]], float)  # a1 a8 h8 h1

WORK = 1000          # long side of the working image, pixels
RING = [(np.cos(a), np.sin(a)) for a in np.arange(16) * np.pi / 8]


# ----------------------------------------------------------------- x-corners

def _ring_test(img, pts, radius):
    """ChESS-like test. Returns (response, transitions) per point."""
    h, w = img.shape
    x, y = pts[:, 0], pts[:, 1]
    ring = np.stack([
        img[np.clip(np.rint(y + radius * s).astype(int), 0, h - 1),
            np.clip(np.rint(x + radius * c).astype(int), 0, w - 1)]
        for c, s in RING], axis=1)
    sr = sum(np.abs(ring[:, n] + ring[:, n + 8] - ring[:, n + 4] - ring[:, n + 12])
             for n in range(4))
    dr = sum(np.abs(ring[:, n] - ring[:, n + 8]) for n in range(8))
    mid = (ring.max(1, keepdims=True) + ring.min(1, keepdims=True)) / 2
    sign = ring > mid
    trans = (sign != np.roll(sign, 1, axis=1)).sum(1)
    contrast = ring.max(1) - ring.min(1)
    return sr - dr, trans, contrast


def x_corners(gray, max_points=260):
    """Sub-pixel saddle points that look like chessboard x-corners."""
    g = cv2.GaussianBlur(gray, (0, 0), 2.0)
    ixx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3)
    iyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3)
    ixy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3)
    sad = ixy * ixy - ixx * iyy
    sad[sad < 0] = 0
    peak = (sad == cv2.dilate(sad, np.ones((9, 9), np.uint8))) & (sad > 0)
    b = 8
    peak[:b] = peak[-b:] = False
    peak[:, :b] = peak[:, -b:] = False
    ys, xs = np.nonzero(peak)
    if len(xs) == 0:
        return np.zeros((0, 2), np.float32)
    order = np.argsort(-sad[ys, xs])[:max_points * 4]
    pts = np.stack([xs[order], ys[order]], 1).astype(np.float32)
    g1 = cv2.GaussianBlur(gray, (0, 0), 1.0)
    keep = np.zeros(len(pts), bool)
    for radius in (4, 7):
        resp, trans, contrast = _ring_test(g1, pts, radius)
        keep |= (trans == 4) & (resp > 0.4 * contrast) & (contrast > 18)
    pts = pts[keep][:max_points]
    if len(pts) == 0:
        return pts
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    before = pts.copy()
    cv2.cornerSubPix(g1, pts, (4, 4), (-1, -1), crit)
    moved = np.linalg.norm(pts - before, axis=1)
    return pts[moved < 4]


# ------------------------------------------------------------------- lattice

def _apply(H, pts):
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:3]


def _inliers(H, pts, tol):
    uv = _apply(H, pts)
    r = np.rint(uv)
    ok = (np.abs(uv - r).max(1) < tol) & (np.abs(uv).max(1) < 12)
    return ok, r.astype(int)


_UNIMODULAR = [np.array([[a, b], [c, d]]) for a in range(-2, 3) for b in range(-2, 3)
               for c in range(-2, 3) for d in range(-2, 3) if a * d - b * c == 1]


def fit_lattice(pts):
    """Homography image -> lattice with most x-corners on integer nodes.
    Returns (H, inlier mask, integer coords) or None."""
    n = len(pts)
    if n < 9:
        return None
    d = np.linalg.norm(pts[:, None] - pts[None], axis=2)
    np.fill_diagonal(d, 1e9)
    nn = np.argsort(d, axis=1)[:, :7]
    hyps = []
    unit = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    for i in range(n):
        p = pts[i]
        for a in range(nn.shape[1]):
            for b in range(a + 1, nn.shape[1]):
                qa, qb = pts[nn[i, a]], pts[nn[i, b]]
                va, vb = qa - p, qb - p
                la, lb = np.linalg.norm(va), np.linalg.norm(vb)
                if abs(va[0] * vb[1] - va[1] * vb[0]) < 0.35 * la * lb:
                    continue
                if max(la, lb) > 6 * min(la, lb):
                    continue
                t = p + va + vb
                dt = np.linalg.norm(pts - t, axis=1)
                j = int(dt.argmin())
                if dt[j] > 0.3 * min(la, lb) or j in (i, nn[i, a], nn[i, b]):
                    continue
                quad = np.array([p, qa, pts[j], qb], np.float32)
                H = cv2.getPerspectiveTransform(quad, unit)
                ok, r = _inliers(H, pts, 0.15)
                # distinct nodes, so a collapsed map cannot win
                hyps.append((len(np.unique(r[ok, 0] * 64 + r[ok, 1])), H))
    if not hyps:
        return None
    hyps.sort(key=lambda t: -t[0])
    best = None
    for count, H in hyps[:12]:
        ok, r = _inliers(H, pts, 0.15)
        for tol in (0.18, 0.18, 0.15, 0.12):
            if ok.sum() < 6:
                break
            H2, _ = cv2.findHomography(pts[ok], r[ok].astype(np.float32), 0)
            if H2 is None:
                break
            H = H2
            ok, r = _inliers(H, pts, tol)
        # one x-corner per node
        if ok.sum() >= 6:
            uniq = len({tuple(c) for c in r[ok]})
            if best is None or uniq > best[0]:
                best = (uniq, H, ok, r)
    if best is None:
        return None
    _, H, ok, r = best
    # The winning lattice may be finer than the board's (stray points on
    # square centres or edge midpoints propose it, and every true x-corner
    # still lands on a node). Then nearly all inliers share one coset of a
    # sublattice of index 2: move to that sublattice.
    for _ in range(3):
        ij = r[ok]
        changed = False
        # (pieces stand on square centres and add points there, so the
        # diagonal case is accepted on a two-thirds majority)
        for M, key, lim in ((np.array([[0.5, 0], [0, 1]]), ij[:, 0] % 2, 0.2),
                            (np.array([[1, 0], [0, 0.5]]), ij[:, 1] % 2, 0.2),
                            (np.array([[0.5, 0.5], [-0.5, 0.5]]), (ij[:, 0] + ij[:, 1]) % 2, 0.34)):
            share = key.mean()
            if lim < share < 1 - lim:
                continue
            c = int(share >= 0.5)                 # the coset the nodes are on
            off = np.array([c, 0]) if M[0, 0] == 0.5 else np.array([0, c])
            keep = np.zeros(len(pts), bool)
            keep[np.nonzero(ok)[0][key == c]] = True
            M3, T3 = np.eye(3), np.eye(3)
            M3[:2, :2] = M
            T3[:2, 2] = -off
            H = M3 @ T3 @ H
            ok = keep
            r = np.rint(_apply(H, pts)).astype(int)
            changed = True
            break
        if not changed:
            break
    # The winning basis may be a sheared one (same lattice, slanted axes):
    # take the unimodular change of basis that makes the inliers' bounding
    # box smallest.
    ij = r[ok]

    def spread(c):      # robust to a few stray inliers off the board
        lo, hi = np.percentile(c, [8, 92], axis=0)
        return float((hi - lo).sum())

    best_u, best_span = np.eye(2, dtype=int), spread(ij)
    for U in _UNIMODULAR:
        span = spread(ij @ U.T)
        if span < best_span - 1e-9:
            best_u, best_span = U, span
    U3 = np.eye(3)
    U3[:2, :2] = best_u
    H = U3 @ H
    r = r @ best_u.T
    return H, ok, r


# ------------------------------------------------------ cells and the window

def _warp(gray, H, x0, y0, nx, ny, px):
    """Lattice rectangle [x0, x0+nx] x [y0, y0+ny] at px pixels per cell."""
    T = np.array([[px, 0, -x0 * px], [0, px, -y0 * px], [0, 0, 1]], float)
    return cv2.warpPerspective(gray, T @ H, (nx * px, ny * px), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=-1)


def _cell_values(gray, H, x0, y0, nx, ny, px=16):
    """Grey of each lattice cell, as the medians of four patches near its
    corners (a piece usually stands on the middle and hides part of the
    cell). NaN where the cell is off the photo or behind the horizon.
    Shape (ny, nx, 4), index [j, i]."""
    Hi = np.linalg.inv(H)
    gx, gy = np.meshgrid(np.arange(x0, x0 + nx) + 0.5, np.arange(y0, y0 + ny) + 0.5)
    c = np.c_[gx.ravel(), gy.ravel(), np.ones(gx.size)] @ Hi.T
    w0 = (np.array([x0 + nx / 2, y0 + ny / 2, 1.0]) @ Hi.T)[2]
    front = (c[:, 2] * np.sign(w0) > 1e-9).reshape(ny, nx)
    W = _warp(gray, H, x0, y0, nx, ny, px)
    cells = W.reshape(ny, px, nx, px).transpose(0, 2, 1, 3)
    a0, a1, b0, b1 = px // 8, px * 7 // 16, px * 9 // 16, px - px // 8
    out = np.zeros((ny, nx, 4), np.float32)
    for n, (ys, xs) in enumerate(((slice(a0, a1), slice(a0, a1)), (slice(a0, a1), slice(b0, b1)),
                                  (slice(b0, b1), slice(a0, a1)), (slice(b0, b1), slice(b0, b1)))):
        patch = cells[:, :, ys, xs].reshape(ny, nx, -1)
        out[:, :, n] = np.median(patch, axis=2)
        out[:, :, n][(patch < 0).mean(axis=2) > 0.2] = np.nan
    out[~front] = np.nan
    return out


def _two_levels(values):
    """Two cluster centres of a 1-D sample (dark, light)."""
    v = values[~np.isnan(values)]
    lo, hi = np.percentile(v, 20), np.percentile(v, 80)
    for _ in range(10):
        t = (lo + hi) / 2
        if (v <= t).sum() == 0 or (v > t).sum() == 0:
            break
        lo, hi = np.median(v[v <= t]), np.median(v[v > t])
    return float(lo), float(hi)


def _good_cells(v, parity, region):
    """Which cells show their expected colour, given that the cells in
    `region` are board cells. A cell is good when at least two of its four
    patches are near the level of its colour (a piece may hide the rest),
    strict when all four are: that is asked of cells outside the board,
    where a frame can imitate half a dark square.
    Returns (good, strict, valid, contrast, is_light) or None."""
    valid = ~np.isnan(v).any(axis=2)
    reg = region & valid
    if reg.sum() < 6:
        return None
    dark, light = _two_levels(v[reg])
    contrast = light - dark
    if contrast < 20:
        return None
    # which parity is the light one: vote over the region
    lightish = (v > (dark + light) / 2).mean(axis=2)
    p0 = lightish[reg & (parity == 0)]
    p1 = lightish[reg & (parity == 1)]
    if len(p0) == 0 or len(p1) == 0:
        return None
    light_parity = 0 if p0.mean() > p1.mean() else 1
    expect = np.where(parity == light_parity, light, dark)
    with np.errstate(invalid="ignore"):
        near = np.abs(v - expect[:, :, None]) < 0.4 * contrast
    good = (near.sum(axis=2) >= 2) & valid
    strict = (near.sum(axis=2) >= 4) & valid
    return good, strict, valid, float(contrast), parity == light_parity


def _alternating(good, light):
    """How many cells of a row take part in a light/dark alternation: twice
    the smaller of the good light and the good dark cells. A white margin or
    a plain floor matches one colour only and scores 0."""
    return 2 * int(min((good & light).sum(), (good & ~light).sum()))


def _ring_sides(a, b, inset=0):
    """Index slices of the four rows just outside the 8x8 block whose first
    cell is [b, a] (inset=1: its four outermost rows instead)."""
    lo_a, hi_a, lo_b, hi_b = a - 1 + inset, a + 8 - inset, b - 1 + inset, b + 8 - inset
    return [(lo_b, slice(a, a + 8)), (hi_b, slice(a, a + 8)),
            (slice(b, b + 8), lo_a), (slice(b, b + 8), hi_a)]


def choose_window(gray, H, ok, r):
    """The 8x8 window of lattice cells that is the board.
    Returns dict(ox, oy, inside, ring, margin, contrast) or None."""
    ij = r[ok]
    nodes = {tuple(c) for c in ij}
    # Extent of the lattice lines that carry at least 3 x-corners: a stray
    # inlier beside the board must not pin the window.
    na = np.array(sorted(nodes))
    lo, hi = [], []
    for ax in (0, 1):
        vals, counts = np.unique(na[:, ax], return_counts=True)
        full = vals[counts >= 3]
        if len(full) < 3:
            return None
        lo.append(int(full.min()))
        hi.append(int(full.max()))
    # window starts to try on each axis: every 8-wide window that holds
    # those lines or, when they span more than 8 (lines of x-corners on a
    # frame or on the floor), every window among them
    # (one more each way once the span exceeds the 7 inner lines: a line
    # of x-corners beside the board then cannot exclude the true window)
    ex = [1 if hi[ax] - lo[ax] > 6 else 0 for ax in (0, 1)]
    sx = range(min(lo[0], hi[0] - 8) - ex[0], max(lo[0], hi[0] - 8) + 1 + ex[0])
    sy = range(min(lo[1], hi[1] - 8) - ex[1], max(lo[1], hi[1] - 8) + 1 + ex[1])
    x0, y0 = sx[0] - 1, sy[0] - 1
    nx, ny = sx[-1] + 9 - x0, sy[-1] + 9 - y0
    if nx > 30 or ny > 30:
        return None
    v = _cell_values(gray, H, x0, y0, nx, ny)
    gi, gj = np.meshgrid(np.arange(x0, x0 + nx), np.arange(y0, y0 + ny))
    parity = (gi + gj) % 2
    # cells with x-corners on all four corners are board cells for sure
    core = np.zeros((ny, nx), bool)
    for (i, j) in nodes:
        if {(i + 1, j), (i, j + 1), (i + 1, j + 1)} <= nodes:
            core[j - y0, i - x0] = True
    res = _good_cells(v, parity, core)
    if res is None:
        return None
    good, strict, valid, contrast, light = res
    cands = []
    for ox in sx:
        for oy in sy:
            a, b = ox - x0, oy - y0
            if (~valid[b:b + 8, a:a + 8]).sum() > 8:
                continue
            inside = int(good[b:b + 8, a:a + 8].sum())
            ring = sum(_alternating(strict[sl], light[sl]) for sl in _ring_sides(a, b))
            n_in = int(((na[:, 0] >= ox) & (na[:, 0] <= ox + 8)
                        & (na[:, 1] >= oy) & (na[:, 1] <= oy + 8)).sum())
            n_strict = int(((na[:, 0] > ox) & (na[:, 0] < ox + 8)
                            & (na[:, 1] > oy) & (na[:, 1] < oy + 8)).sum())
            cands.append((inside - ring + 0.5 * n_in + 0.5 * n_strict, ox, oy, inside, ring))
    if not cands:
        return None
    cands.sort(reverse=True)
    margin = cands[0][0] - cands[1][0] if len(cands) > 1 else 99
    _, ox, oy, inside, ring = cands[0]
    return dict(ox=ox, oy=oy, inside=inside, ring=ring, margin=float(margin),
                contrast=contrast, n_cands=len(cands))


# ---------------------------------------------------------------- refinement

def _line_points(gray, Hb, contrast, px=40, reach=0.3):
    """Edge points along the 18 grid lines of the board frame Hb (image ->
    [0,8]^2). Returns list of (image point, axis, k, is_border)."""
    m = 1  # margin cells
    n = 8 + 2 * m
    W = _warp(gray, Hb, -m, -m, n, n, px)
    W = cv2.GaussianBlur(W, (0, 0), 1.2)
    Hi = np.linalg.inv(Hb)
    out = []
    half = int(reach * px)
    span = px // 5          # averaging along the line
    for axis in (0, 1):     # 0: lines u = k (vertical in W), 1: lines v = k
        Wa = W if axis == 0 else W.T
        for k in range(9):
            c = (k + m) * px
            border = k in (0, 8)
            per_type = {}
            for j in range(8):
                for f in (0.3, 0.5, 0.7):
                    row = int((j + f + m) * px)
                    strip = Wa[row - span:row + span + 1, c - half - 2:c + half + 3]
                    if strip.min() < 0:
                        continue
                    prof = strip.mean(0)
                    if border:
                        # walk outwards from inside the cell; first clear step
                        inward = prof[::-1] if k == 8 else prof
                        # inward[0] is outside for k == 0: flip so index 0 is inside
                        seq = inward[::-1]
                        ref = np.median(seq[:max(3, half // 3)])
                        dev = np.abs(seq - ref)
                        hit = np.nonzero(dev > 0.35 * contrast)[0]
                        if len(hit) == 0:
                            continue
                        e = hit[0]
                        gseq = np.abs(np.diff(seq))
                        lo_, hi_ = max(e - 3, 0), min(e + 1, len(gseq))
                        e = lo_ + int(gseq[lo_:hi_].argmax())
                        w_ = gseq[max(e - 1, 0):e + 2]
                        e = e + 0.5
                        # e counted from the inside end
                        pos = (len(prof) - 1 - e) if k == 0 else e
                        if k == 8:
                            pos = e
                        else:
                            pos = len(prof) - 1 - e
                        ctype = int(ref > np.median(W[W >= 0]))
                        off = pos - (half + 2)
                        per_type.setdefault(ctype, []).append((row, off))
                    else:
                        grad = np.diff(prof)
                        e = int(np.abs(grad).argmax())
                        if abs(grad[e]) < 0.12 * contrast:
                            continue
                        # parabola for sub-pixel
                        a = np.abs(grad)
                        if 0 < e < len(a) - 1:
                            den = a[e - 1] - 2 * a[e] + a[e + 1]
                            sub = 0.5 * (a[e - 1] - a[e + 1]) / den if den != 0 else 0
                        else:
                            continue
                        off = e + 0.5 + sub - (half + 2)
                        per_type.setdefault(2, []).append((row, off))
            if border and per_type:
                # the true edge is the innermost step: keep the cell colour
                # whose steps sit furthest inside
                def inner(offs):
                    med = np.median([o for _, o in offs])
                    return med if k == 0 else -med
                usable = {t: v for t, v in per_type.items() if len(v) >= 4}
                if not usable:
                    continue
                t = max(usable, key=lambda t: inner(usable[t]))
                items = usable[t]
            else:
                items = per_type.get(2, [])
            for row, off in items:
                u = (c + off) / px - m
                v = row / px - m
                lat = (u, v) if axis == 0 else (v, u)
                p = Hi @ np.array([lat[0], lat[1], 1.0])
                out.append((p[:2] / p[2], axis, k, border))
    return out


def _fit_lines(points, H0, iters=6):
    """Homography image -> board from point-on-line constraints, robustly."""
    P = np.array([p for p, *_ in points])
    axis = np.array([a for _, a, _, _ in points])
    k = np.array([kk for _, _, kk, _ in points], float)
    # normalise image coords
    mu, sd = P.mean(0), P.std() + 1e-9
    N = np.array([[1 / sd, 0, -mu[0] / sd], [0, 1 / sd, -mu[1] / sd], [0, 0, 1]])
    Ph = np.c_[(P - mu) / sd, np.ones(len(P))]
    S = np.diag([1 / 4, 1 / 4, 1.0]) @ np.array([[1, 0, -4], [0, 1, -4], [0, 0, 1.0]])
    kn = (k - 4) / 4
    A = np.zeros((len(P), 9))
    for a in (0, 1):
        sel = axis == a
        A[sel, 3 * a:3 * a + 3] = Ph[sel]
        A[sel, 6:9] = -kn[sel, None] * Ph[sel]
    w = np.ones(len(P))
    H = H0
    for _ in range(iters):
        _, _, vt = np.linalg.svd(A * w[:, None], full_matrices=False)
        Hn = vt[-1].reshape(3, 3)
        H = np.linalg.inv(S) @ Hn @ N
        H = H / H[2, 2]
        uv = _apply(H, P)
        res = np.where(axis == 0, uv[:, 0], uv[:, 1]) - k
        s = max(1.4826 * np.median(np.abs(res)), 0.01)
        w = 1.0 / (1.0 + (res / (2 * s)) ** 2)
    return H, res


def refine(gray, Hb, contrast, rounds=2):
    stats = {}
    for _ in range(rounds):
        pts = _line_points(gray, Hb, contrast)
        nb = {(a, k): 0 for a in (0, 1) for k in (0, 8)}
        for _, a, k, border in pts:
            if border:
                nb[(a, k)] += 1
        stats = dict(n_line_pts=len(pts), border_support=min(nb.values()) / 24.0)
        if len(pts) < 60:
            return Hb, stats
        Hn, res = _fit_lines(pts, Hb)
        # refuse a refinement that runs away
        c0 = _apply(np.linalg.inv(Hb), BOARD_CORNERS)
        c1 = _apply(np.linalg.inv(Hn), BOARD_CORNERS)
        shift = np.abs(_apply(Hb, c1) - BOARD_CORNERS).max()
        stats["shift"] = float(shift)
        stats["line_rms"] = float(np.sqrt(np.median(res ** 2)))
        stats["line_support"] = float((np.abs(res) < 0.04).sum() / 432.0)
        if shift > 0.45:
            return Hb, stats
        Hb = Hn
    return Hb, stats


def final_check(gray, Hb):
    """Cells of the final frame: share that show their colour, and how much
    each side of the ring outside still looks like more of the pattern."""
    v = _cell_values(gray, Hb, -1, -1, 10, 10, px=20)
    gi, gj = np.meshgrid(np.arange(-1, 9), np.arange(-1, 9))
    parity = (gi + gj) % 2
    inside = (gi >= 0) & (gi < 8) & (gj >= 0) & (gj < 8)
    res = _good_cells(v, parity, inside)
    if res is None or res[2][inside].sum() < 56:
        return None
    good, strict, valid, contrast, light = res
    # a board shifted by one shows as one side of the ring alternating well
    sides = [_alternating(strict[sl], light[sl]) / 8 for sl in _ring_sides(1, 1)]
    rows = [_alternating(good[sl], light[sl]) / 8 for sl in _ring_sides(1, 1, inset=1)]
    return dict(inside=float(good[inside].mean()), ring_max=float(max(sides)),
                edge_rows_min=float(min(rows)), contrast=contrast)


# ---------------------------------------------------------------------- main

def _prepare(image_bgr, prior_box, pad=0.10):
    h, w = image_bgr.shape[:2]
    if prior_box is not None:
        x1, y1, x2, y2 = [float(t) for t in prior_box]
        px, py = pad * (x2 - x1), pad * (y2 - y1)
        x1, y1 = int(max(0, x1 - px)), int(max(0, y1 - py))
        x2, y2 = int(min(w, x2 + px)), int(min(h, y2 + py))
    else:
        x1, y1, x2, y2 = 0, 0, w, h
    crop = image_bgr[y1:y2, x1:x2]
    s = min(1.0, WORK / max(crop.shape[:2]))
    if s < 1:
        crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return gray, s, (x1, y1)


def _detect(image_bgr, prior_box):
    info = {"stage": "start"}
    try:
        with np.errstate(all="ignore"):
            return _detect_inner(image_bgr, prior_box, info)
    except (np.linalg.LinAlgError, cv2.error, ValueError, IndexError) as e:
        info["stage"] = f"error at {info['stage']}: {type(e).__name__}"
        return None, info


def _detect_inner(image_bgr, prior_box, info):
    gray, s, (x1, y1) = _prepare(image_bgr, prior_box)
    pts = x_corners(gray)
    info["n_xcorners"] = len(pts)
    info["stage"] = "lattice"
    fit = fit_lattice(pts)
    if fit is None:
        return None, info
    H, ok, r = fit
    info["n_inliers"] = len({tuple(c) for c in r[ok]})
    info["stage"] = "window"
    if info["n_inliers"] < 9:
        return None, info
    win = choose_window(gray, H, ok, r)
    if win is None:
        return None, info
    info.update(win)
    info["stage"] = "refine"
    nodes = {tuple(c) for c in r[ok]}
    ox, oy = win["ox"], win["oy"]
    info["border_nodes"] = min(
        sum(1 for c in nodes if c[0] == ox), sum(1 for c in nodes if c[0] == ox + 8),
        sum(1 for c in nodes if c[1] == oy), sum(1 for c in nodes if c[1] == oy + 8))
    inner = [c for c in nodes if ox < c[0] < ox + 8 and oy < c[1] < oy + 8]
    info["first_line_nodes"] = min(
        sum(1 for c in inner if c[0] == ox + 1), sum(1 for c in inner if c[0] == ox + 7),
        sum(1 for c in inner if c[1] == oy + 1), sum(1 for c in inner if c[1] == oy + 7))
    info["inner_nodes"] = sum(1 for c in nodes if ox < c[0] < ox + 8 and oy < c[1] < oy + 8)
    T = np.array([[1, 0, -win["ox"]], [0, 1, -win["oy"]], [0, 0, 1.0]])
    Hb = T @ H
    Hb, st = refine(gray, Hb, win["contrast"])
    info.update(st)
    chk = final_check(gray, Hb)
    if chk is None:
        info["stage"] = "final_check"
        return None, info
    info.update({"f_" + k: v for k, v in chk.items()})
    corners = _apply(np.linalg.inv(Hb), BOARD_CORNERS) / s + np.array([x1, y1])
    # clockwise in the image (y down): positive shoelace area
    a = corners
    area = 0.5 * np.sum(a[:, 0] * np.roll(a[:, 1], -1) - np.roll(a[:, 0], -1) * a[:, 1])
    if area < 0:
        corners = corners[::-1].copy()
    info["area"] = abs(float(area))
    # convexity
    d = np.roll(corners, -1, 0) - corners
    cr = d[:, 0] * np.roll(d[:, 1], -1) - d[:, 1] * np.roll(d[:, 0], -1)
    info["convex"] = bool((cr > 0).all() or (cr < 0).all())
    if prior_box is not None:
        bx1, by1, bx2, by2 = prior_box
        qx1, qy1 = corners.min(0)
        qx2, qy2 = corners.max(0)
        h, w = image_bgr.shape[:2]
        qx1, qy1, qx2, qy2 = max(qx1, 0), max(qy1, 0), min(qx2, w), min(qy2, h)
        iw = max(0, min(bx2, qx2) - max(bx1, qx1))
        ih = max(0, min(by2, qy2) - max(by1, qy1))
        union = (bx2 - bx1) * (by2 - by1) + (qx2 - qx1) * (qy2 - qy1) - iw * ih
        info["box_iou"] = float(iw * ih / union) if union > 0 else 0.0
    info["stage"] = "done"
    return corners, info


def confidence(info, with_prior):
    """0 when a validity check fails, otherwise a 0..1 score."""
    if info.get("stage") != "done" or not info.get("convex"):
        return 0.0
    checks = [
        info["inner_nodes"] >= 20,
        info["first_line_nodes"] >= 2,   # an inner line without x-corners: maybe shifted
        info["f_inside"] >= 0.7,
        info["f_ring_max"] <= 0.25,
        info["f_edge_rows_min"] >= 0.25,
        info["margin"] >= 1,
        info.get("line_support", 0) >= 0.6,
        info.get("border_support", 0) >= 0.16,
        info.get("shift", 1) <= 0.45,
    ]
    if with_prior:
        checks.append(info.get("box_iou", 0) >= 0.5)
    if not all(checks):
        return 0.0
    score = (min(info["inner_nodes"], 40) / 40 * 0.3 + info["f_inside"] * 0.4
             + min(info["margin"], 12) / 12 * 0.3)
    return float(score)


def find_grid(image_bgr, prior_box=None, return_info=False):
    """Outer corners of the 8x8 grid.

    image_bgr: the photo as OpenCV reads it. prior_box: optional (x1, y1, x2,
    y2) of the board in image pixels (e.g. the pose model's box).
    Returns (corners, confidence): corners a 4x2 float array in image pixels,
    clockwise in the image, or None when no grid passed the validity checks
    (confidence 0.0). With return_info=True a third value holds the
    diagnostics, and the unchecked corners are in info['raw_corners']."""
    corners, info = _detect(image_bgr, prior_box)
    conf = confidence(info, prior_box is not None) if corners is not None else 0.0
    if prior_box is None and corners is not None:
        # second pass on the found board alone: more pixels per square
        h, w = image_bgr.shape[:2]
        box = (max(corners[:, 0].min(), 0), max(corners[:, 1].min(), 0),
               min(corners[:, 0].max(), w), min(corners[:, 1].max(), h))
        if box[2] - box[0] > 40 and box[3] - box[1] > 40:
            c2, i2 = _detect(image_bgr, box)
            conf2 = confidence(i2, True) if c2 is not None else 0.0
            i2["pass"] = 2
            if conf2 > 0 or conf == 0:
                corners, info, conf = c2, i2, conf2
    info["raw_corners"] = None if corners is None else corners.tolist()
    out = corners if conf > 0 else None
    return (out, conf, info) if return_info else (out, conf)


# --------------------------------------------------------------- evaluation

def corner_error(pred, truth):
    """Mean corner distance in squares, best of the 4 rotations (the measure
    of corner_error.py)."""
    from chess_pipeline import homography, project
    to_board = homography(truth)
    on_board = np.array([project(to_board, p) for p in pred])
    return min(float(np.linalg.norm(on_board - np.roll(BOARD_CORNERS, -k, axis=0), axis=1).mean())
               for k in range(4))


def _rows(args):
    rows = []
    for row in json.loads(Path(args.positions).read_text()):
        if args.source and row["source"] != args.source:
            continue
        if args.group and row.get("group") != args.group:
            continue
        if args.split != "all" and row["split"] != args.split:
            continue
        rows.append(row)
    return rows[:args.limit] if args.limit else rows


def _prior_boxes(rows, tag, model_path):
    """Board boxes from the pose model, cached under runs/grid/."""
    cache = OUT / f"prior_{tag}.json"
    boxes = json.loads(cache.read_text()) if cache.exists() else {}
    todo = [r for r in rows if (r.get("original") or r["image"]) not in boxes]
    if todo:
        import torch
        from ultralytics import YOLO
        torch.set_num_threads(4)
        model = YOLO(model_path)
        for r in todo:
            src = r.get("original") or r["image"]
            res = model(src, imgsz=640, device="cpu", verbose=False)[0]
            if res.boxes is None or len(res.boxes) == 0:
                boxes[src] = None
                continue
            b = int(res.boxes.conf.argmax())
            boxes[src] = dict(box=res.boxes.xyxy[b].tolist(), conf=float(res.boxes.conf[b]),
                              kpts=res.keypoints.xy[b].tolist())
        cache.write_text(json.dumps(boxes))
    return boxes


def _work(job):
    row, box = job
    cv2.setNumThreads(1)
    from evaluate import merged_labels
    src = Path(row.get("original") or row["image"])
    truth, _ = merged_labels(row, src)
    if not truth:
        return None
    img = cv2.imread(str(src))
    t = time.time()
    corners, conf, info = find_grid(img, prior_box=box, return_info=True)
    dt = time.time() - t
    raw = info.get("raw_corners")
    h, w = img.shape[:2]
    tr = np.array(truth)
    cut = bool((tr[:, 0] < 0.012 * w).any() or (tr[:, 0] > 0.988 * w).any()
               or (tr[:, 1] < 0.012 * h).any() or (tr[:, 1] > 0.988 * h).any())
    return dict(src=str(src), conf=conf, time=dt, truth=truth, raw=raw, cut=cut,
                err=corner_error(np.array(raw), truth) if raw is not None else None,
                info={k: v for k, v in info.items() if k != "raw_corners"})


def _draw(res, path, box=None):
    img = cv2.imread(res["src"])
    t = max(3, img.shape[0] // 350)
    cv2.polylines(img, [np.array(res["truth"], np.int32)], True, (0, 200, 0), t)
    if res["raw"] is not None:
        cv2.polylines(img, [np.array(res["raw"], np.int32)], True, (0, 0, 255), t)
    if box is not None:
        cv2.rectangle(img, tuple(int(v) for v in box[:2]), tuple(int(v) for v in box[2:]),
                      (255, 160, 0), t)
    s = 1100 / max(img.shape[:2])
    img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    err = "none" if res["err"] is None else f"{res['err']:.2f}"
    cv2.putText(img, f"err {err} sq  conf {res['conf']:.2f}  green=label red=found",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imwrite(str(path), img)


def summarise(name, results):
    n = len(results)
    acc = [r for r in results if r["conf"] > 0]
    line = f"{name}: {n} photos, accepted {len(acc)} ({len(acc) / max(n, 1):.0%})"
    if acc:
        e = np.array([r["err"] for r in acc])
        line += (f" | on accepted: median {np.median(e):.3f}, p90 {np.percentile(e, 90):.3f} sq; "
                 f"within 0.25: {np.mean(e < 0.25):.0%}, within 0.5: {np.mean(e < 0.5):.0%}, "
                 f"over 1 (false accepts): {np.sum(e > 1)}/{len(e)} = {np.mean(e > 1):.1%}")
    print(line)
    raw = np.array([r["err"] for r in results if r["err"] is not None])
    if len(raw):
        print(f"    before the validity check: {len(raw)} had some result; median {np.median(raw):.3f}, "
              f"within 0.25: {np.mean(raw < 0.25):.0%}, over 1: {np.mean(raw > 1):.0%}")
    print(f"    time per photo: median {np.median([r['time'] for r in results]):.2f} s")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--positions", default=str(HERE / "datasets/merged/positions.json"))
    ap.add_argument("--source")
    ap.add_argument("--group")
    ap.add_argument("--split", default="all", choices=["train", "val", "test", "all"])
    ap.add_argument("--prior", default="box", choices=["box", "none"],
                    help="box: restrict the search to the pose model's board box")
    ap.add_argument("--prior-model", default=str(HERE / "runs/pose/corners_s_exp1/weights/last.pt"))
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save-fails", type=int, default=0,
                    help="save this many annotated failures of each kind under runs/grid/")
    ap.add_argument("--dump", help="write per-photo results to this json")
    args = ap.parse_args()
    args.procs = min(args.procs, 4)
    OUT.mkdir(parents=True, exist_ok=True)

    rows = _rows(args)
    tag = "_".join(str(x) for x in (args.source, args.group, args.split) if x)
    boxes = {}
    if args.prior == "box":
        boxes = _prior_boxes(rows, tag, args.prior_model)
    jobs = []
    for r in rows:
        b = boxes.get(r.get("original") or r["image"])
        jobs.append((r, b["box"] if b else None))
    from multiprocessing import Pool
    with Pool(args.procs) as pool:
        results = [r for r in pool.imap(_work, jobs, chunksize=2) if r]

    name = f"{tag} prior={args.prior}"
    summarise(name, results)
    whole = [r for r in results if not r["cut"]]
    if len(whole) != len(results):
        summarise(f"{name}, only photos whose labelled corners are inside the frame", whole)
    stages = {}
    for r in results:
        if r["conf"] == 0:
            stages[r["info"].get("stage")] = stages.get(r["info"].get("stage"), 0) + 1
    print("    rejected at stage:", stages)
    if args.dump:
        Path(args.dump).write_text(json.dumps(results))
    if args.save_fails:
        box_of = {str(Path(r.get("original") or r["image"])): b for (r, b) in jobs}
        kinds = {
            "false_accept": [r for r in results if r["conf"] > 0 and r["err"] > 0.5],
            "rejected": [r for r in results if r["conf"] == 0],
        }
        for kind, items in kinds.items():
            items.sort(key=lambda r: -(r["err"] or 99))
            step = max(1, len(items) // args.save_fails)
            for i, r in enumerate(items[::step][:args.save_fails]):
                p = OUT / f"{tag}_{args.prior}_{kind}_{i}.jpg"
                _draw(r, p, box_of.get(r["src"]))
                print("    saved", p)


if __name__ == "__main__":
    main()
