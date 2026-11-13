"""Corner labels for the Roboflow photos, derived without a model.

    python derive_corners.py validate     # the a1 rule on Fenit A + ChessReD train
    python derive_corners.py make         # -> datasets/rf_corners + contact sheets
    python derive_corners.py make --prior box --limit 400 --dry-run   # compare priors

The Roboflow projects have piece boxes but no board corners. For each photo
build_dataset.load_roboflow yields (the same photos, de-duplicated and capped,
that a merge uses):

  1. grid   grid_corners.find_grid finds the 8x8 grid; only results that pass
            its validity check are used. A corner outside the photo rejects
            it (a label there cannot be written).
  2. pieces every piece box is put on the grid the way evaluate.py does it
            (chess_pipeline.pulled_board_point: bottom-centre of the box,
            pull 0.30). The photo is rejected when
              - a piece lands outside the grid (board_square gives None), or
              - the grid is not pinned: shifting it one square towards any
                of its four sides would still hold every piece. A shift
                towards a side loses a piece only if a piece stands in the
                row of squares along the opposite side, so the test is: all
                four outer rows (file a, file h, rank 1, rank 8) hold a
                piece. A grid found one square off (the detector's one known
                wrong answer) leaves an outer row empty or a piece outside.
            --pinned N keeps photos with at least N of the 4 sides pinned
            (default 4).
  3. a1     the corner order must be a1, a8, h8, h1. Two cues:
              colour  a1 is a dark square. Read from the warped grid, this
                      leaves two of the four rotations of the clockwise
                      corner ring, half a turn apart.
              pieces  White stands towards rank 1 and Black towards rank 8:
                      each piece scores +1 at its own end .. -1 at the far
                      end, pawns count 3x, a pawn on rank 1 or 8 costs 3
                      more. Scored for all four rotations.
            Projects whose pieces are scattered at random (--skip-projects)
            are left out: there the pieces cue means nothing.
            Kept only if the rotation the pieces like best of all four is
            one the colours allow (otherwise the photo is mirrored, the
            board is turned a quarter, or the pieces say nothing), it beats
            the other colour-valid one by --margin, and there are at least
            --min-pieces pieces. Note that the colour cue alone can never
            show a mirrored photo (a mirrored board still has two dark
            corners); the disagreement of the two cues does.

  4. thin   video frames: of kept frames of one project with the same pieces
            on the same squares and all four corners within 1% of the image
            size of each other, one stays.

chess-vision-ljby5 contains copies of the chesscog renders and the ChessReD
photos (file names chesscog_*, chessred2k_*), test photos included; they are
left out (--exclude-prefix).

Output: datasets/rf_corners/corners/{images,labels} in the `native` layout
(images are symlinks to the Roboflow files; labels are YOLO pose rows like
datasets/synth), datasets/rf_corners/derived.json with one row per kept
photo, and a contact sheet per project in runs/grid/rf_<project>.jpg.
CPU only, at most 4 processes.
"""

import argparse
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from chess_pipeline import PIECE_CLASSES, board_square, homography, pulled_board_point
from grid_corners import _cell_values, find_grid

HERE = Path(__file__).resolve().parent
OUT = HERE / "datasets" / "rf_corners"
SHEETS = HERE / "runs" / "grid"
PULL = 0.30


# ------------------------------------------------------------------ the rules

def piece_points(H, pieces):
    """Board coordinates of each piece's contact point, as evaluate.py maps it."""
    return [pulled_board_point(H, ((x1 + x2) / 2, y2), PULL) for _, (x1, y1, x2, y2) in pieces]


def check_pieces(corners, pieces):
    """(pieces outside the grid, sides pinned 0..4) for a grid in any rotation."""
    H = homography(corners)
    outside = sum(board_square(H, ((x1 + x2) / 2, y2), PULL) is None
                  for _, (x1, y1, x2, y2) in pieces)
    pts = np.array(piece_points(H, pieces)).reshape(-1, 2)
    pinned = int((pts[:, 0] < 1).any()) + int((pts[:, 0] >= 7).any()) \
        + int((pts[:, 1] < 1).any()) + int((pts[:, 1] >= 7).any())
    return int(outside), pinned


def dark_corner_rotations(image_bgr, ring):
    """Which rotations of the clockwise ring put a dark square on a1, from
    the colours of the 64 cells. Returns (rotations, strength): rotations is
    (0, 2) or (1, 3), strength in 0..1 is how cleanly the two cell colours
    split; (None, strength) when they do not."""
    s = min(1.0, 1000 / max(image_bgr.shape[:2]))
    small = cv2.resize(image_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else image_bgr
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    H = homography(np.asarray(ring) * s)
    v = _cell_values(gray, H, 0, 0, 8, 8)            # [rank, file, 4 patches]
    cell = np.nanmedian(v, axis=2)
    ok = ~np.isnan(cell)
    if ok.sum() < 40:
        return None, 0.0
    gi, gj = np.meshgrid(np.arange(8), np.arange(8))
    even = (gi + gj) % 2 == 0                        # a1's colour for rotation 0
    lo, hi = np.percentile(cell[ok], [25, 75])
    if hi - lo < 15:
        return None, 0.0
    light = cell > (lo + hi) / 2
    share_even, share_odd = light[ok & even].mean(), light[ok & ~even].mean()
    strength = abs(share_even - share_odd)
    if strength < 0.5:
        return None, float(strength)
    # rotation 0 is valid when the cells of a1's colour are the dark ones;
    # a quarter turn swaps the two colours
    return ((0, 2) if share_even < share_odd else (1, 3)), float(strength)


def piece_score(ordered, pieces):
    """How well the pieces fit `ordered` as a1, a8, h8, h1: +1 when every
    piece stands at its own end of the board, -1 at the wrong end."""
    H = homography(ordered)
    total = weight = 0.0
    for (cls, _), (_, by) in zip(pieces, piece_points(H, pieces)):
        rank = min(max(int(np.floor(by)), 0), 7)
        white, pawn = cls.startswith("white"), cls.endswith("pawn")
        w = 3.0 if pawn else 1.0
        t = (3.5 - rank) / 3.5 if white else (rank - 3.5) / 3.5
        if pawn and rank in (0, 7):
            t -= 3.0
        total += w * t
        weight += w
    return total / weight if weight else 0.0


def choose_a1(image_bgr, ring, pieces, margin, min_pieces):
    """(ordered corners or None, reason, details). ring: 4 corners clockwise."""
    ring = np.asarray(ring, float)
    valid, strength = dark_corner_rotations(image_bgr, ring)
    info = {"colour_strength": round(strength, 3), "pieces": len(pieces)}
    if valid is None:
        return None, "a1: square colours unclear", info
    if len(pieces) < min_pieces:
        return None, "a1: too few pieces", info
    scores = [piece_score(np.roll(ring, -k, axis=0), pieces) for k in range(4)]
    best = int(np.argmax(scores))
    info["scores"] = [round(float(x), 3) for x in scores]
    if best not in valid:
        return None, "a1: pieces and colours disagree (mirrored or turned board)", info
    other = (best + 2) % 4
    info["a1_margin"] = round(float(scores[best] - scores[other]), 3)
    if info["a1_margin"] < margin:
        return None, "a1: pieces do not say which end is White's", info
    return np.roll(ring, -best, axis=0), "kept", info


def derive(image_bgr, pieces, args, prior_box=None):
    """pieces: [(class name, (x1, y1, x2, y2) in pixels)].
    Returns dict(status, corners (a1 a8 h8 h1) or None, ...)."""
    ring, conf = find_grid(image_bgr, prior_box=prior_box)
    out = {"status": "no grid accepted", "stage": 0, "confidence": round(float(conf), 3)}
    if ring is None:
        return out
    out["stage"] = 1
    out["ring"] = ring.tolist()
    h, w = image_bgr.shape[:2]
    tol = 0.005
    if (ring[:, 0] < -tol * w).any() or (ring[:, 0] > (1 + tol) * w).any() \
            or (ring[:, 1] < -tol * h).any() or (ring[:, 1] > (1 + tol) * h).any():
        out["status"] = "a corner is outside the photo"
        return out
    ring = np.clip(ring, 0, [w, h])
    # every rule is evaluated (the report counts what a looser setting
    # would keep); the status is the first one that fails
    outside, pinned = check_pieces(ring, pieces)
    out["outside"], out["pinned"] = outside, pinned
    ordered, reason, info = choose_a1(image_bgr, ring, pieces, args.margin, args.min_pieces)
    out.update(info)
    out["a1_ok"] = ordered is not None
    if outside:
        out["status"] = "pieces outside the grid"
    elif pinned < args.pinned:
        out["status"] = "grid not pinned by the pieces"
    else:
        out["stage"] = 2
        out["status"] = reason
        if ordered is not None:
            out["stage"] = 3
            out["corners"] = ordered.tolist()
    return out


# -------------------------------------------------------------------- workers

def _work(job):
    cv2.setNumThreads(1)
    path, pieces, args, extra = job
    img = cv2.imread(str(path))
    if img is None:
        return {"status": "image cannot be read", "stage": 0, **extra}
    h, w = img.shape[:2]
    if extra.get("normalized"):
        pieces = [(c, ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
                  for c, (cx, cy, bw, bh) in pieces]
    t = time.time()
    res = derive(img, pieces, args, extra.get("prior_box"))
    res.update(extra)
    res["time"] = time.time() - t
    res["size"] = (w, h)
    return res


def _run(jobs, procs):
    from multiprocessing import Pool
    with Pool(min(procs, 4)) as pool:
        return list(pool.imap(_work, jobs, chunksize=4))


def _prior_boxes(paths, model_path):
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(4)
    model = YOLO(model_path)
    boxes = {}
    for p in paths:
        r = model(str(p), imgsz=640, device="cpu", verbose=False)[0]
        boxes[str(p)] = (r.boxes.xyxy[int(r.boxes.conf.argmax())].tolist()
                         if r.boxes is not None and len(r.boxes) else None)
    return boxes


# ------------------------------------------------------------------- validate

def validate(args):
    """The whole a1 choice where the truth is known. Fenit B is left out
    unless --source/--group ask for it."""
    from evaluate import merged_labels
    from grid_corners import corner_error
    rows = json.loads((HERE / "datasets/merged/positions.json").read_text())
    sets = ([(args.source, args.group, args.split)] if args.source
            else [("fenit", "A", "all"), ("chessred2k", None, "train")])
    for source, group, split in sets:
        jobs = []
        for row in rows:
            if row["source"] != source or (group and row.get("group") != group):
                continue
            if split != "all" and row["split"] != split:
                continue
            src = Path(row.get("original") or row["image"])
            corners, boxes = merged_labels(row, src)
            if not corners or not boxes:
                continue
            pieces = [(d.cls, tuple(d.bbox)) for d in boxes]
            jobs.append((src, pieces, args, {"truth": corners, "src": str(src)}))
        if args.limit:
            random.Random(0).shuffle(jobs)
            jobs = jobs[:args.limit]
        results = _run(jobs, args.procs)
        name = f"{source}{' ' + group if group else ''} {split}"
        n = len(results)
        stage = Counter(r["stage"] for r in results)
        print(f"\n{name}: {n} photos; grid accepted {n - stage[0]}, "
              f"passed piece check {stage[2] + stage[3]}, passed a1 rule (kept) {stage[3]}")
        for status, c in Counter(r["status"] for r in results).most_common():
            print(f"    {c:5d}  {status}")
        # among photos that reached the a1 step: is the chosen rotation right?
        right = wrong = 0
        errs = []
        for r in results:
            if r["stage"] != 3:
                continue
            truth = np.array(r["truth"])
            c = np.array(r["corners"])
            to_board = homography(truth)
            from chess_pipeline import project
            on_board = np.array([project(to_board, p) for p in c])
            target = np.array([[0, 0], [0, 8], [8, 8], [8, 0]], float)
            e = [np.linalg.norm(on_board - np.roll(target, -k, axis=0), axis=1).mean()
                 for k in range(4)]
            errs.append(min(e))
            if int(np.argmin(e)) == 0:
                right += 1
            else:
                wrong += 1
        if right + wrong:
            errs = np.array(errs)
            print(f"    kept: a1 is the labelled a1 in {right} of {right + wrong} "
                  f"({right / (right + wrong):.1%}); corner error median {np.median(errs):.3f}, "
                  f"over 0.5 squares: {(errs > 0.5).sum()}, over 1: {(errs > 1).sum()}")
        if args.dump:
            Path(args.dump).parent.mkdir(parents=True, exist_ok=True)
            with open(f"{args.dump}_{source}_{group or split}.json", "w") as f:
                json.dump(results, f)


# ----------------------------------------------------------------------- make

def roboflow_samples(caps=None):
    """The Roboflow photos as `build_dataset.py merge` takes them. caps:
    {project: N} replaces that project's max_images for this run only (0 = no
    cap); sources.yaml is not touched. load_roboflow takes the first N of a
    fixed shuffle of the project's files, so a larger N is a superset of a
    smaller one and is spread over the whole project."""
    import yaml
    from build_dataset import load_roboflow
    cfg = yaml.safe_load((HERE / "sources.yaml").read_text())["sources"]["roboflow"]
    for name, n in (caps or {}).items():
        entry = next((e for e in cfg["projects"] if e["id"].split("/")[-1] == name), None)
        if entry is None:
            raise SystemExit(f"--cap: no project {name} in sources.yaml")
        entry["max_images"] = n or 10 ** 9
    skipped = Counter()
    seen = {"names": set(), "hashes": defaultdict(list)}
    samples = list(load_roboflow("roboflow", cfg, HERE / cfg["dir"], skipped, seen))
    return samples, skipped, cfg.get("project_licenses", {})


def project_of(sample, projects):
    return next(p for p in sorted(projects, key=len, reverse=True)
                if sample.stem.startswith(p + "_"))


def contact_sheet(rows, path, count=12, seed=0):
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    tiles = []
    for r in rows[:count]:
        img = cv2.imread(r["original"])
        c = np.array(r["corners"])
        s = 420 / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        c = (c * s).astype(int)
        cv2.polylines(img, [c.reshape(-1, 1, 2)], True, (0, 0, 255), 2)
        cv2.line(img, tuple(c[0]), tuple(c[3]), (0, 255, 255), 3)      # rank 1: White's edge
        cv2.circle(img, tuple(c[0]), 8, (0, 255, 0), -1)
        cv2.putText(img, "a1", (c[0][0] + 8, c[0][1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(img, "h1", (c[3][0] + 8, c[3][1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 2, cv2.LINE_AA)
        tile = np.zeros((420, 420, 3), np.uint8)
        tile[:img.shape[0], :img.shape[1]] = img
        tiles.append(tile)
    if not tiles:
        return False
    while len(tiles) % 4:
        tiles.append(np.zeros((420, 420, 3), np.uint8))
    sheet = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return True


def video_of(stem, project):
    """The part of a file name that names its video or photo series: the
    name without the Roboflow hash and the trailing frame number."""
    name = stem[len(project) + 1:]
    name = re.sub(r"\.rf\.[0-9a-zA-Z]+$", "", name)
    name = re.sub(r"_(jpe?g|png)$", "", name, flags=re.I)
    return re.sub(r"[-_ ]?\d+$", "", name)


def assignment(corners, pieces, size):
    """Which piece stands on which square, as a sorted tuple."""
    w, h = size
    H = homography(corners)
    return tuple(sorted(
        (cls, board_square(H, (cx * w, (cy + bh / 2) * h), PULL) or (-1, -1))
        for cls, (cx, cy, bw, bh) in pieces))


def thin(rows, pieces_of, tolerance=0.01):
    """Drop near-duplicates among kept frames: same project, the same pieces
    on the same squares and all four corners within `tolerance` of the image
    size. The first of each such group (by file name) stays."""
    kept, groups = [], defaultdict(list)
    for r in sorted(rows, key=lambda r: r["stem"]):
        c = np.array(r["corners"]) / r["size"]
        key = (r["project"], assignment(r["corners"], pieces_of[r["stem"]], r["size"]))
        if any(np.abs(c - other).max() <= tolerance for other in groups[key]):
            continue
        groups[key].append(c)
        kept.append(r)
    return kept


def make(args):
    t_start = time.time()
    caps = {}
    for item in args.cap or []:
        name, _, n = item.partition("=")
        caps[name] = int(n)
    samples, skipped, licenses = roboflow_samples(caps)
    projects = sorted(p.name for p in (HERE / "datasets/roboflow").iterdir() if p.is_dir())
    excluded = Counter()
    keep = []
    for s in samples:
        proj = project_of(s, projects)
        name = s.stem[len(proj) + 1:]
        hit = next((x for x in args.exclude_prefix if name.startswith(x)), None)
        if hit:
            excluded[(proj, hit)] += 1
        else:
            keep.append(s)
    samples = keep
    if args.limit:
        random.Random(0).shuffle(samples)
        samples = samples[:args.limit]
    boxes = {}
    if args.prior == "box":
        boxes = _prior_boxes([s.image for s in samples], args.prior_model)
    jobs = []
    for s in samples:
        pieces = [(PIECE_CLASSES[c], (cx, cy, w, h)) for c, cx, cy, w, h in s.boxes]
        jobs.append((s.image, pieces, args, {
            "normalized": True, "stem": s.stem, "original": str(s.image),
            "project": project_of(s, projects), "prior_box": boxes.get(str(s.image))}))
    t0 = time.time()
    cache = SHEETS / "rf_results.json"
    done = {}
    if args.reuse and cache.exists():
        # results of earlier runs, by photo (rules as of the run that made them)
        done = {r["stem"]: r for r in json.loads(cache.read_text())}
    todo = [j for j in jobs if j[3]["stem"] not in done]
    fresh = {r["stem"]: r for r in _run(todo, args.procs)} if todo else {}
    results = [done.get(j[3]["stem"]) or fresh[j[3]["stem"]] for j in jobs]
    took = time.time() - t0
    pieces_of = {j[3]["stem"]: j[1] for j in jobs}
    for (proj, prefix), n in sorted(excluded.items()):
        print(f"left out before detection: {n} files of {proj} named {prefix}* (--exclude-prefix)")
    print(f"detected {len(todo)} photos now, {len(jobs) - len(todo)} taken from the earlier run")

    print(f"{len(results)} photos in {took:.0f} s ({took / max(len(results), 1):.2f} s each, "
          f"{min(args.procs, 4)} processes), prior={args.prior}")
    print(f"{'project':22} {'seen':>5} {'grid':>5} {'pieces':>6} {'a1':>5} {'kept':>5}")
    by = defaultdict(list)
    for r in results:
        by[r["project"]].append(r)
    total_kept = 0
    for p in projects:
        rs = by.get(p, [])
        st = Counter(r["stage"] for r in rs)
        kept = 0 if p in args.skip_projects else st[3]
        total_kept += kept
        print(f"{p:22} {len(rs):5d} {len(rs) - st[0]:5d} {st[2] + st[3]:6d} {st[3]:5d} {kept:5d}"
              + ("   (project left out: --skip-projects)" if p in args.skip_projects else ""))
        for status, c in Counter(r["status"] for r in rs).most_common():
            if status != "kept":
                print(f"      {c:5d}  {status}")
    st = Counter(r["stage"] for r in results)
    print(f"{'all':22} {len(results):5d} {len(results) - st[0]:5d} {st[2] + st[3]:6d} "
          f"{st[3]:5d} {total_kept:5d}")
    for k, v in skipped.items():
        if not k.startswith("note: "):
            print(f"  load_roboflow skipped {v}: {k}")
    alt = Counter()
    for r in results:
        if r.get("a1_ok") and not r.get("outside"):
            for n in (0, 2, 3, 4):
                alt[n] += r["pinned"] >= n
    print("kept at other --pinned settings: " + ", ".join(f"{n}: {alt[n]}" for n in (0, 2, 3, 4)))
    if args.dry_run:
        return
    SHEETS.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(results))

    for kind in ("images", "labels"):
        d = OUT / "corners" / kind
        d.mkdir(parents=True, exist_ok=True)
        for old in d.iterdir():        # only this script's own earlier output
            old.unlink()
    final = [r for r in results if r["stage"] == 3 and r["project"] not in args.skip_projects]
    before = Counter(r["project"] for r in final)
    if not args.no_thin:
        final = thin(final, pieces_of)
    after = Counter(r["project"] for r in final)
    print("near-duplicate frames thinned (kept before -> after):")
    for p in projects:
        if before[p]:
            videos = {video_of(r["stem"], p) for r in final if r["project"] == p}
            print(f"  {p:22} {before[p]:5d} -> {after[p]:5d}   "
                  f"({len(videos)} distinct file-name series)")
    derived = []
    for r in final:
        w, h = r["size"]
        c = np.array(r["corners"]) / [w, h]
        xs, ys = c[:, 0], c[:, 1]
        pad = 0.03
        x1, x2 = max(xs.min() - pad, 0), min(xs.max() + pad, 1)
        y1, y2 = max(ys.min() - pad, 0), min(ys.max() + pad, 1)
        kpts = " ".join(f"{x:.6f} {y:.6f} 2" for x, y in c)
        src = Path(r["original"]).resolve()   # the real file, not a path through a worktree
        (OUT / "corners" / "labels" / f"{r['stem']}.txt").write_text(
            f"0 {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f} {kpts}")
        os.symlink(src, OUT / "corners" / "images" / f"{r['stem']}{src.suffix}")
        derived.append({"stem": r["stem"], "project": r["project"], "original": str(src),
                        "license": licenses.get(r["project"]),
                        "series": video_of(r["stem"], r["project"]),
                        "confidence": r["confidence"], "a1_margin": r["a1_margin"],
                        "colour_strength": r["colour_strength"], "pieces": r["pieces"],
                        "pinned": r["pinned"], "corners": r["corners"]})
    (OUT / "derived.json").write_text(json.dumps(derived, indent=1))
    print(f"wrote {len(derived)} labels to {OUT}; total run time {time.time() - t_start:.0f} s")
    SHEETS.mkdir(parents=True, exist_ok=True)
    for p in projects:
        rows = [d for d in derived if d["project"] == p]
        sheet = SHEETS / f"rf_{p}.jpg"
        if contact_sheet(rows, sheet, count=24 if len(rows) >= 1000 else 12):
            print(f"  contact sheet {sheet} ({len(rows)} kept)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("validate", "make"):
        p = sub.add_parser(name)
        p.add_argument("--pinned", type=int, default=4,
                       help="sides of the grid the pieces must pin (0-4)")
        p.add_argument("--margin", type=float, default=0.5,
                       help="piece score by which the chosen a1 must beat the half-turn")
        p.add_argument("--min-pieces", type=int, default=6)
        p.add_argument("--procs", type=int, default=4)
        p.add_argument("--limit", type=int, default=0)
        if name == "validate":
            p.add_argument("--source")
            p.add_argument("--group")
            p.add_argument("--split", default="all")
            p.add_argument("--dump", help="prefix for per-photo json")
        else:
            p.add_argument("--prior", default="none", choices=["none", "box"])
            p.add_argument("--prior-model",
                           default=str(HERE / "runs/pose/corners_s_exp1/weights/last.pt"))
            p.add_argument("--dry-run", action="store_true", help="count only, write nothing")
            p.add_argument("--reuse", action="store_true",
                           help="take photos already in runs/grid/rf_results.json from "
                                "there and detect only the new ones")
            p.add_argument("--cap", nargs="*", metavar="PROJECT=N",
                           help="this many images of a project instead of its max_images "
                                "in sources.yaml (0 = all), for this run only")
            p.add_argument("--exclude-prefix", nargs="*", default=["chesscog", "chessred2k"],
                           help="file-name prefixes to leave out: chess-vision-ljby5 holds "
                                "copies of the chesscog renders and the ChessReD photos, "
                                "test photos included")
            p.add_argument("--no-thin", action="store_true",
                           help="keep near-duplicate frames")
            p.add_argument("--skip-projects", nargs="*",
                           default=["chess-bprbi", "chess-f0mbu"],
                           help="projects whose pieces are scattered at random, so they "
                                "cannot say where a1 is; their labels are not written")
    args = ap.parse_args()
    {"validate": validate, "make": make}[args.cmd](args)


if __name__ == "__main__":
    main()
