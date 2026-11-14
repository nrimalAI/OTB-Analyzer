"""Two models on the same photos, photo by photo, from `app_check --verbose`.

    python compare_app_check.py runs/app_check/final/now_fenit_B.txt \
                                runs/app_check/final/render3_fenit_B.txt

Each file is the output of `app_check --models <folder> --list <photos>.json
--verbose` for one pair of models on one list of photos, made with the same
build of `app_check`. The first file is the pair to beat, the second the
candidate. Prints on how many photos the candidate has more squares right and
on how many fewer (sign test), the mean difference in points of squares with
a bootstrap interval over photos, and the boards gained and lost.

This is the test the rule for replacing a model in `Models/` is checked with
("beat the ones in `Models/` on the held-out board by more than the noise"):
the count of boards read fully right moves too little on 100 photos to show
a difference, the squares of each photo do. The photos are taken as
independent, which photos of one board are not quite: a difference just over
the line is not a result.

Photos are paired by their place in the file, not by name (the chess-cv
games repeat file names from one game folder to the next), so both files
must come from the same list; the script stops if the names disagree.
"""

import argparse
import math
import random
import re
import sys

LINE = re.compile(r"^(.*): (\d+)/64 ")


def rows(path):
    """(photo name, squares right of 64) for every photo line of the file. A
    photo whose board was not found is listed by app_check with 0/64."""
    out = []
    with open(path, errors="replace") as f:
        for line in f:
            m = LINE.match(line)
            if m:
                out.append((m.group(1), int(m.group(2))))
    return out


def sign_test(a, b):
    """Two-sided p of a split a against b, were each side as likely."""
    n = a + b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def compare(base, candidate, draws=10000, seed=1):
    if [name for name, _ in base] != [name for name, _ in candidate]:
        raise ValueError("the two files do not list the same photos in the same order")
    diffs = [c - b for (_, b), (_, c) in zip(base, candidate)]
    n = len(diffs)
    rng = random.Random(seed)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n / 64 * 100
                   for _ in range(draws))
    return {
        "photos": n,
        "more": sum(d > 0 for d in diffs),
        "fewer": sum(d < 0 for d in diffs),
        "mean": sum(diffs) / n / 64 * 100,
        "low": means[int(draws * 0.025)],
        "high": means[int(draws * 0.975) - 1],
        "gained": sum(b < 64 and c == 64 for (_, b), (_, c) in zip(base, candidate)),
        "lost": sum(b == 64 and c < 64 for (_, b), (_, c) in zip(base, candidate)),
        "base_squares": sum(b for _, b in base) / n / 64 * 100,
        "candidate_squares": sum(c for _, c in candidate) / n / 64 * 100,
        "base_exact": sum(b == 64 for _, b in base),
        "candidate_exact": sum(c == 64 for _, c in candidate),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("base", help="app_check --verbose output of the pair to beat")
    ap.add_argument("candidate", help="the same for the candidate, on the same list of photos")
    ap.add_argument("--draws", type=int, default=10000, help="bootstrap draws")
    args = ap.parse_args()
    base, candidate = rows(args.base), rows(args.candidate)
    if not base or len(base) != len(candidate):
        sys.exit(f"{len(base)} photos in {args.base}, {len(candidate)} in {args.candidate}: "
                 "both must be --verbose outputs of the same list")
    try:
        r = compare(base, candidate, args.draws)
    except ValueError as e:
        sys.exit(str(e))
    print(f"{r['photos']} photos. Squares right: {r['base_squares']:.2f}% -> {r['candidate_squares']:.2f}%; "
          f"boards exact: {r['base_exact']} -> {r['candidate_exact']}")
    print(f"squares: more right on {r['more']} photos, fewer on {r['fewer']} "
          f"(sign test p = {sign_test(r['more'], r['fewer']):.2g}); "
          f"{r['mean']:+.2f} points, 95% bootstrap interval {r['low']:+.2f} to {r['high']:+.2f}")
    print(f"boards exact: {r['gained']} gained, {r['lost']} lost "
          f"(sign test p = {sign_test(r['gained'], r['lost']):.2g})")


if __name__ == "__main__":
    main()
