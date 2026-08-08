"""The distance profile on DECORRELATED data — the measurement the generator is for.

The standard readout since session 22 is not a distance profile on its own. It is
a distance profile **beside what composition alone would predict**, because on the
tree generator those two were the same number and nobody noticed for days.

That comparison stays in the reporting path here even though the generator is
built to make it uninformative. If the composition prediction ever tracks the
observed profile again, something has recoupled and the result is not about
distance.

Only the INTERIOR BAND is analysed — steps at index >= 2 whose consumer draw had
the full bounded range available. Near either end position bounds both variables
and no construction avoids it; those steps are excluded, not repaired. See
docs/paper.md §11.

Usage:
    python scripts/m9_profile.py runs/dec10_early.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from adze.eval.strata import cell

CLASSES = ("both-leaves", "one-leaf", "both-from-earlier")
MIN_CELL = 150


def _mean_se(xs):
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    sd = (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5
    return m, sd / n ** 0.5


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dumps", nargs="+", type=Path)
    p.add_argument("--distance-max", type=int, default=6)
    args = p.parse_args()

    per_seed_class: dict[str, list[float]] = {c: [] for c in CLASSES}
    per_seed_d: dict[int, list[float]] = {}
    shares: dict[int, dict[str, float]] = {}
    counts: dict[int, int] = {}

    for path in args.dumps:
        recs = json.loads(path.read_text())["records"]
        band = [r for r in recs
                if r["block"] >= 2 and r["block"] + args.distance_max <= r["n_steps"] - 1]
        print(f"{path.name}: {len(recs)} records, {len(band)} in the interior band")
        for c in CLASSES:
            sub = [r for r in band if r["provenance"] == c]
            if len(sub) >= MIN_CELL:
                per_seed_class[c].append(cell(c, sub, "result").gap)
        for d in sorted({r["distance"] for r in band if r["distance"]}):
            if d >= args.distance_max:
                continue
            sub = [r for r in band if r["distance"] == d]
            if len(sub) < MIN_CELL:
                continue
            per_seed_d.setdefault(d, []).append(cell(str(d), sub, "result").gap)
            counts[d] = len(sub)
            shares[d] = {c: sum(r["provenance"] == c for r in sub) / len(sub)
                         for c in CLASSES}

    print("\n" + "=" * 78)
    print("BY PROVENANCE — the axis that survived session 22")
    print("=" * 78)
    class_gap = {}
    print(f"  {'class':>20} {'gap':>10} {'+/- SE':>9} {'n runs':>7}")
    for c in CLASSES:
        m, se = _mean_se(per_seed_class[c])
        class_gap[c] = m
        print(f"  {c:>20} {m:>+10.2%} {se:>9.2%} {len(per_seed_class[c]):>7}")

    print("\n" + "=" * 78)
    print("BY DISTANCE — on data where distance is not a proxy for provenance")
    print("=" * 78)
    print(f"  {'d':>3} {'n':>7} {'observed':>19} {'composition-only':>18} "
          f"{'residual':>11}")
    preds = []
    for d in sorted(per_seed_d):
        m, se = _mean_se(per_seed_d[d])
        pred = sum(shares[d][c] * class_gap[c] for c in CLASSES)
        preds.append(pred)
        print(f"  {d:>3} {counts[d]:>7} {m:>+12.2%} +/-{se:>5.2%} "
              f"{pred:>+18.2%} {m - pred:>+11.2%}")
    if preds:
        print(f"\n  composition-only swing across d: {max(preds) - min(preds):.2%}")
        print("    <- the largest profile composition alone could manufacture here")
    print(f"\n  Interior band only, distances 1..{args.distance_max - 1}.")
    if len(args.dumps) < 2:
        print("  ONE RUN — direction only. Shape claims wait for seeds.")


if __name__ == "__main__":
    main()
