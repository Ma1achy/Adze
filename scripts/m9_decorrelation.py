"""Verify the decorrelating generator before anything is trained on it.

The generator exists because a distance profile on `generate.py` was a provenance
profile. So the acceptance criterion is not "the code runs" — it is that
**composition can no longer produce a distance profile on its own.**

That is what the last table measures. Using the measured per-class gaps, it asks
what gap each distance would show if distance had NO effect whatever and only the
class mix varied. On the original generator that number swings +2.79 -> +0.75, and
the entire observed profile fell inside it. Here it should be nearly flat, and the
residual swing is the largest artifact a distance profile could contain.

Usage:
    python scripts/m9_decorrelation.py
"""

from __future__ import annotations

import argparse
from collections import Counter

from adze.data.decorrelated import DISTANCE_MAX, generate_decorrelated_dataset
from adze.data.generate import generate_dataset
from adze.eval.strata import consumer_distance, operand_provenance

CLASSES = ("both-leaves", "one-leaf", "both-from-earlier")

# Measured over six L4 seeds, RESULT gap by provenance class. These are what turn
# a composition imbalance into a spurious effect size.
CLASS_GAP = {"both-leaves": 0.0300, "one-leaf": 0.0210, "both-from-earlier": -0.0286}


def crosstab(traces, interior: bool, min_cell: int = 100):
    cells: Counter = Counter()
    totals: Counter = Counter()
    for t in traces:
        n = len(t.steps)
        for b in range(n):
            if interior and (b < 2 or b + DISTANCE_MAX > n - 1):
                continue
            d = consumer_distance(t, b)
            if d is None:
                continue
            cells[(d, operand_provenance(t.steps[b]))] += 1
            totals[d] += 1
    ds = [d for d in sorted(totals) if totals[d] >= min_cell]
    return cells, totals, ds


def report(label: str, traces, interior: bool) -> tuple[float, float]:
    cells, totals, ds = crosstab(traces, interior)
    tot = sum(totals.values())
    marg = {c: sum(cells[(d, c)] for d in totals) / tot for c in CLASSES}

    print("=" * 84)
    print(label)
    print("=" * 84)
    print(f"  {'d':>3} {'n':>8} " + " ".join(f"{c:>18}" for c in CLASSES)
          + f" {'predicted gap':>14}")
    # Distances beyond the bound exist only because every slot inside it was
    # full. They are a capacity artifact, not part of the designed range, and are
    # printed but excluded from the swing.
    usable = [d for d in ds if not interior or d < DISTANCE_MAX]
    preds = []
    for d in ds:
        shares = {c: cells[(d, c)] / totals[d] for c in CLASSES}
        pred = sum(shares[c] * CLASS_GAP[c] for c in CLASSES)
        if d in usable:
            preds.append(pred)
        flag = "" if d in usable else "   overshoot, excluded"
        print(f"  {d:>3} {totals[d]:>8} "
              + " ".join(f"{shares[c]:>17.1%}" for c in CLASSES)
              + f" {pred:>+14.2%}{flag}")
    print(f"  {'ALL':>3} {tot:>8} " + " ".join(f"{marg[c]:>17.1%}" for c in CLASSES))

    chi = sum((cells[(d, c)] - totals[d] * marg[c]) ** 2 / (totals[d] * marg[c])
              for d in usable for c in CLASSES)
    dof = max((len(usable) - 1) * 2, 1)
    swing = max(preds) - min(preds)
    print(f"\n  chi2 {chi:.1f} on {dof} dof   per-dof {chi / dof:.1f}")
    print(f"  usable distances {usable[0]}..{usable[-1]}")
    print(f"  COMPOSITION-ONLY SWING across d: {swing:.2%}")
    print(f"    <- the largest spurious distance profile this data can produce\n")
    return chi / dof, swing


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--traces", type=int, default=4000)
    args = p.parse_args()

    c_old, s_old = report(
        "ORIGINAL generate.py — what the withdrawn window was measured on",
        generate_dataset(args.traces, seed=0, max_depth=3), interior=False)
    c_new, s_new = report(
        f"DECORRELATED, interior band (index >= 2, index + {DISTANCE_MAX} <= n-1)",
        generate_decorrelated_dataset(args.traces, seed=0), interior=True)

    print("=" * 84)
    print("VERDICT")
    print("=" * 84)
    print(f"  per-dof chi2   {c_old:>8.1f} -> {c_new:>6.1f}   "
          f"({c_old / c_new:.0f}x flatter)")
    print(f"  spurious swing {s_old:>8.2%} -> {s_new:>6.2%}   "
          f"({s_old / s_new:.0f}x smaller)")
    print()
    print(f"  The measured effect is ~2.5pp. A composition artifact of {s_new:.2%}")
    print(f"  cannot manufacture a profile at that scale, so a distance profile")
    print(f"  measured on the interior band means what it says.")
    print()
    print("  NOT a claim that the ends are clean. Near either end, position bounds")
    print("  both variables and no construction avoids it — those steps are")
    print("  excluded by the band, not fixed.")


if __name__ == "__main__":
    main()
