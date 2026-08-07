"""THE DECISIVE PLOT: does global's absolute rate scale with prefix length?

Not the aggregate effect. The aggregate can move for reasons unrelated to the
hypothesis, and it already varies by +/- 0.62pp across seeds.

## What was measured, and what it predicts

M7 found the two arms exploit DISJOINT information sources:

    causal   climbs with prefix length (1.3% -> 7.8%), ignores the pin
    global   set by whether a pin exists (~3-5% with, ~1% without),
             FLAT in prefix length

Global does not use the prefix at all, even where the prefix is informative.

The candidate cause is regime B's erasure shape. Under `random`, mean |S| = 2.24
of ~4.4 real blocks, so roughly half the prefix is itself erased on a typical
refine step and no reliable clean-prefix mapping exists to be learned.

    If that is right, an arm trained with the prefix left CLEAN
    (`single`, `contiguous`) should show global's rate CLIMBING with
    prefix length, the way causal's already does.

Prefix length is the corrupted block's INDEX — how many clean blocks precede the
erased one. `n_steps` is the wrong axis: in the root condition prefix length
equals trace length, but in the `early` condition the corrupted block is b0-b3
and its prefix is short however long the trace.

## Reading the result

The six-seed spread on the aggregate effect is +/- 0.62pp, so a difference below
~1.2pp between arms is scatter and is reported as scatter. The slope is the claim;
the aggregate is context.

Usage:
    python scripts/m6_prefix_use.py runs/seed0_early.json runs/Ssingle_early.json ...
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from adze.eval.strata import cell

SEED_SPREAD = 0.0062      # sample sd of the raw effect over six seeds
MIN_CELL = 100            # below this a per-block rate is not worth printing


def _slope(xs: list[float], ys: list[float]) -> float | None:
    """Least-squares slope of y on x, in percentage points per block."""
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("early", nargs="+", type=Path)
    args = p.parse_args()

    print("PREFIX LENGTH = the corrupted block's index — how many CLEAN blocks")
    print("precede the erased one. The `early` condition, where a pin exists.\n")

    summary = []
    for path in args.early:
        final_path = Path(str(path).replace("_early.json", "_final.json"))
        blob = json.loads(path.read_text())
        recs = blob["records"]
        label = re.sub(r"_early\.json$", "", path.name)

        blocks, c_rates, g_rates = [], [], []
        rows = []
        for b in sorted({r["block"] for r in recs}):
            sub = [r for r in recs if r["block"] == b]
            if len(sub) < MIN_CELL:
                continue
            c = cell(str(b), sub, "result")
            blocks.append(float(b))
            c_rates.append(c.causal)
            g_rates.append(c.glob)
            rows.append((b, c.n, c.causal, c.glob, c.gap))

        print("=" * 78)
        print(f"{label}")
        print("=" * 78)
        print(f"  {'prefix':>7} {'n':>6} {'causal':>9} {'global':>9} {'gap':>9}")
        for b, n, cc, gg, gap in rows:
            print(f"  {b:>7} {n:>6} {cc:>9.1%} {gg:>9.1%} {gap:>+9.1%}")

        cs, gs = _slope(blocks, c_rates), _slope(blocks, g_rates)
        agg = cell("all", recs, "result")
        hcp = (cell("all", json.loads(final_path.read_text())["records"], "result").gap
               if final_path.exists() else None)
        if cs is not None:
            print(f"\n  slope per block   causal {cs * 100:>+6.2f}pp   "
                  f"global {gs * 100:>+6.2f}pp   <- THE CLAIM IS THE GLOBAL SLOPE")
        print(f"  aggregate         effect {agg.gap:>+6.2%}   "
              f"global abs {agg.glob:>6.2%}   chance {agg.chance:.2%}"
              + (f"   handicap {hcp:+.2%}" if hcp is not None else ""))
        print()
        summary.append({"label": label, "c_slope": cs, "g_slope": gs,
                        "effect": agg.gap, "glob": agg.glob,
                        "handicap": hcp})

    print("=" * 78)
    print("ACROSS ARMS")
    print("=" * 78)
    print(f"  {'arm':>22} {'global slope':>13} {'causal slope':>13} "
          f"{'effect':>9} {'global abs':>11}")
    for s in summary:
        gs = "  n/a" if s["g_slope"] is None else f"{s['g_slope']*100:+.2f}pp"
        cs = "  n/a" if s["c_slope"] is None else f"{s['c_slope']*100:+.2f}pp"
        print(f"  {s['label']:>22} {gs:>13} {cs:>13} {s['effect']:>+9.2%} "
              f"{s['glob']:>11.2%}")

    if len(summary) > 1:
        base = summary[0]
        print(f"\n  Differences against {base['label']}, judged against the six-seed")
        print(f"  spread of +/- {SEED_SPREAD:.2%} on the aggregate effect:")
        for s in summary[1:]:
            d = s["effect"] - base["effect"]
            verdict = ("SCATTER — below ~2 sd, do not report as a difference"
                       if abs(d) < 2 * SEED_SPREAD else "EXCEEDS the seed spread")
            print(f"    {s['label']:>22}  effect {d:>+7.2%}   {verdict}")
    print()
    print("  The aggregate is context. The claim is whether GLOBAL'S SLOPE in")
    print("  prefix length turns positive — currently it is flat while causal's")
    print("  climbs, which is what 'disjoint sources' means.")


if __name__ == "__main__":
    main()
