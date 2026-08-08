"""Read the R sweep: does the window widen with refinement passes?

The claim is the DISTANCE PROFILE at each R, with between-seed error bars. The
aggregate rate is context — it can move because passes destroy context, which is
a different fact from the window moving.

Every cell is a mean over the six checkpoints with the between-seed sample sd,
per the standing rule. A shape read off one checkpoint is not a shape: seed 0 has
already supplied two that failed to replicate.

Usage:
    python scripts/m7_reach_read.py runs/reach6.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def mean_sd(xs: list[float]) -> tuple[float, float, float]:
    """Mean, sample sd (n-1), and SE of the mean."""
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    m = sum(xs) / n
    if n < 2:
        return m, 0.0, 0.0
    sd = (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5
    return m, sd, sd / n ** 0.5


def rate(recs: list[dict]) -> float | None:
    return sum(r["hit"] for r in recs) / len(recs) if recs else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dump", type=Path)
    p.add_argument("--min-cell", type=int, default=60,
                   help="skip a per-seed cell thinner than this; a rate over a "
                        "handful of traces is noise wearing a percentage sign")
    args = p.parse_args()

    blob = json.loads(args.dump.read_text())
    recs = blob["records"]
    ckpts = sorted({r["ckpt"] for r in recs})
    arms = sorted({r["arm"] for r in recs}, key=lambda a: ("only-b" in a, a))
    rs = sorted({r["R"] for r in recs})
    ds = sorted({r["distance"] for r in recs if r["distance"] is not None})

    print(f"{len(ckpts)} checkpoints, R = {rs}, arms {arms}")
    print(f"passes {blob['passes']}, nfe {blob['nfe']}, eta {blob['eta']}\n")

    def cells(arm, mask, R, pred):
        """Per-checkpoint rates for one condition, thin cells dropped."""
        out = []
        for c in ckpts:
            sub = [r for r in recs if r["ckpt"] == c and r["arm"] == arm
                   and r["mask"] == mask and r["R"] == R and pred(r)]
            if len(sub) >= args.min_cell:
                out.append(rate(sub))
        return out

    print("=" * 84)
    print("AGGREGATE — the level. Context, not the claim.")
    print("=" * 84)
    print(f"  {'arm':>8} {'R':>3} {'causal':>16} {'global':>16} {'GAP':>17}")
    for arm in arms:
        for R in rs:
            c = cells(arm, "causal", R, lambda r: True)
            g = cells(arm, "global", R, lambda r: True)
            gaps = [gi - ci for gi, ci in zip(g, c)]
            cm, _, cse = mean_sd(c)
            gm, _, gse = mean_sd(g)
            am, asd, ase = mean_sd(gaps)
            print(f"  {arm:>8} {R:>3} {cm:>10.2%} +/-{cse:>5.2%} "
                  f"{gm:>10.2%} +/-{gse:>5.2%} {am:>+10.2%} +/-{ase:>5.2%}")
        print()

    print("=" * 84)
    print("THE CLAIM — the distance profile at each R. GAP, mean +/- SE over seeds")
    print("=" * 84)
    for arm in arms:
        print(f"\n  arm = {arm}")
        print(f"  {'R':>3} " + " ".join(f"{'d=' + str(d):>18}" for d in ds))
        for R in rs:
            row = []
            for d in ds:
                pred = (lambda dd: (lambda r: r["distance"] == dd))(d)
                c = cells(arm, "causal", R, pred)
                g = cells(arm, "global", R, pred)
                if len(c) < 2 or len(c) != len(g):
                    row.append(f"{'thin':>18}")
                    continue
                m, sd, se = mean_sd([gi - ci for gi, ci in zip(g, c)])
                z = m / se if se else float("nan")
                row.append(f"{m:>+9.2%} +/-{se:>4.2%}" if abs(z) < 2
                           else f"{m:>+9.2%} +/-{se:>4.2%}*")
            print(f"  {R:>3} " + " ".join(row))
    print("\n  * marks |z| > 2 against zero, over checkpoints.")
    print("  The prediction under test: d = 3 and d = 4 lift as R rises.")
    print("  Six-seed R = 1 reference, from the committed dumps:")
    print("    d=1 +2.90% (z 6.06)   d=2 +2.84% (z 4.21)   "
          "d=3 +0.29% (z 0.75)   d=4 +0.28% (z 0.67)")


if __name__ == "__main__":
    main()
