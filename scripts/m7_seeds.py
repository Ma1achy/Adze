"""The headline number, with its spread — and whether the recalibration is a
control variate.

The project's own standing rule is that any number gating a decision gets reported
with its budget and its spread. Every effect size so far has had a budget and no
spread, and when two p = 0.50 runs finally existed they gave +3.5pp and +1.6pp —
a spread as wide as an entire four-point mix sweep. So the mix curve was withdrawn
and this is what replaces it: one configuration, many seeds, a paired comparison
inside each checkpoint.

## The control-variate question

The recalibration subtracts the handicap — the global-minus-causal gap measured
where NO downstream evidence exists — from the effect measured where it does. That
was justified as a fairness correction: global carries a condition-level penalty
unrelated to the mechanism, so leaving it in understates the mechanism.

On n = 2 something further showed up. Raw spread 1.9pp, recalibrated spread 0.9pp.
If the handicap and the effect are both driven by how good the global pathway
happens to be on a given run, then subtracting one removes a shared nuisance term
and the difference has lower variance than either part. That is a control variate,
and it would make the recalibrated figure a better ESTIMATOR rather than merely a
fairer number.

Testable across seeds: **the correlation between handicap and effect.**

  strongly positive -> confirmed. The shared nuisance term is real.
  near zero         -> the variance halving was luck at n = 2, and the
                       recalibration is a fairness correction only.

No optimal coefficient is fitted. Six seeds is far too few to estimate one, and a
fitted beta would absorb noise and manufacture the very variance reduction the
test is meant to detect. beta = 1 throughout.

Usage:
    python scripts/m7_seeds.py runs/seed*_early.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from adze.eval.strata import cell


def _mean_sd(xs: list[float]) -> tuple[float, float]:
    """Mean and SAMPLE standard deviation (n-1). With n small the population
    form understates the spread, and the spread is the point here."""
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    return m, (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5


def _corr(xs: list[float], ys: list[float]) -> float | None:
    """Pearson r. None if either side is constant."""
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    return sxy / (sxx * syy) ** 0.5


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("early", nargs="+", type=Path,
                   help="the *_early.json dumps; *_final.json is inferred")
    args = p.parse_args()

    rows = []
    for early_path in sorted(args.early):
        final_path = Path(str(early_path).replace("_early.json", "_final.json"))
        if not final_path.exists():
            print(f"  skipping {early_path.name} — no matching {final_path.name}")
            continue
        e = json.loads(early_path.read_text())
        f = json.loads(final_path.read_text())
        label = re.sub(r"_early\.json$", "", early_path.name)
        eff = cell("effect", e["records"], "result")
        hcp = cell("handicap", f["records"], "result")
        rows.append({
            "label": label,
            "effect": eff.gap,
            "handicap": hcp.gap,
            "recal": eff.gap - hcp.gap,
            "causal": eff.causal,
            "global": eff.glob,
            "chance": eff.chance,
            "n": eff.n,
        })

    if not rows:
        raise SystemExit("no complete seed pairs found")

    print("=" * 92)
    print(f"PER-SEED  —  n = {len(rows)} checkpoints, {rows[0]['n']} traces each, "
          f"oracle selection")
    print("=" * 92)
    print(f"  {'run':>28} {'causal':>8} {'global':>8} {'effect':>9} "
          f"{'handicap':>10} {'recalibrated':>13}")
    for r in rows:
        print(f"  {r['label']:>28} {r['causal']:>8.1%} {r['global']:>8.1%} "
              f"{r['effect']:>+9.2%} {r['handicap']:>+10.2%} {r['recal']:>+13.2%}")

    eff = [r["effect"] for r in rows]
    hcp = [r["handicap"] for r in rows]
    rec = [r["recal"] for r in rows]
    em, es = _mean_sd(eff)
    hm, hs = _mean_sd(hcp)
    rm, rs = _mean_sd(rec)

    print()
    print("=" * 92)
    print("MEAN +/- SPREAD   (sample sd, n-1)")
    print("=" * 92)
    print(f"  raw effect            {em:>+8.2%}  +/- {es:.2%}")
    print(f"  handicap              {hm:>+8.2%}  +/- {hs:.2%}")
    print(f"  RECALIBRATED          {rm:>+8.2%}  +/- {rs:.2%}   <- THE HEADLINE")
    print(f"  chance (permutation)  {sum(r['chance'] for r in rows)/len(rows):>8.2%}")

    print()
    print("=" * 92)
    print("THE CONTROL VARIATE")
    print("=" * 92)
    r = _corr(hcp, eff)
    if r is None:
        print("  correlation undefined (n < 3, or a constant column)")
    else:
        print(f"  corr(handicap, effect) over {len(rows)} seeds = {r:+.3f}")
        if es > 0:
            print(f"  variance ratio  sd(recalibrated) / sd(raw) = {rs / es:.2f}")
        if r > 0.5:
            print("  STRONGLY POSITIVE -> the shared nuisance term is real. The")
            print("  recalibrated figure is a LOWER-VARIANCE ESTIMATOR, not merely a")
            print("  fairer number, and is the right headline for both reasons.")
        elif r > 0.2:
            print("  WEAKLY POSITIVE -> suggestive, not established at this n.")
            print("  Report the recalibrated figure as a fairness correction and say")
            print("  the variance claim is unresolved.")
        else:
            print("  NEAR ZERO OR NEGATIVE -> the variance halving seen at n = 2 was")
            print("  luck. The recalibration remains a FAIRNESS correction only, and")
            print("  the lower-variance claim should be dropped rather than softened.")
    print()
    print("  beta = 1 throughout. No optimal coefficient is fitted: this n is far")
    print("  too small to estimate one, and a fitted beta would absorb noise and")
    print("  manufacture the variance reduction this test exists to detect.")


if __name__ == "__main__":
    main()
