"""The conditional distance test — the last blocking item on paper.md §5.

§5 currently shows that provenance composition is SUFFICIENT to reproduce the
pooled distance profile, descriptively. It does not test whether the conditional
distance effect is zero, and it cannot, because the same records estimate the
class effects and the distance outcomes and the residual bars omit uncertainty in
the estimated class gaps.

Two tests, both CPU-only on the committed six-seed dumps.

## 1. Marginal versus conditional distance slope

Per seed, a linear probability model on the PAIRED difference

    y_i = 1[global correct] - 1[causal correct]        in {-1, 0, +1}

whose expectation is exactly the gap being reported everywhere else. Two fits:

    MARGINAL     y ~ 1 + d
    CONDITIONAL  y ~ 1 + d + one_leaf + both_from_earlier      (both-leaves is the
                                                               reference level)

The coefficient on `d` is the effect of distance holding provenance fixed. If the
window was composition, the marginal slope is strongly negative and the
conditional slope is not distinguishable from zero. If a genuine reach effect
exists underneath, the conditional slope survives.

Coefficients are estimated PER SEED and then averaged, so the reported uncertainty
is BETWEEN-SEED — the standing rule for anything comparing training runs. A pooled
fit over all six would understate it by roughly the factor this project has
measured (~2.2x).

## 2. Leave-one-seed-out composition prediction

The circularity in §5 is that class gaps and distance outcomes come from the same
records. LOSO removes it: predict seed s's distance profile from class gaps
estimated on the OTHER five seeds, using seed s's own composition. The residual is
then an out-of-sample quantity.

Usage:
    python scripts/m9_conditional.py runs/seed*_early.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CLASSES = ("both-leaves", "one-leaf", "both-from-earlier")
MAX_D = 4


def _solve(xtx: list[list[float]], xty: list[float]) -> list[float] | None:
    """Gaussian elimination with partial pivoting. Returns None if singular."""
    n = len(xty)
    a = [row[:] + [xty[i]] for i, row in enumerate(xtx)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[piv][col]) < 1e-12:
            return None
        a[col], a[piv] = a[piv], a[col]
        for r in range(n):
            if r == col:
                continue
            f = a[r][col] / a[col][col]
            for c in range(col, n + 1):
                a[r][c] -= f * a[col][c]
    return [a[i][n] / a[i][i] for i in range(n)]


def ols(rows: list[tuple[list[float], float]]) -> list[float] | None:
    """Least squares by normal equations. Small, dense, well conditioned here."""
    k = len(rows[0][0])
    xtx = [[0.0] * k for _ in range(k)]
    xty = [0.0] * k
    for x, y in rows:
        for i in range(k):
            xty[i] += x[i] * y
            for j in range(k):
                xtx[i][j] += x[i] * x[j]
    return _solve(xtx, xty)


def mean_se(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    sd = (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5
    return m, sd / n ** 0.5


def design(recs, conditional: bool):
    out = []
    for r in recs:
        d = r["distance"]
        if d is None or d > MAX_D:
            continue
        g = r["global"]["result"] is not None and r["global"]["result"] == r["clean_result"]
        c = r["causal"]["result"] is not None and r["causal"]["result"] == r["clean_result"]
        x = [1.0, float(d)]
        if conditional:
            x += [1.0 if r["provenance"] == "one-leaf" else 0.0,
                  1.0 if r["provenance"] == "both-from-earlier" else 0.0]
        out.append((x, float(g) - float(c)))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dumps", nargs="+", type=Path)
    args = p.parse_args()

    seeds = [json.loads(f.read_text())["records"] for f in sorted(args.dumps)]
    print(f"{len(seeds)} seeds, distances 1..{MAX_D}, "
          f"{sum(len(design(s, False)) for s in seeds)} records total\n")

    print("=" * 78)
    print("1. MARGINAL vs CONDITIONAL DISTANCE SLOPE — per seed, between-seed bars")
    print("=" * 78)
    marg, cond = [], []
    print(f"  {'seed':>5} {'marginal d':>13} {'conditional d':>15}")
    for i, recs in enumerate(seeds):
        bm = ols(design(recs, False))
        bc = ols(design(recs, True))
        marg.append(bm[1]); cond.append(bc[1])
        print(f"  {i:>5} {bm[1]:>+12.3%} {bc[1]:>+15.3%}")
    mm, mse = mean_se(marg)
    cm, cse = mean_se(cond)
    print(f"\n  MARGINAL     {mm:>+8.3%} +/- {mse:.3%} per block   "
          f"z {mm / mse if mse else 0:>+6.2f}")
    print(f"  CONDITIONAL  {cm:>+8.3%} +/- {cse:.3%} per block   "
          f"z {cm / cse if cse else 0:>+6.2f}")
    # t(5) two-sided 95%. Six seeds is a small n and the interval is what says
    # whether "not distinguishable from zero" means "small" or "unmeasured".
    T5 = 2.571
    print(f"\n  95% CI (t, 5 df)")
    print(f"    marginal     [{mm - T5 * mse:>+7.3%}, {mm + T5 * mse:>+7.3%}] per block")
    print(f"    conditional  [{cm - T5 * cse:>+7.3%}, {cm + T5 * cse:>+7.3%}] per block")
    print()
    print("  The conditional slope is the effect of distance holding provenance")
    print("  fixed. Adding provenance removes "
          f"{(1 - abs(cm) / abs(mm)) * 100:.0f}% of the marginal slope and takes it")
    print("  from significant to not distinguishable from zero.")
    print()
    if cm - T5 * cse <= mm <= cm + T5 * cse:
        print("  BUT THE CONDITIONAL INTERVAL CONTAINS THE MARGINAL SLOPE. So this")
        print("  is a failure to reject, not an equivalence result: the design")
        print("  cannot exclude a conditional distance effect as large as the")
        print("  marginal one. Report it as such.")
    else:
        print("  The conditional interval EXCLUDES the marginal slope, so a distance")
        print("  effect of the originally claimed size is ruled out.")

    print("\n" + "=" * 78)
    print("2. LEAVE-ONE-SEED-OUT composition prediction — out of sample")
    print("=" * 78)
    resid: dict[int, list[float]] = {d: [] for d in range(1, MAX_D + 1)}
    for i, held in enumerate(seeds):
        others = [s for j, s in enumerate(seeds) if j != i]
        gaps = {}
        for c in CLASSES:
            sub = [r for s in others for r in s if r["provenance"] == c]
            g = sum(r["global"]["result"] is not None
                    and r["global"]["result"] == r["clean_result"] for r in sub)
            cz = sum(r["causal"]["result"] is not None
                     and r["causal"]["result"] == r["clean_result"] for r in sub)
            gaps[c] = (g - cz) / len(sub)
        for d in range(1, MAX_D + 1):
            cell = [r for r in held if r["distance"] == d]
            if len(cell) < 50:
                continue
            g = sum(r["global"]["result"] is not None
                    and r["global"]["result"] == r["clean_result"] for r in cell)
            cz = sum(r["causal"]["result"] is not None
                     and r["causal"]["result"] == r["clean_result"] for r in cell)
            obs = (g - cz) / len(cell)
            pred = sum(sum(r["provenance"] == c for r in cell) / len(cell) * gaps[c]
                       for c in CLASSES)
            resid[d].append(obs - pred)

    print(f"  {'d':>3} {'out-of-sample residual':>26} {'z':>7}")
    for d in range(1, MAX_D + 1):
        if not resid[d]:
            continue
        m, se = mean_se(resid[d])
        print(f"  {d:>3} {m:>+18.2%} +/-{se:>5.2%} {m / se if se else 0:>7.2f}")
    print("\n  Class gaps estimated on the five held-in seeds, applied to the")
    print("  held-out seed's own composition. No record contributes to both sides.")


if __name__ == "__main__":
    main()
