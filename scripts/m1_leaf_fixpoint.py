"""TASK 1 — iterate the leaf distribution to a fixed point against its own results.

The defect: leaves drawn uniform on 1..20 while results reach MAGNITUDE_CAP = 1000.
The model met small values as INPUTS and large values only as OUTPUTS, and the
measured consequence was a 6x arithmetic-truth cliff at operand magnitude ~30
against a decoder that round-trips that range at 87%.

Raising `operand_max` does not fix it for any value: raise it and the cap starts
binding on nearly every operation instead. The self-consistent fix is to draw
leaves from the empirical distribution of results and iterate until the input and
output histograms agree.

This script exists to make the convergence VISIBLE rather than asserted. It prints
both histograms at every iteration, plus the operator mix — because a side effect
of large leaves is that `*` stops fitting under the cap, and that is a real change
to the data worth seeing rather than discovering later.

Usage:
    python scripts/m1_leaf_fixpoint.py
"""

from __future__ import annotations

import argparse

from adze.data.generate import BINS_DOC, fixed_point_leaves

BINS = [(0, 9, "0-9"), (10, 29, "10-29"), (30, 99, "30-99"),
        (100, 299, "100-299"), (300, 999, "300-999"), (1000, 10**9, "1000+")]


def histogram(values) -> list[tuple[str, int, float]]:
    """(bin name, count, share) over |value| bins, same bins as the eval path."""
    out = []
    n = max(len(values), 1)
    for lo, hi, name in BINS:
        c = sum(1 for v in values if lo <= abs(v) <= hi)
        out.append((name, c, c / n))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--n", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--operand-max", type=int, default=20)
    args = p.parse_args()

    print(BINS_DOC)
    print(f"iterating   {args.iterations} rounds x {args.n} traces, seed {args.seed}")
    print("  iteration 0 is the ORIGINAL uniform sampler. The fixed point is reached")
    print("  when the LEAF row and the RESULT row of an iteration agree.")
    print()

    rows = []

    def report(i, leaves, results):
        leaf_h = histogram(leaves) if leaves is not None else None
        res_h = histogram(results)
        rows.append((i, leaf_h, res_h))
        label = "uniform 1..%d" % args.operand_max if leaves is None else \
            f"pool of {len(leaves)}"
        print(f"  --- iteration {i}  (leaves: {label}) " + "-" * 26)
        head = "  ".join(f"{name:>9}" for _, _, name in BINS)
        print(f"    {'':>8} {head}")
        if leaf_h is None:
            uni = histogram(tuple(range(1, args.operand_max + 1)))
            print(f"    {'LEAF':>8} " + "  ".join(f"{s:>9.1%}" for _, _, s in uni))
        else:
            print(f"    {'LEAF':>8} " + "  ".join(f"{s:>9.1%}" for _, _, s in leaf_h))
        print(f"    {'RESULT':>8} " + "  ".join(f"{s:>9.1%}" for _, _, s in res_h))
        if leaf_h is not None:
            drift = sum(abs(a[2] - b[2]) for a, b in zip(leaf_h, res_h)) / 2
            print(f"    total variation distance leaf vs result: {drift:.4f}")
        print()

    leaves = fixed_point_leaves(
        iterations=args.iterations, n=args.n, seed=args.seed,
        max_depth=args.max_depth, operand_max=args.operand_max,
        on_iteration=report,
    )

    print("=" * 74)
    print("CONVERGENCE")
    print("=" * 74)
    print("  Total variation distance between the leaf pool and the results it")
    print("  produces. Falling toward 0 is the fixed point; flat and large means")
    print("  the iteration does not converge and the whole approach is wrong.")
    print()
    for i, leaf_h, res_h in rows:
        if leaf_h is None:
            uni = histogram(tuple(range(1, args.operand_max + 1)))
            d = sum(abs(a[2] - b[2]) for a, b in zip(uni, res_h)) / 2
        else:
            d = sum(abs(a[2] - b[2]) for a, b in zip(leaf_h, res_h)) / 2
        print(f"  iteration {i}   TV distance {d:.4f}")

    print()
    print(f"  final leaf pool: {len(leaves)} values, "
          f"min {min(leaves)}, max {max(leaves)}, "
          f"median {sorted(leaves)[len(leaves) // 2]}")


if __name__ == "__main__":
    main()
