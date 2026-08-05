"""M1 — length statistics. These are the numbers that decide K and B.

Character-level tokenisation is the standing decision: subword splits on numbers
are exactly what this project avoids. So "characters per step" is the quantity K
must cover, and the step-count distribution is what B must cover.

B is fixed for v0, so a variable step count forces a choice — pad every trace up
to B, or filter to exactly B steps. That choice belongs to M2, but it can only be
made against the right numbers, so the step-count table reports both sides of it.

Usage:
    python scripts/m1_stats.py
    python scripts/m1_stats.py --n 10000 --depths 2 3 4
"""

from __future__ import annotations

import argparse
import statistics
from collections import Counter

from adze.data.corrupt import make_pair
from adze.data.generate import Trace, generate_dataset


def _percentile(values: list[int], q: float) -> int:
    """Nearest-rank percentile. Boring and exact; no numpy needed."""
    ordered = sorted(values)
    rank = max(1, min(len(ordered), round(q / 100 * len(ordered))))
    return ordered[rank - 1]


def _summarise(label: str, values: list[int]) -> None:
    print(
        f"  {label:<26} min {min(values):>4}  mean {statistics.mean(values):>7.2f}  "
        f"median {int(statistics.median(values)):>4}  p95 {_percentile(values, 95):>4}  "
        f"max {max(values):>4}"
    )


def _step_count_table(traces: list[Trace]) -> None:
    """Step-count distribution, framed as the B decision rather than a curiosity."""
    counts = [len(t.steps) for t in traces]
    total = len(counts)
    histogram = Counter(counts)
    observed = sorted(histogram)

    print(f"  {'n steps':>7} {'count':>7} {'share':>7} "
          f"{'filter yield':>13} {'pad coverage':>13} {'pad waste':>10}")
    for n in observed:
        exact = histogram[n] / total
        coverage = sum(c for k, c in histogram.items() if k <= n) / total
        # Mean empty blocks per trace if B = n and short traces are padded up.
        usable = [c for c in counts if c <= n]
        waste = statistics.mean([n - c for c in usable]) if usable else 0.0
        print(f"  {n:>7} {histogram[n]:>7} {exact:>6.1%} "
              f"{exact:>12.1%} {coverage:>12.1%} {waste:>10.2f}")

    print("\n    filter yield at B=n — share of traces with exactly n steps; what")
    print("      survives if B is fixed at n and the remainder discarded")
    print("    pad coverage at B=n  — share with at most n steps; what is usable if")
    print("      short traces are padded up to B")
    print("    pad waste at B=n     — mean empty blocks per trace, the cost of padding")


def _report(n: int, depth: int, operand_max: int, seed: int) -> None:
    print(f"\n{'=' * 78}")
    print(f"max_depth={depth}  operand_max={operand_max}  n={n}")
    print("=" * 78)

    traces = generate_dataset(n=n, seed=seed, max_depth=depth, operand_max=operand_max)

    assert all(t.is_valid() for t in traces), "a generated trace is invalid"

    print("\nstep counts (decides B)")
    _step_count_table(traces)

    step_chars = [len(s.render()) for t in traces for s in t.steps]
    trace_chars = [len(t.render()) for t in traces]
    question_chars = [len(t.question) for t in traces]

    print("\nlengths in characters (decides K)")
    _summarise("chars per step", step_chars)
    _summarise("chars per trace", trace_chars)
    _summarise("chars per question", question_chars)

    charset = sorted({c for t in traces for c in t.render()})
    print(f"\ncharacter set ({len(charset)} distinct): {''.join(charset)!r}")

    # Corruption round-trip: every pair must be invalid, differ in exactly one
    # step, at the recorded index, and never at the final step.
    pairs = [make_pair(t, rng_seed=i, prefer_early=True) for i, t in enumerate(traces)]
    for pair in pairs:
        assert pair.clean.is_valid(), "clean side of a pair is invalid"
        assert not pair.corrupted.is_valid(), "corrupted side of a pair is valid"
        differing = [
            i for i, (c, b) in enumerate(zip(pair.clean.steps, pair.corrupted.steps))
            if c != b
        ]
        assert differing == [pair.block_index], "corruption did not round-trip"
        assert pair.block_index < len(pair.clean.steps) - 1, "final step was corrupted"

    positions = [p.block_index / max(len(p.clean.steps) - 1, 1) for p in pairs]
    print(f"\ncorruption: {len(pairs)} pairs, all invalid, all round-trip, "
          f"none on the final step")
    print(f"  mean normalised index  {statistics.mean(positions):.3f}  (early-biased)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10_000)
    p.add_argument("--depths", type=int, nargs="+", default=[2, 3, 4])
    p.add_argument("--operand-max", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    for depth in args.depths:
        _report(args.n, depth, args.operand_max, args.seed)

    print(f"\n{'=' * 78}")
    print("Now go and choose K and B against these numbers.")
    print("=" * 78)


if __name__ == "__main__":
    main()
