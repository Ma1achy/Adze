"""Bin decoded steps by operand magnitude.

The per-block collapse (b0 61.8% true, b6 4.7%) turned out not to be about block
position. Binning generated steps by `max(|lhs|, |rhs|)` and ignoring position
gives a 7x cliff between the 10-29 and 30-99 bins — and `configs/debug.yaml` sets
`operand_max: 20` while `MAGNITUDE_CAP` in `adze.data.generate` is 1000, so the
model meets small values as *inputs* and large ones only as *outputs*.

**Why the loose regex.** `trajectory.STEP_RE` requires the whole `N op N = N`, so
a malformed decode has no magnitude and cannot be binned at all. Binning on it
therefore conditions on well-formedness, and if malformed output correlates with
magnitude the bins are biased in exactly the direction that would manufacture the
cliff. `OPERANDS_RE` needs only the operand pair, so a string whose *result* is
garbage still lands in a bin and still counts against that bin's well-formedness.

Strings with no parseable operand pair are returned as an explicit remainder, not
dropped. If that remainder is large the bins are still biased and the caller has
to say so.
"""

from __future__ import annotations

import re

from adze.sample.trajectory import classify

# Operands only. Deliberately not anchored at the end: the result may be missing,
# truncated, or nonsense, and such a string still has a magnitude.
OPERANDS_RE = re.compile(r"^(-?\d+) ([+\-*]) (-?\d+)")

BINS: list[tuple[int, int, str]] = [
    (0, 9, "0-9"),
    (10, 29, "10-29"),
    (30, 99, "30-99"),
    (100, 299, "100-299"),
    (300, 10**9, "300+"),
]


def magnitude(text: str) -> int | None:
    """max(|lhs|, |rhs|) for a decoded step, or None if no operand pair parses."""
    m = OPERANDS_RE.match(text)
    return None if m is None else max(abs(int(m[1])), abs(int(m[3])))


def magnitude_table(texts: list[str]) -> tuple[list[tuple[str, int, float, float]], int]:
    """Group decodes by operand magnitude.

    Returns:
        (rows, unbinnable) where each row is (bin name, n, well-formed share, true
        share) and `unbinnable` counts strings with no parseable operand pair.
        Empty bins are omitted; well-formedness is measured with the strict
        `classify`, so a bin's `well-formed` share is a real readout of whether
        that magnitude range is decoding correctly at all.
    """
    binned: dict[str, list[str]] = {name: [] for _, _, name in BINS}
    unbinnable = 0
    for text in texts:
        mag = magnitude(text)
        if mag is None:
            unbinnable += 1
            continue
        for lo, hi, name in BINS:
            if lo <= mag <= hi:
                binned[name].append(text)
                break

    rows = []
    for _, _, name in BINS:
        group = binned[name]
        if not group:
            continue
        kinds = [classify(t) for t in group]
        formed = sum(1 for k in kinds if k != "malformed") / len(kinds)
        true = sum(1 for k in kinds if k == "true") / len(kinds)
        rows.append((name, len(group), formed, true))
    return rows, unbinnable


def print_magnitude_table(texts: list[str], title: str) -> None:
    """Print one magnitude table, remainder included."""
    rows, unbinnable = magnitude_table(texts)
    print(f"  {title}")
    print(f"  {'magnitude':>12} {'n':>7} {'well-formed':>12} {'true':>8}")
    for name, n, formed, true in rows:
        print(f"  {name:>12} {n:>7} {formed:>12.1%} {true:>8.1%}")
    share = unbinnable / len(texts) if texts else 0.0
    print(f"  {'unbinnable':>12} {unbinnable:>7} {'':>12} {'':>8}   "
          f"({share:.1%} — no operand pair parsed)")
    print()
