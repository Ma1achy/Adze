"""M7 — stratifying the causal-vs-global gap, and scoring it against chance.

The first M7 number was a single pooled figure: `global - causal` = +2.8pp on
RESULT. A pooled gap says an effect exists somewhere; it does not say the effect
is the one the design predicts. These are the cuts that test the mechanism rather
than merely measure it.

## What the mechanism predicts

Global's only advantage is that it sees the later steps which consumed the
corrupted block's original result. That downstream consumer is the *pin*. So:

  * no consumer (the root)          -> no pin -> gap must be ZERO
  * consumer further away           -> more chain to traverse -> gap decays
  * both operands from earlier      -> causal can recompute -> causal is strong
  * pin moved (consistent corrupt)  -> the gap must MOVE WITH IT

Anything that shows a gap where no pin exists is not downstream evidence, and we
would need to find out what it is before reading the headline.

## Chance is the reference, not `none`

`none` measures the VAE round-trip, not no-revision: the VAE was trained on valid
arithmetic only and silently repairs 19.6% of corruptions by projecting an
off-distribution step onto the nearest valid one. The regeneration arms erase the
block and discard that free repair, so `none` is not their baseline.

The baseline is chance, and chance here is a PERMUTATION null rather than an
analytic guess: score a condition's regenerated block against a *different*
trace's target at the same block index. That measures this model's actual output
marginal against this data's target marginal, which is what "chance" has to mean.
An analytic 1/|support| figure is reported alongside, labelled as the crude
reference it is.

## The pad-attention confound

When a trace is shorter than B, the erased block attends to PAD blocks under
GLOBAL and not under CAUSAL. Pads carry a constant latent so they should carry no
information, but they change the softmax normalisation, and "should" is not a
measurement. Every table therefore also reports the `n_steps == B` subset, which
is pad-free by construction.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from adze.data.generate import Step, Trace

PROVENANCE = ("both-leaves", "one-leaf", "both-from-earlier")


def consumers(trace: Trace, block: int) -> list[int]:
    """Indices of later steps that consume `block`'s result. Provenance, not value.

    Value-matching would false-positive: two steps can hold the same number by
    coincidence. `lhs_from` / `rhs_from` record which producer each operand came
    from, which is the fact being asked about.
    """
    return [
        j for j in range(block + 1, len(trace.steps))
        if trace.steps[j].lhs_from == block or trace.steps[j].rhs_from == block
    ]


def consumer_distance(trace: Trace, block: int) -> int | None:
    """Blocks from `block` to its nearest consumer, or None if nothing consumes it.

    None is the no-pin case. On this generator it occurs only at the root — the
    expression tree is emitted post-order, so every other step has exactly one
    consumer, its parent.
    """
    found = consumers(trace, block)
    return min(found) - block if found else None


def operand_provenance(step: Step) -> str:
    """'both-leaves' | 'one-leaf' | 'both-from-earlier'.

    How much of the step the prefix already determines. With both operands from
    earlier steps a causal regeneration has everything it needs to recompute the
    result; with both from leaves it has nothing, because leaves come from the
    question and question conditioning (M5) is not built.
    """
    n = (step.lhs_from is not None) + (step.rhs_from is not None)
    return PROVENANCE[n]


# --- scoring ----------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """One stratum's paired comparison."""

    key: str
    n: int
    causal: float
    glob: float
    chance: float
    only_global: int
    only_causal: int

    @property
    def gap(self) -> float:
        return self.glob - self.causal

    @property
    def chi2(self) -> float | None:
        """McNemar with continuity correction, or None with no discordant pairs."""
        disc = self.only_global + self.only_causal
        if disc == 0:
            return None
        return (abs(self.only_global - self.only_causal) - 1) ** 2 / disc

    @property
    def significant(self) -> bool:
        chi = self.chi2
        return chi is not None and chi > 3.84


def _hit(got: str | None, want: str | None) -> bool:
    """Whether a decoded field matches its target. None never matches."""
    return got is not None and want is not None and got == want


def chance_rate(
    records: list[dict],
    condition: str,
    field: str,
    target: str = "clean",
    permutations: int = 200,
    seed: int = 0,
) -> tuple[float, float]:
    """Permutation null: mean and standard deviation of the shuffled-pairing rate.

    Scores each record's regenerated output against a DIFFERENT record's target at
    the same block index. Records are grouped by block index first, so the shuffle
    cannot smuggle in a block-position effect — a chance rate that mixed block 0's
    outputs with block 3's targets would be measuring the wrong thing.

    Returns:
        (mean, sd) over `permutations` shuffles.
    """
    rng = random.Random(seed)
    by_block: dict[int, list[dict]] = {}
    for r in records:
        by_block.setdefault(r["block"], []).append(r)

    rates = []
    for _ in range(permutations):
        hits = total = 0
        for group in by_block.values():
            if len(group) < 2:
                continue
            shuffled = group[:]
            rng.shuffle(shuffled)
            for got, want in zip(group, shuffled):
                if got is want:
                    continue
                hits += _hit(got[condition][field], want[f"{target}_{field}"])
                total += 1
        if total:
            rates.append(hits / total)

    if not rates:
        return 0.0, 0.0
    mean = sum(rates) / len(rates)
    var = sum((r - mean) ** 2 for r in rates) / len(rates)
    return mean, var**0.5


def cell(
    key: str,
    records: list[dict],
    field: str,
    target: str = "clean",
    permutations: int = 200,
    seed: int = 0,
) -> Cell:
    """Score one stratum: both arms, their chance rate, and the discordant pairs."""
    want_key = f"{target}_{field}"
    c_hits = g_hits = only_g = only_c = 0
    for r in records:
        c_ok = _hit(r["causal"][field], r[want_key])
        g_ok = _hit(r["global"][field], r[want_key])
        c_hits += c_ok
        g_hits += g_ok
        only_g += g_ok and not c_ok
        only_c += c_ok and not g_ok

    n = max(len(records), 1)
    mean, _sd = chance_rate(records, "causal", field, target, permutations, seed)
    return Cell(
        key=key,
        n=len(records),
        causal=c_hits / n,
        glob=g_hits / n,
        chance=mean,
        only_global=only_g,
        only_causal=only_c,
    )


def stratify(
    records: list[dict],
    key_fn,
    field: str,
    target: str = "clean",
    permutations: int = 200,
    seed: int = 0,
) -> list[Cell]:
    """Group `records` by `key_fn` and score each group. Sorted by key."""
    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(str(key_fn(r)), []).append(r)
    return [
        cell(k, groups[k], field, target, permutations, seed)
        for k in sorted(groups, key=lambda s: (len(s), s))
    ]


def print_cells(title: str, cells: list[Cell], field: str, target: str) -> None:
    """One stratification table. Cell counts always shown — a gap on n = 6 is not
    a gap, and the reader must be able to see that without asking."""
    print(f"\n{title}   [{field} vs {target} target]")
    print(f"  {'cell':>18} {'n':>6} {'causal':>8} {'global':>8} {'gap':>8} "
          f"{'chance':>8} {'only-g':>7} {'only-c':>7} {'chi2':>7}")
    for c in cells:
        chi = "-" if c.chi2 is None else f"{c.chi2:.2f}"
        star = " *" if c.significant else "  "
        print(f"  {c.key:>18} {c.n:>6} {c.causal:>8.1%} {c.glob:>8.1%} "
              f"{c.gap:>+8.1%} {c.chance:>8.1%} {c.only_global:>7} "
              f"{c.only_causal:>7} {chi:>7}{star}")
