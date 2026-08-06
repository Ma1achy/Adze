"""The standard readout: distribution-matched truth.

WHY THIS IS THE HEADLINE AND RAW TRUTH IS NOT.

Every aggregate truth figure in this project up to session 11 was inflated by the
model choosing its own difficulty. Measured: the model put 47.5% of its generated
steps in the 10-29 operand bin against 4.5% of real steps, while its per-bin truth
ran 80.6% at 10-29 and 17.3% at 30-99. So the pooled figure was largely a readout
of *which problems the model picked*, not of how well it did them. Reweighting the
per-bin rates by the real data's magnitude shares turned 51.2% into ~13.2%, and
"69% of ceiling" into "18% of ceiling".

The ceiling has the opposite bias — it is measured on real held-out steps, so it
carries the *data's* magnitude distribution. Reading pooled generated truth against
it was never apples to apples, and this module exists so that comparison cannot be
made carelessly again.

WHAT IS REPORTED

  matched     per-bin truth reweighted to the real magnitude distribution — the
              headline. It answers "how often is the model right on the problems
              it will actually be asked", which is the question that matters.
  raw         the old pooled figure, kept and LABELLED so numbers recorded before
              session 12 stay comparable rather than being quietly redefined.
  per-bin     with the per-bin ceiling beside it, because once a small bin clears
              the pooled ceiling the pooled number has stopped being fine enough
              to read against.
  histograms  generated vs real shares, side by side. The difference between them
              IS the inflation, so it is printed rather than summarised.

A bin the model never generates contributes its real-distribution weight at truth
0 — not skipped. Skipping it would reward the model for avoiding a bin entirely,
which is the exact failure this readout exists to expose. Bins with real weight 0
are dropped from the reweighting, since they are not part of the task.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from adze.eval.magnitude import BINS, magnitude, magnitude_table
from adze.sample.trajectory import classify


@dataclass(frozen=True)
class Readout:
    """One measurement, reported the standard way."""

    raw_true: float
    raw_formed: float
    matched_true: float
    per_bin: dict[str, tuple[int, float, float]]   # name -> (n, formed, true)
    generated_share: dict[str, float]
    real_share: dict[str, float]
    ceiling: dict[str, float] = field(default_factory=dict)
    unbinnable: int = 0
    n: int = 0

    @property
    def matched_ceiling(self) -> float:
        """The ceiling under the same reweighting, so the ratio is well defined."""
        return sum(self.real_share.get(b, 0.0) * c for b, c in self.ceiling.items())


def magnitude_shares(texts: list[str]) -> dict[str, float]:
    """Share of `texts` falling in each magnitude bin. Unbinnable text is excluded
    from the denominator, so shares sum to 1 over what could be binned."""
    counts = {name: 0 for _, _, name in BINS}
    total = 0
    for t in texts:
        mag = magnitude(t)
        if mag is None:
            continue
        total += 1
        for lo, hi, name in BINS:
            if lo <= mag <= hi:
                counts[name] += 1
                break
    return {k: (v / total if total else 0.0) for k, v in counts.items()}


def readout(
    generated: list[str],
    real: list[str],
    ceiling: dict[str, float] | None = None,
) -> Readout:
    """The standard measurement.

    Args:
        generated: raw decoded model output, unfiltered.
        real: held-out data steps, for the reference magnitude distribution.
        ceiling: optional per-bin decoder round-trip truth, from
            `adze.eval.checks.unseen_ceiling_by_magnitude`.
    """
    rows, unbinnable = magnitude_table(generated)
    per_bin = {name: (n, formed, true) for name, n, formed, true in rows}
    gen_share = magnitude_shares(generated)
    real_share = magnitude_shares(real)

    kinds = [classify(t) for t in generated]
    raw_true = sum(1 for k in kinds if k == "true") / max(len(kinds), 1)
    raw_formed = sum(1 for k in kinds if k != "malformed") / max(len(kinds), 1)

    # A bin with real weight but no generated samples scores 0. That is the point:
    # avoiding a bin is a failure on it, not an absence of evidence about it.
    matched = sum(
        w * per_bin.get(name, (0, 0.0, 0.0))[2]
        for name, w in real_share.items() if w > 0
    )
    return Readout(
        raw_true=raw_true, raw_formed=raw_formed, matched_true=matched,
        per_bin=per_bin, generated_share=gen_share, real_share=real_share,
        ceiling=ceiling or {}, unbinnable=unbinnable, n=len(generated),
    )


def print_readout(r: Readout, title: str) -> None:
    """Print the standard readout. Matched first, raw second and labelled."""
    print("=" * 78)
    print(title)
    print("=" * 78)
    print(f"  distribution-matched true   {r.matched_true:>7.1%}   <- THE HEADLINE")
    mc = r.matched_ceiling
    if r.ceiling and mc > 0:
        print(f"  matched ceiling             {mc:>7.1%}   "
              f"({r.matched_true / mc:.0%} of it)")
    print(f"  raw pooled true             {r.raw_true:>7.1%}   "
          f"(pre-session-12 convention; inflated by difficulty selection)")
    print(f"  raw pooled well-formed      {r.raw_formed:>7.1%}")
    print(f"  n {r.n}, unbinnable {r.unbinnable} "
          f"({r.unbinnable / max(r.n, 1):.1%})")
    print()
    print(f"  {'magnitude':>10} {'gen share':>10} {'real share':>11} "
          f"{'n':>7} {'well-formed':>12} {'true':>8} {'ceiling':>9}")
    for _, _, name in BINS:
        g = r.generated_share.get(name, 0.0)
        d = r.real_share.get(name, 0.0)
        if g == 0.0 and d == 0.0:
            continue
        n, formed, true = r.per_bin.get(name, (0, float("nan"), 0.0))
        ceil = r.ceiling.get(name)
        ceil_s = f"{ceil:>9.1%}" if ceil is not None else f"{'—':>9}"
        formed_s = f"{formed:>12.1%}" if n else f"{'—':>12}"
        print(f"  {name:>10} {g:>10.1%} {d:>11.1%} {n:>7} {formed_s} "
              f"{true:>8.1%} {ceil_s}")
    print()
    drift = sum(abs(r.generated_share.get(b, 0.0) - r.real_share.get(b, 0.0))
                for _, _, b in BINS) / 2
    print(f"  magnitude-distribution mismatch (total variation): {drift:.3f}")
    print("  0 means the model generates the difficulty mix it will be asked for;")
    print("  the gap between raw and matched truth is this mismatch cashed out.")
