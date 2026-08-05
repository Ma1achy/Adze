"""M1 — corruption and matched-pair construction for the central experiment.

The corruption is deliberately *inconsistent*: one intermediate value is changed
and nothing downstream is recomputed. Later steps continue to use the CORRECT
value, so the chain contradicts itself and the contradiction is only visible from
downstream context.

That is the whole point. A model that can only see backwards cannot detect the
error; a model that sees the whole chain can.

See tests/test_m1_corrupt.py for the acceptance criteria.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace

from adze.data.generate import Trace


@dataclass(frozen=True)
class CorruptedPair:
    """A clean trace and its corrupted counterpart, with the corruption located."""

    clean: Trace
    corrupted: Trace
    block_index: int      # which step was corrupted (0-indexed)

    def __post_init__(self) -> None:
        if not 0 <= self.block_index < len(self.clean.steps):
            raise ValueError(f"block_index {self.block_index} out of range")


def corrupt_step(
    trace: Trace,
    block_index: int,
    rng_seed: int,
    delta_max: int = 10,
) -> Trace:
    """Change the result of step `block_index` without recomputing anything after it.

    The returned trace has exactly one step whose arithmetic is wrong. All other
    steps are byte-identical to the original, including any that consumed the
    original correct value.

    Args:
        trace: a valid trace.
        block_index: which step to corrupt.
        rng_seed: seed for the perturbation.
        delta_max: maximum magnitude of the change. Must not produce a no-op.
    """
    if not 0 <= block_index < len(trace.steps):
        raise ValueError(f"block_index {block_index} out of range")
    if delta_max < 1:
        raise ValueError(f"delta_max must be >= 1, got {delta_max}")

    rng = random.Random(rng_seed)
    delta = rng.choice([d for d in range(-delta_max, delta_max + 1) if d != 0])

    original = trace.steps[block_index]
    steps = list(trace.steps)
    # Only the result moves. Operands, provenance, and every downstream step are
    # left exactly as they were — that inconsistency is the signal being studied.
    steps[block_index] = replace(original, result=original.result + delta)
    return replace(trace, steps=tuple(steps))


def make_pair(
    trace: Trace,
    rng_seed: int,
    prefer_early: bool = True,
) -> CorruptedPair:
    """Build a matched pair, corrupting an early step by preference.

    Early corruption is what makes the experiment meaningful — the later steps
    then carry the evidence needed to detect and repair it.

    The final step is never corrupted. A contradiction placed there has no
    downstream context to be visible from, so causal and global regeneration face
    identical evidence and the comparison the experiment rests on is vacuous.
    `prefer_early` makes that unlikely but does not prevent it; excluding the last
    index does.

    Args:
        trace: a valid trace with at least 2 steps.
        rng_seed: seed for both step choice and perturbation.
        prefer_early: bias the corrupted index toward the first half of the chain.
    """
    n = len(trace.steps)
    if n < 2:
        raise ValueError(f"need at least 2 steps to build a pair, got {n}")

    rng = random.Random(rng_seed)

    # Candidates exclude the final step in both branches.
    limit = max(1, (n + 1) // 2) if prefer_early else n - 1
    limit = min(limit, n - 1)
    block_index = rng.randrange(limit)

    corrupted = corrupt_step(trace, block_index, rng_seed=rng.randrange(2**31))
    return CorruptedPair(clean=trace, corrupted=corrupted, block_index=block_index)
