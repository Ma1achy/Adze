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

from adze.data.generate import OPS, Trace


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


# --- M7 controls ------------------------------------------------------------
#
# The control the central experiment was designed around — "corrupt a block whose
# result nothing downstream consumes" — is STRUCTURALLY EMPTY on this generator.
# `_emit` walks the expression tree post-order, so every step but the root has
# exactly one later consumer (its parent), and `make_pair` excludes the root.
# Measured on the M7 eval set: 0 of 500 unconsumed.
#
# The two constructors below replace it. They are eval-time controls; neither
# changes generation, so the construct-time-only rule is untouched.


def corrupt_final(trace: Trace, rng_seed: int) -> CorruptedPair:
    """Corrupt the LAST step. The genuine null: nothing consumes the root.

    With no downstream consumer there is no pin, so global regeneration has no
    evidence causal lacks and the gap must be zero. That makes this simultaneously
    the mechanism's null and the harness's leak check — a non-zero gap here is a
    second noise draw or a mask that is not doing what it claims.

    One confound, which is why the caller should also read the `n_steps == B`
    subset: when a trace is shorter than B, the erased block attends to PAD blocks
    under GLOBAL and not under CAUSAL. Pads carry a constant latent so they should
    carry no information, but they do change the softmax normalisation.
    """
    n = len(trace.steps)
    if n < 2:
        raise ValueError(f"need at least 2 steps, got {n}")
    block_index = n - 1
    corrupted = corrupt_step(trace, block_index, rng_seed=rng_seed)
    return CorruptedPair(clean=trace, corrupted=corrupted, block_index=block_index)


def corrupt_consistent(trace: Trace, rng_seed: int) -> CorruptedPair:
    """Perturb one step's result and RECOMPUTE every step downstream of it.

    Not a null — a **redirected pin**, which is the stronger test. Every consumer
    of the perturbed step now holds the PERTURBED value, and every step downstream
    is arithmetically correct given it. So global still has a pin; it has been
    moved. Causal still has nothing.

    `is_valid()` is FALSE on the result, and must be. Step `b`'s operands are left
    alone — that is the whole point, since it is what lets causal recompute the
    clean step in the both-from-earlier cell — so `lhs op rhs != result` at `b`
    exactly. The inconsistency has been *relocated*, from a chain-wide
    contradiction into block `b` itself, and everything outside `b` now agrees on
    the corrupted value. Making `b` self-consistent would mean changing its
    operands or its operator, which would destroy the recomputability the sharp
    cell rests on.

    Scored two ways against the same regenerated block:

      against the CLEAN step       global should be <= causal — downstream is
                                   actively pointing away from it
      against the CORRUPTED step   global should be >  causal — it is following
                                   the pin to where it now points

    An effect that *tracks* the mechanism when the mechanism is moved is a stronger
    causal claim than an effect that merely vanishes without it.

    Sharpest where both of the perturbed step's operands come from earlier steps:
    the prefix is untouched, so CAUSAL can recompute the clean step exactly while
    GLOBAL is pinned to the corrupted one. The two arms are then aimed at different
    targets, and disagree in direction rather than in magnitude.

    Downstream steps are rebuilt from recorded provenance, not by value-matching —
    two steps can hold the same value by coincidence, and `lhs_from` / `rhs_from`
    already say exactly which producer each operand came from.
    """
    n = len(trace.steps)
    if n < 2:
        raise ValueError(f"need at least 2 steps, got {n}")

    rng = random.Random(rng_seed)
    block_index = rng.randrange(max(1, (n + 1) // 2))

    steps = list(corrupt_step(trace, block_index, rng_seed=rng.randrange(2**31)).steps)

    # Forward pass over the chain. Each step's operands are re-read from whichever
    # earlier step produced them; a literal operand is left alone, because a leaf
    # comes from the question and the question has not changed.
    for i in range(block_index + 1, n):
        step = steps[i]
        lhs = steps[step.lhs_from].result if step.lhs_from is not None else step.lhs
        rhs = steps[step.rhs_from].result if step.rhs_from is not None else step.rhs
        steps[i] = replace(step, lhs=lhs, rhs=rhs, result=OPS[step.op](lhs, rhs))

    corrupted = replace(trace, steps=tuple(steps), answer=steps[-1].result)
    return CorruptedPair(clean=trace, corrupted=corrupted, block_index=block_index)
