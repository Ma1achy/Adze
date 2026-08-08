"""A generator where distance-to-consumer and operand provenance are INDEPENDENT.

## Why the existing generator cannot answer reach questions

`generate.py` builds an expression tree and emits it post-order. That ties the two
variables together **by construction**, and no amount of extra data separates them:

    a step with both operands from earlier steps is an internal node with two
    internal children -> it roots a large subtree -> in post-order its own parent
    cannot appear until that whole subtree has been emitted -> its consumer is FAR

    a step with both operands literal is a leaf-level node -> small subtree ->
    its consumer is NEAR

Measured there: d = 1 is 76% both-leaves and 0% both-from-earlier; d = 4 is 20%
both-leaves and 31% both-from-earlier. With gaps of +3.00% and −2.86% for those
classes, plotting the effect against distance manufactures a cliff at d = 2 with
no distance content at all. This repository found that cliff at z = 6.06 over six
seeds, led with it, and withdrew it.

## The construction, and two designs that did not work

The edge set is what has to be controlled. Two node attributes must come apart:

    in-degree of i   -> i's provenance class (0, 1 or 2 operands from earlier)
    out-edge of i    -> distance from i to the step that consumes it

They are different attributes of the same node and *can* be independent, but the
obvious constructions couple them:

**Rejected — greedy class choice.** Drawing each step's class from whatever the
pool allows pushes `k = 2` late (only k = 2 drains the pool toward a single root)
and `k = 0` early (step 0 has an empty pool). Position then predicts class, and
position bounds how far the consumer can be. Measured: both-from-earlier ran
25.0% of d = 1 down to 10.6% at d = 6.

**Rejected — fixed class schedule, uniform draw from the pool.** Fixing the class
multiset first and drawing producers uniformly still couples them, for a subtler
reason: **a step that consumes two results shrinks the pool it then competes in**,
so it is drawn sooner than a step that consumed none and grew the pool. Measured:
both-leaves ran 38.6% at d = 1 up to 74.8% at d = 7 — the original bias, reduced
but alive.

**Rejected — assign consumers by distance, unbounded.** Drawing `d_i` uniformly
over everything still available reintroduces the coupling through POSITION, which
bounds both variables from opposite ends. A late step can be targeted by many
earlier steps, so its in-degree is high (both-from-earlier), and it sits near the
end, so its own consumer is close. An early step is the reverse. Measured: d = 1
was 43.5% both-from-earlier falling to 0.0% by d = 8 — the original bias with its
sign flipped.

**Used — assign consumers by distance, BOUNDED, and measure the interior.** `d_i`
is drawn uniformly from `1 .. DISTANCE_MAX`. That is what removes position from
the picture: for any step at least `DISTANCE_MAX` from the end, the distance range
is the full `1 .. DISTANCE_MAX` regardless of where it sits, and for any step at
index >= 2 the in-degree range is unrestricted. In that INTERIOR BAND both
variables are free and the draws never consult each other —

    d_i        drawn for step i, consulting nothing
    class of i  = how many OTHER steps happened to draw i

— so they are independent by construction rather than by hope. Near the two ends
the bounds genuinely bind and no construction can avoid it; those steps are
reported separately rather than silently pooled.

The result is still a tree: every step but the last has exactly one consumer, so
`consumer_distance` and the pin mean exactly what they meant before.

Magnitude is held as in `generate.py` — the operator is chosen from those keeping
the result within the cap given the operands' actual values. Every operand is
bounded by the cap, so `+`/`-` always yields a candidate and the set is never
empty; the guard is a bug check, not a fallback.
"""

from __future__ import annotations

import random

from adze.data.generate import MAGNITUDE_CAP, OPS, Step, Trace

MAX_IN_DEGREE = 2   # a step has two operands

# The consumer is drawn from 1..DISTANCE_MAX blocks ahead. Bounding it is what
# makes the distance draw position-independent through the interior; see the
# module docstring. It also sets the furthest distance the data can measure, so it
# is the knob that decides how far a reach claim could ever see.
DISTANCE_MAX = 8


def assign_consumers(rng: random.Random, n_steps: int,
                     distance_max: int = DISTANCE_MAX) -> list[int]:
    """`consumer[i]` = the step that consumes step i's result; −1 for the root.

    Each step draws its consumer by DISTANCE, uniformly over the distances still
    available to it, so the draw cannot see the step's own provenance class. The
    last step is the root and is consumed by nothing.

    Steps are processed in DESCENDING index order — most constrained first — and
    that ordering is required, not cosmetic. Step `n-2` can only ever be consumed
    by step `n-1`; if `n-1` fills up first, `n-2` has nowhere to go and the trace
    cannot be completed. A random order strands it in practice, measured.

    Descending order cannot strand anything. When step `i` is reached, the steps
    above it hold `2*(n-1-i)` slots of which `n-2-i` are taken, and
    `2*(n-1-i) > n-2-i` for every `i < n`, so a free slot always remains.
    """
    consumer = [-1] * n_steps
    in_degree = [0] * n_steps

    for i in range(n_steps - 2, -1, -1):
        reach = min(distance_max, n_steps - 1 - i)
        want = i + rng.randint(1, reach)
        # Nearest step with capacity, searching outward from the drawn target.
        # Steps <= i are never eligible: an operand must come from strictly
        # earlier, which is what makes the trace a valid evaluation order. The
        # search may pass DISTANCE_MAX when everything inside it is full — that
        # is the capacity constraint showing, and the realised distance
        # distribution is reported rather than assumed.
        pick = None
        for off in range(n_steps):
            for cand in (want + off, want - off):
                if i < cand < n_steps and in_degree[cand] < MAX_IN_DEGREE:
                    pick = cand
                    break
            if pick is not None:
                break
        if pick is None:
            raise RuntimeError(
                f"no step with capacity for {i} of {n_steps}; descending order "
                f"is supposed to make this unreachable"
            )
        consumer[i] = pick
        in_degree[pick] += 1
    return consumer


def generate_decorrelated_trace(
    rng_seed: int,
    n_steps: int,
    operand_max: int = 100,
    magnitude_cap: int = MAGNITUDE_CAP,
    leaf_values: tuple[int, ...] | None = None,
    distance_max: int = DISTANCE_MAX,
) -> Trace:
    """Build one trace of exactly `n_steps` steps, distance ⟂ provenance.

    Args:
        rng_seed: seed for this trace, so generation is reproducible.
        n_steps: exact step count. Distance to consumer can reach `n_steps - 1`,
            so this is what bounds the measurable range.
        operand_max: upper bound on literal operands. Must not exceed
            `magnitude_cap`, or the operator set can be empty.
        leaf_values: optional pool to draw literals from, as in `generate.py`.

    Returns:
        A Trace with `is_valid()` True, one root, and every other step consumed
        exactly once.
    """
    if n_steps < 2:
        raise ValueError(f"n_steps must be >= 2, got {n_steps}")
    if operand_max > magnitude_cap:
        raise ValueError(
            f"operand_max {operand_max} exceeds magnitude_cap {magnitude_cap}; "
            "the operator set is not guaranteed non-empty"
        )

    rng = random.Random(rng_seed)
    consumer = assign_consumers(rng, n_steps, distance_max)

    producers: list[list[int]] = [[] for _ in range(n_steps)]
    for i, j in enumerate(consumer):
        if j >= 0:
            producers[j].append(i)

    steps: list[Step] = []
    values: list[int] = []
    exprs: list[str] = []

    for j in range(n_steps):
        operands: list[tuple[int, int | None, str]] = [
            (values[i], i, exprs[i]) for i in producers[j]
        ]
        for _ in range(2 - len(operands)):
            v = rng.choice(leaf_values) if leaf_values else rng.randint(1, operand_max)
            operands.append((v, None, str(v)))

        # Which operand lands on the left is a coin flip, so the side an earlier
        # result appears on carries no information beyond the count.
        rng.shuffle(operands)
        (lhs, lhs_from, lhs_expr), (rhs, rhs_from, rhs_expr) = operands

        candidates = [op for op, f in OPS.items()
                      if abs(f(lhs, rhs)) <= magnitude_cap]
        if not candidates:
            raise RuntimeError(f"no operator keeps {lhs} ? {rhs} within {magnitude_cap}")
        op = rng.choice(candidates)
        result = OPS[op](lhs, rhs)

        steps.append(Step(lhs, op, rhs, result, lhs_from, rhs_from))
        values.append(result)
        exprs.append(f"({lhs_expr} {op} {rhs_expr})")

    return Trace(question=exprs[-1][1:-1], steps=tuple(steps), answer=values[-1])


def generate_decorrelated_dataset(
    n: int,
    seed: int = 0,
    min_steps: int = 14,
    max_steps: int = 18,
    operand_max: int = 100,
    magnitude_cap: int = MAGNITUDE_CAP,
    leaf_values: tuple[int, ...] | None = None,
    distance_max: int = DISTANCE_MAX,
) -> list[Trace]:
    """`n` traces with step counts spread over [min_steps, max_steps].

    Length is drawn per trace so every distance from 1 to `max_steps - 1` is
    populated. Seeding follows `generate_dataset`'s convention — trace `i` is
    seeded from `seed * 1_000_003 + i` — so growing `n` leaves earlier traces
    unchanged.
    """
    out = []
    for i in range(n):
        s = seed * 1_000_003 + i
        n_steps = random.Random(s ^ 0x5EED).randint(min_steps, max_steps)
        out.append(generate_decorrelated_trace(
            s, n_steps, operand_max=operand_max, magnitude_cap=magnitude_cap,
            leaf_values=leaf_values, distance_max=distance_max))
    return out
