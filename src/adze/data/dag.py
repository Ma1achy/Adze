"""DAG (directed acyclic graph) arithmetic trace generator.

## Why this generator exists

The tree generator (`generate.py`) and the decorrelated generator
(`decorrelated.py`) both produce traces where each computed value has EXACTLY
ONE consumer — either its tree-parent or the one step that drew it as a consumer.
That makes provenance and consumer-distance properties of the same event:
the move that places a value is the same move that determines how far away its
consumer sits. Decorrelating the two variables did not help; it removed the
adjacency that made the pin exploitable.

The fix is a topology change: allow each value to be consumed by MULTIPLE later
steps at different distances. Then one value is its own control — same provenance,
same corruption, evidence at two distances. The within-record intervention
(mask the near consumer, keep the far one, or vice versa) measures reach with
provenance held constant by construction.

## Structure

A DagTrace contains:
  - the same `Step` objects as the tree generator (lhs, op, rhs, result,
    lhs_from, rhs_from) — no schema change
  - a `consumer_map`: for each step i, the sorted tuple of later step indices
    j that use step i's result as an operand

Consumer assignment is done BEFORE step values are computed (construct-time-only
rule). Operators are then chosen from each step's children's actual values, never
patched afterwards.

## Mask-count discipline for the intervention

The three-arm intervention (both consumers visible, near-only, far-only) requires
that arms (b) and (c) erase the same number of blocks. This is guaranteed by
restricting the intervention to steps with EXACTLY ONE near consumer (d ≤ 2) AND
EXACTLY ONE far consumer (d ≥ 5). Each arm then erases exactly one block.

## Corruption method

Use the same random-delta method as M7's `corrupt.py` — perturb the result by a
signed offset, not by operator swap. Operator swaps produce a constrained set
of wrong answers {a+b, a-b, a*b}; a model that memorised the swap distribution
could narrow the search without reading evidence.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from adze.data.generate import MAGNITUDE_CAP, OPS, Step

MAX_IN_DEGREE = 2   # each step has two operand slots (lhs and rhs)


@dataclass(frozen=True)
class DagTrace:
    """A multi-consumer arithmetic trace — structural analogue of `Trace`.

    Shapes / semantics:
        question     : the final expression as a string (outermost expression)
        steps        : tuple of Step objects, same schema as in generate.py
        answer       : steps[-1].result
        consumer_map : consumer_map[i] = sorted tuple of indices j > i where
                       step j uses step i's result as lhs_from or rhs_from.
                       consumer_map[-1] = () — the root has no consumers.
                       Derived from steps but stored for O(1) access.
    """

    question: str
    steps: tuple[Step, ...]
    answer: int
    consumer_map: tuple[tuple[int, ...], ...]

    def is_valid(self) -> bool:
        """Full correctness check: arithmetic, provenance, and consumer_map."""
        n = len(self.steps)
        if n == 0 or len(self.consumer_map) != n:
            return False

        values = []
        for idx, s in enumerate(self.steps):
            # Arithmetic.
            if s.op not in OPS:
                return False
            if OPS[s.op](s.lhs, s.rhs) != s.result:
                return False
            # Provenance source values are consistent with earlier steps.
            if s.lhs_from is not None:
                if s.lhs_from >= idx or values[s.lhs_from] != s.lhs:
                    return False
            if s.rhs_from is not None:
                if s.rhs_from >= idx or values[s.rhs_from] != s.rhs:
                    return False
            values.append(s.result)

        # Final answer.
        if self.steps[-1].result != self.answer:
            return False

        # consumer_map consistency: reconstruct from lhs_from/rhs_from and compare.
        expected: list[list[int]] = [[] for _ in range(n)]
        for j, s in enumerate(self.steps):
            if s.lhs_from is not None:
                expected[s.lhs_from].append(j)
            if s.rhs_from is not None:
                expected[s.rhs_from].append(j)
        for i in range(n):
            if tuple(sorted(expected[i])) != self.consumer_map[i]:
                return False

        return True


def assign_consumers_dag(
    rng: random.Random,
    n_steps: int,
    min_consumers: int,
    max_consumers: int,
    distance_min: int,
    distance_max: int,
) -> list[list[int]]:
    """Assign consumer lists to each step.

    Returns consumer_lists where consumer_lists[i] is the sorted list of step
    indices j > i that will use step i's result as an operand.

    Each step i draws k = randint(min_consumers, max_consumers) consumers.
    Consumers are drawn by distance, uniformly over distance_min..reach, where
    reach = min(distance_max, n_steps - 1 - i). If a drawn target is already
    full or already assigned to this step, a bidirectional outward search finds
    the nearest free slot. If no slot is found, the attempt is skipped.

    Steps are processed in DESCENDING order (most constrained first), for the
    same reason as `decorrelated.assign_consumers`: step n-2 can only be consumed
    by step n-1, and if n-1 fills up first the step has nowhere to go.

    The last step (root) has an empty consumer list.
    """
    consumer_lists: list[list[int]] = [[] for _ in range(n_steps)]
    in_degree = [0] * n_steps   # how many steps have drawn j as a consumer

    for i in range(n_steps - 2, -1, -1):
        k = rng.randint(min_consumers, max_consumers)
        assigned: set[int] = set()

        for _ in range(k):
            reach = min(distance_max, n_steps - 1 - i)
            if reach < distance_min:
                break
            want = i + rng.randint(distance_min, reach)
            # Bidirectional outward search from `want`.
            pick = None
            for off in range(n_steps):
                for cand in (want + off, want - off):
                    if (i < cand < n_steps
                            and in_degree[cand] < MAX_IN_DEGREE
                            and cand not in assigned):
                        pick = cand
                        break
                if pick is not None:
                    break
            if pick is not None:
                consumer_lists[i].append(pick)
                in_degree[pick] += 1
                assigned.add(pick)

    for lst in consumer_lists:
        lst.sort()

    return consumer_lists


def generate_dag_trace(
    rng_seed: int,
    n_steps: int,
    min_consumers: int = 1,
    max_consumers: int = 2,
    distance_min: int = 1,
    distance_max: int = 8,
    operand_max: int = 100,
    magnitude_cap: int = MAGNITUDE_CAP,
    leaf_values: tuple[int, ...] | None = None,
) -> DagTrace:
    """Build one DAG trace of exactly `n_steps` steps.

    Args:
        rng_seed      : reproducible seed.
        n_steps       : exact number of steps. Must be >= 2.
        min_consumers : minimum consumers to request per step (default 1).
        max_consumers : maximum consumers to request per step (default 2).
        distance_min  : minimum drawn consumer distance.
        distance_max  : maximum drawn consumer distance (actual may exceed under
                        capacity pressure; overshoot is recorded in consumer_map).
        operand_max   : upper bound for fresh literal values. Must be ≤
                        magnitude_cap or the operator set can be empty.
        magnitude_cap : absolute value ceiling on all intermediate results.
        leaf_values   : optional pool to draw literals from (same fixed-point
                        leaves as in generate.py and decorrelated.py).

    Returns:
        DagTrace with is_valid() True.

    Raises:
        ValueError   : on bad parameter combinations.
        RuntimeError : if no operator keeps a step's result within magnitude_cap
                       (a bug, since +/- always satisfies this given the cap).
    """
    if n_steps < 2:
        raise ValueError(f"n_steps must be >= 2, got {n_steps}")
    if operand_max > magnitude_cap:
        raise ValueError(
            f"operand_max {operand_max} exceeds magnitude_cap {magnitude_cap}; "
            "the operator set is not guaranteed non-empty"
        )

    rng = random.Random(rng_seed)

    # ── step 1: pre-assign consumers ──────────────────────────────────────────
    consumer_lists = assign_consumers_dag(
        rng, n_steps, min_consumers, max_consumers, distance_min, distance_max)

    # ── step 2: build producers_of[j] from the consumer lists ─────────────────
    producers_of: list[list[int]] = [[] for _ in range(n_steps)]
    for i, consumers in enumerate(consumer_lists):
        for j in consumers:
            producers_of[j].append(i)
    # producers_of[j] has at most 2 entries (MAX_IN_DEGREE constraint).

    # ── step 3: emit steps in order 0..n_steps-1 ─────────────────────────────
    steps: list[Step] = []
    values: list[int] = []
    exprs: list[str] = []

    for j in range(n_steps):
        # Operands from earlier producers.
        operands: list[tuple[int, int | None, str]] = [
            (values[i], i, exprs[i]) for i in producers_of[j]
        ]
        # Fill remaining slots with fresh literals.
        for _ in range(2 - len(operands)):
            v = (rng.choice(leaf_values) if leaf_values
                 else rng.randint(1, operand_max))
            operands.append((v, None, str(v)))

        # Coin-flip which producer lands on the left (same as decorrelated.py).
        rng.shuffle(operands)
        (lhs, lhs_from, lhs_expr), (rhs, rhs_from, rhs_expr) = operands

        candidates = [op for op, f in OPS.items()
                      if abs(f(lhs, rhs)) <= magnitude_cap]
        if not candidates:
            # +/- always fits since |lhs ± rhs| ≤ |lhs| + |rhs| ≤ 2 * cap.
            raise RuntimeError(
                f"no operator keeps {lhs} ? {rhs} within {magnitude_cap}; "
                "this should be unreachable"
            )
        op = rng.choice(candidates)
        result = OPS[op](lhs, rhs)

        steps.append(Step(lhs, op, rhs, result, lhs_from, rhs_from))
        values.append(result)
        exprs.append(f"({lhs_expr} {op} {rhs_expr})")

    consumer_map = tuple(tuple(lst) for lst in consumer_lists)

    return DagTrace(
        question=exprs[-1][1:-1],
        steps=tuple(steps),
        answer=values[-1],
        consumer_map=consumer_map,
    )


def generate_dag_dataset(
    n: int,
    seed: int = 0,
    n_steps: int = 10,
    min_consumers: int = 1,
    max_consumers: int = 2,
    distance_min: int = 1,
    distance_max: int = 8,
    operand_max: int = 100,
    magnitude_cap: int = MAGNITUDE_CAP,
    leaf_values: tuple[int, ...] | None = None,
) -> list[DagTrace]:
    """Generate `n` DAG traces. Trace i is seeded from `seed * 1_000_003 + i`."""
    return [
        generate_dag_trace(
            rng_seed=seed * 1_000_003 + i,
            n_steps=n_steps,
            min_consumers=min_consumers,
            max_consumers=max_consumers,
            distance_min=distance_min,
            distance_max=distance_max,
            operand_max=operand_max,
            magnitude_cap=magnitude_cap,
            leaf_values=leaf_values,
        )
        for i in range(n)
    ]
