"""The decorrelating generator.

The point of this generator is a statistical property, so the tests check the
property and not only the plumbing. A generator that produced valid traces with
the confound intact would pass every structural test and be useless.
"""

from __future__ import annotations

import random
from collections import Counter

import pytest

from adze.data.decorrelated import (
    DISTANCE_MAX,
    assign_consumers,
    generate_decorrelated_dataset,
    generate_decorrelated_trace,
)
from adze.eval.strata import consumer_distance, consumers, operand_provenance

CLASSES = ("both-leaves", "one-leaf", "both-from-earlier")


def test_traces_are_valid():
    for t in generate_decorrelated_dataset(200, seed=0):
        assert t.is_valid()


def test_every_step_but_the_last_has_exactly_one_consumer():
    """The pin's meaning depends on this. Multiple consumers would make
    `consumer_distance` a minimum over several, and the M7 semantics would shift
    under the same name."""
    for t in generate_decorrelated_dataset(100, seed=1):
        for b in range(len(t.steps) - 1):
            assert len(consumers(t, b)) == 1
        assert consumers(t, len(t.steps) - 1) == []


def test_the_answer_is_the_last_step():
    for t in generate_decorrelated_dataset(100, seed=2):
        assert t.answer == t.steps[-1].result


def test_generation_is_reproducible():
    a = generate_decorrelated_trace(7, n_steps=10)
    b = generate_decorrelated_trace(7, n_steps=10)
    assert a.render() == b.render()


def test_step_count_is_exact():
    for n in (2, 3, 5, 12, 20):
        assert len(generate_decorrelated_trace(0, n_steps=n).steps) == n


def test_producers_are_strictly_earlier():
    for t in generate_decorrelated_dataset(100, seed=3):
        for i, step in enumerate(t.steps):
            for src in (step.lhs_from, step.rhs_from):
                assert src is None or src < i


def test_consumer_assignment_never_strands_a_step():
    """Descending order is load-bearing: step n-2 can only be consumed by n-1, so
    a random processing order strands it once n-1 fills up."""
    for n in range(2, 40):
        for seed in range(20):
            c = assign_consumers(random.Random(seed), n)
            assert c[-1] == -1
            assert all(x > i for i, x in enumerate(c[:-1]))
            assert max(Counter(c[:-1]).values(), default=0) <= 2


def test_distance_respects_the_bound_except_under_capacity_pressure():
    over = 0
    total = 0
    for t in generate_decorrelated_dataset(200, seed=4):
        for b in range(len(t.steps) - 1):
            total += 1
            if consumer_distance(t, b) > DISTANCE_MAX:
                over += 1
    # Overshoot happens only when every slot inside the bound is taken.
    assert over / total < 0.10


def _interior_xtab(traces):
    cells: Counter = Counter()
    totals: Counter = Counter()
    for t in traces:
        n = len(t.steps)
        for b in range(n):
            if b < 2 or b + DISTANCE_MAX > n - 1:
                continue
            d = consumer_distance(t, b)
            if d is None or d > DISTANCE_MAX - 1:
                continue
            cells[(d, operand_provenance(t.steps[b]))] += 1
            totals[d] += 1
    return cells, totals


def test_provenance_shares_are_flat_across_distance_in_the_interior():
    """THE POINT OF THE GENERATOR. Every class must appear at every distance in
    roughly constant proportion — otherwise a distance profile computed on this
    data is measuring composition again."""
    cells, totals = _interior_xtab(generate_decorrelated_dataset(1500, seed=5))
    ds = [d for d in sorted(totals) if totals[d] >= 200]
    assert len(ds) >= 5, "not enough populated distances to check flatness"
    for c in CLASSES:
        shares = [cells[(d, c)] / totals[d] for d in ds]
        assert max(shares) - min(shares) < 0.10, (c, shares)


def test_every_class_is_populated_at_every_distance():
    cells, totals = _interior_xtab(generate_decorrelated_dataset(1500, seed=6))
    for d in [d for d in sorted(totals) if totals[d] >= 200]:
        for c in CLASSES:
            assert cells[(d, c)] > 0, (d, c)


def test_it_is_far_flatter_than_the_original_generator():
    """The original is the baseline the generator exists to beat, so the
    comparison is asserted rather than left to a report nobody reruns."""
    from adze.data.generate import generate_dataset

    def spread(traces, interior):
        cells: Counter = Counter()
        totals: Counter = Counter()
        for t in traces:
            n = len(t.steps)
            for b in range(n):
                if interior and (b < 2 or b + DISTANCE_MAX > n - 1):
                    continue
                d = consumer_distance(t, b)
                if d is None:
                    continue
                cells[(d, operand_provenance(t.steps[b]))] += 1
                totals[d] += 1
        ds = [d for d in sorted(totals) if totals[d] >= 200]
        sh = [cells[(d, "both-leaves")] / totals[d] for d in ds]
        return max(sh) - min(sh)

    old = spread(generate_dataset(1500, seed=0, max_depth=3), interior=False)
    new = spread(generate_decorrelated_dataset(1500, seed=7), interior=True)
    assert new < old / 3, (new, old)


def test_operand_max_above_the_cap_is_rejected():
    with pytest.raises(ValueError):
        generate_decorrelated_trace(0, n_steps=5, operand_max=2000,
                                    magnitude_cap=1000)
