"""The empirical leaf sampler.

The defect it repairs: leaves uniform on 1..operand_max while results reach
MAGNITUDE_CAP, so the model met small values as inputs and large values only as
outputs. These tests hold the two properties that make the fix a fix rather than a
reparameterisation — that the pool is actually used, and that iterating converges —
plus the invariants the rest of M1 already depends on and must not lose.
"""

from __future__ import annotations

import pytest

from adze.data.generate import (
    MAGNITUDE_CAP,
    configured_leaves,
    fixed_point_leaves,
    generate_dataset,
    result_magnitudes,
)


def test_uniform_is_unchanged_by_the_new_argument() -> None:
    """leaf_values=None must reproduce the original sampler exactly."""
    a = generate_dataset(n=50, seed=3, max_depth=3, operand_max=20)
    b = generate_dataset(n=50, seed=3, max_depth=3, operand_max=20, leaf_values=None)
    assert [t.render() for t in a] == [t.render() for t in b]


def test_leaves_come_from_the_pool() -> None:
    """A literal operand — one with no producer — must be a pool member.

    Operands WITH a producer are earlier results and are not constrained by the
    pool, so the check is restricted to `*_from is None`.
    """
    pool = (7, 11, 13)
    traces = generate_dataset(n=60, seed=1, max_depth=3, operand_max=20,
                              leaf_values=pool)
    literals = [
        v for t in traces for s in t.steps
        for v, src in ((s.lhs, s.lhs_from), (s.rhs, s.rhs_from)) if src is None
    ]
    assert literals, "no literal leaves in the sample — the test proves nothing"
    assert set(literals) <= set(pool)


def test_traces_stay_valid_under_the_new_sampler() -> None:
    """The whole M1 contract — arithmetic and provenance — must survive."""
    pool = fixed_point_leaves(iterations=1, n=300, seed=0, operand_max=20)
    traces = generate_dataset(n=200, seed=5, max_depth=3, operand_max=20,
                              leaf_values=pool)
    assert all(t.is_valid() for t in traces)
    assert all(abs(s.result) <= MAGNITUDE_CAP for t in traces for s in t.steps)


def test_the_iteration_converges() -> None:
    """Input and output magnitude distributions must approach each other.

    This is the property the fix rests on: if the iteration did not converge, leaf
    and result distributions would never agree and the approach would be wrong.
    Measured in total variation over decade bins, which is what the reporting
    script prints.
    """
    bins = [(0, 9), (10, 29), (30, 99), (100, 299), (300, 10**9)]

    def shares(values):
        n = max(len(values), 1)
        return [sum(1 for v in values if lo <= abs(v) <= hi) / n for lo, hi in bins]

    def tv(pool):
        results = result_magnitudes(
            generate_dataset(n=1500, seed=11, max_depth=3, operand_max=20,
                             leaf_values=pool)
        )
        return sum(abs(a - b) for a, b in zip(shares(pool), shares(results))) / 2

    early = fixed_point_leaves(iterations=1, n=1500, seed=0, operand_max=20)
    late = fixed_point_leaves(iterations=5, n=1500, seed=0, operand_max=20)
    assert tv(late) < tv(early)
    assert tv(late) < 0.10


def test_configured_leaves_validates_and_is_deterministic() -> None:
    a = configured_leaves("empirical", 1, 0, 3, 20)
    b = configured_leaves("empirical", 1, 0, 3, 20)
    assert a == b and a is not None
    assert configured_leaves("uniform", 1, 0, 3, 20) is None
    with pytest.raises(ValueError):
        configured_leaves("gaussian", 1, 0, 3, 20)
