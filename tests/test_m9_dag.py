"""Tests for the DAG generator and strata helpers."""

from __future__ import annotations

import pytest

from adze.data.dag import (
    DagTrace,
    generate_dag_dataset,
    generate_dag_trace,
)
from adze.data.generate import OPS, MAGNITUDE_CAP, Step
from adze.eval.dag_strata import (
    consumer_distances,
    exactly_one_near_one_far,
    farthest_consumer_distance,
    has_near_and_far,
    nearest_consumer_distance,
)
from adze.eval.strata import PROVENANCE, operand_provenance


# ── helpers ───────────────────────────────────────────────────────────────────

def make_dag_trace(steps_args, consumer_map_raw) -> DagTrace:
    """Hand-build a DagTrace for testing.

    steps_args: list of (lhs, op, rhs, result, lhs_from, rhs_from)
    consumer_map_raw: list of lists of ints
    """
    steps = tuple(
        Step(lhs, op, rhs, result, lhs_from, rhs_from)
        for lhs, op, rhs, result, lhs_from, rhs_from in steps_args
    )
    consumer_map = tuple(tuple(sorted(cs)) for cs in consumer_map_raw)
    answer = steps[-1].result
    question = steps[-1].render()
    return DagTrace(question=question, steps=steps, answer=answer,
                    consumer_map=consumer_map)


# ── is_valid ──────────────────────────────────────────────────────────────────

def test_is_valid_hand_built_dag():
    """Step 0's result (8) is used by both step 1 and step 3."""
    # step 0: 5 + 3 = 8            lhs_from=None, rhs_from=None
    # step 1: 8 + 2 = 10           lhs_from=0
    # step 2: 4 - 1 = 3            lhs_from=None, rhs_from=None
    # step 3: 8 * 3 = 24           lhs_from=0, rhs_from=2  (uses step 0 and step 2)
    steps_args = [
        (5, "+", 3, 8, None, None),
        (8, "+", 2, 10, 0, None),
        (4, "-", 1, 3, None, None),
        (8, "*", 3, 24, 0, 2),
    ]
    consumer_map = [[1, 3], [3], [], []]   # step 0 → {1, 3}; step 1 → {3}
    # wait — step 3 uses lhs_from=0 (step0) and rhs_from=2 (step2)
    # so consumer_map[0] = [1, 3], consumer_map[1] = [], consumer_map[2] = [3]
    consumer_map = [[1, 3], [], [3], []]
    tr = make_dag_trace(steps_args, consumer_map)
    assert tr.is_valid()


def test_is_valid_consumer_map_inconsistency():
    """consumer_map says step 0 has consumer 2, but step 2 has no lhs_from/rhs_from = 0."""
    steps_args = [
        (5, "+", 3, 8, None, None),
        (8, "+", 2, 10, 0, None),
        (4, "-", 1, 3, None, None),   # does NOT use step 0
        (10, "*", 3, 30, 1, None),
    ]
    consumer_map = [[1, 2], [], [], []]   # wrong: 2 is not a consumer of 0
    tr = make_dag_trace(steps_args, consumer_map)
    assert not tr.is_valid()


def test_is_valid_wrong_arithmetic():
    """One step has a wrong result; is_valid must catch it."""
    steps_args = [
        (5, "+", 3, 99, None, None),   # wrong: 5+3=8, not 99
        (99, "+", 2, 101, 0, None),
    ]
    consumer_map = [[1], []]
    tr = make_dag_trace(steps_args, consumer_map)
    assert not tr.is_valid()


# ── consumer_distances ────────────────────────────────────────────────────────

def test_consumer_distances_multi():
    """Step 0 consumed by steps 1 and 3 → distances (1, 3)."""
    steps_args = [
        (5, "+", 3, 8, None, None),
        (8, "+", 2, 10, 0, None),
        (4, "-", 1, 3, None, None),
        (8, "*", 3, 24, 0, 2),
    ]
    consumer_map = [[1, 3], [], [3], []]
    tr = make_dag_trace(steps_args, consumer_map)
    assert consumer_distances(tr, 0) == (1, 3)
    assert consumer_distances(tr, 1) == ()   # step 1 has no consumers
    assert consumer_distances(tr, 2) == (1,)
    assert consumer_distances(tr, 3) == ()   # root


def test_root_consumer_distances_empty():
    tr = generate_dag_trace(rng_seed=0, n_steps=5)
    assert consumer_distances(tr, len(tr.steps) - 1) == ()


def test_nearest_farthest():
    """nearest and farthest consumer distances on a known trace."""
    steps_args = [
        (5, "+", 3, 8, None, None),
        (8, "+", 2, 10, 0, None),
        (4, "-", 1, 3, None, None),
        (8, "*", 3, 24, 0, 2),
    ]
    consumer_map = [[1, 3], [], [3], []]
    tr = make_dag_trace(steps_args, consumer_map)
    assert nearest_consumer_distance(tr, 0) == 1
    assert farthest_consumer_distance(tr, 0) == 3
    assert nearest_consumer_distance(tr, 3) is None   # root


# ── has_near_and_far / exactly_one_near_one_far ───────────────────────────────

def test_has_near_and_far_true():
    """A step with consumers at d=1 and d=6 passes has_near_and_far."""
    # Build manually: step 0 → consumers at step 1 (d=1) and step 6 (d=6)
    # Need a 7-step trace where step 0's result is used by step 1 and step 6.
    # Simple: step 0 = 3+4=7; steps 1-5 = literals; step 6 uses step 0
    steps_args = [
        (3, "+", 4, 7, None, None),      # 0
        (7, "+", 1, 8, 0, None),          # 1: uses step 0 (d=1)
        (2, "+", 1, 3, None, None),       # 2
        (3, "+", 1, 4, None, None),       # 3
        (4, "+", 1, 5, None, None),       # 4
        (5, "+", 1, 6, None, None),       # 5
        (7, "-", 1, 6, 0, None),          # 6: uses step 0 (d=6)
    ]
    consumer_map = [[1, 6], [], [], [], [], [], []]
    tr = make_dag_trace(steps_args, consumer_map)
    assert tr.is_valid()
    assert has_near_and_far(tr, 0, near_max=2, far_min=5)
    assert exactly_one_near_one_far(tr, 0, near_max=2, far_min=5)


def test_has_near_and_far_false_no_far():
    """A step with consumers at d=1 and d=3 only (no d>=5) is False."""
    steps_args = [
        (3, "+", 4, 7, None, None),      # 0
        (7, "+", 1, 8, 0, None),          # 1: d=1
        (2, "+", 1, 3, None, None),       # 2
        (7, "-", 1, 6, 0, None),          # 3: d=3 — not far enough
    ]
    consumer_map = [[1, 3], [], [], []]
    tr = make_dag_trace(steps_args, consumer_map)
    assert not has_near_and_far(tr, 0, near_max=2, far_min=5)


def test_exactly_one_near_one_far_two_near():
    """Two near consumers and one far → NOT exactly_one_near_one_far."""
    steps_args = [
        (3, "+", 4, 7, None, None),      # 0
        (7, "+", 1, 8, 0, None),          # 1: d=1 (near)
        (7, "-", 1, 6, 0, None),          # 2: d=2 (near) — second near
        (2, "+", 1, 3, None, None),       # 3
        (3, "+", 1, 4, None, None),       # 4
        (4, "+", 1, 5, None, None),       # 5: d=5 (far)
    ]
    consumer_map = [[1, 2, 5], [], [], [], [], []]
    # Step 5 uses step 0 AND step? Let's fix step 5's lhs_from
    steps_args[-1] = (7, "+", 1, 8, 0, None)   # step 5 uses step 0
    tr = make_dag_trace(steps_args, consumer_map)
    # Two near (d=1, d=2) and one far (d=5) → not exactly one near
    assert not exactly_one_near_one_far(tr, 0, near_max=2, far_min=5)


# ── operand_provenance works on DagTrace steps ───────────────────────────────

def test_provenance_both_leaves():
    """A step using two literals → both-leaves."""
    tr = generate_dag_trace(rng_seed=0, n_steps=5)
    # Step 0 always has lhs_from=None and rhs_from=None (nothing before it).
    s0 = tr.steps[0]
    assert s0.lhs_from is None and s0.rhs_from is None
    assert operand_provenance(s0) == "both-leaves"


def test_provenance_valid_for_all_steps():
    """All generated steps return a valid PROVENANCE label."""
    tr = generate_dag_trace(rng_seed=42, n_steps=8)
    for s in tr.steps:
        assert operand_provenance(s) in PROVENANCE


# ── generated traces ──────────────────────────────────────────────────────────

def test_generated_traces_are_valid():
    """100 seeds, all traces pass is_valid()."""
    for seed in range(100):
        tr = generate_dag_trace(rng_seed=seed, n_steps=8)
        assert tr.is_valid(), f"trace seed={seed} failed is_valid()"


def test_fan_out_distributed():
    """Over 200 traces, at least some steps have k=2 consumers."""
    traces = generate_dag_dataset(n=200, seed=0, n_steps=10, max_consumers=2)
    k2_count = sum(
        1 for tr in traces
        for i in range(len(tr.steps) - 1)
        if len(tr.consumer_map[i]) >= 2
    )
    assert k2_count > 0, "no step with 2 consumers in 200 traces"


def test_reproducibility():
    """Same seed → same trace."""
    t1 = generate_dag_trace(rng_seed=7, n_steps=6)
    t2 = generate_dag_trace(rng_seed=7, n_steps=6)
    assert t1 == t2


def test_step_count_is_exact():
    """generate_dag_trace always produces exactly n_steps steps."""
    for n in (2, 5, 10, 15):
        tr = generate_dag_trace(rng_seed=0, n_steps=n)
        assert len(tr.steps) == n
        assert len(tr.consumer_map) == n


def test_root_has_no_consumer():
    """The last step (root) always has an empty consumer tuple."""
    for seed in range(20):
        tr = generate_dag_trace(rng_seed=seed, n_steps=8)
        assert tr.consumer_map[-1] == ()


def test_producers_are_strictly_earlier():
    """Every lhs_from / rhs_from is strictly less than the step index."""
    for seed in range(50):
        tr = generate_dag_trace(rng_seed=seed, n_steps=8)
        for idx, s in enumerate(tr.steps):
            if s.lhs_from is not None:
                assert s.lhs_from < idx
            if s.rhs_from is not None:
                assert s.rhs_from < idx


def test_operand_max_above_cap_rejected():
    """operand_max > magnitude_cap must raise ValueError."""
    with pytest.raises(ValueError, match="magnitude_cap"):
        generate_dag_trace(rng_seed=0, n_steps=5, operand_max=MAGNITUDE_CAP + 1)


def test_answer_is_last_step():
    """trace.answer == trace.steps[-1].result."""
    for seed in range(20):
        tr = generate_dag_trace(rng_seed=seed, n_steps=7)
        assert tr.answer == tr.steps[-1].result
