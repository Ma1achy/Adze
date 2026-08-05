"""M1 acceptance — synthetic trace generation.

These fully specify the generator. Make them pass; do not edit them to match an
implementation.
"""

from __future__ import annotations

import pytest

from adze.data.generate import generate_dataset, generate_trace


def test_generated_trace_is_valid() -> None:
    trace = generate_trace(rng_seed=0, max_depth=3, operand_max=100)
    assert trace.is_valid()


def test_answer_matches_final_step() -> None:
    trace = generate_trace(rng_seed=1, max_depth=3, operand_max=100)
    assert trace.answer == trace.steps[-1].result


def test_every_step_arithmetic_is_correct() -> None:
    trace = generate_trace(rng_seed=2, max_depth=3, operand_max=100)
    for step in trace.steps:
        expected = {"+": step.lhs + step.rhs,
                    "-": step.lhs - step.rhs,
                    "*": step.lhs * step.rhs}[step.op]
        assert step.result == expected


def test_generation_is_reproducible() -> None:
    a = generate_trace(rng_seed=7, max_depth=3, operand_max=100)
    b = generate_trace(rng_seed=7, max_depth=3, operand_max=100)
    assert a.render() == b.render()


def test_different_seeds_differ() -> None:
    traces = {generate_trace(rng_seed=s, max_depth=3).render() for s in range(20)}
    assert len(traces) > 15, "generator is not producing enough variety"


def test_render_is_one_line_per_step() -> None:
    trace = generate_trace(rng_seed=3, max_depth=3)
    lines = [ln for ln in trace.render().splitlines() if ln.strip()]
    assert len(lines) == len(trace.steps)


def test_dataset_size_and_validity() -> None:
    traces = generate_dataset(n=200, seed=0, max_depth=3)
    assert len(traces) == 200
    assert all(t.is_valid() for t in traces)


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_depth_controls_step_count(depth: int) -> None:
    """Deeper trees produce more steps, on average."""
    traces = generate_dataset(n=50, seed=0, max_depth=depth)
    mean_steps = sum(len(t.steps) for t in traces) / len(traces)
    assert mean_steps >= depth * 0.5


def test_length_statistics_are_reportable() -> None:
    """M1 must surface the numbers that decide K and B. This asserts they exist
    and are sane — then go and LOOK at them before choosing K and B in the config."""
    traces = generate_dataset(n=500, seed=0, max_depth=3)
    step_counts = [len(t.steps) for t in traces]
    assert min(step_counts) >= 1
    assert max(step_counts) <= 32, "traces longer than any sane B"
