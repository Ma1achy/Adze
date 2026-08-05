"""M1 acceptance — corruption and matched pairs.

The critical property: corruption must be INCONSISTENT. One step's result is
changed and nothing downstream is recomputed, so later steps continue to use the
correct value. That contradiction is what a global model can see and a causal one
cannot.
"""

from __future__ import annotations

from adze.data.corrupt import corrupt_step, make_pair
from adze.data.generate import generate_trace


def test_corrupted_trace_is_invalid() -> None:
    trace = generate_trace(rng_seed=0, max_depth=3)
    bad = corrupt_step(trace, block_index=0, rng_seed=1)
    assert not bad.is_valid()


def test_exactly_one_step_is_wrong() -> None:
    trace = generate_trace(rng_seed=0, max_depth=3)
    idx = 0
    bad = corrupt_step(trace, block_index=idx, rng_seed=1)
    wrong = [
        i for i, s in enumerate(bad.steps)
        if s.result != {"+": s.lhs + s.rhs, "-": s.lhs - s.rhs, "*": s.lhs * s.rhs}[s.op]
    ]
    assert wrong == [idx]


def test_downstream_steps_are_untouched() -> None:
    """The whole point: later steps keep the ORIGINAL correct value, so the chain
    contradicts itself. If downstream were recomputed the corruption would be
    invisible from later context and the experiment would be meaningless."""
    trace = generate_trace(rng_seed=0, max_depth=3)
    idx = 0
    bad = corrupt_step(trace, block_index=idx, rng_seed=1)
    for i, (clean_step, bad_step) in enumerate(zip(trace.steps, bad.steps)):
        if i != idx:
            assert clean_step == bad_step, f"step {i} changed but only {idx} should have"


def test_corruption_is_not_a_noop() -> None:
    trace = generate_trace(rng_seed=0, max_depth=3)
    bad = corrupt_step(trace, block_index=0, rng_seed=1)
    assert bad.steps[0].result != trace.steps[0].result


def test_corruption_is_reproducible() -> None:
    trace = generate_trace(rng_seed=0, max_depth=3)
    a = corrupt_step(trace, block_index=0, rng_seed=5)
    b = corrupt_step(trace, block_index=0, rng_seed=5)
    assert a.render() == b.render()


def test_pair_records_the_index() -> None:
    trace = generate_trace(rng_seed=0, max_depth=3)
    pair = make_pair(trace, rng_seed=2)
    assert 0 <= pair.block_index < len(trace.steps)
    assert pair.clean.is_valid()
    assert not pair.corrupted.is_valid()


def test_pair_index_round_trips() -> None:
    """The corrupted index must be recoverable — the central experiment erases
    exactly this block."""
    trace = generate_trace(rng_seed=0, max_depth=3)
    pair = make_pair(trace, rng_seed=2)
    differing = [
        i for i, (c, b) in enumerate(zip(pair.clean.steps, pair.corrupted.steps)) if c != b
    ]
    assert differing == [pair.block_index]


def test_prefer_early_biases_toward_first_half() -> None:
    """Early corruption is what makes later blocks carry the repair evidence."""
    indices = []
    for s in range(200):
        trace = generate_trace(rng_seed=s, max_depth=3)
        if len(trace.steps) < 2:
            continue
        pair = make_pair(trace, rng_seed=s, prefer_early=True)
        indices.append(pair.block_index / max(len(trace.steps) - 1, 1))
    assert sum(indices) / len(indices) < 0.5
