"""M7 — the stratification controls.

Two constructors and a set of pure cuts. What must hold:

  * `corrupt_final` targets the root, and the root has no consumer — that is the
    genuine null, and it is only a null if nothing consumes it
  * `corrupt_consistent` MOVES the pin rather than removing it: every downstream
    consumer holds the perturbed value, every downstream step is arithmetically
    correct given it, and the perturbed step's own operands are untouched
  * the chance rate is a permutation null and reads as one on a case with a
    known answer
"""

from __future__ import annotations

import pytest

from adze.data.corrupt import corrupt_consistent, corrupt_final, make_pair
from adze.data.generate import Step, Trace, generate_dataset
from adze.eval.strata import (
    Cell,
    chance_rate,
    cell,
    consumer_distance,
    consumers,
    operand_provenance,
    stratify,
)


def _chain() -> Trace:
    """2 + 3 = 5 ; 5 * 4 = 20 ; 20 - 1 = 19. Step 1 consumes step 0; step 2 step 1."""
    steps = (
        Step(2, "+", 3, 5),
        Step(5, "*", 4, 20, lhs_from=0),
        Step(20, "-", 1, 19, lhs_from=1),
    )
    return Trace(question="((2 + 3) * 4) - 1", steps=steps, answer=19)


# --- the cuts ---------------------------------------------------------------


def test_consumers_are_found_by_provenance_not_by_value() -> None:
    """A coincidental value match must not count as consumption."""
    t = _chain()
    assert consumers(t, 0) == [1]
    assert consumers(t, 1) == [2]
    assert consumers(t, 2) == []          # the root: nothing downstream

    # Step 2's lhs is 20, which equals step 1's result — but its provenance says
    # so explicitly. Now a step whose value coincides with an earlier result while
    # recording no producer must NOT be picked up.
    coincidence = Trace(
        question="q",
        steps=(Step(2, "+", 3, 5), Step(5, "+", 1, 6)),   # 5 matches, lhs_from None
        answer=6,
    )
    assert consumers(coincidence, 0) == []


def test_consumer_distance_and_the_no_pin_case() -> None:
    t = _chain()
    assert consumer_distance(t, 0) == 1
    assert consumer_distance(t, 1) == 1
    assert consumer_distance(t, 2) is None      # no pin


def test_operand_provenance_classes() -> None:
    assert operand_provenance(Step(2, "+", 3, 5)) == "both-leaves"
    assert operand_provenance(Step(5, "*", 4, 20, lhs_from=0)) == "one-leaf"
    assert operand_provenance(
        Step(5, "*", 4, 20, lhs_from=0, rhs_from=1)
    ) == "both-from-earlier"


def test_the_unconsumed_cell_is_structurally_empty_on_this_generator() -> None:
    """The control the design was built around does not exist here, and the reason
    is structural rather than a sampling accident.

    `_emit` walks the tree post-order, so every step but the root has exactly one
    later consumer. `make_pair` excludes the root. This test exists so that if the
    generator ever changes, the fact that the null cell reappears is announced
    rather than discovered by a puzzling table.
    """
    traces = generate_dataset(n=200, seed=11, max_depth=3, operand_max=20)
    pairs = [make_pair(t, rng_seed=i) for i, t in enumerate(traces)
             if len(t.steps) >= 2]
    assert pairs
    assert all(consumers(p.clean, p.block_index) for p in pairs)


# --- corrupt_final: the genuine null ----------------------------------------


def test_corrupt_final_targets_the_root_and_leaves_it_unconsumed() -> None:
    traces = generate_dataset(n=50, seed=12, max_depth=3, operand_max=20)
    for i, t in enumerate(traces):
        if len(t.steps) < 2:
            continue
        pair = corrupt_final(t, rng_seed=i)
        assert pair.block_index == len(t.steps) - 1
        assert consumers(pair.clean, pair.block_index) == []
        # Exactly one step differs, and it is the root.
        differ = [j for j, (a, b) in enumerate(zip(t.steps, pair.corrupted.steps))
                  if a != b]
        assert differ == [pair.block_index]


# --- corrupt_consistent: the pin is MOVED, not removed ----------------------


def test_consistent_corruption_moves_the_pin_to_the_perturbed_value() -> None:
    """THE PROPERTY THE WHOLE CONDITION RESTS ON.

    If a downstream consumer still holds the ORIGINAL value, the pin was not moved
    and the condition measures nothing. Checked through recorded provenance, never
    by scanning for a value — two steps can share a number by coincidence.
    """
    traces = generate_dataset(n=200, seed=13, max_depth=3, operand_max=20)
    checked = 0
    for i, t in enumerate(traces):
        if len(t.steps) < 2:
            continue
        pair = corrupt_consistent(t, rng_seed=i)
        b, steps = pair.block_index, pair.corrupted.steps

        for j in range(b + 1, len(steps)):
            step = steps[j]
            if step.lhs_from is not None:
                assert step.lhs == steps[step.lhs_from].result
            if step.rhs_from is not None:
                assert step.rhs == steps[step.rhs_from].result
        checked += 1
    assert checked > 50


def test_consistent_corruption_leaves_the_perturbed_step_recomputable() -> None:
    """The sharp cell depends on this: block b's OPERANDS and OPERATOR are
    untouched, so a causal arm that can see the prefix can recompute the clean
    result exactly while global is pinned to the corrupted one."""
    traces = generate_dataset(n=100, seed=14, max_depth=3, operand_max=20)
    for i, t in enumerate(traces):
        if len(t.steps) < 2:
            continue
        pair = corrupt_consistent(t, rng_seed=i)
        b = pair.block_index
        before, after = t.steps[b], pair.corrupted.steps[b]
        assert (after.lhs, after.op, after.rhs) == (before.lhs, before.op, before.rhs)
        assert after.result != before.result
        assert after.lhs_from == before.lhs_from
        assert after.rhs_from == before.rhs_from


def test_consistent_corruption_relocates_the_inconsistency_into_block_b() -> None:
    """`is_valid()` is False, and MUST be.

    Leaving block b's operands alone is what makes it recomputable, and that means
    `lhs op rhs != result` at b exactly. The point of the condition is not that the
    trace becomes valid — it is that everything OUTSIDE b now agrees on the
    corrupted value instead of contradicting it. Making b self-consistent would
    require changing its operands or operator and would destroy the sharp cell.
    """
    from adze.data.generate import OPS

    traces = generate_dataset(n=100, seed=15, max_depth=3, operand_max=20)
    for i, t in enumerate(traces):
        if len(t.steps) < 2:
            continue
        pair = corrupt_consistent(t, rng_seed=i)
        b, steps = pair.block_index, pair.corrupted.steps

        assert not pair.corrupted.is_valid()
        for j, s in enumerate(steps):
            arithmetic_holds = s.result == OPS[s.op](s.lhs, s.rhs)
            assert arithmetic_holds == (j != b), f"step {j} (b={b})"
        assert pair.corrupted.answer == steps[-1].result


def test_consistent_corruption_never_targets_the_last_step() -> None:
    """With nothing downstream there is no pin to redirect, so the condition would
    silently degenerate into the null."""
    traces = generate_dataset(n=100, seed=16, max_depth=3, operand_max=20)
    for i, t in enumerate(traces):
        if len(t.steps) < 2:
            continue
        pair = corrupt_consistent(t, rng_seed=i)
        assert pair.block_index < len(t.steps) - 1
        assert consumers(pair.clean, pair.block_index)


def test_both_constructors_reject_traces_too_short_to_be_meaningful() -> None:
    one = Trace(question="q", steps=(Step(1, "+", 1, 2),), answer=2)
    with pytest.raises(ValueError):
        corrupt_final(one, rng_seed=0)
    with pytest.raises(ValueError):
        corrupt_consistent(one, rng_seed=0)


# --- chance ------------------------------------------------------------------


def _rec(block: int, clean: str, causal: str, glob: str) -> dict:
    return {
        "block": block,
        "clean_result": clean,
        "causal": {"result": causal},
        "global": {"result": glob},
    }


def test_chance_is_a_permutation_null_and_reads_as_one() -> None:
    """Every record shares a target, so every shuffled pairing matches: chance 1.0.

    A chance rate that came out below 1.0 here would mean the permutation is not
    scoring against another record's target at all.
    """
    records = [_rec(0, "X", "X", "X") for _ in range(20)]
    mean, sd = chance_rate(records, "causal", "result", permutations=20)
    assert mean == 1.0 and sd == 0.0

    # Distinct targets, and an arm that always emits the same thing: exactly one
    # of n shuffled pairings can match, so chance is 1/n and the true rate is too.
    distinct = [_rec(0, str(i), "3", "3") for i in range(10)]
    mean, _ = chance_rate(distinct, "causal", "result", permutations=400, seed=1)
    assert 0.02 < mean < 0.20
    c = cell("all", distinct, "result")
    assert c.causal == 0.1 and c.glob == 0.1


def test_chance_does_not_mix_block_positions() -> None:
    """Block 0's outputs scored against block 3's targets would measure a
    block-position effect and call it chance."""
    records = [_rec(0, "A", "A", "A") for _ in range(8)]
    records += [_rec(3, "B", "B", "B") for _ in range(8)]
    mean, _ = chance_rate(records, "causal", "result", permutations=20)
    assert mean == 1.0     # would fall to ~0.5 if the groups were pooled


def test_mcnemar_counts_only_discordant_pairs() -> None:
    records = (
        [_rec(0, "A", "A", "A") for _ in range(30)]      # concordant hit
        + [_rec(0, "A", "z", "z") for _ in range(30)]    # concordant miss
        + [_rec(0, "A", "z", "A") for _ in range(12)]    # global only
        + [_rec(0, "A", "A", "z") for _ in range(2)]     # causal only
    )
    c = cell("all", records, "result")
    assert isinstance(c, Cell)
    assert c.only_global == 12 and c.only_causal == 2
    assert c.chi2 == pytest.approx((abs(12 - 2) - 1) ** 2 / 14)
    assert c.significant
    assert c.gap == pytest.approx(10 / 74)

    none_disc = [_rec(0, "A", "A", "A") for _ in range(5)]
    assert cell("all", none_disc, "result").chi2 is None


def test_stratify_groups_and_keeps_counts() -> None:
    records = [_rec(0, "A", "A", "A") for _ in range(3)]
    records += [_rec(2, "B", "z", "B") for _ in range(5)]
    cells = stratify(records, lambda r: r["block"], "result")
    assert {c.key: c.n for c in cells} == {"0": 3, "2": 5}
