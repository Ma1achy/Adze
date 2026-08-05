"""Shape invariants. These should pass on a fresh clone — if they don't, the
environment is wrong, not the code."""

from __future__ import annotations

from adze.invariants import MaskMode, Regime, Shapes


def test_n_positions(shapes: Shapes) -> None:
    assert shapes.n_positions == shapes.blocks * shapes.latents_per_block == 16


def test_block_ids_length(shapes: Shapes) -> None:
    assert len(shapes.block_ids()) == shapes.n_positions


def test_block_ids_non_decreasing(shapes: Shapes) -> None:
    ids = shapes.block_ids()
    assert all(a <= b for a, b in zip(ids, ids[1:]))


def test_block_ids_contiguous(shapes: Shapes) -> None:
    """Each block occupies exactly K contiguous positions."""
    ids = shapes.block_ids()
    for b in range(shapes.blocks):
        positions = [i for i, x in enumerate(ids) if x == b]
        assert len(positions) == shapes.latents_per_block
        assert positions == list(range(positions[0], positions[0] + len(positions)))


def test_enums_exist() -> None:
    assert MaskMode.CAUSAL != MaskMode.GLOBAL
    assert Regime.DRAFT != Regime.REFINE
