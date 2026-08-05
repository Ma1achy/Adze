"""M3 acceptance — attention masks.

Small, fully specifiable, and one of the two things most likely to be silently
wrong. Write this before the denoiser.

    causal : i attends to j iff block(j) <= block(i)
    global : i attends to j always
"""

from __future__ import annotations

import torch

from adze.invariants import MaskMode, Shapes
from adze.model.masks import build_mask, visible_prefix_mask


def _ids(shapes: Shapes) -> torch.Tensor:
    return torch.tensor(shapes.block_ids(), dtype=torch.long)


def test_global_mask_is_all_true(shapes: Shapes) -> None:
    m = build_mask(_ids(shapes), MaskMode.GLOBAL)
    assert m.shape == (shapes.n_positions, shapes.n_positions)
    assert m.dtype == torch.bool
    assert m.all()


def test_causal_mask_shape_and_dtype(shapes: Shapes) -> None:
    m = build_mask(_ids(shapes), MaskMode.CAUSAL)
    assert m.shape == (shapes.n_positions, shapes.n_positions)
    assert m.dtype == torch.bool


def test_causal_is_bidirectional_within_a_block(shapes: Shapes) -> None:
    """Positions in the same block see each other, both directions."""
    ids = _ids(shapes)
    m = build_mask(ids, MaskMode.CAUSAL)
    for i in range(shapes.n_positions):
        for j in range(shapes.n_positions):
            if ids[i] == ids[j]:
                assert m[i, j], f"({i},{j}) same block but masked"


def test_causal_blocks_the_future(shapes: Shapes) -> None:
    ids = _ids(shapes)
    m = build_mask(ids, MaskMode.CAUSAL)
    for i in range(shapes.n_positions):
        for j in range(shapes.n_positions):
            if ids[j] > ids[i]:
                assert not m[i, j], f"({i},{j}) attends to a future block"


def test_causal_allows_the_past(shapes: Shapes) -> None:
    ids = _ids(shapes)
    m = build_mask(ids, MaskMode.CAUSAL)
    for i in range(shapes.n_positions):
        for j in range(shapes.n_positions):
            if ids[j] < ids[i]:
                assert m[i, j], f"({i},{j}) should attend to an earlier block"


def test_causal_is_a_strict_subset_of_global(shapes: Shapes) -> None:
    ids = _ids(shapes)
    causal = build_mask(ids, MaskMode.CAUSAL)
    glob = build_mask(ids, MaskMode.GLOBAL)
    assert (causal <= glob).all()
    assert causal.sum() < glob.sum()


def test_first_block_sees_only_itself(shapes: Shapes) -> None:
    ids = _ids(shapes)
    m = build_mask(ids, MaskMode.CAUSAL)
    k = shapes.latents_per_block
    assert m[:k, :k].all()
    assert not m[:k, k:].any()


def test_visible_prefix_excludes_later_blocks(shapes: Shapes) -> None:
    """Regime A: blocks after the one being denoised are absent entirely."""
    ids = _ids(shapes)
    up_to = 1
    m = visible_prefix_mask(ids, up_to_block=up_to)
    later = ids > up_to
    assert not m[:, later].any()
    assert not m[later, :].any()
