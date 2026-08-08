"""The banded refine mask.

The identity at `width >= B` is the load-bearing test: if a band at full width is
not byte-identical to the global mask, then every banded number is measured
against a different baseline than the one the project has been reporting, and the
comparison is silently invalid.
"""

from __future__ import annotations

import torch

from adze.invariants import MaskMode
from adze.model.masks import banded_mask, build_mask


def _ids(blocks: int = 5, k: int = 3) -> torch.Tensor:
    return torch.repeat_interleave(torch.arange(blocks), k)


def test_full_width_is_exactly_the_global_mask():
    ids = _ids()
    for width in (5, 6, 99):
        assert torch.equal(banded_mask(ids, width), build_mask(ids, MaskMode.GLOBAL))


def test_zero_width_is_block_diagonal():
    ids = _ids()
    mask = banded_mask(ids, 0)
    assert torch.equal(mask, ids.unsqueeze(0) == ids.unsqueeze(1))


def test_it_is_symmetric_at_every_width():
    """Bidirectional means symmetric. An asymmetric band would be a causal mask
    with extra steps, and would confound the arms it is meant to separate."""
    ids = _ids()
    for width in range(0, 6):
        mask = banded_mask(ids, width)
        assert torch.equal(mask, mask.T)


def test_widths_are_nested():
    ids = _ids()
    for width in range(0, 5):
        assert (banded_mask(ids, width) <= banded_mask(ids, width + 1)).all()


def test_the_band_is_measured_in_blocks_not_positions():
    ids = _ids(blocks=4, k=3)
    mask = banded_mask(ids, 1)
    # Position 0 is in block 0; block 1 spans 3..5 and block 2 spans 6..8.
    assert bool(mask[0, 5]) and not bool(mask[0, 6])


def test_it_is_a_pure_function_of_block_ids():
    ids = _ids()
    assert torch.equal(banded_mask(ids, 2), banded_mask(ids.clone(), 2))


def test_a_negative_width_is_rejected():
    try:
        banded_mask(_ids(), -1)
    except ValueError:
        return
    raise AssertionError("negative width should raise")
