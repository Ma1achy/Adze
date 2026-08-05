"""M3 — attention mask construction.

Both modes are pure functions of block_ids. One function, one switch. This is
small, easy to get silently wrong, and fully specified by tests — write it first.

    causal : bidirectional WITHIN a block, causal ACROSS blocks
             position i attends to j iff block(j) <= block(i)
    global : fully bidirectional
             position i attends to j always

See tests/test_m3_masks.py for the acceptance criteria.
"""

from __future__ import annotations

import torch

from adze.invariants import MaskMode


def build_mask(block_ids: torch.Tensor, mode: MaskMode) -> torch.Tensor:
    """Build an attention mask from block structure.

    Args:
        block_ids: [N] long tensor, which block each position belongs to.
                   Must be non-decreasing.
        mode: CAUSAL or GLOBAL.

    Returns:
        [N, N] bool tensor. True means "position i may attend to position j".
    """
    raise NotImplementedError


def visible_prefix_mask(block_ids: torch.Tensor, up_to_block: int) -> torch.Tensor:
    """Mask for regime A drafting: blocks after `up_to_block` are absent entirely.

    Args:
        block_ids: [N]
        up_to_block: the block currently being denoised. Blocks strictly after it
                     are masked out completely in both directions.

    Returns:
        [N, N] bool tensor.
    """
    raise NotImplementedError
