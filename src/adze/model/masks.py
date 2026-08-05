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
    if block_ids.ndim != 1:
        raise ValueError(f"block_ids must be [N], got shape {tuple(block_ids.shape)}")
    if (block_ids[1:] < block_ids[:-1]).any():
        raise ValueError("block_ids must be non-decreasing")

    n = block_ids.shape[0]

    if mode is MaskMode.GLOBAL:
        return torch.ones(n, n, dtype=torch.bool, device=block_ids.device)

    if mode is MaskMode.CAUSAL:
        # query block on rows, key block on columns; j is visible from i when its
        # block is not in i's future. Equality keeps a block bidirectional inside
        # itself, which is what makes this block-causal rather than token-causal.
        return block_ids.unsqueeze(0) <= block_ids.unsqueeze(1)

    raise ValueError(f"unknown mask mode: {mode}")


def visible_prefix_mask(block_ids: torch.Tensor, up_to_block: int) -> torch.Tensor:
    """Mask for regime A drafting: blocks after `up_to_block` are absent entirely.

    "Absent" is stronger than "not attended to": later blocks are removed in *both*
    directions, so they neither inform nor are informed by the prefix. Within the
    surviving prefix the mask is fully bidirectional — block `up_to_block` is the
    one being denoised and it may read every earlier block, while the earlier blocks
    are clean context.

    Args:
        block_ids: [N]
        up_to_block: the block currently being denoised. Blocks strictly after it
                     are masked out completely in both directions.

    Returns:
        [N, N] bool tensor.
    """
    if block_ids.ndim != 1:
        raise ValueError(f"block_ids must be [N], got shape {tuple(block_ids.shape)}")

    visible = block_ids <= up_to_block
    return visible.unsqueeze(0) & visible.unsqueeze(1)


def regime_a_mask(block_ids: torch.Tensor, block: int) -> torch.Tensor:
    """The mask regime A trains under, and therefore the one the sampler must use.

    Causal across blocks AND every block after `block` absent entirely. Both halves
    are needed: the build plan says "causal mask", the mask tests say later blocks
    are absent, and regime A is both at once.

    This exists so training and sampling cannot drift apart. A sampler that built
    its own mask would be measuring something the model was never taught, and the
    discrepancy would present as "the denoiser doesn't work" rather than as a mask
    bug. One definition, called from `regime_a_batch` and from `draft`.

    Args:
        block_ids: [N]
        block: the block being denoised.

    Returns:
        [N, N] bool tensor.
    """
    return build_mask(block_ids, MaskMode.CAUSAL) & visible_prefix_mask(block_ids, block)


def vectorised_regime_a_mask(block_ids: torch.Tensor) -> torch.Tensor:
    """Mask for the vectorised regime A: every block denoised in ONE forward pass.

    The naive algorithm scores one block per step, because the clean context
    x^<b differs for each b. BD3-LM (arXiv 2503.09573, Suppl. B.6) removes that by
    running the model over the concatenation [z_t ; z0] of length 2N and shaping
    the attention so all B conditionals are computed at once:

        M_full = [[M_BD, M_OBC],
                  [0,    M_BC ]]

        M_BD  [i,j] = block(j) == block(i)   noised -> noised, own block only
        M_OBC [i,j] = block(j) <  block(i)   noised -> clean, STRICTLY earlier
        0                                    clean  -> noised, never
        M_BC  [i,j] = block(j) <= block(i)   clean  -> clean, block-causal

    M_BD is what makes this correct: a noised block sees no other block's noised
    state, so its prediction depends only on itself and the clean prefix — exactly
    the conditional the naive path computes with `regime_a_mask`, and exactly the
    conditional the sampler faces. M_BC makes the clean half's representations
    identical to what a block-causal pass over the clean sequence alone produces.

    Args:
        block_ids: [N] REAL block ids for one copy. Not the offset ids used for the
            timestep gather — those exist only so the clean half can be given t=0
            through the same [batch, 2B] tensor. Passing offset ids here would put
            every clean block at index >= B, `M_OBC` would never fire, and noised
            blocks would attend to no prefix at all while still training happily.

    Returns:
        [2N, 2N] bool.
    """
    if block_ids.ndim != 1:
        raise ValueError(f"block_ids must be [N], got shape {tuple(block_ids.shape)}")

    q, kk = block_ids.unsqueeze(1), block_ids.unsqueeze(0)
    bd = kk == q
    obc = kk < q
    bc = kk <= q
    zero = torch.zeros_like(bd)
    return torch.cat(
        [torch.cat([bd, obc], dim=1), torch.cat([zero, bc], dim=1)], dim=0
    )
