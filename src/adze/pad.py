"""Padding discipline. Every loss and every metric routes through here.

Three separate bugs in this repo have shared one shape — padding masquerading as
data:

  - pad blocks scored in the denoiser loss. They hold one constant vector, which
    is trivially predictable, so including them deflated the loss and turned a
    failing result into a reported PASS.
  - pad rows miscounted in a diagnostic. Pad blocks are filled with the K rows of
    `pad_latent`, so there are K distinct pad vectors and a most-common-row test
    finds only 1/K of them. The "real" set read 90.6% when the truth was ~61%.
  - pad blocks consuming training steps at zero gradient. ~11% of steps, and ~61%
    of block-6 draws, starving the last blocks of the gradient they needed.

Each was invisible in aggregate and surfaced only when one specific number
disagreed with theory. Three means a fourth exists, so this closes the class
rather than the instance.

The structural part is that `keep` has NO DEFAULT. The bug was always forgetting
to mask; an API that cannot be called without a mask cannot be forgotten.
"""

from __future__ import annotations

import torch


def real_positions(block_mask: torch.Tensor, latents_per_block: int) -> torch.Tensor:
    """[batch, B] block mask -> [batch, N, 1] position mask. The only expansion.

    Args:
        block_mask: [batch, B] bool, True where the block holds a real step.
        latents_per_block: K.

    Returns:
        [batch, N, 1] bool, True at positions belonging to a real block.

    Do not hand-roll this with repeat_interleave at a call site. It is one line and
    it has been written wrong in three different places; having one spelling of it
    is the entire point.
    """
    if block_mask.ndim != 2:
        raise ValueError(
            f"block_mask must be [batch, B], got shape {tuple(block_mask.shape)}"
        )
    if block_mask.dtype is not torch.bool:
        raise TypeError(f"block_mask must be bool, got {block_mask.dtype}")
    return block_mask.repeat_interleave(latents_per_block, dim=1).unsqueeze(-1)


def masked_mean(values: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """Mean of `values` over kept positions only.

    Args:
        values: [batch, N, D] per-element quantity, e.g. squared error.
        keep:   [batch, N, 1] or [1, N, 1] bool. REQUIRED — there is no default and
                no None. Build it with `real_positions`.

    Returns:
        Scalar tensor.

    Raises:
        ValueError: if nothing is kept. A fully-padded batch has no target, and
            returning 0.0 there is exactly how a masked-out block once read as a
            perfect score. Callers that can legitimately hit this must skip the
            step explicitly rather than average a zero in.

    The divisor is `keep.sum() * D`, not `values.numel()`. Dividing by the full
    element count is the specific error that inflated losses 8x in a diagnostic
    earlier in this project: it counts padded and unpadded positions alike, so the
    figure moves with the padding ratio rather than with the model.
    """
    if values.ndim != 3:
        raise ValueError(f"values must be [batch, N, D], got {tuple(values.shape)}")
    if keep.dtype is not torch.bool:
        raise TypeError(f"keep must be bool, got {keep.dtype}")
    if keep.ndim != 3 or keep.shape[-1] != 1:
        raise ValueError(f"keep must be [batch, N, 1], got {tuple(keep.shape)}")
    if keep.shape[1] != values.shape[1]:
        raise ValueError(
            f"keep has {keep.shape[1]} positions, values has {values.shape[1]}"
        )
    if keep.shape[0] not in (1, values.shape[0]):
        raise ValueError(
            f"keep batch {keep.shape[0]} broadcasts against neither 1 nor "
            f"{values.shape[0]}"
        )

    n_kept = int(keep.sum()) * (values.shape[0] // keep.shape[0])
    if n_kept == 0:
        raise ValueError(
            "masked_mean got an all-False mask: nothing to average. A fully-padded "
            "batch has no target — skip the step rather than averaging in a zero."
        )
    return (values * keep).sum() / (n_kept * values.shape[-1])
