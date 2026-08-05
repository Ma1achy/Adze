"""M7 — pass two. Global refinement by complete erasure.

v0 uses t = 1 on selected blocks (complete erasure), NOT partial re-noising.
That matches regime B training exactly: the flow path from pure noise is what the
model has learned. Partial re-noising is a later measured change, not a starting
assumption.

For v0 the block selection is ORACLE — you know which block was corrupted and
erase that one. This answers "can global regeneration repair a known-bad block
using future context?" It does NOT answer "can the system detect which block
needs repair?" Label results accordingly.
"""

from __future__ import annotations

import torch

from adze.invariants import MaskMode
from adze.model.denoiser import Denoiser


def refine(
    denoiser: Denoiser,
    latents: torch.Tensor,
    erase_indices: torch.Tensor,
    context: torch.Tensor,
    nfe: int,
    mode: MaskMode = MaskMode.GLOBAL,
) -> torch.Tensor:
    """Erase the selected blocks and regenerate them.

    Args:
        latents: [batch, N, D] the drafted chain.
        erase_indices: [batch, n_erase] which blocks to erase.
        context: question conditioning.
        nfe: denoising steps.
        mode: GLOBAL for the real thing, CAUSAL for the comparison arm of the
              central experiment. Everything else must be held identical.

    Returns:
        latents: [batch, N, D] with erased blocks regenerated.
    """
    raise NotImplementedError
