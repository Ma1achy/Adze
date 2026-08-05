"""M4 — pass one. Block-causal drafting.

Generate blocks in order. Each block is denoised from pure noise over `nfe` Euler
steps, conditioned on the question and all previous (clean, already-generated)
blocks, under the causal mask.
"""

from __future__ import annotations

import torch

from adze.model.denoiser import Denoiser


def draft(
    denoiser: Denoiser,
    context: torch.Tensor,
    blocks: int,
    latents_per_block: int,
    latent_dim: int,
    nfe: int,
    device: str = "mps",
) -> torch.Tensor:
    """Returns:
        latents: [batch, N, D] the drafted reasoning chain.
    """
    raise NotImplementedError
