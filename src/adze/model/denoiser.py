"""M3 — the shared denoiser.

DiT backbone with one non-standard extension: PER-BLOCK timestep conditioning.

Vanilla DiT takes a scalar t per sample and broadcasts it through adaLN globally.
Regime B needs different blocks at different noise levels in the SAME forward
pass, so the modulation must be computed per block and broadcast to the K
positions within it.

This is the single most likely thing to be silently miswired, and it is what the
overfit-one-batch gate at M3 exists to catch.

Prediction target is VELOCITY (design §3.1). Not x-prediction.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from adze.invariants import MaskMode


class PerBlockAdaLN(nn.Module):
    """adaLN modulation computed per block, broadcast to positions within the block.

    The deviation from standard DiT. Given [batch, B] timesteps and a mode
    embedding, produce [batch, N, ...] modulation parameters.
    """

    def __init__(self, d_model: int, latents_per_block: int) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(
        self,
        timesteps: torch.Tensor,
        mode: MaskMode,
        block_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Args:
            timesteps: [batch, B] in [0, 1]
            mode: draft or refine, embedded alongside t
            block_ids: [N]

        Returns:
            shift, scale, gate tensors each [batch, N, d_model]
        """
        raise NotImplementedError


class DenoiserBlock(nn.Module):
    """One DiT block: modulated self-attention + modulated MLP."""

    def __init__(self, d_model: int, n_heads: int, latents_per_block: int) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        mask: torch.Tensor,
        block_ids: torch.Tensor,
        mode: MaskMode,
    ) -> torch.Tensor:
        raise NotImplementedError


class Denoiser(nn.Module):
    """Shared denoiser for both regimes. Predicts velocity.

    Same weights handle drafting (causal mask) and refinement (global mask). The
    mode is signalled through conditioning, not through separate parameters.
    """

    def __init__(
        self,
        latent_dim: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        latents_per_block: int,
        blocks: int,
    ) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        block_ids: torch.Tensor,
        mode: MaskMode,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Args:
            latents:   [batch, N, D]  noised
            timesteps: [batch, B]     per block
            block_ids: [N]
            mode:      CAUSAL or GLOBAL
            context:   [batch, T_ctx, d_model] question conditioning, prefix-concatenated

        Returns:
            velocity: [batch, N, D]
        """
        raise NotImplementedError
