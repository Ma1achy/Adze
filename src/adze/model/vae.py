"""M2 — VAE mapping reasoning steps to latent blocks and back.

Encoder: a step's tokens -> K learned Perceiver query tokens cross-attending over
them -> [K, D]. Perceiver queries handle variable-length input natively, which
matters because a reasoning step is however many tokens the text is while a block
is exactly K latents.

Decoder: mirror. K latents -> token sequence for that step.

The hard gate for this milestone is the latent-use check, not reconstruction
accuracy. See adze.eval.checks.latent_use_check.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class StepEncoder(nn.Module):
    """Tokens for one step -> K latents.

    Small bidirectional transformer over the step's tokens, then K learned query
    tokens cross-attend over the result.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        latents_per_block: int,
        latent_dim: int,
    ) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Args:
            tokens: [batch, T] token ids for one step.

        Returns:
            mu:     [batch, K, D]
            logvar: [batch, K, D]
        """
        raise NotImplementedError


class StepDecoder(nn.Module):
    """K latents -> tokens for one step."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        latents_per_block: int,
        latent_dim: int,
        max_len: int,
    ) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """Args:
            latents: [batch, K, D]

        Returns:
            logits: [batch, T, vocab_size]
        """
        raise NotImplementedError


class StepVAE(nn.Module):
    """Encoder + decoder + reparameterisation. Loss is CE reconstruction + beta * KL."""

    def __init__(self, encoder: StepEncoder, decoder: StepDecoder, kl_beta: float) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        """Returns dict with keys: loss, recon_loss, kl_loss, logits, latents."""
        raise NotImplementedError
