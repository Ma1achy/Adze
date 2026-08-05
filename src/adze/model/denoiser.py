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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from adze.invariants import MaskMode
from adze.model.masks import build_mask

# Order of the modulation parameters produced per block. Named rather than
# positional because a transposed chunk here is exactly the kind of silent
# miswiring the M3 gate exists to catch.
_MOD_FIELDS = ("shift_attn", "scale_attn", "gate_attn", "shift_mlp", "scale_mlp", "gate_mlp")


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10_000.0) -> torch.Tensor:
    """Sinusoidal embedding of continuous timesteps.

    Args:
        t:   [...] timesteps in [0, 1].
        dim: embedding width, must be even.

    Returns:
        [..., dim]
    """
    if dim % 2 != 0:
        raise ValueError(f"embedding dim must be even, got {dim}")

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / half
    )
    args = t.unsqueeze(-1).float() * freqs
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """adaLN: scale and shift a normalised activation. Both are per-position."""
    return x * (1 + scale) + shift


class PerBlockAdaLN(nn.Module):
    """adaLN modulation computed per block, broadcast to positions within the block.

    The deviation from standard DiT. Given [batch, B] timesteps and a mode
    embedding, produce [batch, N, ...] modulation parameters.

    The broadcast is a gather on `block_ids`, NOT a reshape. A reshape happens to
    agree while K divides N evenly and silently disagrees the moment anything else
    is true; the gather states the intent. See adze.model.flow.broadcast_t, which
    is the same operation for scalars and is tested directly.
    """

    def __init__(self, d_model: int, latents_per_block: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.latents_per_block = latents_per_block

        self.t_mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        # Mode is signalled through conditioning rather than separate parameters,
        # so one set of weights serves drafting and refinement.
        self.mode_embed = nn.Embedding(len(MaskMode), d_model)
        self.to_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(d_model, len(_MOD_FIELDS) * d_model)
        )
        # adaLN-Zero: every block starts as the identity, residual stream untouched.
        nn.init.zeros_(self.to_modulation[1].weight)
        nn.init.zeros_(self.to_modulation[1].bias)

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
            six tensors, each [batch, N, d_model], in _MOD_FIELDS order.
        """
        if timesteps.ndim != 2:
            raise ValueError(f"timesteps must be [batch, B], got {tuple(timesteps.shape)}")
        if int(block_ids.max()) >= timesteps.shape[1]:
            raise ValueError(
                f"block_ids reference block {int(block_ids.max())} but timesteps "
                f"cover only {timesteps.shape[1]}"
            )

        mode_idx = torch.tensor(
            list(MaskMode).index(mode), device=timesteps.device, dtype=torch.long
        )
        per_block = self.t_mlp(timestep_embedding(timesteps, self.d_model))
        per_block = per_block + self.mode_embed(mode_idx).view(1, 1, -1)

        # Position p takes block_ids[p]'s conditioning. The one line that matters.
        per_position = per_block[:, block_ids]
        return self.to_modulation(per_position).chunk(len(_MOD_FIELDS), dim=-1)


class MaskedSelfAttention(nn.Module):
    """Multi-head self-attention over an [N, N] boolean mask. True means attend."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Args:
            x:    [batch, N, d_model]
            mask: [N, N] bool, True where attention is permitted.
        """
        batch, n, _ = x.shape
        qkv = self.qkv(x).view(batch, n, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        # A row with no permitted keys is a softmax over nothing — NaN. Regime A
        # produces exactly that: `visible_prefix_mask` removes later blocks in both
        # directions, so their rows are entirely False. Those positions are outside
        # the loss and cannot be read by visible rows (their columns are masked
        # there), so opening their row changes nothing that is used, and their
        # output is zeroed below. Fixing it here keeps `visible_prefix_mask` a clean
        # statement of what is visible rather than a compromise with an NaN.
        row_live = mask.any(dim=-1, keepdim=True)
        safe = mask | ~row_live

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=safe)
        out = out.transpose(1, 2).reshape(batch, n, -1)
        return self.proj(out) * row_live.view(1, n, 1)


class DenoiserBlock(nn.Module):
    """One DiT block: modulated self-attention + modulated MLP."""

    def __init__(self, d_model: int, n_heads: int, latents_per_block: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.attn = MaskedSelfAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        modulation: tuple[torch.Tensor, ...],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Args:
            x: [batch, N, d_model]
            modulation: six [batch, N, d_model] tensors in _MOD_FIELDS order,
                already broadcast from per-block conditioning.
            mask: [N, N] bool.
        """
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = modulation
        x = x + gate_a * self.attn(_modulate(self.norm1(x), shift_a, scale_a), mask)
        x = x + gate_m * self.mlp(_modulate(self.norm2(x), shift_m, scale_m))
        return x


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
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.latents_per_block = latents_per_block
        self.blocks = blocks
        self.n_positions = blocks * latents_per_block

        self.in_proj = nn.Linear(latent_dim, d_model)
        self.pos = nn.Parameter(torch.randn(1, self.n_positions, d_model) * 0.02)

        # One conditioning module per layer, plus one for the final head. Each
        # produces its own modulation from the same per-block timesteps.
        self.cond = nn.ModuleList(
            [PerBlockAdaLN(d_model, latents_per_block) for _ in range(n_layers)]
        )
        self.layers = nn.ModuleList(
            [DenoiserBlock(d_model, n_heads, latents_per_block) for _ in range(n_layers)]
        )

        self.final_cond = PerBlockAdaLN(d_model, latents_per_block)
        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.out_proj = nn.Linear(d_model, latent_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        block_ids: torch.Tensor,
        mode: MaskMode,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Args:
            latents:   [batch, N, D]  noised
            timesteps: [batch, B]     per block
            block_ids: [N]
            mode:      CAUSAL or GLOBAL
            context:   [batch, T_ctx, d_model] question conditioning, prefix-concatenated
            mask:      [N, N] bool, overriding the mask `mode` would build. Regime A
                       needs later blocks absent entirely, which is a prefix mask
                       rather than a plain causal one; `mode` still selects the
                       conditioning embedding.

        Returns:
            velocity: [batch, N, D]
        """
        if context is not None:
            raise NotImplementedError("question conditioning is M5, not M3")

        attn_mask = build_mask(block_ids, mode) if mask is None else mask

        x = self.in_proj(latents) + self.pos[:, : latents.shape[1]]
        for cond, layer in zip(self.cond, self.layers):
            x = layer(x, cond(timesteps, mode, block_ids), attn_mask)

        shift, scale, *_ = self.final_cond(timesteps, mode, block_ids)
        x = _modulate(self.final_norm(x), shift, scale)
        return self.out_proj(x)
