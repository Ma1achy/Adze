"""Shape and convention invariants. Import these rather than redefining them.

    B  = blocks per sequence      (fixed for v0)
    K  = latents per block        (4)
    D  = latent channel dim       (64)
    N  = B * K                    total latent positions

    latents    : [batch, N, D]
    timesteps  : [batch, B]       PER BLOCK, broadcast to K within the block
    block_ids  : [N]              which block each position belongs to
    mask       : [N, N]           bool, True = attend

Diffusion convention: t = 0 is clean, t = 1 is pure noise.

The per-block timestep is the main deviation from standard DiT, which takes one
scalar t per sample. Regime B needs different blocks at different noise levels in
the same forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MaskMode(str, Enum):
    """Attention scope. Both are pure functions of block_ids."""

    CAUSAL = "causal"   # bidirectional within a block, causal across blocks
    GLOBAL = "global"   # fully bidirectional


class Regime(str, Enum):
    """Which training regime a batch belongs to. See design §3.1."""

    DRAFT = "draft"     # regime A: noise one block, previous clean, causal mask
    REFINE = "refine"   # regime B: erase subset, others clean, global mask


@dataclass(frozen=True)
class Shapes:
    """Resolved shape constants for a run."""

    blocks: int              # B
    latents_per_block: int   # K
    latent_dim: int          # D

    @property
    def n_positions(self) -> int:
        """N = B * K."""
        return self.blocks * self.latents_per_block

    def block_ids(self) -> list[int]:
        """Which block each of the N positions belongs to."""
        return [b for b in range(self.blocks) for _ in range(self.latents_per_block)]
