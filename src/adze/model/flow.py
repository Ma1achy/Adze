"""M3 — rectified flow utilities. Convention: t = 0 clean, t = 1 pure noise.

    z_t    = (1 - t) * z0 + t * eps
    target = eps - z0
    step   = z_t - dt * velocity          (integrating t: 1 -> 0)
"""

from __future__ import annotations

import torch


def interpolate(z0: torch.Tensor, eps: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Forward corruption.

    Args:
        z0:  [batch, N, D] clean latents
        eps: [batch, N, D] gaussian noise
        t:   [batch, B] per-block timestep, broadcast to K within each block

    Returns:
        z_t: [batch, N, D]
    """
    raise NotImplementedError


def velocity_target(z0: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """The regression target: eps - z0. Independent of t."""
    raise NotImplementedError


def euler_step(z_t: torch.Tensor, velocity: torch.Tensor, dt: float) -> torch.Tensor:
    """One reverse Euler step, integrating from t=1 toward t=0."""
    raise NotImplementedError


def broadcast_t(t: torch.Tensor, block_ids: torch.Tensor) -> torch.Tensor:
    """Expand [batch, B] per-block timesteps to [batch, N, 1] per-position.

    Small, used everywhere, easy to get wrong. Tested directly.
    """
    raise NotImplementedError
