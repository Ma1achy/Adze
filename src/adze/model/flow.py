"""M3 — rectified flow utilities. Convention: t = 0 clean, t = 1 pure noise.

    z_t    = (1 - t) * z0 + t * eps
    target = eps - z0
    step   = z_t - dt * velocity          (integrating t: 1 -> 0)
"""

from __future__ import annotations

import torch


def sample_timesteps(
    shape: tuple[int, ...],
    shift: float = 1.5,
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Logit-normal timesteps with an SD3-style shift. Standard for rectified flow.

    Uniform `t` spends a large share of the budget on the worst-conditioned part of
    the schedule. Recovering `eps` from `z_t` requires `(z_t - (1-t)z0)/t`, so error
    is amplified by 1/t as t -> 0; measured on this model, loss at t=0.02 was 5.33
    against 0.21 for t >= 0.4. Sampling t ~ sigmoid(N(0,1)) concentrates draws
    mid-schedule and away from both ends.

    The shift then reweights toward higher noise:

        t' = shift * t / (1 + (shift - 1) * t)

    Args:
        shape: shape of the timestep tensor, e.g. (batch,).
        shift: 1.0 leaves the logit-normal untouched; ~1.5 follows SD3.

    Returns:
        Tensor of `shape` with values in (0, 1).
    """
    if shift <= 0:
        raise ValueError(f"shift must be positive, got {shift}")

    normal = torch.randn(shape, device=device, generator=generator)
    t = torch.sigmoid(normal)
    return shift * t / (1 + (shift - 1) * t)


def broadcast_t(t: torch.Tensor, block_ids: torch.Tensor) -> torch.Tensor:
    """Expand [batch, B] per-block timesteps to [batch, N, 1] per-position.

    Small, used everywhere, easy to get wrong. Tested directly.

    The failure this guards against: indexing `t` by *position* instead of by
    *block*, which silently gives position p the timestep of block p. With K=4 that
    is wrong for three quarters of positions and still trains, just badly.

    Args:
        t:         [batch, B] per-block timesteps.
        block_ids: [N] which block each position belongs to.

    Returns:
        [batch, N, 1]
    """
    if t.ndim != 2:
        raise ValueError(f"t must be [batch, B], got shape {tuple(t.shape)}")
    if block_ids.ndim != 1:
        raise ValueError(f"block_ids must be [N], got shape {tuple(block_ids.shape)}")
    if int(block_ids.max()) >= t.shape[1]:
        raise ValueError(
            f"block_ids reference block {int(block_ids.max())} but t has only "
            f"{t.shape[1]} blocks"
        )

    # Gather along the block axis: position p takes t[:, block_ids[p]].
    return t[:, block_ids].unsqueeze(-1)


def _broadcast_uniform(t: torch.Tensor, n_positions: int) -> torch.Tensor:
    """Expand [batch, B] to [batch, N, 1] assuming K = N // B positions per block.

    The tested signatures of `interpolate` do not carry `block_ids`, so the block
    structure has to come from the shapes. K is fixed for v0 and blocks are
    contiguous, which makes this exact — but it is an assumption, so it is stated
    here rather than buried. Anything with a non-uniform layout must call
    `broadcast_t` with explicit ids.
    """
    batch, blocks = t.shape
    if n_positions % blocks != 0:
        raise ValueError(
            f"{n_positions} positions do not divide evenly into {blocks} blocks; "
            f"pass explicit block_ids to broadcast_t instead"
        )
    return t.repeat_interleave(n_positions // blocks, dim=1).unsqueeze(-1)


def interpolate(z0: torch.Tensor, eps: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Forward corruption.

    Args:
        z0:  [batch, N, D] clean latents
        eps: [batch, N, D] gaussian noise
        t:   [batch, B] per-block timestep, broadcast to K within each block

    Returns:
        z_t: [batch, N, D]
    """
    if z0.shape != eps.shape:
        raise ValueError(f"z0 {tuple(z0.shape)} and eps {tuple(eps.shape)} must match")

    t_pos = _broadcast_uniform(t, z0.shape[1])
    return (1 - t_pos) * z0 + t_pos * eps


def velocity_target(z0: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """The regression target: eps - z0. Independent of t."""
    return eps - z0


def euler_step(z_t: torch.Tensor, velocity: torch.Tensor, dt: float) -> torch.Tensor:
    """One reverse Euler step, integrating from t=1 toward t=0."""
    return z_t - dt * velocity
