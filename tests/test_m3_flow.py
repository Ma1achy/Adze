"""M3 acceptance — rectified flow utilities.

Convention: t = 0 clean, t = 1 pure noise.

broadcast_t is the OTHER thing most likely to be silently wrong. It is four lines
and it will cost you a weekend if the axes are transposed.
"""

from __future__ import annotations

import torch

from adze.invariants import Shapes
from adze.model.flow import broadcast_t, euler_step, interpolate, velocity_target


def _ids(shapes: Shapes) -> torch.Tensor:
    return torch.tensor(shapes.block_ids(), dtype=torch.long)


def test_broadcast_t_shape(shapes: Shapes) -> None:
    t = torch.rand(2, shapes.blocks)
    out = broadcast_t(t, _ids(shapes))
    assert out.shape == (2, shapes.n_positions, 1)


def test_broadcast_t_repeats_within_block(shapes: Shapes) -> None:
    """Every position in block b must carry block b's timestep — not position b's."""
    t = torch.arange(shapes.blocks, dtype=torch.float32).unsqueeze(0)
    out = broadcast_t(t, _ids(shapes)).squeeze(0).squeeze(-1)
    k = shapes.latents_per_block
    for b in range(shapes.blocks):
        chunk = out[b * k : (b + 1) * k]
        assert torch.allclose(chunk, torch.full_like(chunk, float(b)))


def test_interpolate_at_t_zero_is_clean(shapes: Shapes) -> None:
    z0 = torch.randn(2, shapes.n_positions, shapes.latent_dim)
    eps = torch.randn_like(z0)
    t = torch.zeros(2, shapes.blocks)
    assert torch.allclose(interpolate(z0, eps, t), z0, atol=1e-6)


def test_interpolate_at_t_one_is_noise(shapes: Shapes) -> None:
    z0 = torch.randn(2, shapes.n_positions, shapes.latent_dim)
    eps = torch.randn_like(z0)
    t = torch.ones(2, shapes.blocks)
    assert torch.allclose(interpolate(z0, eps, t), eps, atol=1e-6)


def test_interpolate_respects_per_block_t(shapes: Shapes) -> None:
    """The reason per-block timesteps exist: block 0 clean, block 1 pure noise,
    in the SAME forward pass."""
    z0 = torch.randn(1, shapes.n_positions, shapes.latent_dim)
    eps = torch.randn_like(z0)
    t = torch.zeros(1, shapes.blocks)
    t[0, 1] = 1.0
    out = interpolate(z0, eps, t)
    k = shapes.latents_per_block
    assert torch.allclose(out[:, :k], z0[:, :k], atol=1e-6)
    assert torch.allclose(out[:, k : 2 * k], eps[:, k : 2 * k], atol=1e-6)


def test_velocity_target_is_eps_minus_z0(shapes: Shapes) -> None:
    z0 = torch.randn(2, shapes.n_positions, shapes.latent_dim)
    eps = torch.randn_like(z0)
    assert torch.allclose(velocity_target(z0, eps), eps - z0)


def test_velocity_target_is_t_independent(shapes: Shapes) -> None:
    z0 = torch.randn(2, shapes.n_positions, shapes.latent_dim)
    eps = torch.randn_like(z0)
    assert velocity_target(z0, eps).shape == z0.shape


def test_euler_step_integrates_toward_clean(shapes: Shapes) -> None:
    """With the true velocity, a full step from t=1 must land exactly on z0."""
    z0 = torch.randn(1, shapes.n_positions, shapes.latent_dim)
    eps = torch.randn_like(z0)
    v = velocity_target(z0, eps)
    z1 = eps
    assert torch.allclose(euler_step(z1, v, dt=1.0), z0, atol=1e-5)
