"""SDE sampling — a MEASUREMENT, not an adopted feature.

An SDE sampler is on design §8's not-in-v0 list. This module exists to measure
whether promoting it would be justified, and is deliberately NOT wired into
`adze.sample.draft`. The v0 sampling path stays ODE-only until that decision is
taken explicitly. Nothing outside the unconditional probe should import this.

Why it is worth measuring here specifically: the deterministic ODE is a bijection
from noise to data, so hitting a strongly multimodal target requires a very
contorted map, with nearby noise points sent to distant modes. The measured latent
clustering (nearest-neighbour / mean pairwise distance 0.280 against a 0.744
Gaussian null) says the target is exactly that shape. Stochastic sampling
re-injects noise at every step, which keeps trajectories inside regions the model
actually saw and corrects accumulated error instead of compounding it.

The parameterisation is predict-then-renoise, with one knob:

    z0_hat = z_t - t*v                 eps_hat = z_t + (1-t)*v
    z_s    = (1-s)*z0_hat + s*( sqrt(1-eta^2)*eps_hat + eta*fresh )

At eta = 0 this collapses algebraically to `z_t - (t-s)*v`, the exact Euler step
`draft` takes:

    (1-s)(z_t - t v) + s(z_t + (1-t) v) = z_t + v(s - t) = z_t - (t-s) v

so the eta = 0 row of any sweep is a free correctness check against the ODE rather
than a claim that the two agree.
"""

from __future__ import annotations

import torch

from adze.invariants import MaskMode
from adze.model.denoiser import Denoiser
from adze.model.flow import schedule
from adze.model.masks import regime_a_mask


@torch.no_grad()
def sample_stochastic(
    denoiser: Denoiser,
    blocks: int,
    latents_per_block: int,
    latent_dim: int,
    nfe: int,
    eta: float = 0.0,
    shift: float = 1.0,
    device: str = "mps",
    batch: int = 1,
) -> torch.Tensor:
    """Block-causal sampling with a stochastic step. Returns [batch, N, D].

    Args:
        eta: 0 reproduces the Euler ODE exactly; 1 resamples the noise component
            completely at every step. Values between interpolate.

    The mask and the block ordering are identical to `draft`'s — only the update
    rule differs — so a difference in the result is attributable to the sampler
    and not to a conditioning change.
    """
    if not 0.0 <= eta <= 1.0:
        raise ValueError(f"eta must be in [0, 1], got {eta}")

    denoiser.eval()
    dev = torch.device(device)
    n_positions = blocks * latents_per_block
    block_ids = torch.repeat_interleave(torch.arange(blocks), latents_per_block).to(dev)
    latents = torch.randn(batch, n_positions, latent_dim, device=dev)
    knots = schedule(nfe, shift, device=dev)

    for b in range(blocks):
        mask = regime_a_mask(block_ids, b)
        is_active = (block_ids == b).view(1, n_positions, 1)

        for i in range(nfe):
            t_val, s_val = float(knots[i]), float(knots[i + 1])
            t = torch.zeros(batch, blocks, device=dev)
            t[:, b] = t_val
            t[:, b + 1 :] = 1.0

            v = denoiser(latents, t, block_ids, MaskMode.CAUSAL, mask=mask)
            z0_hat = latents - t_val * v
            eps_hat = latents + (1 - t_val) * v
            noise = (
                (1 - eta**2) ** 0.5 * eps_hat + eta * torch.randn_like(latents)
                if eta > 0
                else eps_hat
            )
            stepped = (1 - s_val) * z0_hat + s_val * noise
            latents = torch.where(is_active, stepped, latents)

    return latents
