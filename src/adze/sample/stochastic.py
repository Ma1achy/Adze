"""The stochastic step: predict-then-renoise, plus EDM-style churn.

PROMOTED out of design §8. It was measured first as a probe, deliberately kept out
of `draft`, and promoted on the evidence: unconditional truth went 38.6% -> 70.7%
at eta = 1 — 95% of the measured 74.5% unseen-step ceiling — with the eta = 0 row
reproducing the ODE to max|diff| = 1.6e-05.

Why it works here. The deterministic ODE is a bijection from noise to data, so
hitting a strongly multimodal target requires a very contorted map, with nearby
noise points sent to distant modes. The measured latent clustering (nearest-
neighbour / mean pairwise distance 0.280 against a 0.744 Gaussian null) says the
target is exactly that shape. Stochastic sampling re-injects noise at every step,
which keeps trajectories inside regions the model actually saw and corrects
accumulated error instead of compounding it.

`draft` owns the block loop and calls `renoise_step`. There is deliberately no
second block loop in this module: two loops that must stay identical are the joint
that gives way three sessions later, and the mask, the block ordering and the
prefix handling all have to match training exactly.

## The step

    z0_hat = z_t - t*v                 eps_hat = z_t + (1-t)*v
    z_s    = (1-s)*z0_hat + s*( sqrt(1-eta^2)*eps_hat + eta*fresh )

At eta = 0 this collapses algebraically to `z_t - (t-s)*v`, the exact Euler step:

    (1-s)(z_t - t v) + s(z_t + (1-t) v) = z_t + v(s - t) = z_t - (t-s) v

so the eta = 0 row of any sweep is a free correctness check rather than a claim
that the two agree, and `tests/test_m4_stochastic.py` holds it as a permanent
regression guard.

## Churn

eta = 1 is the hard boundary of the parameterisation — the predicted eps is
discarded entirely — so a truth curve that rises monotonically to eta = 1 says the
optimum may lie outside the range. Going further means noising *up* past what the
schedule calls for, then integrating back down. That is EDM's S_churn (Karras et
al. 2022, arXiv 2206.00364, Algorithm 2), transcribed here from the reference
implementation:

    gamma  = min(S_churn / N, sqrt(2) - 1)   if S_tmin <= sigma <= S_tmax else 0
    s_hat  = sigma + gamma * sigma
    x_hat  = x + sqrt(s_hat^2 - sigma^2) * S_noise * randn_like(x)

EDM is variance-exploding in sigma; we are rectified flow in t. The map is exact:
z_t = (1-t)(z0 + sigma*eps) with sigma = t/(1-t), so a VE point at noise level
sigma is our point scaled by (1-t). Churn therefore becomes

    sigma_hat = (1 + gamma) * sigma
    t_hat     = sigma_hat / (1 + sigma_hat)
    z_hat     = (1-t_hat)/(1-t) * ( z_t + (1-t)*sqrt(sigma_hat^2 - sigma^2)*S_noise*fresh )

and the ordinary step then runs from `t_hat` down to `t_next`. The noise-up costs
no function evaluation. **S_churn = 0 gives t_hat == t identically**, so the ODE
identity check survives churn.

The second-order Heun correction in EDM's Algorithm 2 is NOT taken. It is a change
to the integrator, not to the stochasticity, and folding it in here would confound
the churn measurement with a better ODE solver.
"""

from __future__ import annotations

import torch

# EDM's cap on gamma. sqrt(2)-1 is the largest churn for which a single step
# cannot more than double the variance; the reference implementation clamps here.
MAX_GAMMA = 2.0**0.5 - 1.0

# t is clamped strictly below this when churning. t = 1 is pure noise, sigma is
# infinite there, and the rescale by (1-t_hat)/(1-t) would divide by zero.
T_CEILING = 1.0 - 1e-4


def churn_t(t: float, s_churn: float, nfe: int,
            t_min: float = 0.0, t_max: float = 1.0) -> tuple[float, float]:
    """The noised-up timestep and the gamma that produced it.

    Args:
        t: current timestep, in [0, 1].
        s_churn: EDM's S_churn, spread over `nfe` steps. 0 disables churn.
        nfe: steps in the schedule, so a given S_churn means the same total
            stochasticity regardless of budget — EDM divides by N for this reason.
        t_min, t_max: churn only inside this t window. EDM churns in a middle band
            of sigma and leaves the ends alone.

    Returns:
        (t_hat, gamma). gamma is 0 and t_hat == t exactly when churn is disabled or
        t is outside the window, which is what makes S_churn = 0 an identity.
    """
    if s_churn <= 0.0 or not (t_min <= t <= t_max) or t <= 0.0:
        return t, 0.0
    gamma = min(s_churn / nfe, MAX_GAMMA)
    sigma = t / (1.0 - min(t, T_CEILING))
    sigma_hat = (1.0 + gamma) * sigma
    return min(sigma_hat / (1.0 + sigma_hat), T_CEILING), gamma


def churn_up(z_t: torch.Tensor, t: float, t_hat: float,
             s_noise: float = 1.0) -> torch.Tensor:
    """Noise `z_t` from level t up to level t_hat. No function evaluation.

    Returns z_t unchanged when t_hat == t, so this is a no-op under S_churn = 0
    rather than a near-no-op that drifts.
    """
    if t_hat <= t:
        return z_t
    one_t = 1.0 - min(t, T_CEILING)
    sigma, sigma_hat = t / one_t, t_hat / (1.0 - t_hat)
    added = (sigma_hat**2 - sigma**2) ** 0.5
    return (1.0 - t_hat) / one_t * (
        z_t + one_t * added * s_noise * torch.randn_like(z_t)
    )


def renoise_step(
    z_t: torch.Tensor, velocity: torch.Tensor, t: float, s: float, eta: float
) -> torch.Tensor:
    """One step from timestep `t` down to `s`. Returns a tensor shaped like `z_t`.

    Args:
        eta: 0 reproduces the Euler ODE exactly (see the module docstring for the
            algebra); 1 discards the predicted eps and resamples the noise
            component completely. Values between interpolate.
    """
    if not 0.0 <= eta <= 1.0:
        raise ValueError(f"eta must be in [0, 1], got {eta}")
    z0_hat = z_t - t * velocity
    eps_hat = z_t + (1 - t) * velocity
    if eta > 0.0:
        eps_hat = (1 - eta**2) ** 0.5 * eps_hat + eta * torch.randn_like(z_t)
    return (1 - s) * z0_hat + s * eps_hat


@torch.no_grad()
def sample_stochastic(
    denoiser,
    blocks: int,
    latents_per_block: int,
    latent_dim: int,
    nfe: int,
    eta: float = 0.0,
    shift: float = 1.0,
    device: str = "mps",
    batch: int = 1,
) -> torch.Tensor:
    """Kept so the unconditional probe script and its recorded numbers still run.

    A thin wrapper over `draft` — the block loop lives there and only there.
    """
    from adze.sample.draft import draft

    return draft(
        denoiser, None, blocks, latents_per_block, latent_dim, nfe,
        device=device, batch=batch, shift=shift, eta=eta,
    )
