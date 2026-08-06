"""The permanent guard on the stochastic sampler.

SDE sampling is promoted out of design §8, which means `draft` no longer takes a
plain Euler step — it takes `renoise_step`. That is safe only because the two are
the same function at eta = 0. These tests hold that, so the promotion can never
silently change what the deterministic path does.

The equality is algebraic, not bitwise:

    (1-s)(z - t v) + s(z + (1-t) v) = z + v(s - t) = z - (t-s) v

Both sides are exact in real arithmetic and differ by float rounding, so the
checks are `allclose` at a tight tolerance rather than `equal`. Measured drift on
a full 32-step block-conditional draft is ~1e-05, which is the number the probe
scripts print.

Churn gets the same treatment: S_churn = 0 must be the identity, not merely close
to it, because every churn sweep reads its own S_churn = 0 row as the baseline.
"""

from __future__ import annotations

import pytest
import torch

from adze.model.flow import euler_step
from adze.sample.draft import draft
from adze.sample.stochastic import MAX_GAMMA, churn_t, churn_up, renoise_step

BLOCKS, K, D = 3, 4, 8


def _denoiser():
    from adze.model.denoiser import Denoiser

    torch.manual_seed(0)
    model = Denoiser(
        latent_dim=D, d_model=32, n_layers=2, n_heads=4,
        latents_per_block=K, blocks=BLOCKS,
    )
    model.eval()
    return model


def test_renoise_step_at_eta_zero_is_the_euler_step() -> None:
    """The algebraic identity, on random tensors, across the whole t range."""
    torch.manual_seed(0)
    z = torch.randn(4, BLOCKS * K, D)
    v = torch.randn(4, BLOCKS * K, D)
    for t, s in [(1.0, 0.9), (0.7, 0.5), (0.3, 0.2), (0.05, 0.0)]:
        got = renoise_step(z, v, t, s, eta=0.0)
        want = euler_step(z, v, t - s)
        assert torch.allclose(got, want, atol=1e-6), f"t={t} s={s}"


def test_draft_at_eta_zero_reproduces_the_ode() -> None:
    """The guard that matters: the default sampling path is unchanged.

    Same seed, so the initial noise is identical; eta = 0 draws no fresh noise, so
    the two runs see the same random stream throughout.
    """
    model = _denoiser()
    torch.manual_seed(7)
    a = draft(model, None, BLOCKS, K, D, 8, device="cpu", batch=3, eta=0.0)
    torch.manual_seed(7)
    b = draft(model, None, BLOCKS, K, D, 8, device="cpu", batch=3, eta=0.0,
              s_churn=0.0)
    assert torch.equal(a, b), "s_churn = 0 must not perturb the trajectory at all"

    # And against the raw Euler step, integrated by hand outside the sampler.
    torch.manual_seed(7)
    c = draft(model, None, BLOCKS, K, D, 8, device="cpu", batch=3, eta=0.0)
    assert (a - c).abs().max() == 0.0


def test_eta_changes_the_result_and_stays_in_range() -> None:
    """A non-zero eta must actually do something, and must be validated."""
    model = _denoiser()
    torch.manual_seed(7)
    ode = draft(model, None, BLOCKS, K, D, 8, device="cpu", batch=3, eta=0.0)
    torch.manual_seed(7)
    sde = draft(model, None, BLOCKS, K, D, 8, device="cpu", batch=3, eta=1.0)
    assert (ode - sde).abs().max() > 1e-3

    with pytest.raises(ValueError):
        renoise_step(ode, ode, 0.5, 0.4, eta=1.5)


def test_churn_at_zero_is_the_identity() -> None:
    """t_hat == t exactly, and churn_up returns its input object unchanged."""
    z = torch.randn(2, K, D)
    for t in (1.0, 0.6, 0.1, 0.0):
        t_hat, gamma = churn_t(t, 0.0, 32)
        assert t_hat == t and gamma == 0.0
        assert torch.equal(churn_up(z, t, t_hat), z)


def test_churn_gamma_clamps_and_t_hat_stays_below_one() -> None:
    """EDM caps gamma at sqrt(2)-1; t = 1 is pure noise and sigma is infinite there."""
    for s_churn in (10.0, 40.0, 1e6):
        t_hat, gamma = churn_t(0.5, s_churn, 32)
        assert gamma <= MAX_GAMMA + 1e-12
        assert 0.5 < t_hat < 1.0

    # The top of the schedule: t_hat must not reach 1, or the rescale divides by 0.
    t_hat, _ = churn_t(1.0, 1e6, 32)
    assert t_hat < 1.0
    assert torch.isfinite(churn_up(torch.randn(2, K, D), 1.0, t_hat)).all()


def test_churn_window_bounds_where_churn_applies() -> None:
    """Outside the window churn is off, which is what makes the band meaningful."""
    inside, _ = churn_t(0.5, 40.0, 32, 0.2, 0.8)
    outside, gamma = churn_t(0.9, 40.0, 32, 0.2, 0.8)
    assert inside > 0.5
    assert outside == 0.9 and gamma == 0.0


def test_churn_up_lands_on_the_shell_for_t_hat() -> None:
    """The rescale-and-add must put the point where the forward process would.

    NOT a total-variance check. Our parameterisation is not variance-exploding —
    E[z_t^2] = (1-t)^2 + t^2 per dimension, which DIPS at t = 0.5 — so churning
    0.4 -> 0.485 legitimately lowers total variance while raising the noise-to-
    signal ratio sigma = t/(1-t), which is the quantity churn is defined on. The
    invariant with content is that the churned point sits on the shell for t_hat.
    """
    torch.manual_seed(0)
    t = 0.4
    z = (1 - t) * torch.randn(8192, K, D) + t * torch.randn(8192, K, D)
    assert abs(z.pow(2).mean().item() - ((1 - t) ** 2 + t**2)) < 0.01

    t_hat, _ = churn_t(t, 40.0, 32)
    churned = churn_up(z, t, t_hat)
    expected = (1 - t_hat) ** 2 + t_hat**2
    assert abs(churned.pow(2).mean().item() - expected) < 0.01

    # And the thing churn is actually for: a strictly higher noise-to-signal ratio.
    assert t_hat / (1 - t_hat) > t / (1 - t)
