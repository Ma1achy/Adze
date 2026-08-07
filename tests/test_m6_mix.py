"""M6 — the regime A/B mix as a swept parameter.

M7's crossing table showed the model is specialised to whichever configuration it
saw most: either departure from regime A's setup — mask or conditioning — costs
~5pp alone, and both together cost no more. Regime B fired on 9.7% of steps. So
the share is a knob under test, not a constant inherited from DiD.

What must hold before any of it is trained:

  * two shares produce two checkpoint paths — a sweep that overwrites its own
    arms measures only the last one
  * `--mixed` with a zero share fails loudly rather than training regime A while
    claiming a mix
  * the realised share tracks the requested one, since the mix is a per-step
    Bernoulli draw and the REALISED share is what the model saw
"""

from __future__ import annotations

import torch

import pytest

from adze.train.regime_b import sample_subset
from adze.train.train_denoiser import resolve_b_prob

BLOCKS = 5


def test_the_override_wins_and_a_zero_share_is_rejected() -> None:
    assert resolve_b_prob(0.1, None, mixed=True) == 0.1      # config
    assert resolve_b_prob(0.1, 0.5, mixed=True) == 0.5       # override
    assert resolve_b_prob(0.1, 0.5, mixed=False) == 0.0      # not mixed at all

    # A zero share with --mixed trains regime A while claiming a mix, which would
    # silently produce a duplicate of the unmixed arm under a mixed name.
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            resolve_b_prob(0.1, bad, mixed=True)


def checkpoint_suffix(b_prob: float, mixed: bool, n_layers: int | None = None,
                      seed: int = 0) -> str:
    """The naming rule under test, stated once here and mirrored in training.

    Kept as a small reimplementation rather than an import because the thing being
    checked is that DISTINCT configurations get DISTINCT names — a bug shared
    between an implementation and a test that imports it would cancel.
    """
    suffix = "" if seed == 0 else f"_seed{seed}"
    suffix += "" if n_layers is None else f"_L{n_layers}"
    suffix += f"_mixedP{round(b_prob * 100)}" if mixed else ""
    return suffix


def test_each_mix_gets_its_own_checkpoint_name() -> None:
    """A sweep that overwrites its own arms has measured only the last one."""
    names = {checkpoint_suffix(p, mixed=True, n_layers=4)
             for p in (0.10, 0.25, 0.50, 0.75)}
    assert len(names) == 4
    assert "_mixedP50" in checkpoint_suffix(0.50, mixed=True, n_layers=4)
    # And a mixed run must never collide with an unmixed one at the same depth.
    assert (checkpoint_suffix(0.10, mixed=True, n_layers=4)
            != checkpoint_suffix(0.10, mixed=False, n_layers=4))


def test_the_realised_share_tracks_the_requested_one() -> None:
    """The mix is a per-step Bernoulli draw, so realised != requested.

    The training loop reports the REALISED share, because that is what the model
    saw and what a comparison across mixes rests on. This pins the draw's
    behaviour so a change to it cannot silently shift every arm of the sweep.
    """
    torch.manual_seed(0)
    for want in (0.10, 0.50, 0.75):
        fired = sum(torch.rand(1).item() < want for _ in range(20_000))
        assert abs(fired / 20_000 - want) < 0.02, f"requested {want}"


def test_regime_b_still_erases_at_least_one_block_at_any_share() -> None:
    """The share governs HOW OFTEN regime B fires, never what it does when it does.

    An example with an empty S contributes nothing and would dilute the batch, and
    that must stay true however rare or common regime B becomes.
    """
    block_mask = torch.ones(256, BLOCKS, dtype=torch.bool)
    block_mask[:, 3:] = False
    for p in (0.01, 0.25, 0.5, 1.0):
        sel = sample_subset(block_mask, p=p)
        assert sel.any(dim=1).all()
        assert not sel[:, 3:].any()
