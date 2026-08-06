"""M6 — regime B, the refine objective.

The gate on this milestone is the equivalence test: the batched multi-block loss
must equal a hand-computed per-block reference. Same discipline as the regime-A
equivalence check, and it catches the same class of bug — a loss that silently
averages the wrong positions, or blocks leaking into each other's score.
"""

from __future__ import annotations

import pytest
import torch

from adze.invariants import MaskMode
from adze.model.denoiser import Denoiser
from adze.model.flow import interpolate, velocity_target
from adze.model.masks import build_mask
from adze.train.regime_b import (
    regime_b_batch,
    regime_b_loss,
    sample_subset,
)

BATCH, BLOCKS, K, D = 6, 5, 4, 8
N = BLOCKS * K


def _model():
    torch.manual_seed(0)
    m = Denoiser(latent_dim=D, d_model=32, n_layers=2, n_heads=4,
                 latents_per_block=K, blocks=BLOCKS)
    m.eval()
    return m


def _ids():
    return torch.repeat_interleave(torch.arange(BLOCKS), K)


def test_selected_blocks_are_pure_noise_and_the_rest_are_untouched() -> None:
    """t = 1 must mean COMPLETE erasure, and t = 0 must mean exactly z0.

    Partial re-noising is not in v0; if `t` ever drifted off the endpoints this
    would be the first thing to catch it.
    """
    torch.manual_seed(1)
    z0 = torch.randn(BATCH, N, D)
    b = regime_b_batch(z0, _ids(), BLOCKS)
    sel = b["selected"].repeat_interleave(K, dim=1).unsqueeze(-1)

    assert torch.equal(b["t"], b["selected"].float())
    assert torch.allclose(b["z_t"][sel.expand_as(z0)], b["eps"][sel.expand_as(z0)])
    assert torch.allclose(b["z_t"][~sel.expand_as(z0)], z0[~sel.expand_as(z0)])


def test_the_mask_is_global() -> None:
    """Regime B attends everywhere — that is the whole point of pass two."""
    b = regime_b_batch(torch.randn(BATCH, N, D), _ids(), BLOCKS)
    assert b["mask"].all()
    assert torch.equal(b["mask"], build_mask(_ids(), MaskMode.GLOBAL))


def test_loss_equals_a_hand_computed_per_block_reference() -> None:
    """THE M6 EQUIVALENCE TEST.

    Erase several blocks in one pass, then recompute the same loss by hand: run
    the model once, pull out each erased block's positions individually, and
    average the squared error over exactly those. If the batched loss weights
    positions differently, or lets an unerased block contribute, this diverges.

    The reference deliberately does NOT reuse `masked_mean` — it sums and divides
    explicitly, so a bug inside the helper cannot cancel on both sides.
    """
    torch.manual_seed(3)
    model, ids = _model(), _ids()
    z0 = torch.randn(BATCH, N, D)
    eps = torch.randn(BATCH, N, D)
    selected = torch.zeros(BATCH, BLOCKS, dtype=torch.bool)
    selected[:, 1] = True
    selected[0, 3] = True
    selected[2, 0] = True

    batch = regime_b_batch(z0, ids, BLOCKS, eps=eps, selected=selected)
    got = regime_b_loss(model, batch, ids)

    # --- hand-computed reference ---------------------------------------------
    t = selected.float()
    z_t = interpolate(z0, eps, t)
    target = velocity_target(z0, eps)
    with torch.no_grad():
        pred = model(z_t, t, ids, MaskMode.GLOBAL,
                     mask=build_mask(ids, MaskMode.GLOBAL))

    total, count = 0.0, 0
    for i in range(BATCH):
        for b in range(BLOCKS):
            if not selected[i, b]:
                continue
            lo, hi = b * K, (b + 1) * K
            err = (pred[i, lo:hi] - target[i, lo:hi]) ** 2
            total += err.sum().item()
            count += err.numel()
    want = total / count

    assert abs(got.item() - want) < 1e-6, f"batched {got.item()} vs reference {want}"


def test_unerased_blocks_do_not_contribute_to_the_loss() -> None:
    """Perturbing a clean block's TARGET must not move the loss.

    A target only matters where the loss reads it. If changing an unerased
    block's target changes the number, the loss mask is wrong.
    """
    torch.manual_seed(4)
    model, ids = _model(), _ids()
    z0 = torch.randn(BATCH, N, D)
    eps = torch.randn(BATCH, N, D)
    selected = torch.zeros(BATCH, BLOCKS, dtype=torch.bool)
    selected[:, 2] = True

    batch = regime_b_batch(z0, ids, BLOCKS, eps=eps, selected=selected)
    before = regime_b_loss(model, batch, ids).item()

    batch["target"] = batch["target"].clone()
    batch["target"][:, 0:K] += 100.0          # block 0, not erased
    after = regime_b_loss(model, batch, ids).item()
    assert abs(before - after) < 1e-9


def test_padding_is_never_erased_or_scored() -> None:
    torch.manual_seed(5)
    block_mask = torch.ones(BATCH, BLOCKS, dtype=torch.bool)
    block_mask[:, 3:] = False
    sel = sample_subset(block_mask, p=1.0)
    assert not sel[:, 3:].any()

    b = regime_b_batch(torch.randn(BATCH, N, D), _ids(), BLOCKS,
                       block_mask=block_mask, p=1.0)
    assert not b["loss_mask"].view(BATCH, BLOCKS, K, 1)[:, 3:].any()


def test_every_example_erases_at_least_one_block() -> None:
    """An example with an empty S contributes nothing and would dilute the batch."""
    block_mask = torch.ones(64, BLOCKS, dtype=torch.bool)
    block_mask[:, 2:] = False
    for p in (0.01, 0.5, 1.0):
        assert sample_subset(block_mask, p=p).any(dim=1).all()

    with pytest.raises(ValueError):
        sample_subset(block_mask, p=0.0)


def test_subset_varies_across_examples() -> None:
    """Regime B's global mask permits a per-example S; check it actually uses one.

    Regime A cannot do this — its mask depends on the block index — so a copied
    implementation would show identical rows here.
    """
    torch.manual_seed(6)
    sel = sample_subset(torch.ones(128, BLOCKS, dtype=torch.bool), p=0.5)
    assert len({tuple(row.tolist()) for row in sel}) > 1
