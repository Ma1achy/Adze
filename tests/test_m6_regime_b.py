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
    sample_subset_structured,
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


# --- erasure STRUCTURE ------------------------------------------------------
#
# M7 found the two arms exploit disjoint sources: global's accuracy is flat in
# prefix length while causal's climbs with it. The candidate cause is here —
# `random` erases mean |S| = 2.24 of ~4.4 real blocks, so roughly half the prefix
# is itself erased on a typical refine step and a reliable clean-prefix mapping
# never exists to be learned.


def _bm(rows: list[int]) -> torch.Tensor:
    m = torch.zeros(len(rows), BLOCKS, dtype=torch.bool)
    for i, n in enumerate(rows):
        m[i, :n] = True
    return m


def test_single_erases_exactly_one_real_block() -> None:
    """`single` matches M7's inference condition exactly."""
    torch.manual_seed(0)
    mask = _bm([3, 4, 5, 5, 7, 2] * 20)
    sel = sample_subset_structured(mask, structure="single")
    assert (sel.sum(dim=1) == 1).all()
    assert not (sel & ~mask).any()


def test_contiguous_erases_a_run_and_leaves_prefix_and_pin_clean() -> None:
    """THE PROPERTY THE ARM EXISTS FOR.

    Everything before the run is clean, so the prefix is reliable; everything
    after it is clean, so the downstream pin survives. `random` gives neither.
    """
    torch.manual_seed(1)
    mask = _bm([3, 4, 5, 6, 7] * 40)
    sel = sample_subset_structured(mask, structure="contiguous")
    for row, real in zip(sel, mask):
        idx = row.nonzero().flatten()
        assert len(idx) >= 1
        # A single run: the selected indices are consecutive.
        assert torch.equal(idx, torch.arange(int(idx[0]), int(idx[-1]) + 1))
        # And it stays inside the real blocks.
        assert bool(real[idx].all())


def test_contiguous_matches_randoms_size_distribution() -> None:
    """The two arms must differ in SHAPE, not in how much is erased — otherwise
    a difference between them is a difference in erasure volume."""
    torch.manual_seed(2)
    mask = _bm([3, 4, 5, 6, 7] * 400)
    a = sample_subset_structured(mask, structure="random").sum(dim=1).float().mean()
    b = sample_subset_structured(mask, structure="contiguous").sum(dim=1).float().mean()
    assert abs(float(a) - float(b)) < 0.15, f"random {a:.2f} vs contiguous {b:.2f}"


def test_random_damages_the_prefix_and_the_structured_arms_do_not() -> None:
    """States the mechanism under test as an assertion.

    'Prefix damaged' = at least one CLEAN-side hole, i.e. an unselected real block
    sitting between selected ones. That is the thing a model cannot learn to rely
    on.
    """
    torch.manual_seed(3)
    mask = _bm([5, 6, 7] * 200)

    def holed(sel):
        out = 0
        for row in sel:
            idx = row.nonzero().flatten()
            if len(idx) > 1 and int(idx[-1]) - int(idx[0]) + 1 != len(idx):
                out += 1
        return out / len(sel)

    assert holed(sample_subset_structured(mask, structure="random")) > 0.3
    assert holed(sample_subset_structured(mask, structure="contiguous")) == 0.0
    assert holed(sample_subset_structured(mask, structure="single")) == 0.0


def test_an_unknown_structure_is_rejected() -> None:
    # NOT "suffix" — that was the invalid name when this test was written, on the
    # reasoning that erasing the suffix removes the pin. It does, and that turned
    # out to be the point of a later experiment, so it is a real structure now.
    with pytest.raises(ValueError):
        sample_subset_structured(_bm([4, 4]), structure="every-other-block")


def test_suffix_erases_to_the_end_and_always_leaves_a_prefix() -> None:
    """Suffix erasure removes the PIN — everything after b is gone — while
    leaving blocks < b clean, so the prefix is the only remaining route.

    b never reaches 0: erasing from block 0 leaves neither prefix nor pin and
    would train reconstruction from nothing.
    """
    torch.manual_seed(7)
    mask = _bm([3, 4, 5, 6, 7] * 60)
    sel = sample_subset_structured(mask, structure="suffix")
    for row, real in zip(sel, mask):
        idx = row.nonzero().flatten()
        assert len(idx) >= 1
        n_real = int(real.sum())
        assert int(idx[-1]) == n_real - 1, "must run to the last real block"
        assert torch.equal(idx, torch.arange(int(idx[0]), n_real))
        assert int(idx[0]) >= 1, "a clean prefix block must survive"
        assert not (row & ~real).any()


def test_suffix_erasure_leaves_no_consumer_for_any_erased_block() -> None:
    """The property the arm exists for, checked through provenance.

    Every consumer of an erased block lies strictly after it, so a suffix
    erasure guarantees the consumer is erased too. Pin availability must be 0.
    """
    from adze.config import load_config, trace_kwargs
    from adze.data.generate import generate_dataset
    from adze.eval.strata import consumers

    cfg = load_config("configs/debug.yaml")
    traces = [t for t in generate_dataset(n=300, seed=5, **trace_kwargs(cfg))
              if len(t.steps) <= 7]
    mask = torch.zeros(len(traces), 7, dtype=torch.bool)
    for i, t in enumerate(traces):
        mask[i, :len(t.steps)] = True

    torch.manual_seed(8)
    sel = sample_subset_structured(mask, structure="suffix")
    for i, t in enumerate(traces):
        for b in sel[i].nonzero().flatten().tolist():
            for c in consumers(t, b):
                assert sel[i, c], f"trace {i} block {b} kept a consumer at {c}"


def test_pinmix_is_about_half_suffix() -> None:
    """The split is what the report states, so it has to be what the code does."""
    torch.manual_seed(9)
    mask = _bm([3, 4, 5, 6, 7] * 200)
    sel = sample_subset_structured(mask, structure="pinmix")
    n_real = mask.sum(dim=1)
    is_suffix = torch.tensor([
        bool(row.nonzero().flatten()[-1] == n - 1) and bool(row.sum() == n - int(row.nonzero().flatten()[0]))
        for row, n in zip(sel, n_real.tolist())
    ])
    assert 0.40 < is_suffix.float().mean() < 0.75
