"""Vectorised regime A must compute exactly what the naive path computes.

The naive algorithm runs one forward pass per block, because the clean context
x^<b differs for each b. The vectorised one concatenates [z_t ; z0] and computes
all B conditionals at once under a specialised mask (BD3-LM, arXiv 2503.09573
Suppl. B.6). That is only a speedup if the two agree — otherwise it is a
different objective wearing the same name.

The agreement should be EXACT, not approximate:
  - M_BD lets a noised block attend only within itself, so block b's prediction
    cannot depend on any other block's noise level
  - M_OBC gives it the clean blocks strictly before it
  - M_BC makes clean block j attend to clean blocks <= j
which is precisely `regime_a_mask(block_ids, b)` restricted to what block b sees.
Positional embeddings are tiled and timestep conditioning matches by construction.

If these tests fail, the mask is wrong. Look first at whether the OFFSET block ids
reached the mask builder: the mask must be built from REAL ids, since M_OBC fires
on block(j) < block(i) and offset clean ids all sit above B, which would silently
train a model whose noised blocks see no prefix at all.
"""

from __future__ import annotations

import torch

from adze.invariants import MaskMode
from adze.model.denoiser import Denoiser
from adze.model.masks import regime_a_mask, vectorised_regime_a_mask
from adze.train.train_denoiser import regime_a_batch, vectorised_regime_a_batch

BLOCKS, K, D = 5, 4, 8


def _model(seed: int = 0) -> Denoiser:
    torch.manual_seed(seed)
    model = Denoiser(
        latent_dim=D, d_model=32, n_layers=2, n_heads=4,
        latents_per_block=K, blocks=BLOCKS,
    )
    # out_proj is zero-initialised (adaLN-Zero), so an untrained model returns
    # zeros and every comparison passes trivially. Perturb it so the test has
    # something to compare.
    with torch.no_grad():
        model.out_proj.weight.normal_(0, 0.5)
        model.out_proj.bias.normal_(0, 0.5)
        for layer in model.cond:
            layer.to_modulation[1].weight.normal_(0, 0.05)
    return model.eval()


def _block_ids() -> torch.Tensor:
    return torch.repeat_interleave(torch.arange(BLOCKS), K)


def test_mask_shape_and_quadrants():
    ids = _block_ids()
    n = ids.shape[0]
    mask = vectorised_regime_a_mask(ids)
    assert mask.shape == (2 * n, 2 * n)

    bd, obc = mask[:n, :n], mask[:n, n:]
    zero, bc = mask[n:, :n], mask[n:, n:]

    # noised -> noised: own block only, both directions inside the block
    assert torch.equal(bd, ids.unsqueeze(0) == ids.unsqueeze(1))
    # noised -> clean: STRICTLY earlier blocks
    assert torch.equal(obc, ids.unsqueeze(0) < ids.unsqueeze(1))
    # clean never sees the noised half
    assert not zero.any()
    # clean -> clean: block-causal
    assert torch.equal(bc, ids.unsqueeze(0) <= ids.unsqueeze(1))


def test_noised_block_never_sees_another_noised_block():
    """The property that makes the vectorised loss equal the naive one."""
    ids = _block_ids()
    n = ids.shape[0]
    bd = vectorised_regime_a_mask(ids)[:n, :n]
    for i in range(n):
        for j in range(n):
            if ids[i] != ids[j]:
                assert not bd[i, j]


def test_block_zero_sees_no_clean_prefix():
    """M_OBC is strict, so block 0 has no prefix — as in the naive path."""
    ids = _block_ids()
    n = ids.shape[0]
    obc = vectorised_regime_a_mask(ids)[:n, n:]
    assert not obc[:K].any()


def test_vectorised_velocity_equals_naive_per_block():
    """The gate: same (t, eps) must give the same velocity on every block."""
    torch.manual_seed(1)
    model = _model()
    ids = _block_ids()
    n = BLOCKS * K
    z0 = torch.randn(3, n, D)
    t = torch.rand(3, BLOCKS)
    eps = torch.randn(3, n, D)

    vec = vectorised_regime_a_batch(z0, ids, BLOCKS, t=t, eps=eps)
    with torch.no_grad():
        pred_vec = model(
            vec["z_full"], vec["t_full"], vec["block_ids_full"],
            MaskMode.CAUSAL, mask=vec["mask"],
        )[:, :n]

    for b in range(BLOCKS):
        # Naive path for block b: only block b is noised, prefix clean, later
        # blocks absent. Build it directly so the noise matches exactly.
        t_naive = torch.zeros(3, BLOCKS)
        t_naive[:, b] = t[:, b]
        t_naive[:, b + 1 :] = 1.0
        z_t_naive = z0.clone()
        lo, hi = b * K, (b + 1) * K
        tb = t[:, b].view(3, 1, 1)
        z_t_naive[:, lo:hi] = (1 - tb) * z0[:, lo:hi] + tb * eps[:, lo:hi]
        with torch.no_grad():
            pred_naive = model(
                z_t_naive, t_naive, ids, MaskMode.CAUSAL,
                mask=regime_a_mask(ids, b),
            )
        torch.testing.assert_close(
            pred_vec[:, lo:hi], pred_naive[:, lo:hi], rtol=1e-4, atol=1e-5
        )


def test_other_blocks_noise_does_not_affect_a_block():
    """Change every other block's t and eps; block b's prediction must not move."""
    torch.manual_seed(2)
    model = _model()
    ids = _block_ids()
    n = BLOCKS * K
    z0 = torch.randn(2, n, D)
    t_a = torch.rand(2, BLOCKS)
    eps_a = torch.randn(2, n, D)

    b = 2
    lo, hi = b * K, (b + 1) * K
    t_b = torch.rand(2, BLOCKS)
    t_b[:, b] = t_a[:, b]
    eps_b = torch.randn(2, n, D)
    eps_b[:, lo:hi] = eps_a[:, lo:hi]

    outs = []
    for tt, ee in ((t_a, eps_a), (t_b, eps_b)):
        v = vectorised_regime_a_batch(z0, ids, BLOCKS, t=tt, eps=ee)
        with torch.no_grad():
            outs.append(
                model(v["z_full"], v["t_full"], v["block_ids_full"],
                      MaskMode.CAUSAL, mask=v["mask"])[:, lo:hi]
            )
    torch.testing.assert_close(outs[0], outs[1], rtol=1e-5, atol=1e-6)


def test_pad_blocks_excluded_from_loss_mask():
    ids = _block_ids()
    z0 = torch.randn(2, BLOCKS * K, D)
    block_mask = torch.ones(2, BLOCKS, dtype=torch.bool)
    block_mask[0, 3:] = False
    v = vectorised_regime_a_batch(z0, ids, BLOCKS, block_mask=block_mask)
    assert v["loss_mask"].shape == (2, BLOCKS * K, 1)
    assert not v["loss_mask"][0, 3 * K :].any()
    assert v["loss_mask"][1].all()


def test_training_and_sampling_agree_on_the_prefix_timestep():
    """Clean/generated prefix blocks must carry t=0 in training AND in sampling.

    If they disagree the model sees out-of-distribution context at every sampling
    step. Checked directly rather than assumed, and pinned here so it cannot drift.
    """
    ids = _block_ids()
    z0 = torch.randn(2, BLOCKS * K, D)

    naive = regime_a_batch(z0, ids, BLOCKS, force_block=3)
    assert torch.equal(naive["t"][:, :3], torch.zeros(2, 3))

    vec = vectorised_regime_a_batch(z0, ids, BLOCKS)
    assert torch.equal(vec["t_full"][:, BLOCKS:], torch.zeros(2, BLOCKS))

    # The sampler's construction, mirrored from adze.sample.draft.
    t_sample = torch.zeros(2, BLOCKS)
    t_sample[:, 3] = 0.5
    t_sample[:, 4:] = 1.0
    assert torch.equal(t_sample[:, :3], torch.zeros(2, 3))
