"""The multi-pass reach harness.

Two failures here would be silent and would invalidate the result rather than
crash it: an erase set that drops the target (the scored block would never be
regenerated, so the VAE round-trip would be measured instead), and a pass loop
that reuses its noise (R passes would collapse to one pass repeated).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

_spec = importlib.util.spec_from_file_location(
    "m7_reach", Path(__file__).resolve().parents[1] / "scripts" / "m7_reach.py")
m7_reach = importlib.util.module_from_spec(_spec)
sys.modules["m7_reach"] = m7_reach
_spec.loader.exec_module(m7_reach)

pass_erase = m7_reach.pass_erase
run_passes = m7_reach.run_passes
SEED_STRIDE = m7_reach.SEED_STRIDE


def _fixture():
    block_mask = torch.tensor([
        [True, True, True, True, True],
        [True, True, True, False, False],
        [True, True, False, False, False],
    ])
    return torch.tensor([0, 1, 1]), block_mask


def test_the_target_is_always_erased_in_both_arms():
    target, block_mask = _fixture()
    for arm in m7_reach.ARMS:
        gen = torch.Generator().manual_seed(0)
        erase = pass_erase(target, block_mask, arm, gen)
        assert erase.gather(1, target.unsqueeze(1)).all()


def test_padding_is_never_erased():
    """Erasing a pad block asks the model to rebuild a constant from noise and
    would count that as refinement skill."""
    target, block_mask = _fixture()
    for arm in m7_reach.ARMS:
        gen = torch.Generator().manual_seed(0)
        assert not (pass_erase(target, block_mask, arm, gen) & ~block_mask).any()


def test_the_control_erases_the_target_alone():
    target, block_mask = _fixture()
    erase = pass_erase(target, block_mask, "only-b", None)
    assert (erase.sum(dim=1) == 1).all()


def test_the_main_arm_redraws_its_subset_each_pass():
    """A fixed subset would erase the same neighbours every pass, which is a
    weaker test of propagation than the one intended."""
    target, block_mask = _fixture()
    gen = torch.Generator().manual_seed(0)
    draws = [pass_erase(target, block_mask, "p50", gen) for _ in range(6)]
    assert any(not torch.equal(draws[0], d) for d in draws[1:])


def test_the_one_arm_erases_the_target_plus_exactly_one_where_one_exists():
    """An example whose only real block IS the target has nothing to add, so its
    realised size is 1. Every other example must be exactly 2 — a silent 1 there
    would make `one` a second copy of the control."""
    target, block_mask = _fixture()
    gen = torch.Generator().manual_seed(0)
    erase = pass_erase(target, block_mask, "one", gen)
    n_other = (block_mask.sum(dim=1) - 1).clamp(min=0)
    assert torch.equal(erase.sum(dim=1), torch.where(n_other > 0,
                                                     torch.tensor(2),
                                                     torch.tensor(1)))


def test_an_unknown_arm_is_rejected():
    target, block_mask = _fixture()
    try:
        pass_erase(target, block_mask, "every-other-block", None)
    except ValueError:
        return
    raise AssertionError("unknown arm should raise")


class _Recorder:
    """Stands in for the denoiser, recording the seed each pass was given.

    `regenerate` reseeds from its `seed` argument, so identical seeds across
    passes would mean identical noise and the chain would be R draws of the same
    thing. The stride is checked directly rather than inferred from outputs.
    """

    def __init__(self):
        self.seeds = []

    def __call__(self, z, t, block_ids, mode, mask=None):
        self.seeds.append(int(torch.initial_seed()))
        return torch.zeros_like(z)


def test_the_seed_advances_every_pass():
    target, block_mask = _fixture()
    blocks, k = 5, 2
    latents = torch.zeros(3, blocks * k, 4)
    block_ids = torch.repeat_interleave(torch.arange(blocks), k)
    rec = _Recorder()
    gen = torch.Generator().manual_seed(0)
    from adze.invariants import MaskMode
    outs = list(run_passes(rec, latents, block_ids, target, blocks, k, 3,
                           MaskMode.GLOBAL, "only-b", block_mask, nfe=1,
                           eta=0.0, seed=7, generator=gen))
    assert [r for r, _, _ in outs] == [1, 2, 3]
    assert rec.seeds == [7, 7 + SEED_STRIDE, 7 + 2 * SEED_STRIDE]


def test_blocks_outside_the_erase_set_survive_a_pass_unchanged():
    """Chaining is only meaningful if untouched blocks carry through exactly. If
    they drift, later passes read a context that no pass wrote."""
    from adze.eval.central import regenerate
    from adze.invariants import MaskMode

    blocks, k, d = 5, 2, 4
    torch.manual_seed(0)
    latents = torch.randn(3, blocks * k, d)
    block_ids = torch.repeat_interleave(torch.arange(blocks), k)
    target, block_mask = _fixture()
    erase = pass_erase(target, block_mask, "only-b", None)

    out = regenerate(lambda z, t, bi, m, mask=None: torch.zeros_like(z),
                     latents, block_ids, target, blocks, 2, MaskMode.GLOBAL,
                     eta=0.0, seed=0, erase=erase)
    keep = ~erase.repeat_interleave(k, dim=1).unsqueeze(-1).expand_as(latents)
    assert torch.equal(out[keep], latents[keep])
