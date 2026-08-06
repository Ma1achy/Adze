"""M7 — the central-experiment harness, validated against a STUB denoiser.

The point of a stub is that it makes the harness testable before any model exists.
A denoiser that returns a fixed, known velocity turns `regenerate` into arithmetic
we can predict exactly, so erasure, timestep handling, write-back and mask
switching are all checked without a checkpoint — and when a real one lands, a
failure is attributable to the model rather than the harness.

What must hold regardless of the model:

  * only the target block changes; every other position is bit-identical
  * `causal` and `global` differ ONLY in the mask, given the same seed
  * the causal mask genuinely cannot see downstream blocks
  * scoring counts the right things and ignores padding
"""

from __future__ import annotations

import torch

from adze.eval.central import Repair, regenerate, revision_mask, score
from adze.invariants import MaskMode

BATCH, BLOCKS, K, D = 4, 5, 4, 8
N = BLOCKS * K


class StubDenoiser(torch.nn.Module):
    """Returns a constant velocity, and records what it was asked to attend to.

    Deliberately ignores its input, so `regenerate`'s output is a pure function of
    the schedule and the erasure — which is what makes the arithmetic checkable.
    """

    def __init__(self, value: float = 0.0) -> None:
        super().__init__()
        self.value = value
        self.seen_masks: list[torch.Tensor] = []
        self.seen_t: list[torch.Tensor] = []

    def forward(self, z, t, block_ids, mode, mask=None):
        self.seen_masks.append(mask.clone())
        self.seen_t.append(t.clone())
        return torch.full_like(z, self.value)

    def eval(self):
        return self


def _ids():
    return torch.repeat_interleave(torch.arange(BLOCKS), K)


def _targets():
    return torch.tensor([0, 2, 4, 1])


def test_only_the_target_block_is_written() -> None:
    """A revision pass must not disturb the blocks it was not asked to revise."""
    torch.manual_seed(0)
    z0 = torch.randn(BATCH, N, D)
    out = regenerate(StubDenoiser(0.3), z0, _ids(), _targets(), BLOCKS, 8,
                     MaskMode.GLOBAL, seed=1)
    for i, b in enumerate(_targets().tolist()):
        lo, hi = b * K, (b + 1) * K
        untouched = torch.cat([out[i, :lo], out[i, hi:]])
        original = torch.cat([z0[i, :lo], z0[i, hi:]])
        assert torch.equal(untouched, original), f"example {i} leaked outside block {b}"
        assert not torch.equal(out[i, lo:hi], z0[i, lo:hi])


def test_causal_and_global_differ_only_in_the_mask() -> None:
    """The paired comparison rests on this: same seed, same noise, same everything."""
    z0 = torch.randn(BATCH, N, D)
    kw = dict(blocks=BLOCKS, nfe=6, seed=7)
    a = regenerate(StubDenoiser(0.0), z0, _ids(), _targets(),
                   mode=MaskMode.CAUSAL, **kw)
    b = regenerate(StubDenoiser(0.0), z0, _ids(), _targets(),
                   mode=MaskMode.GLOBAL, **kw)
    # A stub ignores the mask, so with identical seeds the two must coincide
    # exactly. Any difference here is the harness leaking a second noise draw.
    assert torch.equal(a, b)

    ca, gl = StubDenoiser(), StubDenoiser()
    regenerate(ca, z0, _ids(), _targets(), mode=MaskMode.CAUSAL, **kw)
    regenerate(gl, z0, _ids(), _targets(), mode=MaskMode.GLOBAL, **kw)
    assert not torch.equal(ca.seen_masks[0], gl.seen_masks[0])


def test_the_causal_mask_cannot_see_downstream() -> None:
    """If it could, the experiment would have no control arm."""
    ids = _ids()
    causal = revision_mask(ids, 0, MaskMode.CAUSAL)
    glob = revision_mask(ids, 0, MaskMode.GLOBAL)
    assert glob.all()
    q, kk = ids.unsqueeze(1), ids.unsqueeze(0)
    assert not causal[kk > q].any(), "causal mask attends forward"
    assert causal[kk <= q].all(), "causal mask should see itself and earlier"


def test_timesteps_are_zero_off_the_target_and_run_to_zero_on_it() -> None:
    """Clean context must be presented at t = 0, exactly as training presents it."""
    stub = StubDenoiser()
    regenerate(stub, torch.randn(BATCH, N, D), _ids(), _targets(), BLOCKS, 5,
               MaskMode.GLOBAL, seed=2)
    tgt = _targets()
    for t in stub.seen_t:
        for i, b in enumerate(tgt.tolist()):
            others = [j for j in range(BLOCKS) if j != b]
            assert (t[i, others] == 0).all(), "context block carried a non-zero t"
    first, last = stub.seen_t[0], stub.seen_t[-1]
    for i, b in enumerate(tgt.tolist()):
        assert first[i, b] == 1.0          # starts from complete erasure
        assert 0.0 <= last[i, b] < 1.0     # and descends


def test_scoring_counts_the_right_things_and_ignores_padding() -> None:
    clean = [
        ["1 + 1 = 2", "2 + 2 = 4", "4 + 4 = 8", "PAD", "PAD"],
        ["3 + 3 = 6", "6 + 6 = 12", "PAD", "PAD", "PAD"],
    ]
    # ex0: target block 1 repaired, everything else intact -> repaired, preserved
    # ex1: target block 0 NOT repaired, block 1 also wrong  -> neither
    decoded = [
        ["1 + 1 = 2", "2 + 2 = 4", "4 + 4 = 8", "junk", "junk"],
        ["9 + 9 = 99", "6 + 6 = 13", "junk", "junk", "junk"],
    ]
    r = score("global", decoded, clean, torch.tensor([1, 0]), [3, 2])
    assert isinstance(r, Repair)
    assert r.repaired == 0.5
    assert r.preserved == 0.5      # ex0 yes; ex1's block 1 differs
    assert r.answer == 0.5         # ex0's last real block matches; ex1's does not
    assert r.n == 2


def test_a_stub_that_reproduces_the_clean_latent_scores_perfectly() -> None:
    """End-to-end sanity: if regeneration were perfect, the harness must say so.

    Guards against a scorer that can never reach 1.0 — an off-by-one in the target
    index or the step count would show up here and nowhere else.
    """
    clean = [["a", "b", "c", "PAD", "PAD"]] * 3
    decoded = [row[:] for row in clean]
    r = score("global", decoded, clean, torch.tensor([1, 1, 2]), [3, 3, 3])
    assert r.repaired == 1.0 and r.preserved == 1.0 and r.answer == 1.0
