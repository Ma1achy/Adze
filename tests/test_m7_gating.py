"""The gating test's erase-set builders.

`regenerate` scores `target_block` and requires it to be inside `erase`. A builder
that drops the target would score a block that was never regenerated — the model
would be graded on the VAE round-trip of the corrupted text instead. That failure
is silent, so it is tested rather than assumed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

_spec = importlib.util.spec_from_file_location(
    "m7_gating", Path(__file__).resolve().parents[1] / "scripts" / "m7_gating.py")
m7_gating = importlib.util.module_from_spec(_spec)
sys.modules["m7_gating"] = m7_gating
_spec.loader.exec_module(m7_gating)

build_suffix_erase = m7_gating.build_suffix_erase
build_single_erase = m7_gating.build_single_erase


def _fixture():
    """Three examples, B = 5, with 5, 3 and 2 real blocks and targets 0, 1, 1."""
    block_mask = torch.tensor([
        [True, True, True, True, True],
        [True, True, True, False, False],
        [True, True, False, False, False],
    ])
    target = torch.tensor([0, 1, 1])
    return target, block_mask


def test_suffix_contains_the_target():
    target, block_mask = _fixture()
    erase = build_suffix_erase(target, block_mask)
    assert erase.gather(1, target.unsqueeze(1)).all()


def test_suffix_erases_nothing_below_the_target():
    target, block_mask = _fixture()
    erase = build_suffix_erase(target, block_mask)
    idx = torch.arange(5).unsqueeze(0)
    assert not (erase & (idx < target.unsqueeze(1))).any()


def test_suffix_is_contiguous_from_the_target_to_the_last_real_block():
    target, block_mask = _fixture()
    erase = build_suffix_erase(target, block_mask)
    expected = torch.tensor([
        [True, True, True, True, True],      # b = 0, 5 real
        [False, True, True, False, False],   # b = 1, 3 real
        [False, True, False, False, False],  # b = 1, 2 real
    ])
    assert torch.equal(erase, expected)


def test_suffix_never_erases_padding():
    target, block_mask = _fixture()
    assert not (build_suffix_erase(target, block_mask) & ~block_mask).any()


def test_single_is_the_target_alone():
    target, block_mask = _fixture()
    erase = build_single_erase(target, block_mask)
    assert (erase.sum(dim=1) == 1).all()
    assert erase.gather(1, target.unsqueeze(1)).all()


def test_the_two_shapes_agree_when_the_target_is_the_last_real_block():
    """A suffix starting at the final real block IS a single-block erasure. The
    two builders must not disagree there — if they do, one of them is counting
    padding."""
    block_mask = torch.tensor([[True, True, True, False, False]])
    target = torch.tensor([2])
    assert torch.equal(build_suffix_erase(target, block_mask),
                       build_single_erase(target, block_mask))
