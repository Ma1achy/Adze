"""Shared fixtures."""

from __future__ import annotations

import pytest

from adze.invariants import Shapes


@pytest.fixture
def shapes() -> Shapes:
    """Small debug-scale shapes: B=4, K=4, D=64 -> N=16."""
    return Shapes(blocks=4, latents_per_block=4, latent_dim=64)
