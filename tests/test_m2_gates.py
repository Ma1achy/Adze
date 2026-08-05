"""M2/M3 gate semantics.

These test that the GATES behave correctly, not that the model passes them.
A gate that can't fail is not a gate.
"""

from __future__ import annotations

import pytest

from adze.eval.checks import latent_use_check, overfit_one_batch


@pytest.mark.slow
def test_latent_use_check_returns_gap() -> None:
    """Must report clean vs shuffled accuracy and the gap between them.

    PASS means a LARGE gap — shuffled latents should destroy reconstruction.
    A small gap means posterior collapse: the decoder is modelling steps
    unconditionally and ignoring the latent entirely.
    """
    pytest.skip("requires a trained VAE — run manually at end of M2")


@pytest.mark.slow
def test_overfit_one_batch_reaches_near_zero() -> None:
    """8 examples, near-zero loss. If this fails, suspect broadcast_t or the mask
    before suspecting the idea."""
    pytest.skip("requires a denoiser — run manually at end of M3")


def test_gate_functions_exist() -> None:
    assert callable(latent_use_check)
    assert callable(overfit_one_batch)
