"""Hard gates. Do not proceed past a failing gate.

These are not metrics. They are pass/fail conditions on whether the thing you
built is capable of meaning anything.
"""

from __future__ import annotations

import torch


def latent_use_check(
    vae: torch.nn.Module,
    tokens: torch.Tensor,
    n_shuffles: int = 8,
) -> dict[str, float]:
    """M2 GATE — does the decoder actually use the latent?

    Decode normally, then decode from shuffled/random latents. If quality barely
    drops, the decoder has learned to model steps unconditionally and is ignoring
    the latent entirely (posterior collapse). Everything downstream — the
    denoiser, both passes, the whole experiment — is then measuring nothing.

    This is a five-minute test that saves a week. Run it before M3.

    Returns:
        {"clean_acc": float, "shuffled_acc": float, "gap": float}

    PASS: gap is large (shuffled accuracy collapses).
    FAIL: gap is small. Stop and fix the VAE.
    """
    raise NotImplementedError


def overfit_one_batch(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    steps: int = 500,
    lr: float = 1e-3,
) -> dict[str, float]:
    """M3 GATE — can the model drive loss to near zero on 8 examples?

    If it cannot, something is miswired. In this architecture the two usual
    suspects are:
      - the per-block timestep broadcast (adze.model.flow.broadcast_t)
      - the attention mask (adze.model.masks.build_mask)

    Both present as "diffusion is just hard", which is why this gate exists.

    Returns:
        {"initial_loss": float, "final_loss": float}

    PASS: final loss near zero.
    FAIL: do not proceed to full training.
    """
    raise NotImplementedError
