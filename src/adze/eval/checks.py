"""Hard gates. Do not proceed past a failing gate.

These are not metrics. They are pass/fail conditions on whether the thing you
built is capable of meaning anything.

Everything here measures the model's raw output. Nothing is clamped, retried,
resampled, or constrained to well-formed arithmetic on the way out. A gate that
grades a cleaned-up version of the output is not measuring the model.
"""

from __future__ import annotations

import torch


def _accuracies(logits: torch.Tensor, tokens: torch.Tensor) -> tuple[float, float]:
    """Argmax reconstruction accuracy against the true tokens.

    Returns:
        (token_acc, seq_acc) — per-token agreement, and the share of steps
        reconstructed exactly at every position.
    """
    pred = logits.argmax(dim=-1)
    correct = pred == tokens
    token_acc = correct.float().mean().item()
    seq_acc = correct.all(dim=-1).float().mean().item()
    return token_acc, seq_acc


@torch.no_grad()
def reconstruction_accuracy(
    vae: torch.nn.Module,
    tokens: torch.Tensor,
) -> dict[str, float]:
    """M2 GATE 1 — held-out reconstruction. Encode, decode, compare.

    Returns:
        {"token_acc": float, "exact_match": float}

    PASS: exact_match > 0.95.
    """
    vae.eval()
    mu, _ = vae.encoder(tokens)
    logits = vae.decoder(mu)
    token_acc, seq_acc = _accuracies(logits, tokens)
    return {"token_acc": token_acc, "exact_match": seq_acc}


@torch.no_grad()
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

    `shuffled_acc` pairs each step with another *real* step's latent, averaged over
    `n_shuffles` permutations — the harder and more honest test, since a random
    Gaussian latent may simply fall off the posterior manifold and produce garbage
    for reasons that have nothing to do with whether the latent is being read.
    `random_acc` is reported alongside as the weaker cross-check.
    """
    vae.eval()
    mu, _ = vae.encoder(tokens)

    clean_logits = vae.decoder(mu)
    _, clean_acc = _accuracies(clean_logits, tokens)

    shuffled_accs: list[float] = []
    for _ in range(n_shuffles):
        # A derangement is not enforced; with batch >> 1 the fixed-point rate is
        # ~1/batch and would flatter the result by well under a percentage point.
        perm = torch.randperm(mu.shape[0], device=mu.device)
        _, acc = _accuracies(vae.decoder(mu[perm]), tokens)
        shuffled_accs.append(acc)
    shuffled_acc = sum(shuffled_accs) / len(shuffled_accs)

    random_accs: list[float] = []
    for _ in range(n_shuffles):
        _, acc = _accuracies(vae.decoder(torch.randn_like(mu)), tokens)
        random_accs.append(acc)
    random_acc = sum(random_accs) / len(random_accs)

    return {
        "clean_acc": clean_acc,
        "shuffled_acc": shuffled_acc,
        "gap": clean_acc - shuffled_acc,
        "random_acc": random_acc,
    }


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
    raise NotImplementedError("M3 — not this milestone")
