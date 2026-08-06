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


@torch.no_grad()
def unseen_decode_ceiling(
    vae: torch.nn.Module,
    tokeniser,
    train_texts: set[str],
    held_texts: list[str],
) -> dict[str, float]:
    """The ceiling a GENERATED step can realistically hit.

    Decoding real cached latents gives ~97.5% truth, but that mostly measures
    reconstruction of steps the VAE has memorised — the overwhelming majority of
    held-out steps also appear verbatim in training, because the space of short
    arithmetic statements is small. Anything the model invents is by construction
    novel, so the number to read generated output against is the decode rate on
    steps the VAE has never seen.

    Returns:
        {"seen_share", "unseen_true", "unseen_formed", "seen_true", "n_unseen"}

    `unseen_true` is the ceiling. `seen_true` is reported beside it so the size of
    the memorisation effect is visible rather than asserted.
    """
    from adze.sample.trajectory import rates

    device = next(vae.parameters()).device
    unseen = [t for t in held_texts if t not in train_texts]
    seen = [t for t in held_texts if t in train_texts]

    def decode_rate(texts: list[str]) -> tuple[float, float]:
        if not texts:
            return float("nan"), float("nan")
        out: list[str] = []
        for i in range(0, len(texts), 4096):
            chunk = tokeniser.encode_batch(texts[i : i + 4096]).to(device)
            mu, _ = vae.encoder(chunk)
            out += [tokeniser.decode(r) for r in vae.decoder(mu).argmax(dim=-1)]
        return rates(out)

    unseen_formed, unseen_true = decode_rate(unseen)
    _, seen_true = decode_rate(seen)
    return {
        "seen_share": len(seen) / max(len(held_texts), 1),
        "unseen_formed": unseen_formed,
        "unseen_true": unseen_true,
        "seen_true": seen_true,
        "n_unseen": len(unseen),
    }


def _overfit_vectorised(
    model, batch, steps, lr, t_shift, threshold,
    make_batch, loss_fn, MaskMode, masked_mean, real_positions,
) -> dict[str, float]:
    """Overfit gate on the vectorised training path.

    Every block is denoised every step, so every block gets a loss reading every
    step. The naive gate's per-block ratios were sampled — a block was only
    measured on the steps where its index happened to be drawn — which made the
    result depend on the draw sequence as well as on the model. Reading all blocks
    every step removes that source of variance without touching the threshold.
    """
    import torch as _torch

    z0 = batch["z0"]
    block_ids = batch["block_ids"]
    blocks = int(batch["blocks"])
    block_mask = batch.get("block_mask")
    k = z0.shape[1] // blocks

    def per_block_losses(one) -> dict[int, float]:
        """Loss for each block from a single forward pass."""
        with _torch.no_grad():
            pred = model(
                one["z_full"], one["t_full"], one["block_ids_full"],
                MaskMode.CAUSAL, mask=one["mask"],
            )[:, : z0.shape[1]]
        err = (pred - one["target"]) ** 2
        out = {}
        for b in range(blocks):
            keep = one["loss_mask"] & (block_ids == b).view(1, -1, 1)
            if keep.sum() > 0:
                out[b] = masked_mean(err, keep).item()
        return out

    opt = _torch.optim.AdamW(model.parameters(), lr=lr)

    probe = make_batch(z0, block_ids, blocks, block_mask=block_mask, t_shift=t_shift)
    initial = per_block_losses(probe)

    recent: dict[int, list[float]] = {b: [] for b in initial}
    crossed: dict[int, int] = {}

    model.train()
    for step in range(steps):
        one = make_batch(z0, block_ids, blocks, block_mask=block_mask, t_shift=t_shift)
        loss = loss_fn(model, one)
        opt.zero_grad()
        loss.backward()
        _torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 10 == 0 or step == steps - 1:
            for b, v in per_block_losses(one).items():
                recent[b].append(v)
                if b not in crossed and len(recent[b]) >= 20:
                    window = recent[b][-20:]
                    if sum(window) / len(window) < threshold * initial[b]:
                        crossed[b] = step

    def tail(values: list[float]) -> float:
        window = values[-max(1, len(values) // 10) :]
        return sum(window) / len(window)

    per_block = {b: tail(v) / initial[b] for b, v in recent.items() if v}
    conditioned = [v for b, v in crossed.items() if b > 0]
    all_conditioned = [b for b in initial if b > 0]

    return {
        "initial_loss": sum(initial.values()) / len(initial),
        "final_loss": sum(tail(v) for v in recent.values() if v) / len(per_block),
        "per_block": per_block,
        "steps_to_threshold": crossed,
        "slowest_crossing": (
            max(conditioned)
            if len(conditioned) == len(all_conditioned)
            else float("inf")
        ),
    }


def overfit_one_batch(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    steps: int = 500,
    lr: float = 1e-3,
    t_shift: float | None = 1.5,
    threshold: float = 0.02,
    vectorised: bool = True,
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

    Args:
        model: the denoiser.
        batch: {"z0": [8, N, D] clean latents, "block_ids": [N], "blocks": B}.
        steps: optimiser steps on that single batch.
        lr: learning rate for this check only.

    The batch is fixed but the *noise* is resampled every step, as it is in real
    training. Freezing eps too would let the model memorise one input-output pair
    and pass a gate that no longer tests the conditioning path — the very thing
    this is here to check.

    **Block 0 has an irreducible floor.** Regime A gives the denoised block its
    preceding blocks as clean context, and block 0 has none. At high `t` its input
    is nearly pure noise with nothing to identify which example it belongs to, so
    the best available prediction is the mean over the batch. Measured on 8
    examples, b=0 settles near 5% of its initial loss while b=3 and b=6 reach ~1%.
    That is the unconditional case, not a wiring fault, and it is what question
    conditioning fixes at M5. `per_block` is returned so the two are never
    confused.
    """
    from adze.train.train_denoiser import (
        regime_a_batch,
        regime_a_loss,
        vectorised_regime_a_batch,
        vectorised_regime_a_loss,
    )
    from adze.invariants import MaskMode
    from adze.pad import masked_mean, real_positions

    if vectorised:
        return _overfit_vectorised(
            model, batch, steps, lr, t_shift, threshold,
            vectorised_regime_a_batch, vectorised_regime_a_loss,
            MaskMode, masked_mean, real_positions,
        )

    z0 = batch["z0"]
    block_ids = batch["block_ids"]
    blocks = int(batch["blocks"])
    block_mask = batch.get("block_mask")

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    # Measure the baseline for every block BEFORE any optimisation. Recording each
    # block's first *observed* loss instead is wrong by a factor of ten: blocks are
    # sampled at random, so by the time block b first comes up the shared weights
    # have already trained on other blocks and collapsed toward predicting the
    # mean. That deflates the baseline and makes every ratio look worse than it is.
    initial: dict[int, float] = {}
    with torch.no_grad():
        for b in range(blocks):
            probe = regime_a_batch(
                z0, block_ids, blocks, force_block=b, block_mask=block_mask,
                t_shift=t_shift,
            )
            # A block that is padding for every example in the batch has nothing to
            # score. Skip it rather than record a zero that reads as a pass.
            if probe["loss_mask"].sum() == 0:
                continue
            initial[b] = regime_a_loss(model, probe, block_ids).item()

    recent: dict[int, list[float]] = {b: [] for b in initial}
    # Step at which each block's running mean first crosses the threshold. This is
    # the quantity that compares timestep schedules: final loss at a fixed budget
    # conflates "converges lower" with "converges sooner".
    crossed: dict[int, int] = {}

    for step in range(steps):
        one = regime_a_batch(
            z0, block_ids, blocks, block_mask=block_mask, t_shift=t_shift
        )
        if one["loss_mask"].sum() == 0:
            continue
        loss = regime_a_loss(model, one, block_ids)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        b_i = int(one["block"])
        if b_i in recent:
            recent[b_i].append(loss.item())
            if b_i not in crossed and len(recent[b_i]) >= 20:
                window = recent[b_i][-20:]
                if sum(window) / len(window) < threshold * initial[b_i]:
                    crossed[b_i] = step

    # Average the tail rather than reading one final step: a single step's loss is
    # dominated by whichever t and block it happened to draw, and reporting that
    # makes the gate a coin flip.
    def tail(values: list[float]) -> float:
        window = values[-max(1, len(values) // 10):]
        return sum(window) / len(window)

    per_block = {b: tail(v) / initial[b] for b, v in recent.items() if v}
    initial_loss = sum(initial.values()) / len(initial)
    final_loss = sum(tail(v) for v in recent.values() if v) / len(per_block)

    # Conditioned blocks only: block 0 has no prefix and may never cross.
    conditioned_crossings = [v for b, v in crossed.items() if b > 0]
    all_conditioned = [b for b in initial if b > 0]

    return {
        "initial_loss": float(initial_loss),
        "final_loss": float(final_loss),
        "per_block": per_block,
        "steps_to_threshold": crossed,
        "slowest_crossing": (
            max(conditioned_crossings)
            if len(conditioned_crossings) == len(all_conditioned)
            else float("inf")
        ),
    }
