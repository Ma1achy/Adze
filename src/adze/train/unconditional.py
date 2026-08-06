"""Unconditional single-block training — the clean setting for probes.

No prefix, no positional structure, no per-block budget split. Regime A with
blocks=1 degenerates to plain rectified flow on [K, D] block latents, which makes
this the place to test hypotheses about the target distribution and the sampler
without any block-conditional machinery confounding the result.

It reuses `regime_a_batch` / `regime_a_loss` deliberately rather than
reimplementing the loop: a bug in the real training path then shows up here too,
instead of being masked by a clean reimplementation that happens to be correct.
"""

from __future__ import annotations

import time

import torch

from adze.model.denoiser import Denoiser
from adze.train.train_denoiser import regime_a_batch, regime_a_loss


@torch.no_grad()
def decode_rates(decoder, tokeniser, latents: torch.Tensor, batch: int = 4096):
    """(well-formed, true) for [M, K, D] latents already in the UNSCALED space."""
    from adze.sample.trajectory import rates

    texts: list[str] = []
    for i in range(0, latents.shape[0], batch):
        texts += [
            tokeniser.decode(row)
            for row in decoder(latents[i : i + batch]).argmax(dim=-1)
        ]
    return rates(texts)


def train_unconditional(
    z0: torch.Tensor,
    k: int,
    d: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    steps: int,
    batch_size: int,
    lr: float,
    t_shift: float | None,
    device: torch.device,
    seed: int = 0,
    jitter: float = 0.0,
    quiet: bool = False,
) -> Denoiser:
    """Plain rectified flow on [M, K, D] block latents.

    Args:
        jitter: standard deviation of isotropic Gaussian noise added to z0 on every
            draw. 0 disables it. This is augmentation, not a fixed perturbation —
            resampled per draw — so it smooths the target distribution rather than
            moving it. Used to test whether latent clustering is what the flow
            cannot hit.
    """
    torch.manual_seed(seed)
    model = Denoiser(
        latent_dim=d, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
        latents_per_block=k, blocks=1,
    ).to(device)
    block_ids = torch.zeros(k, dtype=torch.long, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    model.train()
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        idx = torch.randint(0, z0.shape[0], (batch_size,), device=device)
        target = z0[idx]
        if jitter > 0:
            target = target + torch.randn_like(target) * jitter
        batch = regime_a_batch(target, block_ids, 1, t_shift=t_shift)
        loss = regime_a_loss(model, batch, block_ids)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if not quiet and (step % max(1, steps // 6) == 0 or step == 1):
            print(f"    step {step:>6}  loss {loss.item():.6f}", flush=True)
    if not quiet:
        print(f"    trained in {time.perf_counter() - t0:.1f}s", flush=True)
    return model.eval()
