"""Task 4 — does the diffusion reproduce the marginal at ALL?

Train a denoiser on a SINGLE block position with no prefix, no conditioning, no
causal structure: pure unconditional generation of "a valid arithmetic step".
Same data, same architecture, same frozen VAE.

This is the cleanest available test of "does the diffusion reproduce the
marginal", isolated from every architectural choice. The training data contains
only true steps, so a working unconditional model should produce true steps at
close to the rate real latents decode at.

  truth near the real-latent decode rate  -> the machinery works, and the fault
                                             is in the block-conditional setup
  truth at the ~6% noise floor            -> the fault is upstream of everything
                                             block-related: the flow, the sampler,
                                             or the latent space itself

CONFOUND, stated up front: pooling every block position gives this model far more
data per parameter than the block-conditional model gets per position. So "the
unconditional model does better" is partly an artefact of that and is NOT a clean
win. It does not touch the decisive case — if this also lands at 6%, more data per
parameter did not rescue it and the floor is real.

Run at two timestep shifts. 1.5 concentrates draws at high t (the M3/M4 setting);
0.5 inverts it toward small t, where final sharpening happens and where the
under-norm appears. Answering that here costs minutes rather than the three full
block-conditional training runs.

Usage:
    python scripts/m4_unconditional.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from adze.config import load_config
from adze.data.dataset import LatentCache
from adze.data.tokeniser import CharTokeniser
from adze.eval.load import load_vae
from adze.model.denoiser import Denoiser
from adze.sample.draft import draft
from adze.sample.trajectory import noise_floor, rates
from adze.train.train_denoiser import regime_a_batch, regime_a_loss

CACHE_DIR = Path("data/cache")
CHECKPOINT_DIR = Path("checkpoints")


@torch.no_grad()
def decode_rates(decoder, tokeniser, latents, batch=4096):
    """(well-formed, true) for [M, K, D] UNSCALED latents."""
    texts = []
    for i in range(0, latents.shape[0], batch):
        texts += [
            tokeniser.decode(row)
            for row in decoder(latents[i : i + batch]).argmax(dim=-1)
        ]
    return rates(texts)


def train_unconditional(z0, k, d, config, steps, batch_size, lr, t_shift, device, seed):
    """Plain rectified flow on [M, K, D] block latents. Returns the trained model.

    Reuses `regime_a_batch` / `regime_a_loss` unchanged: with blocks=1 the sampled
    block is always 0, the causal mask over a single block is all-True, and the
    prefix is empty. Regime A degenerates to unconditional training, which is
    exactly what this needs — and reusing the real training path means a bug in it
    shows up here too, rather than being masked by a clean reimplementation.
    """
    torch.manual_seed(seed)
    model = Denoiser(
        latent_dim=d,
        d_model=config.model.denoiser.d_model,
        n_layers=config.model.denoiser.n_layers,
        n_heads=config.model.denoiser.n_heads,
        latents_per_block=k,
        blocks=1,
    ).to(device)
    block_ids = torch.zeros(k, dtype=torch.long, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    model.train()
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        idx = torch.randint(0, z0.shape[0], (batch_size,), device=device)
        batch = regime_a_batch(z0[idx], block_ids, 1, t_shift=t_shift)
        loss = regime_a_loss(model, batch, block_ids)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % max(1, steps // 8) == 0 or step == 1:
            print(f"    step {step:>6}  loss {loss.item():.6f}")
    print(f"    trained in {time.perf_counter() - t0:.1f}s")
    return model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--vae", type=Path, default=Path("checkpoints/vae_debug_d16.pt"))
    p.add_argument("--config", type=Path, default=Path("configs/debug.yaml"))
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--samples", type=int, default=2000)
    p.add_argument("--nfe", type=int, default=32)
    p.add_argument("--shifts", type=float, nargs="+", default=[1.5, 0.5])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    config = load_config(args.config)
    device = torch.device(config.device)
    tokeniser = CharTokeniser()

    vae, vae_arch = load_vae(args.vae, device)
    d = vae_arch["latent_dim"]
    k = vae_arch["latents_per_block"]
    cache = LatentCache(CACHE_DIR / f"latents_{config.name}_d{d}.pt")
    scale = cache.scale

    # Every REAL block latent, pooled across positions and traces. Pad blocks are
    # excluded — they hold one constant vector and would be a trivially learnable
    # mode that is not a valid arithmetic step.
    latents = cache.load().to(device)
    block_mask = cache.load_block_mask().to(device)
    z0 = latents.view(-1, k, d)[block_mask.reshape(-1)]

    print(f"vae           {args.vae}  D={d} K={k}")
    print(f"latent scale  {scale:.4f}")
    print(f"data          {z0.shape[0]} real block latents, pooled over positions")
    print(f"training      {args.steps} steps, batch {args.batch}, lr {args.lr}")
    print(f"sampling      nfe {args.nfe}, {args.samples} samples")
    print()

    # ---- The two reference points, both measured -----------------------------
    floor, floor_true, _ = noise_floor(vae.decoder, tokeniser, (k, d), 2000, device)
    real_wf, real_true = decode_rates(vae.decoder, tokeniser, z0[:8000] * scale)
    print("=" * 74)
    print("REFERENCE POINTS")
    print("=" * 74)
    print(f"  real latents    well-formed {real_wf:6.1%}   true {real_true:6.1%}"
          f"   <- the ceiling")
    print(f"  random latents  well-formed {floor:6.1%}   true {floor_true:6.1%}"
          f"   <- the floor")
    print()

    results = []
    for shift in args.shifts:
        print("=" * 74)
        print(f"UNCONDITIONAL MODEL — t_shift {shift}")
        print("=" * 74)
        model = train_unconditional(
            z0, k, d, config, args.steps, args.batch, args.lr, shift, device, args.seed
        )
        torch.manual_seed(args.seed)
        gen = draft(model, None, 1, k, d, args.nfe,
                    device=str(device), batch=args.samples)
        wf, tr = decode_rates(vae.decoder, tokeniser, gen.view(-1, k, d) * scale)

        gen_var = gen.reshape(-1, d).var(0, unbiased=False)
        real_var = z0.reshape(-1, d).var(0, unbiased=False)
        print(f"    well-formed {wf:.1%}  (floor {floor:.1%}, ceiling {real_wf:.1%})")
        print(f"    true        {tr:.1%}  (floor {floor_true:.1%}, "
              f"ceiling {real_true:.1%})")
        print(f"    total variance gen/real  {(gen_var.sum() / real_var.sum()):.3f}")

        with torch.no_grad():
            texts = [tokeniser.decode(r)
                     for r in vae.decoder(gen[:6].view(-1, k, d) * scale).argmax(-1)]
        print("    raw samples: " + "  ".join(repr(t) for t in texts))
        print()
        results.append((shift, wf, tr))

        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "arch": {
                    "latent_dim": d,
                    "d_model": config.model.denoiser.d_model,
                    "n_layers": config.model.denoiser.n_layers,
                    "n_heads": config.model.denoiser.n_heads,
                    "latents_per_block": k,
                    "blocks": 1,
                },
                "latent_scale": scale,
                "t_shift": shift,
            },
            CHECKPOINT_DIR / f"uncond_debug_d{d}_shift{shift}.pt",
        )

    print("=" * 74)
    print("SUMMARY")
    print("=" * 74)
    print(f"  {'shift':>6} {'well-formed':>12} {'true':>8}")
    for shift, wf, tr in results:
        print(f"  {shift:>6.1f} {wf:>12.1%} {tr:>8.1%}")
    print(f"  {'REAL':>6} {real_wf:>12.1%} {real_true:>8.1%}   <- ceiling")
    print(f"  {'NOISE':>6} {floor:>12.1%} {floor_true:>8.1%}   <- floor")
    print()
    print("  Truth near the ceiling: the diffusion machinery works and the fault is")
    print("  in the block-conditional setup. Truth at the floor: the fault is")
    print("  upstream of everything block-related, and the data-per-parameter")
    print("  confound is irrelevant because more data did not rescue it.")


if __name__ == "__main__":
    main()
