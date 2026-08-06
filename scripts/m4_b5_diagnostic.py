"""B=5 diagnostic — is the b5/b6 collapse gradient starvation or DATA starvation?

Vectorised training gives every block gradient every step, so gradient starvation
is gone. But block 6 is real in only ~6% of traces: its target distribution is
estimated from far fewer distinct examples no matter how much gradient it gets.
Those are different diagnoses with different fixes — the second is a
B-versus-coverage question, not a training one.

The test: rebuild at B=5 over traces of at most 5 steps, so EVERY block position
is real in every trace. Block coverage becomes uniform while everything else —
architecture, budget, algorithm — is held fixed. If the gradient across block
index flattens, the collapse was coverage. If block 4 still trails block 0, it is
something about depth in the chain itself.

MEASUREMENT ONLY. `configs/debug.yaml` is not modified; B is overridden here and
a separate cache is built under a distinct name.

Usage:
    python scripts/m4_b5_diagnostic.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from adze.config import load_config
from adze.data.corrupt import make_pair
from adze.data.dataset import LatentCache, TraceDataset
from adze.data.generate import generate_dataset
from adze.data.tokeniser import CharTokeniser
from adze.eval.load import load_vae
from adze.model.denoiser import Denoiser
from adze.sample.draft import draft
from adze.sample.trajectory import noise_floor, rates
from adze.train.train_denoiser import (
    vectorised_regime_a_batch,
    vectorised_regime_a_loss,
)

CACHE_DIR = Path("data/cache")
B5 = 5


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--vae", type=Path, default=Path("checkpoints/vae_debug_d16.pt"))
    p.add_argument("--config", type=Path, default=Path("configs/debug.yaml"))
    p.add_argument("--n-train", type=int, default=60_000)
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--samples", type=int, default=500)
    p.add_argument("--nfe", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    config = load_config(args.config)
    device = torch.device(config.device)
    tokeniser = CharTokeniser()
    vae, vae_arch = load_vae(args.vae, device)
    d, k = vae_arch["latent_dim"], vae_arch["latents_per_block"]

    # TraceDataset drops traces longer than `blocks`, so passing blocks=5 filters
    # to <=5 steps for free. That is the point: coverage becomes uniform.
    traces = generate_dataset(
        n=args.n_train, seed=config.data.seed,
        max_depth=config.data.max_depth, operand_max=config.data.operand_max,
    )
    pairs = [make_pair(t, rng_seed=i) for i, t in enumerate(traces) if len(t.steps) >= 2]
    dataset = TraceDataset(pairs, blocks=B5, latents_per_block=k)

    path = CACHE_DIR / f"latents_{config.name}_d{d}_b5.pt"
    cache = LatentCache(path)
    if not path.exists():
        cache.build(dataset, vae)
    latents = cache.load().to(device)
    block_mask = cache.load_block_mask().to(device)
    scale = cache.scale

    coverage = block_mask.float().mean(0)
    print(f"B=5 cache     {tuple(latents.shape)}  scale {scale:.4f}")
    print(f"dropped       {dataset.n_dropped} traces with more than {B5} steps")
    print("per-block coverage (share of traces where the block is real):")
    for b in range(B5):
        print(f"  block {b}  {coverage[b]:6.1%}")
    print()

    block_ids = torch.repeat_interleave(torch.arange(B5), k).to(device)
    torch.manual_seed(args.seed)
    model = Denoiser(
        latent_dim=d, d_model=config.model.denoiser.d_model,
        n_layers=config.model.denoiser.n_layers,
        n_heads=config.model.denoiser.n_heads,
        latents_per_block=k, blocks=B5,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    model.train()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, latents.shape[0], (args.batch,), device=device)
        batch = vectorised_regime_a_batch(
            latents[idx], block_ids, B5, block_mask=block_mask[idx]
        )
        loss = vectorised_regime_a_loss(model, batch)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % max(1, args.steps // 8) == 0 or step == 1:
            print(f"  step {step:>6}  loss {loss.item():.6f}", flush=True)

    model.eval()
    torch.manual_seed(args.seed)
    gen = draft(model, None, B5, k, d, args.nfe,
                device=str(device), batch=args.samples)
    with torch.no_grad():
        texts = [
            tokeniser.decode(r)
            for r in vae.decoder((gen * scale).view(-1, k, d)).argmax(dim=-1)
        ]
    floor, floor_true, _ = noise_floor(vae.decoder, tokeniser, (k, d), 2000, device)
    wf, tr = rates(texts)

    print()
    print("=" * 70)
    print(f"B=5, UNIFORM COVERAGE — {args.samples} traces x {B5} blocks")
    print("=" * 70)
    print(f"  well-formed          {wf:7.1%}   (floor {floor:.1%})")
    print(f"  arithmetically true  {tr:7.1%}   (floor {floor_true:.1%})")
    grid = [texts[i * B5 : (i + 1) * B5] for i in range(args.samples)]
    print("\n  by block:")
    for b in range(B5):
        bwf, btr = rates([row[b] for row in grid])
        print(f"    block {b}  well-formed {bwf:6.1%}  true {btr:6.1%}")
    print()
    print("  Flat across blocks -> the B=7 collapse was COVERAGE, and the fix is a")
    print("  B-versus-coverage decision. Still declining -> depth in the chain")
    print("  costs something on its own, independent of how much data each")
    print("  position sees.")


if __name__ == "__main__":
    main()
