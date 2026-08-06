"""TASK 3 — find where truth actually saturates at eta = 1, then report there.

The nfe sweep at eta = 0 flattened by nfe 32. At eta = 1 it was still climbing at
nfe 100 (37.6% -> 49.1%), which means every eta = 1 headline in session 11 was
measured below saturation. "nfe 100 is probably enough" is the same kind of
assumption that made those numbers wrong, so this script looks for the plateau
instead of assuming one.

Why the stochastic sampler wants budget the ODE does not: each step is a fresh
draw rather than a link in a chain, so more steps is more averaging, not merely
finer integration. There is no a-priori reason for that to stop at 100.

Saturation is declared on DISTRIBUTION-MATCHED truth, not raw pooled truth — raw
truth can rise while matched truth is flat if the extra budget is spent steering
into easy bins, and that is exactly the failure mode this project has already been
caught by once.

Usage:
    python scripts/m4_saturation.py --denoiser checkpoints/denoiser_matched_d16.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from adze.config import leaf_pool, load_config
from adze.data.dataset import LatentCache
from adze.data.generate import generate_dataset
from adze.data.tokeniser import CharTokeniser
from adze.eval.checks import unseen_ceiling_by_magnitude
from adze.eval.load import load_denoiser, load_vae
from adze.eval.readout import print_readout, readout
from adze.sample.draft import draft

CACHE_DIR = Path("data/cache")

NFE_VALUES = [8, 16, 32, 64, 100, 150, 200, 300]

# Saturation: matched truth gains less than this over a DOUBLING of budget. Stated
# up front so the plateau is read against a fixed rule rather than eyeballed.
SATURATION_GAIN = 0.01


@torch.no_grad()
def sample(denoiser, vae, tokeniser, arch, scale, nfe, eta, samples, device, seed):
    torch.manual_seed(seed)
    blocks = arch["blocks"]
    latents = draft(
        denoiser, None, blocks, arch["latents_per_block"], arch["latent_dim"], nfe,
        device=str(device), batch=samples, eta=eta,
    )
    per = (latents * scale).view(samples * blocks, arch["latents_per_block"], -1)
    return [tokeniser.decode(r) for r in vae.decoder(per).argmax(dim=-1)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--vae", type=Path, default=Path("checkpoints/vae_matched_d16.pt"))
    p.add_argument("--denoiser", type=Path,
                   default=Path("checkpoints/denoiser_matched_d16.pt"))
    p.add_argument("--config", type=Path, default=Path("configs/debug.yaml"))
    p.add_argument("--samples", type=int, default=400)
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    config = load_config(args.config)
    leaves = leaf_pool(config)
    device = torch.device(config.device)
    tokeniser = CharTokeniser()
    vae, _ = load_vae(args.vae, device)

    train_texts = {
        s.render() for t in generate_dataset(
            n=60_000, seed=config.data.seed, max_depth=config.data.max_depth,
            operand_max=config.data.operand_max, leaf_values=leaves) for s in t.steps
    }
    held = [
        s.render() for t in generate_dataset(
            n=8_000, seed=config.data.seed + 909_091,
            max_depth=config.data.max_depth,
            operand_max=config.data.operand_max, leaf_values=leaves) for s in t.steps
    ]
    ceiling = unseen_ceiling_by_magnitude(vae, tokeniser, train_texts, held)
    real_unseen = [t for t in held if t not in train_texts]

    denoiser, arch, scale = load_denoiser(args.denoiser, device)
    if scale is None:
        scale = LatentCache(
            CACHE_DIR / f"latents_{config.name}_d{arch['latent_dim']}.pt"
        ).scale

    print(f"denoiser      {args.denoiser}  {arch['n_layers']}L x {arch['d_model']}w")
    print(f"eta           {args.eta}")
    print(f"reference     {len(real_unseen)} unseen held-out steps")
    print(f"saturation    matched-truth gain < {SATURATION_GAIN:.1%} over a doubling")
    print()
    print("=" * 78)
    print("SATURATION SWEEP")
    print("=" * 78)
    print(f"  {'nfe':>5} {'RAW true':>10} {'MATCHED true':>14} "
          f"{'gain':>8} {'secs':>7}")

    results = {}
    prev = None
    saturated_at = None
    for nfe in NFE_VALUES:
        t0 = time.perf_counter()
        texts = sample(denoiser, vae, tokeniser, arch, scale, nfe, args.eta,
                       args.samples, device, args.seed)
        r = readout(texts, real_unseen, ceiling)
        results[nfe] = r
        gain = "" if prev is None else f"{r.matched_true - prev:>+8.1%}"
        print(f"  {nfe:>5} {r.raw_true:>10.1%} {r.matched_true:>14.1%} "
              f"{gain:>8} {time.perf_counter() - t0:>7.1f}", flush=True)
        if prev is not None and r.matched_true - prev < SATURATION_GAIN \
                and saturated_at is None:
            saturated_at = nfe
        prev = r.matched_true

    print()
    if saturated_at is None:
        print(f"  NOT SATURATED by nfe {NFE_VALUES[-1]} — matched truth is still")
        print("  climbing, and every headline below is a lower bound.")
    else:
        print(f"  SATURATED at nfe {saturated_at}: the doubling into it bought")
        print(f"  less than {SATURATION_GAIN:.1%} matched truth.")
    print()
    best = max(results, key=lambda k: results[k].matched_true)
    print_readout(results[best], f"BEST CELL — nfe {best}, eta {args.eta}")


if __name__ == "__main__":
    main()
