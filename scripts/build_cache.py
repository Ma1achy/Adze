"""Rebuild a latent cache from a frozen VAE checkpoint.

From M3 the VAE is frozen, so re-encoding the dataset must not require retraining
it. This exists for the cases that change the cache without changing the VAE:
a different B, a different dataset size, or the scaling constant.

Usage:
    python scripts/build_cache.py --checkpoint checkpoints/vae_debug_d64.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from adze.config import load_config
from adze.data.corrupt import make_pair
from adze.data.dataset import LatentCache, TraceDataset
from adze.data.generate import generate_dataset
from adze.model.vae import build_vae

CACHE_DIR = Path("data/cache")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/debug.yaml"))
    p.add_argument("--n-train", type=int, default=60_000)
    args = p.parse_args()

    config = load_config(args.config)
    device = torch.device(config.device)

    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    arch = state["arch"]
    vae = build_vae(**arch).to(device)
    vae.load_state_dict(state["model"])
    vae.eval()

    traces = generate_dataset(
        n=args.n_train,
        seed=config.data.seed,
        max_depth=config.data.max_depth,
        operand_max=config.data.operand_max,
    )
    pairs = [make_pair(t, rng_seed=i) for i, t in enumerate(traces) if len(t.steps) >= 2]
    dataset = TraceDataset(
        pairs,
        blocks=config.data.blocks_per_sequence,
        latents_per_block=config.model.latents_per_block,
    )

    path = CACHE_DIR / f"latents_{config.name}_d{arch['latent_dim']}.pt"
    cache = LatentCache(path)
    cache.build(dataset, vae)
    latents = cache.load()

    print(f"arch          {arch['n_layers']}L x {arch['d_model']}w, D={arch['latent_dim']}")
    print(f"cache         {path}  {tuple(latents.shape)}  [n, N=B*K, D]")
    print(f"scale         {cache.scale:.4f}")
    print(f"dropped       {dataset.n_dropped} traces "
          f"(more steps than B={config.data.blocks_per_sequence})")

    # Effective dimensionality. The pooled scale makes the radius match sqrt(D) by
    # construction, but a D far above the rank the VAE actually uses still wastes
    # most dimensions on near-zero variance. Measured rank was ~14-15 whatever D
    # was offered, which is what the bottleneck sweep settled.
    #
    # Real positions come from block_mask. Identifying pad rows by value does not
    # work: each pad block is filled with the K rows of pad_latent, so there are K
    # distinct pad vectors, not one, and a most-common-row test finds only 1/K of
    # them — which silently inflates the "real" set with padding.
    real_mask = torch.zeros(len(dataset), dataset.n_positions, dtype=torch.bool)
    for i in range(len(dataset)):
        real_mask[i] = dataset[i]["block_mask"].repeat_interleave(
            dataset.latents_per_block
        )
    real = latents[real_mask]
    root_d = arch["latent_dim"] ** 0.5
    print(f"scaled std    {real.std().item():.4f}  (target 1.0)")
    print(f"per-position  RMS norm {real.pow(2).sum(-1).mean().sqrt().item():.3f} "
          f"vs sqrt(D)={root_d:.3f} = "
          f"{real.pow(2).sum(-1).mean().sqrt().item() / root_d:.2f}x")
    sd = real.std(0)
    cum = (sd.sort(descending=True).values ** 2).cumsum(0) / (sd ** 2).sum()
    print(f"effective D   {(cum < 0.99).sum().item() + 1} of {sd.shape[0]} dims carry "
          f"99% of variance ({(sd < 0.1 * sd.max()).sum().item()} below 10% of max)")


if __name__ == "__main__":
    main()
