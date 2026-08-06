"""M4 — draft traces and print the trajectory.

Acceptance for M4 is syntactic well-formedness, not correctness. But the decoder
emits well-formed arithmetic from almost any latent, so a raw well-formedness
figure means nothing on its own. This script MEASURES the noise floor for the
loaded decoder — decode random Gaussian latents, count how many parse — and
reports the sampler against that. Below the floor means the sampler is doing
worse than noise.

Nothing here filters, retries, or snaps output. What is printed is what the model
produced.

Usage:
    python scripts/m4_sample.py --denoiser checkpoints/denoiser_debug_d16.pt \
        --vae checkpoints/vae_debug_d16.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from adze.config import load_config
from adze.data.dataset import LatentCache
from adze.data.tokeniser import CharTokeniser
from adze.model.denoiser import Denoiser
from adze.model.vae import build_vae
from adze.sample.draft import draft
from adze.sample.trajectory import TrajectoryRecorder, classify, noise_floor

CACHE_DIR = Path("data/cache")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--denoiser", type=Path, default=Path("checkpoints/denoiser_debug_d16.pt"))
    p.add_argument("--vae", type=Path, default=Path("checkpoints/vae_debug_d16.pt"))
    p.add_argument("--config", type=Path, default=Path("configs/debug.yaml"))
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--nfe", type=int, default=None)
    p.add_argument("--every", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--zero-prefix", action="store_true",
                   help="match a model trained with --zero-prefix")
    args = p.parse_args()

    config = load_config(args.config)
    device = torch.device(config.device)
    tokeniser = CharTokeniser()
    torch.manual_seed(args.seed)

    vae_state = torch.load(args.vae, map_location=device, weights_only=False)
    vae = build_vae(**vae_state["arch"]).to(device)
    vae.load_state_dict(vae_state["model"])
    vae.eval()

    den_state = torch.load(args.denoiser, map_location=device, weights_only=False)
    arch = den_state["arch"]
    denoiser = Denoiser(**arch).to(device)
    denoiser.load_state_dict(den_state["model"])
    denoiser.eval()

    scale = den_state.get("latent_scale") or LatentCache(
        CACHE_DIR / f"latents_{config.name}_d{arch['latent_dim']}.pt"
    ).scale
    nfe = args.nfe if args.nfe is not None else config.sample.nfe

    print(f"denoiser      {args.denoiser}  {arch['n_layers']}L x {arch['d_model']}w")
    print(f"vae           {args.vae}  D={arch['latent_dim']}")
    print(f"latent scale  {scale:.4f}")
    print(f"sampling      B={arch['blocks']} K={arch['latents_per_block']} nfe={nfe} "
          f"({nfe * arch['blocks']} forward passes per trace)")
    print()

    # ---- Noise floor, measured for THIS decoder ------------------------------
    floor, floor_true, examples = noise_floor(
        vae.decoder,
        tokeniser,
        (arch["latents_per_block"], arch["latent_dim"]),
        1000,
        device,
    )
    print("=" * 74)
    print("NOISE FLOOR — random Gaussian latents through this decoder")
    print("=" * 74)
    print(f"  well-formed          {floor:7.1%}   <- the bar any sample must beat")
    print(f"  arithmetically true  {floor_true:7.1%}")
    for text in examples:
        print(f"    e.g. {text!r}")

    # ---- Trajectory for one sample -------------------------------------------
    recorder = TrajectoryRecorder(
        vae.decoder,
        tokeniser,
        arch["latents_per_block"],
        latent_scale=scale,
        noise_floor=floor,
    )
    draft(
        denoiser,
        None,
        arch["blocks"],
        arch["latents_per_block"],
        arch["latent_dim"],
        nfe,
        device=str(device),
        batch=1,
        recorder=recorder,
        zero_prefix=args.zero_prefix,
    )
    print()
    print("=" * 74)
    print("TRAJECTORY — one drafted trace, every denoising step")
    print("=" * 74)
    recorder.print(every=args.every)

    # ---- Well-formedness over many samples -----------------------------------
    latents = draft(
        denoiser,
        None,
        arch["blocks"],
        arch["latents_per_block"],
        arch["latent_dim"],
        nfe,
        device=str(device),
        batch=args.samples,
        zero_prefix=args.zero_prefix,
    )
    per_block = (latents * scale).view(
        args.samples * arch["blocks"], arch["latents_per_block"], -1
    )
    with torch.no_grad():
        texts = [tokeniser.decode(row) for row in vae.decoder(per_block).argmax(dim=-1)]
    kinds = [classify(t) for t in texts]
    formed = sum(1 for k in kinds if k != "malformed") / len(kinds)
    true = sum(1 for k in kinds if k == "true") / len(kinds)

    print()
    print("=" * 74)
    print(f"WELL-FORMEDNESS — {args.samples} traces x {arch['blocks']} blocks")
    print("=" * 74)
    print(f"  well-formed          {formed:7.1%}   (noise floor {floor:.1%})")
    print(f"  arithmetically true  {true:7.1%}   (noise floor {floor_true:.1%})")
    print(f"  margin over floor    {formed - floor:+7.1%}")

    # Per block, because block 0 has no prefix and is expected to be the weakest.
    print("\n  by block:")
    grid = [kinds[i * arch["blocks"] : (i + 1) * arch["blocks"]] for i in range(args.samples)]
    for b in range(arch["blocks"]):
        col = [row[b] for row in grid]
        wf = sum(1 for k in col if k != "malformed") / len(col)
        tr = sum(1 for k in col if k == "true") / len(col)
        note = "   <- no prefix; floors until M5 question conditioning" if b == 0 else ""
        print(f"    block {b}  well-formed {wf:6.1%}  true {tr:6.1%}{note}")


if __name__ == "__main__":
    main()
