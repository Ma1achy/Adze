"""TASK 3 — is eta = 1 the optimum, or just the edge of the parameterisation?

Truth rises monotonically to eta = 1 in every sweep run so far. But eta = 1 is a
hard boundary, not a discovered optimum: at eta = 1 the predicted eps is discarded
entirely and there is nothing further to give. A monotone climb to a boundary
usually means the optimum lies outside the range.

The way past it is churn — add noise BEYOND what the schedule calls for, then
denoise back down. EDM's S_churn (Karras et al. 2022, arXiv 2206.00364,
Algorithm 2), mapped from their variance-exploding sigma to our rectified-flow t
by sigma = t/(1-t). See `adze.sample.stochastic` for the transcription and the
map; the second-order Heun correction is deliberately not taken, because that is a
better integrator rather than more stochasticity and would confound the reading.

THE PREDICTION THIS TESTS. Churn's mechanism in EDM is correcting ACCUMULATED ODE
error. At eta = 1 there is nothing accumulating: eps_hat is discarded every step
and only z0_hat carries forward. All churn changes there is *where* z0_hat is
evaluated — at t_hat > t, a noisier input, hence a worse estimate. So churn should
be neutral-to-harmful at eta = 1 and useful at eta = 0 and 0.5, where the
mechanism it was designed for actually operates. If churn helps at eta = 1 anyway,
that is evidence about something other than error accumulation and is worth
chasing.

Hence the full eta x S_churn grid rather than a churn sweep at the best eta alone.

S_churn = 0 is the identity — t_hat == t exactly — so every row's S_churn = 0 cell
reproduces the plain eta sweep and is a free correctness check.

Usage:
    python scripts/m4_churn.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from adze.config import load_config
from adze.data.dataset import LatentCache
from adze.data.tokeniser import CharTokeniser
from adze.eval.load import load_denoiser, load_vae
from adze.sample.draft import draft
from adze.sample.trajectory import rates

CACHE_DIR = Path("data/cache")

ETAS = [0.0, 0.5, 1.0]
CHURNS = [0.0, 10.0, 20.0, 40.0, 80.0]

# EDM churns in a middle band of sigma and leaves the ends alone: at the top of the
# schedule the point is already pure noise and there is nothing to correct, and at
# the bottom churn undoes the sharpening the last steps just did. In t, with
# sigma = t/(1-t), this band is roughly the middle of [0, 1].
WINDOW = (0.05, 0.95)


@torch.no_grad()
def score(denoiser, vae, tokeniser, arch, scale, nfe, eta, s_churn,
          samples, device, seed, blocks=None):
    """Draft and classify at one (eta, S_churn) cell. Returns (wf, true, per block)."""
    torch.manual_seed(seed)
    b = blocks if blocks is not None else arch["blocks"]
    latents = draft(
        denoiser, None, b, arch["latents_per_block"], arch["latent_dim"], nfe,
        device=str(device), batch=samples, eta=eta,
        s_churn=s_churn, churn_window=WINDOW,
    )
    per = (latents * scale).view(samples * b, arch["latents_per_block"], -1)
    texts = [tokeniser.decode(r) for r in vae.decoder(per).argmax(dim=-1)]
    formed, true = rates(texts)
    grid = [texts[i * b : (i + 1) * b] for i in range(samples)]
    return formed, true, [rates([r[i] for r in grid])[1] for i in range(b)]


def grid(name, denoiser, vae, tokeniser, arch, scale, nfe, samples, device, seed,
         blocks=None):
    print("=" * 78)
    print(f"{name} — arithmetic truth, eta x S_churn")
    print("=" * 78)
    header = "  ".join(f"S={c:g}".rjust(8) for c in CHURNS)
    print(f"  {'eta':>5}  {header}")
    out = {}
    for eta in ETAS:
        cells = []
        for s_churn in CHURNS:
            _, true, per = score(denoiser, vae, tokeniser, arch, scale, nfe, eta,
                                 s_churn, samples, device, seed, blocks)
            out[(eta, s_churn)] = (true, per)
            cells.append(true)
        print(f"  {eta:>5.2f}  " + "  ".join(f"{c:>8.1%}" for c in cells), flush=True)
    print()
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--vae", type=Path, default=Path("checkpoints/vae_debug_d16.pt"))
    p.add_argument("--denoiser", type=Path,
                   default=Path("checkpoints/denoiser_debug_d16.pt"))
    p.add_argument("--uncond", type=Path,
                   default=Path("checkpoints/uncond_debug_d16_shift1.5.pt"))
    p.add_argument("--config", type=Path, default=Path("configs/debug.yaml"))
    p.add_argument("--samples", type=int, default=500)
    p.add_argument("--nfe", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    config = load_config(args.config)
    device = torch.device(config.device)
    tokeniser = CharTokeniser()
    vae, _ = load_vae(args.vae, device)

    denoiser, arch, scale = load_denoiser(args.denoiser, device)
    if scale is None:
        scale = LatentCache(
            CACHE_DIR / f"latents_{config.name}_d{arch['latent_dim']}.pt"
        ).scale

    print(f"denoiser      {args.denoiser}  {arch['n_layers']}L x {arch['d_model']}w")
    print(f"grid          eta {ETAS} x S_churn {CHURNS}, nfe {args.nfe}, "
          f"{args.samples} traces per cell")
    print(f"churn window  t in {WINDOW}")
    print("  S_churn = 0 is the identity, so that column must reproduce the plain")
    print("  eta sweep exactly. Prediction: churn helps at eta 0 and 0.5, and is")
    print("  neutral-to-harmful at eta = 1 where nothing accumulates.")
    print()

    bc = grid("BLOCK-CONDITIONAL", denoiser, vae, tokeniser, arch, scale,
              args.nfe, args.samples, device, args.seed)

    if args.uncond.exists():
        ud, uarch, uscale = load_denoiser(args.uncond, device)
        uscale = uscale if uscale is not None else scale
        grid("UNCONDITIONAL (single block, comparable to last night's 70.7%)",
             ud, vae, tokeniser, uarch, uscale, args.nfe, 2000, device, args.seed,
             blocks=1)
    else:
        print(f"  MISSING {args.uncond} — unconditional grid skipped")

    best = max(bc, key=lambda k: bc[k][0])
    print("=" * 78)
    print("BEST BLOCK-CONDITIONAL CELL")
    print("=" * 78)
    print(f"  eta {best[0]:.2f}, S_churn {best[1]:g}  ->  {bc[best][0]:.1%} true")
    print("  per block: " + "  ".join(f"{v:.1%}" for v in bc[best][1]))
    print()
    print("  Truth climbing past the S_churn = 0 column at eta = 1 would mean the")
    print("  74.5% ceiling is not the sampler's limit and the constraint we have")
    print("  been reading is the wrong one.")


if __name__ == "__main__":
    main()
