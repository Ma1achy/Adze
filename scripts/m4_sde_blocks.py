"""TASK 1 — does the SDE gain transfer to block-conditional generation?

Everything gates on this. Unconditionally, switching the update rule from Euler
ODE to predict-then-renoise took truth 38.6% -> 70.7% at eta = 1, against a 74.5%
unseen-step ceiling. The block-conditional model sits at 29.5% under the ODE. The
two have never been compared with the sampler held fixed, so the 29.5-vs-38.6
comparison that has been gating M5 was varying two things at once.

No retraining: the three seed checkpoints from last night are loaded as-is, so the
eta sweep is the only thing that moves.

THE PREDICTION THIS TESTS. SDE fixes a sampler-manifold problem. It cannot fix
"the model never learned arithmetic above 30". So the lift should track OPERAND
MAGNITUDE rather than block index — large at b0-b2 (small operands, sampler-
limited), small at b5-b6 (large operands, knowledge-limited). If that holds, the
two failure modes separate cleanly and the magnitude cliff is the real remaining
blocker. If b5-b6 lift proportionally instead, part of the cliff was a sampler
artifact too. Hence the magnitude grouping alongside the per-block one: same
samples, one extra grouping, and it is what distinguishes the two readings.

Nothing here filters, retries or snaps. eta = 0 is checked against `draft`'s ODE
path and printed with its max|diff| rather than asserted.

Usage:
    python scripts/m4_sde_blocks.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from adze.config import load_config
from adze.data.dataset import LatentCache
from adze.data.tokeniser import CharTokeniser
from adze.eval.load import load_denoiser, load_vae
from adze.eval.magnitude import print_magnitude_table
from adze.sample.draft import draft
from adze.sample.trajectory import noise_floor, rates

CACHE_DIR = Path("data/cache")

ETAS = [0.0, 0.5, 0.75, 1.0]

# Measured last night: 87.1% of held-out steps appear verbatim in training, and the
# D=16 VAE decodes the genuinely unseen ones at 74.5%. Anything generated is novel,
# so this is the bar, not the ~97.5% round-trip figure on cached latents.
CEILING = 0.745


@torch.no_grad()
def sample_texts(denoiser, vae, tokeniser, arch, scale, nfe, eta, samples, device, seed):
    """Draft `samples` traces at this eta and decode every block. Returns texts."""
    torch.manual_seed(seed)
    latents = draft(
        denoiser, None, arch["blocks"], arch["latents_per_block"],
        arch["latent_dim"], nfe, device=str(device), batch=samples, eta=eta,
    )
    per_block = (latents * scale).view(
        samples * arch["blocks"], arch["latents_per_block"], -1
    )
    return [tokeniser.decode(r) for r in vae.decoder(per_block).argmax(dim=-1)], latents


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--vae", type=Path, default=Path("checkpoints/vae_debug_d16.pt"))
    p.add_argument("--config", type=Path, default=Path("configs/debug.yaml"))
    p.add_argument("--samples", type=int, default=500)
    p.add_argument("--nfe", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoints", type=Path, nargs="+", default=[
        Path("checkpoints/denoiser_debug_d16.pt"),
        Path("checkpoints/denoiser_debug_d16_seed1.pt"),
        Path("checkpoints/denoiser_debug_d16_seed2.pt"),
    ])
    args = p.parse_args()

    config = load_config(args.config)
    device = torch.device(config.device)
    tokeniser = CharTokeniser()
    vae, _ = load_vae(args.vae, device)

    models = []
    for path in args.checkpoints:
        if not path.exists():
            print(f"  MISSING {path} — skipped")
            continue
        denoiser, arch, scale = load_denoiser(path, device)
        if scale is None:
            scale = LatentCache(
                CACHE_DIR / f"latents_{config.name}_d{arch['latent_dim']}.pt"
            ).scale
        models.append((path, denoiser, arch, scale))
    if not models:
        raise SystemExit("no checkpoints found")

    _, _, arch, scale = models[0]
    blocks = arch["blocks"]
    floor, floor_true, _ = noise_floor(
        vae.decoder, tokeniser,
        (arch["latents_per_block"], arch["latent_dim"]), 2000, device,
    )

    print(f"vae           {args.vae}  D={arch['latent_dim']}")
    print(f"checkpoints   {len(models)} seeds, {arch['n_layers']}L x {arch['d_model']}w")
    print(f"sampling      B={blocks} nfe={args.nfe} eta {ETAS}, "
          f"{args.samples} traces per cell")
    print(f"floor         {floor:.1%} well-formed, {floor_true:.1%} true")
    print(f"ceiling       {CEILING:.1%} true on UNSEEN steps")
    print()

    # eta -> per seed (formed, true); eta -> per seed per block true
    agg: dict[float, list[tuple[float, float]]] = {e: [] for e in ETAS}
    per_block: dict[float, list[list[float]]] = {e: [] for e in ETAS}
    texts_by_eta: dict[float, list[str]] = {e: [] for e in ETAS}

    for path, denoiser, march, mscale in models:
        for eta in ETAS:
            texts, latents = sample_texts(
                denoiser, vae, tokeniser, march, mscale, args.nfe, eta,
                args.samples, device, args.seed,
            )
            formed, true = rates(texts)
            agg[eta].append((formed, true))
            grid = [texts[i * blocks : (i + 1) * blocks] for i in range(args.samples)]
            per_block[eta].append([rates([r[b] for r in grid])[1] for b in range(blocks)])
            texts_by_eta[eta].extend(texts)

            if eta == 0.0 and path == models[0][0]:
                # Free correctness check: the promoted sampler must reduce to the
                # ODE the earlier numbers were measured under.
                torch.manual_seed(args.seed)
                ode = draft(denoiser, None, blocks, march["latents_per_block"],
                            march["latent_dim"], args.nfe, eta=0.0,
                            device=str(device), batch=args.samples)
                print(f"identity check  eta=0 vs draft ODE: "
                      f"max|diff| = {(latents - ode).abs().max().item():.2e}")
                print()
            print(f"  {path.name:<38} eta {eta:>4.2f}  "
                  f"wf {formed:>6.1%}  true {true:>6.1%}", flush=True)

    def spread(vals: list[float]) -> str:
        lo, hi = min(vals), max(vals)
        return f"{sum(vals) / len(vals):>6.1%} +/- {(hi - lo) / 2:>4.1%}"

    print()
    print("=" * 78)
    print(f"AGGREGATE — mean over {len(models)} seeds, half-range as spread")
    print("=" * 78)
    print(f"  {'eta':>5} {'well-formed':>18} {'true':>18} {'% of ceiling':>14}")
    for eta in ETAS:
        formed = [f for f, _ in agg[eta]]
        true = [t for _, t in agg[eta]]
        mean_true = sum(true) / len(true)
        print(f"  {eta:>5.2f} {spread(formed):>18} {spread(true):>18} "
              f"{mean_true / CEILING:>13.0%}")

    print()
    print("=" * 78)
    print("PER BLOCK — arithmetic truth, mean over seeds")
    print("=" * 78)
    header = "  ".join(f"b{b}".rjust(6) for b in range(blocks))
    print(f"  {'eta':>5}  {header}")
    base = None
    for eta in ETAS:
        cols = [sum(s[b] for s in per_block[eta]) / len(per_block[eta])
                for b in range(blocks)]
        if base is None:
            base = cols
        print(f"  {eta:>5.2f}  " + "  ".join(f"{c:>6.1%}" for c in cols))
    print(f"  {'lift':>5}  " + "  ".join(
        f"{c - b:>+6.1%}" for c, b in zip(
            [sum(s[i] for s in per_block[ETAS[-1]]) / len(per_block[ETAS[-1]])
             for i in range(blocks)], base)))

    print()
    print("=" * 78)
    print("BY OPERAND MAGNITUDE — the same samples, grouped the other way")
    print("=" * 78)
    print("  If the lift concentrates in the small-operand bins, the SDE fixed a")
    print("  sampler problem and the magnitude cliff is a separate, knowledge-side")
    print("  limit. If every bin lifts proportionally, part of the cliff was the")
    print("  sampler too.")
    print()
    for eta in ETAS:
        print_magnitude_table(texts_by_eta[eta], f"eta = {eta:.2f}  (all seeds pooled)")


if __name__ == "__main__":
    main()
