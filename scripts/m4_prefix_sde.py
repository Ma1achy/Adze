"""TASK 2 — does a better sampler shrink the exposure-bias gap?

Last session found the ordering `correct prefix >> no prefix > generated prefix`:
zeroing the prefix entirely BEAT conditioning on a generated one, at every block.
That is exposure bias in pure form — the generated prefix is in-distribution (so
the distributional tests could not see it) but wrong, and wrong context is worse
than no context.

A better sampler produces better prefixes, so more of them should be correct, so
the generated-prefix penalty should shrink. If SDE closes the gap, the exposure
bias was downstream of sampler quality rather than a standing property of the
architecture, and design §3.1's stage 2 stays deferred. If it persists, stage 2 is
the answer and gets promoted.

Three arms, all under the same eta sweep:

  correct    real cached latents as the prefix (teacher forcing) — the upper bound
  generated  ordinary drafting — what inference actually does
  none       prefix zeroed, on the model TRAINED with zero_prefix, because a
             sampler whose prefix differs from training's measures something the
             model was never taught

The `none` arm necessarily uses a different checkpoint. That is not a confound to
apologise for — it is the only way to ask the question honestly — but it does mean
`none` vs the other two compares two models, while `correct` vs `generated`
compares one model against itself. The second comparison is the one that carries
the argument.

Usage:
    python scripts/m4_prefix_sde.py
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
from adze.sample.trajectory import rates

CACHE_DIR = Path("data/cache")
ETAS = [0.0, 0.5, 1.0]


@torch.no_grad()
def score(denoiser, vae, tokeniser, arch, scale, nfe, eta, samples, device, seed,
          clean_prefix=None, zero_prefix=False):
    """Draft and classify. Returns (well-formed, true, per-block true)."""
    torch.manual_seed(seed)
    blocks = arch["blocks"]
    latents = draft(
        denoiser, None, blocks, arch["latents_per_block"], arch["latent_dim"], nfe,
        device=str(device), batch=samples, eta=eta,
        clean_prefix=clean_prefix, zero_prefix=zero_prefix,
    )
    per = (latents * scale).view(samples * blocks, arch["latents_per_block"], -1)
    texts = [tokeniser.decode(r) for r in vae.decoder(per).argmax(dim=-1)]
    formed, true = rates(texts)
    grid = [texts[i * blocks : (i + 1) * blocks] for i in range(samples)]
    return formed, true, [rates([r[b] for r in grid])[1] for b in range(blocks)], texts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--vae", type=Path, default=Path("checkpoints/vae_debug_d16.pt"))
    p.add_argument("--denoiser", type=Path,
                   default=Path("checkpoints/denoiser_debug_d16.pt"))
    p.add_argument("--zeroprefix-denoiser", type=Path,
                   default=Path("checkpoints/denoiser_debug_d16_zeroprefix.pt"))
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
    cache = LatentCache(CACHE_DIR / f"latents_{config.name}_d{arch['latent_dim']}.pt")
    if scale is None:
        scale = cache.scale
    blocks = arch["blocks"]

    # Real latents for the teacher-forced arm. Only real (non-padded) traces, so
    # the "correct" prefix is genuinely a correct reasoning chain and not padding.
    all_latents = cache.load().to(device)
    block_mask = cache.load_block_mask().to(device)
    full = block_mask.all(dim=1).nonzero().squeeze(-1)
    if full.numel() < args.samples:
        print(f"  only {full.numel()} fully-real traces; using them all")
    idx = full[torch.randperm(full.numel(), generator=torch.Generator().manual_seed(0))
               [: args.samples]]
    real = all_latents[idx]
    n = real.shape[0]

    zp = None
    if args.zeroprefix_denoiser.exists():
        zp, zp_arch, zp_scale = load_denoiser(args.zeroprefix_denoiser, device)
        zp_scale = zp_scale if zp_scale is not None else scale
    else:
        print(f"  MISSING {args.zeroprefix_denoiser} — 'none' arm skipped")

    print(f"denoiser      {args.denoiser}  {arch['n_layers']}L x {arch['d_model']}w")
    print(f"traces        {n} fully-real traces for the teacher-forced arm")
    print(f"sampling      B={blocks} nfe={args.nfe} eta {ETAS}")
    print()

    rows: dict[str, dict[float, tuple[float, float, list[float]]]] = {}
    for name in ("correct", "generated", "none"):
        rows[name] = {}
        for eta in ETAS:
            if name == "correct":
                out = score(denoiser, vae, tokeniser, arch, scale, args.nfe, eta,
                            n, device, args.seed, clean_prefix=real)
            elif name == "generated":
                out = score(denoiser, vae, tokeniser, arch, scale, args.nfe, eta,
                            n, device, args.seed)
            else:
                if zp is None:
                    continue
                out = score(zp, vae, tokeniser, zp_arch, zp_scale, args.nfe, eta,
                            n, device, args.seed, zero_prefix=True)
            rows[name][eta] = out
            print(f"  {name:>10}  eta {eta:>4.2f}  wf {out[0]:>6.1%}  "
                  f"true {out[1]:>6.1%}", flush=True)

    print()
    print("=" * 74)
    print("THE GAP — correct prefix minus generated prefix, SAME model")
    print("=" * 74)
    print(f"  {'eta':>5} {'correct':>10} {'generated':>10} {'gap':>8} "
          f"{'none':>10} {'gen - none':>11}")
    for eta in ETAS:
        c = rows["correct"][eta][1]
        g = rows["generated"][eta][1]
        nn = rows["none"].get(eta, (0.0, float("nan"), [], []))[1]
        print(f"  {eta:>5.2f} {c:>10.1%} {g:>10.1%} {c - g:>+8.1%} "
              f"{nn:>10.1%} {g - nn:>+11.1%}")

    print()
    print("  Gap shrinking with eta -> exposure bias was downstream of sampler")
    print("  quality; stage 2 stays deferred. Gap holding or widening -> it is a")
    print("  standing property and stage 2 is the answer.")

    print()
    print("=" * 74)
    print("PER BLOCK — arithmetic truth")
    print("=" * 74)
    header = "  ".join(f"b{b}".rjust(6) for b in range(blocks))
    print(f"  {'arm':>10} {'eta':>5}  {header}")
    for name in ("correct", "generated", "none"):
        for eta in ETAS:
            if eta not in rows[name]:
                continue
            cells = "  ".join(f"{v:>6.1%}" for v in rows[name][eta][2])
            print(f"  {name:>10} {eta:>5.2f}  {cells}")

    print()
    print("=" * 74)
    print("BY OPERAND MAGNITUDE, at eta = 1")
    print("=" * 74)
    print("  The aggregate gap is small while b4 gains +20pp from the correct")
    print("  prefix and b2 loses 11pp. If the correct prefix pushes generation")
    print("  toward the REAL magnitude distribution — which is dominated by large")
    print("  operands the model cannot do — then the teacher-forced arm is simply")
    print("  a harder task, and the small aggregate gap is two effects cancelling")
    print("  rather than the prefix not mattering.")
    print()
    for name in ("correct", "generated", "none"):
        if 1.0 in rows[name]:
            print_magnitude_table(rows[name][1.0][3], f"{name} prefix, eta = 1")


if __name__ == "__main__":
    main()
