"""The 74.5% unseen-step ceiling, broken out by operand magnitude.

The pooled ceiling stopped being fine enough to read against once the SDE result
came in by magnitude bin. At eta = 1 the 0-9 bin generates 80.2% true — ABOVE the
pooled ceiling — which is not a contradiction but a sign the pooled number is an
average over bins the decoder handles very differently. Short statements are
easier to reconstruct than long ones, so the ceiling has to be binned the same way
the generations are before "at ceiling" means anything.

Also prints the eta = 0 identity check, so the promoted sampler's reduction to the
ODE is a printed measurement rather than a claim.

Usage:
    python scripts/m4_ceiling_by_magnitude.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from adze.config import load_config, trace_kwargs
from adze.data.dataset import LatentCache
from adze.data.generate import generate_dataset
from adze.data.tokeniser import CharTokeniser
from adze.eval.load import load_denoiser, load_vae
from adze.eval.magnitude import print_magnitude_table
from adze.sample.draft import draft

CACHE_DIR = Path("data/cache")


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--vae", type=Path, default=Path("checkpoints/vae_debug_d16.pt"))
    p.add_argument("--denoiser", type=Path,
                   default=Path("checkpoints/denoiser_debug_d16.pt"))
    p.add_argument("--config", type=Path, default=Path("configs/debug.yaml"))
    p.add_argument("--samples", type=int, default=500)
    p.add_argument("--nfe", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    config = load_config(args.config)

    tkw = trace_kwargs(config)
    device = torch.device(config.device)
    tokeniser = CharTokeniser()
    vae, _ = load_vae(args.vae, device)

    # ---- identity check ------------------------------------------------------
    denoiser, arch, scale = load_denoiser(args.denoiser, device)
    if scale is None:
        scale = LatentCache(
            CACHE_DIR / f"latents_{config.name}_d{arch['latent_dim']}.pt"
        ).scale
    kw = dict(device=str(device), batch=args.samples)
    torch.manual_seed(args.seed)
    a = draft(denoiser, None, arch["blocks"], arch["latents_per_block"],
              arch["latent_dim"], args.nfe, eta=0.0, **kw)
    torch.manual_seed(args.seed)
    b = draft(denoiser, None, arch["blocks"], arch["latents_per_block"],
              arch["latent_dim"], args.nfe, eta=0.0, s_churn=0.0, **kw)
    print("=" * 74)
    print("IDENTITY CHECK — the promoted sampler must reduce to the ODE")
    print("=" * 74)
    print(f"  eta=0 vs eta=0,s_churn=0   max|diff| = {(a - b).abs().max().item():.2e}")
    print("  (the algebraic reduction to euler_step is held by")
    print("   tests/test_m4_stochastic.py; measured drift there is ~1e-6)")
    print()

    # ---- ceiling, per bin ----------------------------------------------------
    train_traces = generate_dataset(
        n=60_000, seed=config.data.seed,
        **tkw,
    )
    train_texts = {s.render() for t in train_traces for s in t.steps}
    held_traces = generate_dataset(
        n=8_000, seed=config.data.seed + 909_091,
        **tkw,
    )
    held_texts = [s.render() for t in held_traces for s in t.steps]
    unseen = [t for t in held_texts if t not in train_texts]

    out: list[str] = []
    for i in range(0, len(unseen), 4096):
        chunk = tokeniser.encode_batch(unseen[i : i + 4096]).to(device)
        mu, _ = vae.encoder(chunk)
        out += [tokeniser.decode(r) for r in vae.decoder(mu).argmax(dim=-1)]

    print("=" * 74)
    print(f"CEILING BY MAGNITUDE — {len(unseen)} held-out steps never seen in training")
    print("=" * 74)
    print("  Binned by the TRUE operands of the source step, so a bin's `true`")
    print("  share is the decoder's round-trip accuracy at that magnitude — the")
    print("  bar a generated step in that bin has to be read against.")
    print()
    # Bin on the source text, not the decode: the decode may be malformed, and the
    # question is what the decoder does to inputs OF a given magnitude.
    from adze.eval.magnitude import BINS, magnitude
    from adze.sample.trajectory import rates

    print(f"  {'magnitude':>12} {'n':>7} {'well-formed':>12} {'true':>8}")
    for lo, hi, name in BINS:
        pairs = [(s, d) for s, d in zip(unseen, out)
                 if (m := magnitude(s)) is not None and lo <= m <= hi]
        if not pairs:
            continue
        wf, tr = rates([d for _, d in pairs])
        print(f"  {name:>12} {len(pairs):>7} {wf:>12.1%} {tr:>8.1%}")
    print()
    print_magnitude_table(out, "the same, binned by the DECODED operands instead")


if __name__ == "__main__":
    main()
