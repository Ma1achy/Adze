"""TASK 6 — re-read the OLD results under distribution-matched truth.

Every aggregate figure recorded before session 12 was raw pooled truth, which is
inflated whenever the model picks its own difficulty. This script re-reads the key
ones against the readout that replaced it, on the ORIGINAL data and the ORIGINAL
checkpoints — which is why `name: debug` artefacts were kept rather than
overwritten when the config was renamed.

It answers one question: which session-11 conclusions survive the correction, and
which were artifacts of the pooled convention?

The old data is regenerated with `leaf_values=None` — the uniform sampler — NOT
with the config's current setting. Reading old checkpoints against the new data
distribution would compare a model to a task it never saw.

Usage:
    python scripts/m4_reread.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from adze.config import load_config
from adze.data.dataset import LatentCache
from adze.data.generate import generate_dataset
from adze.data.tokeniser import CharTokeniser
from adze.eval.checks import unseen_ceiling_by_magnitude
from adze.eval.load import load_denoiser, load_vae
from adze.eval.readout import print_readout, readout
from adze.sample.draft import draft

CACHE_DIR = Path("data/cache")

# The session-11 headline configurations, all on the ORIGINAL uniform-leaf data.
ARMS = [
    ("2L eta=0  (the M4 headline, 29.5%)", "checkpoints/denoiser_debug_d16.pt", 0.0, {}),
    ("2L eta=1  (session 11 headline, 46.3%)", "checkpoints/denoiser_debug_d16.pt", 1.0, {}),
    ("4L eta=1  (the capacity arm, 51.2%)", "checkpoints/denoiser_debug_d16_L4.pt", 1.0, {}),
    ("zero-prefix eta=1  (the 53.9%)", "checkpoints/denoiser_debug_d16_zeroprefix.pt",
     1.0, {"zero_prefix": True}),
]


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--vae", type=Path, default=Path("checkpoints/vae_debug_d16.pt"))
    p.add_argument("--config", type=Path, default=Path("configs/debug.yaml"))
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument("--nfe", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    config = load_config(args.config)
    device = torch.device(config.device)
    tokeniser = CharTokeniser()
    vae, _ = load_vae(args.vae, device)

    # ORIGINAL data: uniform leaves, explicitly. Not config.leaf_distribution.
    train_texts = {
        s.render() for t in generate_dataset(
            n=60_000, seed=config.data.seed, max_depth=config.data.max_depth,
            operand_max=config.data.operand_max, leaf_values=None,
            magnitude_cap=1000) for s in t.steps
    }
    held = [
        s.render() for t in generate_dataset(
            n=8_000, seed=config.data.seed + 909_091,
            max_depth=config.data.max_depth,
            operand_max=config.data.operand_max, leaf_values=None,
            magnitude_cap=1000) for s in t.steps
    ]
    ceiling = unseen_ceiling_by_magnitude(vae, tokeniser, train_texts, held)
    real_unseen = [t for t in held if t not in train_texts]

    print("RE-READING SESSION 11 UNDER DISTRIBUTION-MATCHED TRUTH")
    print(f"  original data, uniform leaves 1..{config.data.operand_max}")
    print(f"  {len(real_unseen)} unseen held-out steps set the weights and ceiling")
    print(f"  nfe {args.nfe}, {args.samples} traces per arm")
    print()

    rows = []
    for label, path, eta, kw in ARMS:
        if not Path(path).exists():
            print(f"  MISSING {path} — skipped")
            continue
        denoiser, arch, scale = load_denoiser(Path(path), device)
        if scale is None:
            scale = LatentCache(
                CACHE_DIR / f"latents_debug_d{arch['latent_dim']}.pt"
            ).scale
        torch.manual_seed(args.seed)
        latents = draft(
            denoiser, None, arch["blocks"], arch["latents_per_block"],
            arch["latent_dim"], args.nfe, device=str(device), batch=args.samples,
            eta=eta, **kw,
        )
        per = (latents * scale).view(
            args.samples * arch["blocks"], arch["latents_per_block"], -1
        )
        texts = [tokeniser.decode(r) for r in vae.decoder(per).argmax(dim=-1)]
        r = readout(texts, real_unseen, ceiling)
        rows.append((label, r))
        print(f"  {label:<42} raw {r.raw_true:>6.1%}   matched {r.matched_true:>6.1%}",
              flush=True)

    print()
    print("=" * 78)
    print("THE CORRECTION")
    print("=" * 78)
    print(f"  {'arm':<42} {'RAW':>8} {'MATCHED':>9} {'inflation':>11}")
    for label, r in rows:
        factor = r.raw_true / r.matched_true if r.matched_true > 0 else float("inf")
        print(f"  {label:<42} {r.raw_true:>8.1%} {r.matched_true:>9.1%} "
              f"{factor:>10.1f}x")
    print()
    print("  An ORDERING that survives the correction is a real conclusion.")
    print("  One that inverts was an artifact of difficulty selection.")
    print()
    for label, r in rows:
        print_readout(r, label)


if __name__ == "__main__":
    main()
