"""The COST side of the regime A/B mix sweep.

The mix is being swept because M7's crossing table showed the model is specialised
to whichever configuration it saw most, and regime B fired on only 9.7% of steps.
Raising regime B's share should shrink that handicap. This script measures what it
costs: free-running draft quality, which is the thing 90/10 was chosen to protect.

DiD found a uniform mixture WORSE at drafting than a bimodal 90/10. If drafting
collapses as the share rises, that is the trade, not a bug — and it is the number
that decides where the optimum sits.

**Distribution-matched truth is the headline**, per the standing convention: the
model's generated steps do not land in the same magnitude bins as real data, so a
pooled truth figure is partly a measure of which problems the model chose. Raw
pooled truth is kept and labelled; per-bin truth is reported against per-bin
decoder ceilings.

At fixed total steps, raising the share also CUTS regime-A steps. So each
checkpoint's realised A-step and B-step counts are printed alongside its quality:
a drop at high share could be interference or could be undertrained regime A, and
those cannot be separated without a matched-A control. Nothing here attributes it.

Usage:
    python scripts/m6_draft_quality.py --checkpoints checkpoints/*_mixedP*.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from adze.config import load_config, trace_kwargs
from adze.data.dataset import LatentCache
from adze.data.generate import generate_dataset
from adze.data.tokeniser import CharTokeniser
from adze.eval.checks import unseen_ceiling_by_magnitude
from adze.eval.draft_quality import sample_draft
from adze.eval.load import load_denoiser, load_vae
from adze.eval.readout import print_readout, readout

CACHE_DIR = Path("data/cache")


def recipe(path: Path) -> dict:
    """The training recipe a checkpoint records about itself.

    Older checkpoints record nothing, which is why the reference arm of the sweep
    had to be retrained rather than reused. Missing keys are reported as unknown
    rather than filled with a plausible guess.
    """
    blob = torch.load(path, weights_only=False, map_location="cpu")
    keys = ("regime_b_prob", "regime_b_steps", "steps", "batch_size", "lr", "seed")
    return {k: blob.get(k) for k in keys}


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--vae", type=Path, default=Path("checkpoints/vae_cap100_d16.pt"))
    p.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    p.add_argument("--config", type=Path, default=Path("configs/debug.yaml"))
    p.add_argument("--samples", type=int, default=500)
    p.add_argument("--nfe", type=int, default=32)
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    config = load_config(args.config)
    tkw = trace_kwargs(config)
    device = torch.device(config.device)
    tokeniser = CharTokeniser()
    vae, _ = load_vae(args.vae, device)

    train_texts = {
        s.render() for t in generate_dataset(n=60_000, seed=config.data.seed, **tkw)
        for s in t.steps
    }
    held = [
        s.render() for t in generate_dataset(
            n=8_000, seed=config.data.seed + 909_091, **tkw) for s in t.steps
    ]
    ceiling = unseen_ceiling_by_magnitude(vae, tokeniser, train_texts, held)
    real_unseen = [t for t in held if t not in train_texts]

    print(f"vae           {args.vae}")
    print(f"reference     {len(real_unseen)} unseen held-out steps")
    print(f"sampling      nfe {args.nfe}, eta {args.eta}, seed {args.seed}, "
          f"{args.samples} traces")
    print()

    summary = []
    for ckpt in args.checkpoints:
        denoiser, arch, scale = load_denoiser(ckpt, device)
        if scale is None:
            scale = LatentCache(
                CACHE_DIR / f"latents_{config.name}_d{arch['latent_dim']}.pt").scale

        r = recipe(ckpt)
        texts = sample_draft(denoiser, vae, tokeniser, arch, scale, args.nfe,
                             args.eta, args.samples, device, args.seed)
        out = readout(texts, real_unseen, ceiling)

        b_steps = r["regime_b_steps"]
        total = r["steps"]
        a_steps = None if (b_steps is None or total is None) else total - b_steps
        share = ("unknown" if r["regime_b_prob"] is None
                 else f"{r['regime_b_prob']:.0%}")
        print_readout(out, f"{ckpt.name}   regime B share {share}")
        print(f"  regime A steps  {a_steps if a_steps is not None else 'unknown'}"
              f"   regime B steps {b_steps if b_steps is not None else 'unknown'}"
              f"   total {total if total is not None else 'unknown'}")
        print()
        summary.append((ckpt.name, share, a_steps, b_steps, out))

    print("=" * 96)
    print("DRAFT QUALITY BY MIX — the cost side")
    print("=" * 96)
    print(f"  {'checkpoint':>44} {'share':>7} {'A-steps':>9} {'B-steps':>9} "
          f"{'MATCHED':>9} {'raw':>8} {'formed':>8}")
    for name, share, a_steps, b_steps, out in summary:
        print(f"  {name:>44} {share:>7} "
              f"{(a_steps if a_steps is not None else '?'):>9} "
              f"{(b_steps if b_steps is not None else '?'):>9} "
              f"{out.matched_true:>9.1%} {out.raw_true:>8.1%} "
              f"{out.raw_formed:>8.1%}")
    print()
    print("  MATCHED is the headline. Raw is inflated by difficulty selection and")
    print("  is kept only so the two eras of this project stay comparable.")
    print("  A-steps fall as the share rises at fixed total steps. A drop in draft")
    print("  quality is therefore NOT attributable to the mix without a control")
    print("  that holds A-steps constant.")


if __name__ == "__main__":
    main()
