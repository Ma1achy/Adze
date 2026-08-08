"""Does the refine pathway want the subset size it was TRAINED on?

The mix pilot found something the mix cannot explain. At p = 0.50, every arm
carrying GLOBAL CONDITIONING scores exactly 0.0% while the global MASK under
causal conditioning IMPROVED (1.6% -> 2.4%). Draft quality is intact (23.9%
matched, 98.9% well-formed), so the checkpoint is not degenerate. More regime-B
training made the refine-mode pathway worse while making the wider receptive
field better.

THE HYPOTHESIS. Regime B trains on SUBSETS. `sample_subset` draws each real block
at p = 0.5, so |S| ~ Binomial(n_real, 0.5): mean 2.24, and only 30.5% of steps at
|S| = 1. M7 erases EXACTLY ONE block. So `mode = REFINE` means "several blocks are
gone" in training and "one block is gone" at evaluation.

The within-mode distribution is IDENTICAL at both mixes — the mix share does not
touch `sample_subset` — so p = 0.50 simply sees five times more of a distribution
that is 69.5% multi-block. More exposure to |S| >= 2 predicts more specialisation
to |S| >= 2, and worse performance at |S| = 1.

THE TEST. Corrupt one block as usual. Erase that block PLUS k−1 random other real
blocks, so |S| matches training. Score ONLY the corrupted block. Sweep |S| = 1..4.

  strong at |S| ~ 2-3, zero at |S| = 1   -> confirmed. The fix is matching
                                            training's rho to inference's
                                            selection, not the mix.
  bad at every |S|                       -> the checkpoint is genuinely
                                            degenerate; interference is back.
  p = 0.10 flat across |S|                -> it never specialised. That arm is
                                            the control for this test.

Extra erased blocks are drawn from REAL blocks only — erasing padding would ask
the model to reconstruct a constant from noise and count that as refinement.

Nothing is snapped, filtered, retried or cleaned. A block that decodes to garbage
counts as garbage.

Usage:
    python scripts/m6_subset_size.py --denoiser checkpoints/..._mixedP50.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from adze.config import load_config, trace_kwargs
from adze.data.corrupt import make_pair
from adze.data.dataset import LatentCache, TraceDataset
from adze.data.build import make_dataset
from adze.data.tokeniser import CharTokeniser
from adze.eval.central import _decode, _parse, encode_traces, regenerate, score
from adze.eval.load import load_denoiser, load_vae
from adze.invariants import MaskMode

CACHE_DIR = Path("data/cache")


def build_erase(target: torch.Tensor, block_mask: torch.Tensor, size: int,
                seed: int) -> torch.Tensor:
    """[batch, B] bool — the target plus (size − 1) other REAL blocks.

    An example with fewer real blocks than `size` erases all of them; its actual
    |S| is reported rather than silently counted as `size`.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    batch, blocks = block_mask.shape
    erase = torch.nn.functional.one_hot(target.cpu(), blocks).bool()
    eligible = block_mask.cpu() & ~erase
    # Random priority per block, +inf where ineligible, then take the lowest.
    priority = torch.rand(batch, blocks, generator=g)
    priority[~eligible] = float("inf")
    order = priority.argsort(dim=1)
    for j in range(size - 1):
        pick = order[:, j]
        ok = priority.gather(1, pick.unsqueeze(1)).squeeze(1) < float("inf")
        erase[ok, pick[ok]] = True
    return erase.to(block_mask.device)


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--vae", type=Path, default=Path("checkpoints/vae_cap100_d16.pt"))
    p.add_argument("--denoiser", type=Path, nargs="+", required=True)
    p.add_argument("--config", type=Path, default=Path("configs/debug.yaml"))
    p.add_argument("--traces", type=int, default=1000)
    p.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--nfe", type=int, default=32)
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    config = load_config(args.config)
    tkw = trace_kwargs(config)
    device = torch.device(config.device)
    tokeniser = CharTokeniser()
    vae, _ = load_vae(args.vae, device)

    traces = make_dataset(config, args.traces * 2, config.data.seed + 909_091)
    pairs = [make_pair(t, rng_seed=i) for i, t in enumerate(traces)
             if len(t.steps) >= 2][: args.traces]

    print(f"vae           {args.vae}")
    print(f"traces        {len(pairs)} held-out corrupted pairs")
    print(f"sampling      nfe {args.nfe}, eta {args.eta}, erasure seed {args.seed}")
    print(f"training      |S| ~ Binomial(n_real, 0.5): mean 2.24, "
          f"P(|S|=1) = 30.5%")
    print(f"scoring       ONLY the corrupted block, whatever else was erased")
    print()

    for ckpt in args.denoiser:
        denoiser, arch, scale = load_denoiser(ckpt, device)
        blocks, k, d = arch["blocks"], arch["latents_per_block"], arch["latent_dim"]
        if scale is None:
            scale = LatentCache(CACHE_DIR / f"latents_{config.name}_d{d}.pt").scale
        blob = torch.load(ckpt, weights_only=False, map_location="cpu")
        share = blob.get("regime_b_prob")

        usable = [q for q in pairs if len(q.clean.steps) <= blocks]
        ds = TraceDataset(usable, blocks=blocks, latents_per_block=k,
                          use_corrupted=True)
        items = [ds[i] for i in range(len(ds))]
        tokens = torch.stack([it["tokens"] for it in items]).to(device)
        block_mask = torch.stack([it["block_mask"] for it in items]).to(device)
        latents = encode_traces(vae, tokens, block_mask, scale)
        target = torch.stack([it["corrupted_idx"] for it in items]).to(device)
        n_steps = [len(q.clean.steps) for q in usable]
        clean_steps = [[s.render() for s in q.clean.steps]
                       + ["<pad>"] * (blocks - len(q.clean.steps)) for q in usable]
        block_ids = torch.repeat_interleave(torch.arange(blocks), k).to(device)

        print("=" * 86)
        print(f"{ckpt.name}   regime B share "
              f"{'unknown' if share is None else f'{share:.0%}'}")
        print("=" * 86)
        print(f"  {'|S| asked':>10} {'|S| real':>9} {'causal RESULT':>14} "
              f"{'global RESULT':>14} {'gap':>8} {'g formed':>9}")

        for size in args.sizes:
            erase = build_erase(target, block_mask, size, seed=args.seed + size)
            realised = erase.sum(1).float().mean().item()
            row = {}
            for name, mode in (("causal", MaskMode.CAUSAL),
                               ("global", MaskMode.GLOBAL)):
                out = regenerate(denoiser, latents, block_ids, target, blocks,
                                 args.nfe, mode, eta=args.eta, seed=args.seed,
                                 erase=erase)
                decoded = _decode(vae, tokeniser, out, scale, blocks, k)
                row[name] = score(name, decoded, clean_steps, target, n_steps)
            gap = row["global"].result - row["causal"].result
            print(f"  {size:>10} {realised:>9.2f} {row['causal'].result:>14.1%} "
                  f"{row['global'].result:>14.1%} {gap:>+8.1%} "
                  f"{row['global'].well_formed:>9.1%}")
        print()

    print("  Chance on RESULT is 0.6-0.7% (permutation null, adze.eval.strata).")
    print("  Only the CORRUPTED block is scored at every |S|, so the rows are")
    print("  comparable — what changes is how much context was erased alongside it.")
    print("  Note the confound: larger |S| also removes more CLEAN context, which")
    print("  should HURT. A rise with |S| therefore cannot be a context effect.")


if __name__ == "__main__":
    main()
