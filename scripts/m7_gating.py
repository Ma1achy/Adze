"""THE GATING TEST — is the mode embedding suppressing a capability the network has?

## The question

M7 measured that the two arms exploit disjoint sources: causal's accuracy climbs
with prefix length and ignores the pin; global's is set by whether a pin exists
and is FLAT in prefix length. The natural reading was that the global pathway
cannot use the prefix.

But "cannot" and "does not" are different claims, and the standard erasure
condition cannot separate them. When ONE block is erased, causal sees the prefix
only and global sees the prefix PLUS the pin. The two arms face different
information, so a difference between them is confounded with what they were shown.

**Suffix erasure removes the confound.** Erase `b .. end`, and everything after
`b` is noise for both arms. Causal sees the clean prefix; global sees the clean
prefix plus noise, which carries nothing. The two conditions are now
INFORMATIONALLY IDENTICAL and differ only in the mode flag and the mask.

    they CONVERGE                  -> the prefix skill is shared. Whatever the
                                      mode partition was doing, it is not
                                      withholding this.
    global flat, causal fine,      -> the mode embedding is GATING a capability
    ON THE SAME INPUTS                the network demonstrably has. Suppression,
                                      not absence.

The capability is not in doubt: the same weights reach 7.8% in causal mode at
long prefix. Suppression is therefore a live outcome, not a fallback explanation.

## Why both erasure shapes are run

The `single` row is the standard M7 condition and is the reference. The `suffix`
row is the test. Reading `suffix` without `single` beside it would leave the
absolute rates uninterpretable — suffix erasure also destroys far more context,
which depresses both arms, and the claim is about the DIFFERENCE between them.

Scoring is on block `b` only, in both rows, so the two are directly comparable.

Usage:
    python scripts/m7_gating.py --denoiser checkpoints/a.pt checkpoints/b.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from adze.config import load_config, trace_kwargs
from adze.data.corrupt import make_pair
from adze.data.dataset import LatentCache, TraceDataset
from adze.data.generate import generate_dataset
from adze.data.tokeniser import CharTokeniser
from adze.eval.central import _decode, _parse, encode_traces, regenerate, score
from adze.eval.load import load_denoiser, load_vae
from adze.invariants import MaskMode

CACHE_DIR = Path("data/cache")


def build_suffix_erase(target: torch.Tensor, block_mask: torch.Tensor) -> torch.Tensor:
    """[batch, B] bool — every REAL block from `target` onwards.

    Padding is never erased: asking the model to reconstruct a constant from noise
    would count as refinement skill. The target is always included, which
    `regenerate` requires — it scores `target` and would otherwise be scoring a
    block that was never regenerated.
    """
    blocks = block_mask.shape[1]
    idx = torch.arange(blocks, device=block_mask.device).unsqueeze(0)
    return (idx >= target.unsqueeze(1)) & block_mask


def build_single_erase(target: torch.Tensor, block_mask: torch.Tensor) -> torch.Tensor:
    """[batch, B] bool — the target block alone. The standard M7 condition."""
    blocks = block_mask.shape[1]
    return torch.nn.functional.one_hot(target, blocks).bool() & block_mask


SHAPES = {"single": build_single_erase, "suffix": build_suffix_erase}


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--vae", type=Path, default=Path("checkpoints/vae_cap100_d16.pt"))
    p.add_argument("--denoiser", type=Path, nargs="+", required=True)
    p.add_argument("--config", type=Path, default=Path("configs/debug.yaml"))
    p.add_argument("--traces", type=int, default=2000)
    p.add_argument("--nfe", type=int, default=32)
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=Path("runs"))
    args = p.parse_args()

    config = load_config(args.config)
    tkw = trace_kwargs(config)
    device = torch.device(config.device)
    tokeniser = CharTokeniser()
    vae, _ = load_vae(args.vae, device)

    # The same held-out pool and the same seed offset as m7_central.py, so the
    # traces here are the traces there.
    traces = generate_dataset(n=args.traces * 2, seed=config.data.seed + 909_091,
                              **tkw)
    pairs = [make_pair(t, rng_seed=i) for i, t in enumerate(traces)
             if len(t.steps) >= 2][: args.traces]

    print(f"vae           {args.vae}")
    print(f"traces        {len(pairs)} held-out corrupted pairs")
    print(f"sampling      nfe {args.nfe}, eta {args.eta}, erasure seed {args.seed}")
    print(f"scoring       block b only, in both shapes, so the rows compare")
    print(f"the point     under `suffix` the two arms see IDENTICAL information —")
    print(f"              clean prefix, noise after. Any gap is the mode alone.")
    print()

    for ckpt in args.denoiser:
        denoiser, arch, scale = load_denoiser(ckpt, device)
        blocks, k, d = arch["blocks"], arch["latents_per_block"], arch["latent_dim"]
        if scale is None:
            scale = LatentCache(CACHE_DIR / f"latents_{config.name}_d{d}.pt").scale
        blob = torch.load(ckpt, weights_only=False, map_location="cpu")

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

        print("=" * 88)
        print(f"{ckpt.name}   B share "
              f"{blob.get('regime_b_prob')}   structure {blob.get('b_structure')}")
        print("=" * 88)
        print(f"  {'shape':>8} {'|S|':>5} {'causal':>8} {'global':>8} {'gap':>8} "
              f"{'g formed':>9} {'c formed':>9}")

        records = {}
        for shape, build in SHAPES.items():
            erase = build(target, block_mask)
            row = {}
            for name, mode in (("causal", MaskMode.CAUSAL),
                               ("global", MaskMode.GLOBAL)):
                out = regenerate(denoiser, latents, block_ids, target, blocks,
                                 args.nfe, mode, eta=args.eta, seed=args.seed,
                                 erase=erase)
                decoded = _decode(vae, tokeniser, out, scale, blocks, k)
                row[name] = score(name, decoded, clean_steps, target, n_steps)
            gap = row["global"].result - row["causal"].result
            print(f"  {shape:>8} {erase.sum(1).float().mean():>5.2f} "
                  f"{row['causal'].result:>8.1%} {row['global'].result:>8.1%} "
                  f"{gap:>+8.1%} {row['global'].well_formed:>9.1%} "
                  f"{row['causal'].well_formed:>9.1%}")
            records[shape] = row

        # Per-prefix-length dump. The aggregate can hide the shape that matters:
        # the claim under test is about whether the PREFIX is used, so the rate
        # by how much prefix there is, is the readout.
        out_path = args.out_dir / f"gating_{ckpt.stem}.json"
        rows = []
        for i in range(len(usable)):
            b = int(target[i])
            rec = {"block": b, "n_steps": n_steps[i],
                   "clean_result": (lambda q: None if q is None else str(q[3]))(
                       _parse(clean_steps[i][b]))}
            for shape in SHAPES:
                for name in ("causal", "global"):
                    got = _parse(records[shape][name].texts[i][b])
                    rec[f"{shape}_{name}"] = None if got is None else str(got[3])
            rows.append(rec)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(
            {"denoiser": str(ckpt), "nfe": args.nfe, "eta": args.eta,
             "seed": args.seed, "records": rows}, indent=1))
        print(f"  wrote {len(rows)} records to {out_path}")

        print(f"  {'prefix':>8} {'n':>6}"
              + "".join(f" {s + ' ' + a:>16}" for s in SHAPES
                        for a in ("causal", "global")))
        for b in sorted({r["block"] for r in rows}):
            sub = [r for r in rows if r["block"] == b]
            if len(sub) < 100:
                continue
            cells = []
            for shape in SHAPES:
                for name in ("causal", "global"):
                    hit = sum(r[f"{shape}_{name}"] is not None
                              and r[f"{shape}_{name}"] == r["clean_result"]
                              for r in sub)
                    cells.append(f"{hit / len(sub):>16.1%}")
            print(f"  {b:>8} {len(sub):>6}" + "".join(cells))
        print()

    print("  Chance on RESULT is 0.6-0.7% (permutation null, adze.eval.strata).")
    print("  `single` is the reference condition; `suffix` is the test. Suffix")
    print("  erasure destroys far more context, so both arms fall — the claim is")
    print("  the GAP between them, where the information is now equalised.")


if __name__ == "__main__":
    main()
