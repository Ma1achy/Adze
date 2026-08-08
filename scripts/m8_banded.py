"""§8 — does the refine mask need to be global, or will a band do?

The refine pass attends over the whole sequence at O(N^2). If the advantage comes
only from nearby blocks, a band buys the same effect far cheaper, and hierarchy
drops from load-bearing to optional in the endpoint design.

## What this test asks changed on 8 Aug 2026

It was written to check whether `w = 2` matches full global, on the reading that
the effect lived inside a two-block window. **That window turned out to be
provenance composition, not distance.** Within a fixed provenance class the
advantage survives at every distance the data can measure — one-leaf at d = 4 is
+2.59% (z = 3.00), the cell that reads null when pooled.

So the prediction has flipped. A band is now expected to COST something, and the
question is how much, and where. If `w = 2` still matches global, that is a
genuine surprise and worth more than the efficiency it buys.

## The distribution, not the mean

"Contributes nothing on average" is not "never contributes". A band that matches
global on 95% of records and loses badly on 5% is a long-range tail — a different
conclusion, and one with different consequences for code and prose. So every
width is compared to full global **on the same records**, reporting

    agree-right / agree-wrong / banded-only / global-only

with the global-only cases broken out by distance and by provenance. Those are the
records where the band cost something, and what they have in common is the result.

Usage:
    python scripts/m8_banded.py --denoiser checkpoints/*.pt --widths 1 2 3 5
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
from adze.eval.strata import consumer_distance, operand_provenance
from adze.invariants import MaskMode
from adze.model.masks import banded_mask

CACHE_DIR = Path("data/cache")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--vae", type=Path, default=Path("checkpoints/vae_cap100_d16.pt"))
    p.add_argument("--denoiser", type=Path, nargs="+", required=True)
    p.add_argument("--config", type=Path, default=Path("configs/debug.yaml"))
    p.add_argument("--traces", type=int, default=2000)
    p.add_argument("--widths", type=int, nargs="+", default=[1, 2, 3, 5])
    p.add_argument("--nfe", type=int, default=32)
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("runs/banded.json"))
    args = p.parse_args()

    config = load_config(args.config)
    tkw = trace_kwargs(config)
    device = torch.device(config.device)
    tokeniser = CharTokeniser()
    vae, _ = load_vae(args.vae, device)

    traces = generate_dataset(n=args.traces * 2, seed=config.data.seed + 909_091,
                              **tkw)
    pairs = [make_pair(t, rng_seed=i) for i, t in enumerate(traces)
             if len(t.steps) >= 2][: args.traces]

    print(f"vae           {args.vae}")
    print(f"traces        {len(pairs)} held-out corrupted pairs")
    print(f"sampling      nfe {args.nfe}, eta {args.eta}, erasure seed {args.seed}")
    print(f"conditioning  REFINE throughout — only the visible mask changes, so")
    print(f"              the arms differ in reach and in nothing else")
    print()

    out: list[dict] = []
    for ckpt in args.denoiser:
        denoiser, arch, scale = load_denoiser(ckpt, device)
        blocks, k, d = arch["blocks"], arch["latents_per_block"], arch["latent_dim"]
        if scale is None:
            scale = LatentCache(CACHE_DIR / f"latents_{config.name}_d{d}.pt").scale

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
        meta = [{"distance": consumer_distance(q.clean, int(target[i])),
                 "provenance": operand_provenance(q.clean.steps[int(target[i])]),
                 "block": int(target[i])}
                for i, q in enumerate(usable)]

        def run(mode, mask):
            z = regenerate(denoiser, latents, block_ids, target, blocks,
                           args.nfe, mode, eta=args.eta, seed=args.seed, mask=mask)
            dec = _decode(vae, tokeniser, z, scale, blocks, k)
            sc = score("x", dec, clean_steps, target, n_steps)
            hits = []
            for i in range(len(usable)):
                b = int(target[i])
                got, want = _parse(dec[i][b]), _parse(clean_steps[i][b])
                hits.append(got is not None and want is not None and got[3] == want[3])
            return sc, hits

        causal, c_hits = run(MaskMode.CAUSAL, None)
        full, g_hits = run(MaskMode.GLOBAL, None)

        print("=" * 88)
        print(f"{ckpt.name}   {arch['n_layers']}L")
        print("=" * 88)
        print(f"  {'width':>7} {'global':>8} {'gap':>8} {'vs full':>9} "
              f"{'agree':>7} {'band-only':>10} {'FULL-only':>10}")
        print(f"  {'full':>7} {full.result:>8.2%} "
              f"{full.result - causal.result:>+8.2%} {'—':>9} "
              f"{'—':>7} {'—':>10} {'—':>10}")

        for w in args.widths:
            if w >= blocks:
                continue
            sc, hits = run(MaskMode.GLOBAL, banded_mask(block_ids, w))
            band_only = sum(h and not g for h, g in zip(hits, g_hits))
            full_only = sum(g and not h for h, g in zip(hits, g_hits))
            agree = sum(h == g for h, g in zip(hits, g_hits))
            print(f"  {w:>7} {sc.result:>8.2%} "
                  f"{sc.result - causal.result:>+8.2%} "
                  f"{sc.result - full.result:>+9.2%} {agree / len(hits):>7.1%} "
                  f"{band_only:>10} {full_only:>10}")
            for i in range(len(usable)):
                out.append({"ckpt": ckpt.name, "width": w, "i": i,
                            "band": hits[i], "full": g_hits[i],
                            "causal": c_hits[i], **meta[i]})
        print(f"  causal {causal.result:>8.2%}")
        print()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"widths": args.widths, "nfe": args.nfe,
                                    "eta": args.eta, "records": out}))
    print(f"wrote {len(out)} records to {args.out}")


if __name__ == "__main__":
    main()
