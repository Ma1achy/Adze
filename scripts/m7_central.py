"""M7 — THE CENTRAL EXPERIMENT.

> Given the corrupted block's location, does global regeneration repair it more
> reliably than causal regeneration?

Three conditions on the same corrupted traces and the same checkpoint. Only the
attention mask differs between `causal` and `global`; the traces, the corrupted
indices, the erasure noise and the schedule are all held identical, so the
comparison is PAIRED and the large cross-run variance this project has measured
cancels out.

Oracle block selection: the corrupted index is known and erased. Every number here
is therefore an UPPER BOUND on what uncertainty-steered selection could achieve,
and is labelled as one.

Nothing is snapped, filtered, retried or cleaned. A regenerated block that decodes
to garbage counts as garbage.

Usage:
    python scripts/m7_central.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from adze.config import load_config, trace_kwargs
from adze.data.corrupt import make_pair
from adze.data.dataset import LatentCache, TraceDataset
from adze.data.generate import generate_dataset
from adze.data.tokeniser import CharTokeniser
from adze.eval.central import _decode, encode_traces, regenerate, score
from adze.eval.load import load_denoiser, load_vae
from adze.invariants import MaskMode

CACHE_DIR = Path("data/cache")


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--vae", type=Path, default=Path("checkpoints/vae_cap100_d16.pt"))
    p.add_argument("--denoiser", type=Path,
                   default=Path("checkpoints/denoiser_cap100_d16_L4_mixed.pt"))
    p.add_argument("--config", type=Path, default=Path("configs/debug.yaml"))
    p.add_argument("--traces", type=int, default=500)
    p.add_argument("--nfe", type=int, default=32)
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    config = load_config(args.config)
    tkw = trace_kwargs(config)
    device = torch.device(config.device)
    tokeniser = CharTokeniser()
    vae, _ = load_vae(args.vae, device)
    denoiser, arch, scale = load_denoiser(args.denoiser, device)
    blocks, k, d = arch["blocks"], arch["latents_per_block"], arch["latent_dim"]
    if scale is None:
        scale = LatentCache(CACHE_DIR / f"latents_{config.name}_d{d}.pt").scale

    # HELD-OUT traces, a seed disjoint from training's.
    traces = generate_dataset(n=args.traces * 2, seed=config.data.seed + 909_091, **tkw)
    pairs = [make_pair(t, rng_seed=i) for i, t in enumerate(traces)
             if len(t.steps) >= 2]
    pairs = [p for p in pairs if len(p.clean.steps) <= blocks][: args.traces]
    if not pairs:
        raise SystemExit("no usable pairs")

    # Encode the CORRUPTED traces — that is the state the revision pass starts from.
    corrupt_ds = TraceDataset(pairs, blocks=blocks, latents_per_block=k,
                              use_corrupted=True)
    items = [corrupt_ds[i] for i in range(len(corrupt_ds))]
    tokens = torch.stack([it["tokens"] for it in items]).to(device)
    block_mask = torch.stack([it["block_mask"] for it in items]).to(device)
    latents = encode_traces(vae, tokens, block_mask, scale)

    # The dataset's own record of which block was corrupted, not a recomputation —
    # a second source for the same fact is a second thing that can disagree.
    target_block = torch.stack([it["corrupted_idx"] for it in items]).to(device)
    n_steps = [len(p.clean.steps) for p in pairs]
    clean_steps = [
        [s.render() for s in p.clean.steps] + ["<pad>"] * (blocks - len(p.clean.steps))
        for p in pairs
    ]
    assert len(items) == len(pairs)

    print(f"denoiser      {args.denoiser}  {arch['n_layers']}L x {arch['d_model']}w")
    print(f"vae           {args.vae}")
    print(f"traces        {len(pairs)} held-out corrupted pairs, B={blocks}")
    print(f"sampling      nfe {args.nfe}, eta {args.eta}, erasure seed {args.seed}")
    print(f"selection     ORACLE — the corrupted index is known. Upper bound.")
    print()

    results = {}

    # --- none: the corrupted trace, untouched --------------------------------
    decoded = _decode(vae, tokeniser, latents, scale, blocks, k)
    results["none"] = score("none", decoded, clean_steps, target_block, n_steps)

    # --- causal / global: identical but for the mask -------------------------
    for name, mode in (("causal", MaskMode.CAUSAL), ("global", MaskMode.GLOBAL)):
        out = regenerate(
            denoiser, latents, torch.repeat_interleave(
                torch.arange(blocks), k).to(device),
            target_block, blocks, args.nfe, mode,
            eta=args.eta, seed=args.seed,
        )
        decoded = _decode(vae, tokeniser, out, scale, blocks, k)
        results[name] = score(name, decoded, clean_steps, target_block, n_steps)

    print("=" * 78)
    print("THE CENTRAL EXPERIMENT")
    print("=" * 78)
    print(f"  {'condition':>10} {'exact':>8} {'RESULT':>8} {'operands':>9} "
          f"{'formed':>8} {'answer':>8} {'preserved':>10}")
    for name in ("none", "causal", "global"):
        r = results[name]
        print(f"  {name:>10} {r.repaired:>8.1%} {r.result:>8.1%} "
              f"{r.operands:>9.1%} {r.well_formed:>8.1%} {r.answer:>8.1%} "
              f"{r.preserved:>10.1%}")

    gap = results["global"].repaired - results["causal"].repaired
    gap_result = results["global"].result - results["causal"].result
    gap_ops = results["global"].operands - results["causal"].operands
    print()
    print("=" * 78)
    print(f"  GAP (global - causal)   exact {gap:+.1%}   "
          f"RESULT {gap_result:+.1%}   operands {gap_ops:+.1%}")
    print("=" * 78)
    print("  RESULT is the arm the experiment's logic points at: the corruption")
    print("  changes a result, downstream steps consume the original value, so")
    print("  downstream context identifies the RESULT and nothing else. Operands")
    print("  are fixed by the question, and question conditioning (M5) is not")
    print("  built — so a positive RESULT gap with a flat operands gap is exactly")
    print("  the signature the claim predicts.")
    print("  Positive -> regenerating with downstream context in view repairs")
    print("  the corrupted block more reliably than regenerating causally. That")
    print("  is the claim the whole repo was built to test.")
    print()
    print("  Read as an UPPER BOUND: block selection is oracle here.")
    print("  `none` is the floor — the corrupted block, unrevised.")
    print()

    # McNemar on the paired RESULT outcomes. The comparison is paired by
    # construction, so the discordant pairs are the whole of the evidence —
    # traces where both conditions succeed or both fail carry none.
    from adze.eval.central import _parse
    only_g = only_c = 0
    for i in range(len(pairs)):
        b = int(target_block[i])
        want = _parse(clean_steps[i][b])
        gc = _parse(results["causal"].texts[i][b])
        gg = _parse(results["global"].texts[i][b])
        c_ok = gc is not None and want is not None and gc[3] == want[3]
        g_ok = gg is not None and want is not None and gg[3] == want[3]
        only_g += g_ok and not c_ok
        only_c += c_ok and not g_ok
    n_disc = only_g + only_c
    print(f"PAIRED (McNemar) on RESULT: global-only {only_g}, causal-only {only_c},")
    if n_disc:
        chi = (abs(only_g - only_c) - 1) ** 2 / n_disc
        print(f"  discordant {n_disc}, chi2(1) with continuity correction = {chi:.2f}")
        print(f"  {'p < 0.05' if chi > 3.84 else 'NOT significant at 0.05'}")
    else:
        print("  no discordant pairs — no evidence either way")
    print()

    # A few paired examples, raw, so the numbers have something behind them.
    print("EXAMPLES — same trace, same erasure, only the mask differs")
    print("-" * 78)
    shown = 0
    for i in range(len(pairs)):
        b = int(target_block[i])
        c, g = results["causal"].texts[i][b], results["global"].texts[i][b]
        if c == g or shown >= 6:
            continue
        want = clean_steps[i][b]
        mark_c = "OK " if c == want else "   "
        mark_g = "OK " if g == want else "   "
        print(f"  block {b}  want {want!r}")
        print(f"    {mark_c}causal  {c!r}")
        print(f"    {mark_g}global  {g!r}")
        shown += 1


if __name__ == "__main__":
    main()
