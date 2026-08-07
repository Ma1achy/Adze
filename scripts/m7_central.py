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
import json
from pathlib import Path

import torch

from adze.config import load_config, trace_kwargs
from adze.data.corrupt import corrupt_consistent, corrupt_final, make_pair
from adze.eval.strata import consumer_distance, operand_provenance
from adze.data.dataset import LatentCache, TraceDataset
from adze.data.generate import generate_dataset
from adze.data.tokeniser import CharTokeniser
from adze.eval.central import _decode, _parse, encode_traces, regenerate, score
from adze.eval.load import load_denoiser, load_vae
from adze.invariants import MaskMode

CACHE_DIR = Path("data/cache")

CONSTRUCTORS = {
    "early": make_pair,              # the committed condition
    "final": corrupt_final,          # the null: root, no consumer, no pin
    "consistent": corrupt_consistent,  # the pin, moved
}


def select(pairs, blocks: int, want: int, full_length_only: bool,
           require_provenance: str | None) -> list:
    """Rejection-sample the eval set.

    Thin strata are fixed by targeting them, not by scaling every run. Eval-trace
    GENERATION is CPU-only and free; the GPU cost is per *sampled* trace. Scaling
    500 -> 2000 takes the both-from-earlier cell from 22 to ~88 — still marginal
    for McNemar on a small effect — at 16x the GPU. A targeted pool buys 500 in
    that cell for the price of 500.

    A targeted pool's rates are NOT comparable to the unbiased ones and are never
    pooled with them. The dump records which selector produced it.
    """
    kept = []
    for pair in pairs:
        n = len(pair.clean.steps)
        if n > blocks:
            continue
        if full_length_only and n != blocks:
            continue
        if require_provenance is not None:
            step = pair.clean.steps[pair.block_index]
            if operand_provenance(step) != require_provenance:
                continue
        kept.append(pair)
        if len(kept) >= want:
            break
    return kept


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
    p.add_argument("--corrupt", choices=sorted(CONSTRUCTORS), default="early")
    p.add_argument("--spread-index", action="store_true",
                   help="corrupt a UNIFORM index rather than preferring early. A "
                        "pin still exists, but prefix length now ranges over 0..B-2 "
                        "instead of 0..2 — which is what makes a slope in prefix "
                        "length measurable rather than resting on three cells")
    p.add_argument("--full-length-only", action="store_true",
                   help="keep only n_steps == B — pad-free, no softmax confound")
    p.add_argument("--require-provenance", choices=["both-leaves", "one-leaf",
                                                    "both-from-earlier"])
    p.add_argument("--out", type=Path, help="per-trace records, for m7_strata.py")
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

    # HELD-OUT traces, a seed disjoint from training's. The pool grows until the
    # selector has filled its quota — trace indices are stable as it grows, since
    # `generate_dataset` seeds each trace from `seed * 1_000_003 + i`, so an
    # unfiltered run reproduces the committed numbers exactly.
    build = CONSTRUCTORS[args.corrupt]
    pool, pairs = args.traces * 2, []
    while True:
        traces = generate_dataset(n=pool, seed=config.data.seed + 909_091, **tkw)
        kw = {"prefer_early": False} if (args.spread_index
                                         and args.corrupt == "early") else {}
        candidates = [build(t, rng_seed=i, **kw) for i, t in enumerate(traces)
                      if len(t.steps) >= 2]
        pairs = select(candidates, blocks, args.traces,
                       args.full_length_only, args.require_provenance)
        if len(pairs) >= args.traces or pool >= 400_000:
            break
        pool *= 4
    if not pairs:
        raise SystemExit("no usable pairs")
    if len(pairs) < args.traces:
        print(f"WARNING: selector filled only {len(pairs)}/{args.traces} "
              f"from a pool of {pool} traces")

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
    def _render(trace):
        return ([s.render() for s in trace.steps]
                + ["<pad>"] * (blocks - len(trace.steps)))

    clean_steps = [_render(p.clean) for p in pairs]
    # Under `consistent` the corrupted trace is a SECOND, different target: the
    # pin has been moved to it. Under the other conditions it differs from clean
    # only at the target block, and scoring against it is meaningless by design.
    corrupt_steps = [_render(p.corrupted) for p in pairs]
    assert len(items) == len(pairs)

    selector = ("full-length-only " if args.full_length_only else "") + (
        "spread-index " if args.spread_index else "") + (
        args.require_provenance or "")
    print(f"denoiser      {args.denoiser}  {arch['n_layers']}L x {arch['d_model']}w")
    print(f"vae           {args.vae}")
    print(f"condition     --corrupt {args.corrupt}"
          + (f"   [TARGETED: {selector.strip()}]" if selector.strip() else ""))
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
    print()
    print("  `none` IS NOT THE BASELINE FOR THESE ARMS. It measures the VAE")
    print("  round-trip, not no-revision: the VAE was trained on valid arithmetic")
    print("  only, so it projects an off-distribution corrupted step onto the")
    print("  nearest valid one and silently repairs 19.6% of corruptions. The")
    print("  regeneration arms erase the block and discard that free repair.")
    print("  The reference is CHANCE — see scripts/m7_strata.py, which scores")
    print("  every rate against a permutation null.")
    print()

    # McNemar on the paired RESULT outcomes. The comparison is paired by
    # construction, so the discordant pairs are the whole of the evidence —
    # traces where both conditions succeed or both fail carry none.
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

    # --- the records dump ----------------------------------------------------
    # Every stratification is an analysis-side cut over this file. Re-cutting the
    # data must not cost another GPU run, and scoring against a second target
    # (the redirected pin) must not either.
    if args.out is not None:
        records = []
        for i, pair in enumerate(pairs):
            b = int(target_block[i])
            rec = {
                "block": b,
                "n_steps": n_steps[i],
                "full_length": n_steps[i] == blocks,
                "distance": consumer_distance(pair.clean, b),
                "provenance": operand_provenance(pair.clean.steps[b]),
                "clean_text": clean_steps[i][b],
                "corrupt_text": corrupt_steps[i][b],
            }
            for target, text in (("clean", clean_steps[i][b]),
                                 ("corrupt", corrupt_steps[i][b])):
                parsed = _parse(text)
                rec[f"{target}_result"] = None if parsed is None else str(parsed[3])
                rec[f"{target}_operands"] = (
                    None if parsed is None else f"{parsed[0]} {parsed[1]} {parsed[2]}")
                rec[f"{target}_exact"] = text
            for name in ("none", "causal", "global"):
                got = results[name].texts[i][b]
                parsed = _parse(got)
                rec[name] = {
                    "text": got,
                    "exact": got,
                    "result": None if parsed is None else str(parsed[3]),
                    "operands": (None if parsed is None
                                 else f"{parsed[0]} {parsed[1]} {parsed[2]}"),
                }
            records.append(rec)

        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "corrupt": args.corrupt,
            "targeted": selector.strip() or None,
            "spread_index": args.spread_index,
            "denoiser": str(args.denoiser),
            "vae": str(args.vae),
            "blocks": blocks,
            "nfe": args.nfe,
            "eta": args.eta,
            "seed": args.seed,
            "records": records,
        }, indent=1))
        print(f"wrote {len(records)} records to {args.out}\n")

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
