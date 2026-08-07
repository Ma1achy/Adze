"""M7 — why the null is not null.

The leak check said: corrupt the ROOT, where nothing consumes it, on FULL-LENGTH
traces, where there is no padding. Causal and global give the erased block exactly
the same input positions there, so the gap must be zero.

Measured: RESULT gap **-5.6pp**, global-only 10 vs causal-only 38, chi2 = 15.19.
Not zero, and negative.

The harness tests rule out the obvious causes — a stub denoiser returns
bit-identical output under both masks at the same seed, so there is no second
noise draw, and the causal mask provably cannot attend forward. So this asks the
third question:

    Under GLOBAL, every CONTEXT block attends to the erased block, which holds
    pure noise. Under CAUSAL, only blocks after the target do — and the root has
    none. So global's clean context is computed with the erasure's noise mixed in
    and causal's is not.

That is a difference between the arms with nothing to do with downstream evidence,
and it contaminates every M7 number, not just this one.

Three arms, same traces, same erasure seed:

    causal            build_mask(CAUSAL)
    global            build_mask(GLOBAL)
    global-shielded   GLOBAL, but nothing outside the target attends TO it

If the negative gap is noise pollution, shielding removes it. If it survives
shielding, the cause is something else and the M7 headline is not readable.

Usage:
    python scripts/m7_shield.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from adze.config import load_config, trace_kwargs
from adze.data.corrupt import corrupt_final
from adze.data.dataset import LatentCache, TraceDataset
from adze.data.generate import generate_dataset
from adze.data.tokeniser import CharTokeniser
from adze.eval.central import (
    _decode,
    _parse,
    encode_traces,
    regenerate,
    revision_mask,
    score,
    shielded_mask,
)
from adze.eval.load import load_denoiser, load_vae
from adze.invariants import MaskMode

CACHE_DIR = Path("data/cache")


def _mcnemar(a, b, clean_steps, target_block) -> tuple[int, int, float | None]:
    """Discordant pairs on RESULT between two conditions, and chi2."""
    only_b = only_a = 0
    for i in range(len(clean_steps)):
        blk = int(target_block[i])
        want = _parse(clean_steps[i][blk])
        pa, pb = _parse(a.texts[i][blk]), _parse(b.texts[i][blk])
        a_ok = pa is not None and want is not None and pa[3] == want[3]
        b_ok = pb is not None and want is not None and pb[3] == want[3]
        only_b += b_ok and not a_ok
        only_a += a_ok and not b_ok
    disc = only_a + only_b
    chi = None if disc == 0 else (abs(only_b - only_a) - 1) ** 2 / disc
    return only_a, only_b, chi


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

    # Root corruption on FULL-LENGTH traces only: the target block is then the
    # same index for every example, which is what lets one [N, N] shielded mask
    # stand in for a per-example one.
    pool, pairs = args.traces * 8, []
    while len(pairs) < args.traces and pool <= 400_000:
        traces = generate_dataset(n=pool, seed=config.data.seed + 909_091, **tkw)
        pairs = [corrupt_final(t, rng_seed=i) for i, t in enumerate(traces)
                 if len(t.steps) == blocks][: args.traces]
        pool *= 4
    if not pairs:
        raise SystemExit("no full-length pairs")

    ds = TraceDataset(pairs, blocks=blocks, latents_per_block=k, use_corrupted=True)
    items = [ds[i] for i in range(len(ds))]
    tokens = torch.stack([it["tokens"] for it in items]).to(device)
    block_mask = torch.stack([it["block_mask"] for it in items]).to(device)
    latents = encode_traces(vae, tokens, block_mask, scale)
    target_block = torch.stack([it["corrupted_idx"] for it in items]).to(device)

    target = blocks - 1
    assert (target_block == target).all(), "shielded mask needs a constant target"
    assert block_mask.all(), "full-length traces should have no padding"

    n_steps = [len(p.clean.steps) for p in pairs]
    clean_steps = [[s.render() for s in p.clean.steps] for p in pairs]
    block_ids = torch.repeat_interleave(torch.arange(blocks), k).to(device)

    print(f"denoiser      {args.denoiser}  {arch['n_layers']}L x {arch['d_model']}w")
    print(f"traces        {len(pairs)} full-length, ROOT corrupted, B={blocks}")
    print(f"sampling      nfe {args.nfe}, eta {args.eta}, erasure seed {args.seed}")
    print("the null      nothing consumes the root and there is no padding, so")
    print("              causal and global give the erased block identical inputs")
    print()

    causal_m = revision_mask(block_ids, 0, MaskMode.CAUSAL)
    global_m = revision_mask(block_ids, 0, MaskMode.GLOBAL)

    # `mode` is not only the mask — it is also embedded into the conditioning
    # (`Denoiser.mode_embed`), so the two arms differ in TWO ways at once. Crossing
    # them separates the attention pattern from the conditioning vector. If the
    # handicap follows the conditioning, it is the mode embedding; if it follows
    # the mask, it is what the model learned to do under that attention pattern —
    # and regime B fired on only 9.7% of training steps, so global is the less
    # trained of the two.
    arms = {
        "causal": (MaskMode.CAUSAL, causal_m),
        "global": (MaskMode.GLOBAL, global_m),
        "global-shielded": (MaskMode.GLOBAL,
                            shielded_mask(block_ids, target, MaskMode.GLOBAL)),
        "causal-cond/global-mask": (MaskMode.CAUSAL, global_m),
        "global-cond/causal-mask": (MaskMode.GLOBAL, causal_m),
    }

    results = {}
    for name, (mode, mask) in arms.items():
        out = regenerate(denoiser, latents, block_ids, target_block, blocks,
                         args.nfe, mode, eta=args.eta, seed=args.seed, mask=mask)
        decoded = _decode(vae, tokeniser, out, scale, blocks, k)
        results[name] = score(name, decoded, clean_steps, target_block, n_steps)

    print(f"  {'arm':>17} {'exact':>8} {'RESULT':>8} {'operands':>9} {'formed':>8}")
    for name, r in results.items():
        print(f"  {name:>17} {r.repaired:>8.1%} {r.result:>8.1%} "
              f"{r.operands:>9.1%} {r.well_formed:>8.1%}")

    print()
    print("=" * 78)
    for name in [a for a in arms if a != "causal"]:
        gap = results[name].result - results["causal"].result
        only_c, only_g, chi = _mcnemar(results["causal"], results[name],
                                       clean_steps, target_block)
        verdict = ("NOT significant" if chi is None or chi <= 3.84 else "p < 0.05")
        print(f"  {name:>17} - causal   RESULT {gap:>+7.1%}   "
              f"only-{name[:3]} {only_g:>3}  only-causal {only_c:>3}  "
              f"chi2 {('-' if chi is None else f'{chi:.2f}'):>6}  {verdict}")
    print("=" * 78)
    print("  Shielded gap ~ 0  -> the null is a null once the erased block stops")
    print("     polluting its own context. The plain-mask gap is contaminated and")
    print("     every M7 number needs re-reading under the shielded mask.")
    print("  Shielded gap still negative -> noise pollution is not the cause and")
    print("     the headline is not readable until we know what is.")


if __name__ == "__main__":
    main()
