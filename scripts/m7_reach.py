"""DOES THE REACH COMPOUND WITH REFINEMENT PASSES?

Six seeds established the window as a CLIFF: d = 1 at z = 6.06, d = 2 at z = 4.21,
d = 3 and d = 4 null. That is the best-supported number in the project. What it
does not say is whether the cliff belongs to the ARCHITECTURE or to a SINGLE PASS.

The mechanism says per-pass. After one pass, block b has absorbed information from
b+1 and b+2. On the next pass, regenerating b−1 reads a b that already contains
b+2 — effective reach 3, then 4.

    REACH ~ 2R

If that holds, the outer loop is already the answer to the cliff, and its
rationale changes from "more compute" to "more range".

## The design, and why the schedule matters

Information only propagates if b's NEIGHBOURS are regenerated too. Erasing and
regenerating b alone, repeatedly, propagates nothing.

So each pass erases b PLUS a fresh random subset of other real blocks, drawn from
`sample_subset` at the same p = 0.5 the model was trained on, and regenerates all
of them under one mask. b is scored after EVERY pass, so a single 3-pass run
yields R = 1, 2, 3 — R = 2 is a prefix of R = 3 and running them separately would
be the same computation three times over.

The subset AND the noise are redrawn per pass. `regenerate` calls
`torch.manual_seed` at entry, so a fixed seed would give every pass identical
noise; the seed advances by pass.

## The only-b control, and the one thing it cannot do

Erase b alone, R times. Neighbours never update, so nothing can propagate.

**It is flat in R by construction, not by expectation.** Pass r starts from a
state where every block but b is clean and unchanged, and b is immediately
overwritten with fresh noise, so each pass is an i.i.d. draw of the single-block
condition. That makes it a strong harness check — IF IT RISES, THERE IS A BUG —
but it cannot distinguish propagation from compute, because no mechanism exists
by which compute could accumulate in it.

The compute question is answered by the main arm's own structure: b is re-erased
and regenerated on every pass there too, so it receives exactly the same R
independent regenerations as the control. The only difference between the arms is
whether the neighbours updated.

## The outcome to watch for

Global's per-block absolute accuracy is ~5%, so a regenerated neighbour is usually
WRONG. Each pass replaces clean context with mostly-incorrect context, and
degradation may outrun propagation. `preserved` — the rate at which non-target
blocks still match — is reported per pass for exactly this reason. It separates
"reach does not compound" from "reach could not be tested because the context fell
apart".

Usage:
    python scripts/m7_reach.py --denoiser checkpoints/*.pt --passes 3
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
from adze.eval.strata import consumer_distance
from adze.invariants import MaskMode
from adze.train.regime_b import sample_subset

CACHE_DIR = Path("data/cache")

# Pass p advances the noise seed by this much. Any constant works; it is fixed so
# a rerun reproduces, and large enough that consecutive passes cannot collide.
SEED_STRIDE = 10_000


ARMS = ("p50", "one", "only-b")


def pass_erase(target: torch.Tensor, block_mask: torch.Tensor, arm: str,
               generator: torch.Generator | None) -> torch.Tensor:
    """[batch, B] bool — what this pass erases.

    `p50`    — b plus a fresh subset of other real blocks at p = 0.5, the
               distribution regime B was trained on.
    `one`    — b plus exactly ONE other real block.
    `only-b` — the control: b alone.

    ## Why `one` exists, measured rather than anticipated

    The smoke run showed `p50` destroying the context outright: `preserved` falls
    9.5% -> 2.5% -> 0.5% over three passes, because ~2.2 blocks are erased per
    pass and each is regenerated at ~5% accuracy. Propagation cannot be observed
    through a context that is being demolished, so `p50` alone would measure a
    floor rather than answer the question.

    `one` erases the minimum that still permits propagation — a neighbour must
    update for anything to move through it — and so gives the mechanism its
    fairest available test at this model's accuracy.

    b is always included. `regenerate` scores `target` and requires it to be
    inside the erase set, or the scored block was never regenerated.
    """
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    blocks = block_mask.shape[1]
    one_hot = torch.nn.functional.one_hot(target, blocks).bool() & block_mask
    if arm == "only-b":
        return one_hot
    if arm == "p50":
        return one_hot | sample_subset(block_mask, p=0.5, generator=generator)

    # Exactly one other REAL block, uniform among those available. An example
    # with only the target real erases the target alone — its realised size is
    # 1, which the printed mean reports rather than hiding.
    eligible = block_mask & ~one_hot
    priority = torch.rand(block_mask.shape, generator=generator,
                          device=block_mask.device)
    priority[~eligible] = float("inf")
    pick = priority.argmin(dim=1)
    chosen = torch.nn.functional.one_hot(pick, blocks).bool()
    chosen &= priority.gather(1, pick.unsqueeze(1)) < float("inf")
    return one_hot | chosen


@torch.no_grad()
def run_passes(denoiser, latents, block_ids, target, blocks, k, n_passes,
               mode, arm, block_mask, nfe, eta, seed, generator):
    """Regenerate for `n_passes`, yielding the latents after each pass.

    Non-erased blocks are carried through untouched — `regenerate` returns the
    full [batch, N, D] state with them preserved exactly — so the chain is what
    makes propagation possible at all.
    """
    z = latents
    for p in range(n_passes):
        erase = pass_erase(target, block_mask, arm, generator)
        z = regenerate(denoiser, z, block_ids, target, blocks, nfe, mode,
                       eta=eta, seed=seed + p * SEED_STRIDE, erase=erase)
        yield p + 1, z, float(erase.sum(1).float().mean())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--vae", type=Path, default=Path("checkpoints/vae_cap100_d16.pt"))
    p.add_argument("--denoiser", type=Path, nargs="+", required=True)
    p.add_argument("--config", type=Path, default=Path("configs/debug.yaml"))
    p.add_argument("--traces", type=int, default=2000)
    p.add_argument("--passes", type=int, default=3)
    p.add_argument("--nfe", type=int, default=32)
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("runs/reach.json"))
    args = p.parse_args()

    config = load_config(args.config)
    tkw = trace_kwargs(config)
    device = torch.device(config.device)
    tokeniser = CharTokeniser()
    vae, _ = load_vae(args.vae, device)

    # The same held-out pool and seed offset as m7_central.py, so R = 1 here is
    # comparable to the committed dumps.
    traces = generate_dataset(n=args.traces * 2, seed=config.data.seed + 909_091,
                              **tkw)
    pairs = [make_pair(t, rng_seed=i) for i, t in enumerate(traces)
             if len(t.steps) >= 2][: args.traces]

    print(f"vae           {args.vae}")
    print(f"traces        {len(pairs)} held-out corrupted pairs")
    print(f"sampling      nfe {args.nfe}, eta {args.eta}, seed {args.seed} "
          f"+ {SEED_STRIDE}/pass")
    print(f"arms          p50 = b + subset at p=0.5 (the training distribution)")
    print(f"              one = b + exactly one other real block")
    print(f"              only-b = control, flat in R BY CONSTRUCTION — a rise")
    print(f"                       in it is a bug, not a compute effect")
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
        distance = [consumer_distance(q.clean, int(target[i]))
                    for i, q in enumerate(usable)]

        print("=" * 90)
        print(f"{ckpt.name}")
        print("=" * 90)
        print(f"  {'arm':>8} {'mask':>7} {'R':>3} {'|S|':>5} {'RESULT':>8} "
              f"{'preserved':>10} {'formed':>8}")

        for arm in ARMS:
            for mode, mname in ((MaskMode.GLOBAL, "global"),
                                (MaskMode.CAUSAL, "causal")):
                # One generator per (arm, mask) so the subset sequence is
                # identical across masks — the gap must not include a different
                # draw of which neighbours were erased.
                gen = torch.Generator(device=device).manual_seed(args.seed)
                for r, z, size in run_passes(denoiser, latents, block_ids,
                                             target, blocks, k, args.passes,
                                             mode, arm, block_mask, args.nfe,
                                             args.eta, args.seed, gen):
                    decoded = _decode(vae, tokeniser, z, scale, blocks, k)
                    sc = score(mname, decoded, clean_steps, target, n_steps)
                    print(f"  {arm:>8} {mname:>7} {r:>3} {size:>5.2f} "
                          f"{sc.result:>8.2%} {sc.preserved:>10.2%} "
                          f"{sc.well_formed:>8.1%}")
                    for i in range(len(usable)):
                        b = int(target[i])
                        got = _parse(decoded[i][b])
                        want = _parse(clean_steps[i][b])
                        out.append({
                            "ckpt": ckpt.name, "arm": arm, "mask": mname, "R": r,
                            "i": i, "block": b, "distance": distance[i],
                            "n_steps": n_steps[i],
                            "hit": got is not None and want is not None
                                   and got[3] == want[3],
                        })
        print()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"passes": args.passes, "nfe": args.nfe,
                                    "eta": args.eta, "seed": args.seed,
                                    "records": out}))
    print(f"wrote {len(out)} records to {args.out}")
    print("  Analyse with scripts/m7_reach_read.py — between-seed error bars,")
    print("  per the standing rule. A shape read off one checkpoint is not a shape.")


if __name__ == "__main__":
    main()
