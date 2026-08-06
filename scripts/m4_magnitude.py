"""TASK 4 — the magnitude cliff: capacity or data?

Binning generated steps by `max(|lhs|, |rhs|)` and ignoring block position gives a
7x cliff between the 10-29 and 30-99 bins. That is the largest single effect
measured on this model, and it survived the sampler fix intact — the SDE lifted the
small bins by ~21pp and the large ones by ~2pp.

But every number behind it is on `configs/debug.yaml` at 2 layers x 128 wide.
"The model cannot do arithmetic above 30" and "a 2-layer model cannot" are
different findings, and only one of them is about the design.

  cliff MOVES with depth  -> capacity
  cliff STAYS             -> data design

The data hypothesis, if it stays: `debug.yaml` has `operand_max: 20` while
`MAGNITUDE_CAP` in `adze.data.generate` is 1000. Leaf operands are drawn from
1..20, so values above ~30 only ever appear as *results* — the model meets small
values as inputs and large values only as outputs. `configs/v0.yaml` already uses
`operand_max: 100`, so the cliff may be milder there. That is a recommendation to
measure, not a change to make here.

Binning uses `adze.eval.magnitude`, whose looser regex needs only the operand pair
and not a well-formed result — so a malformed decode still lands in a bin and still
counts against that bin's well-formedness, and the bins do not silently condition
on the thing they are trying to explain. The unbinnable remainder is printed.

Usage:
    python scripts/m4_magnitude.py \
        --denoisers checkpoints/denoiser_debug_d16.pt \
                    checkpoints/denoiser_debug_d16_L4.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from adze.config import load_config
from adze.data.dataset import LatentCache
from adze.data.tokeniser import CharTokeniser
from adze.eval.load import load_denoiser, load_vae
from adze.eval.magnitude import BINS, magnitude, magnitude_table
from adze.sample.draft import DEFAULT_ETA, draft
from adze.sample.trajectory import rates

CACHE_DIR = Path("data/cache")


@torch.no_grad()
def generate(denoiser, vae, tokeniser, arch, scale, nfe, eta, samples, device, seed):
    """Returns (all texts, texts grouped by block)."""
    torch.manual_seed(seed)
    blocks = arch["blocks"]
    latents = draft(
        denoiser, None, blocks, arch["latents_per_block"], arch["latent_dim"], nfe,
        device=str(device), batch=samples, eta=eta,
    )
    per = (latents * scale).view(samples * blocks, arch["latents_per_block"], -1)
    texts = [tokeniser.decode(r) for r in vae.decoder(per).argmax(dim=-1)]
    grid = [texts[i * blocks : (i + 1) * blocks] for i in range(samples)]
    return texts, [[r[b] for r in grid] for b in range(blocks)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--vae", type=Path, default=Path("checkpoints/vae_debug_d16.pt"))
    p.add_argument("--denoisers", type=Path, nargs="+", default=[
        Path("checkpoints/denoiser_debug_d16.pt"),
        Path("checkpoints/denoiser_debug_d16_L4.pt"),
    ])
    p.add_argument("--config", type=Path, default=Path("configs/debug.yaml"))
    p.add_argument("--samples", type=int, default=1500)
    p.add_argument("--nfe", type=int, default=32)
    p.add_argument("--eta", type=float, default=DEFAULT_ETA)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    config = load_config(args.config)
    device = torch.device(config.device)
    tokeniser = CharTokeniser()
    vae, _ = load_vae(args.vae, device)

    print(f"sampling      nfe {args.nfe}, eta {args.eta}, {args.samples} traces")
    print(f"data          operand_max {config.data.operand_max}, "
          f"magnitude cap 1000 (adze.data.generate)")
    print()

    results = {}
    for path in args.denoisers:
        if not path.exists():
            print(f"  MISSING {path} — skipped")
            continue
        denoiser, arch, scale = load_denoiser(path, device)
        if scale is None:
            scale = LatentCache(
                CACHE_DIR / f"latents_{config.name}_d{arch['latent_dim']}.pt"
            ).scale
        texts, by_block = generate(denoiser, vae, tokeniser, arch, scale,
                                   args.nfe, args.eta, args.samples, device, args.seed)
        label = f"{arch['n_layers']}L x {arch['d_model']}w"
        results[label] = (texts, by_block)
        wf, tr = rates(texts)
        print(f"  {label:<12} {path.name:<36} wf {wf:>6.1%}  true {tr:>6.1%}",
              flush=True)

    if not results:
        raise SystemExit("no checkpoints found")

    print()
    print("=" * 78)
    print("THE CLIFF — arithmetic truth by operand magnitude")
    print("=" * 78)
    labels = list(results)
    head = "  ".join(f"{lab:>18}" for lab in labels)
    print(f"  {'magnitude':>12} {head}")
    tables = {lab: dict((r[0], r) for r in magnitude_table(results[lab][0])[0])
              for lab in labels}
    for _, _, name in BINS:
        cells = []
        for lab in labels:
            row = tables[lab].get(name)
            cells.append("—".rjust(18) if row is None
                         else f"{row[3]:>8.1%} (n={row[1]:>5})")
        print(f"  {name:>12} " + "  ".join(cells))

    print()
    print("=" * 78)
    print("WELL-FORMEDNESS BY MAGNITUDE — is the binning biased?")
    print("=" * 78)
    print("  The bins are built from a looser regex than `classify`, so a decode")
    print("  with garbage on the right of the `=` still lands in a bin. If")
    print("  well-formedness were much lower in the large bins, the truth cliff")
    print("  would be partly an artifact of what fails to parse.")
    print()
    print(f"  {'magnitude':>12} {head}")
    for _, _, name in BINS:
        cells = []
        for lab in labels:
            row = tables[lab].get(name)
            cells.append("—".rjust(18) if row is None else f"{row[2]:>18.1%}")
        print(f"  {name:>12} " + "  ".join(cells))
    for lab in labels:
        _, unbinnable = magnitude_table(results[lab][0])
        share = unbinnable / len(results[lab][0])
        print(f"  {lab:>12} unbinnable remainder {unbinnable} ({share:.1%})")

    print()
    print("=" * 78)
    print("MAGNITUDE DISTRIBUTION — what the model chooses to generate")
    print("=" * 78)
    print("  Compared against the real data's shares. A model that avoids large")
    print("  operands inflates its own pooled truth figure.")
    print()
    print(f"  {'magnitude':>12} {head}")
    for _, _, name in BINS:
        cells = []
        for lab in labels:
            row = tables[lab].get(name)
            total = len(results[lab][0])
            cells.append("—".rjust(18) if row is None
                         else f"{row[1] / total:>18.1%}")
        print(f"  {name:>12} " + "  ".join(cells))

    print()
    print("=" * 78)
    print("BLOCK vs MAGNITUDE — which one is doing the work")
    print("=" * 78)
    for lab in labels:
        _, by_block = results[lab]
        print(f"\n  {lab}")
        print(f"    {'block':>6} {'n':>6} {'true':>7} {'median |operand|':>18}")
        for b, col in enumerate(by_block):
            mags = [m for t in col if (m := magnitude(t)) is not None]
            med = sorted(mags)[len(mags) // 2] if mags else float("nan")
            print(f"    {b:>6} {len(col):>6} {rates(col)[1]:>7.1%} {med:>18}")

    print()
    print("  Cliff moving with depth -> capacity. Cliff staying -> data design,")
    print("  and the fix is matching the leaf-operand range to the result range")
    print("  rather than only raising operand_max.")


if __name__ == "__main__":
    main()
