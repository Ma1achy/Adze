"""Sampler diagnosis — sweep the ODE budget and the schedule shape.

M4 landed at 85.6% well-formed against an 82.6% floor and 6.6% true against a
6.7% floor: truth at chance. Real latents decode at ~96%, so if the sampler were
landing on the manifold well-formedness would be near 96%, not 85.6%. The failure
is "the sampler is not reaching the manifold", which is a different fault from
"reaches it with wrong content".

Two candidate causes, swept as two axes against the SAME checkpoint:

  nfe    — 8 Euler steps is coarse (LaDiR uses 30, SD3-family 28-50). Truncation
           error compounds and walks the trajectory off-manifold.
  shift  — training draws t from a logit-normal with shift 1.5, so the denoiser is
           best trained mid-schedule and worst at both ends. A uniformly-spaced
           sampler spends equal budget where the model is unequally reliable, and
           accumulates error exactly where the model is weakest. SD3 shifts the
           inference schedule for this reason.

Nothing here retrains, filters, retries or snaps. What is measured is what the
model produced.

Usage:
    python scripts/m4_nfe_sweep.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from adze.config import load_config
from adze.data.dataset import LatentCache
from adze.data.tokeniser import CharTokeniser
from adze.eval.load import load_denoiser, load_vae
from adze.sample.draft import draft
from adze.sample.trajectory import TrajectoryRecorder, noise_floor, rates

CACHE_DIR = Path("data/cache")

NFE_VALUES = [8, 16, 32, 50, 100]
SHIFTS = [1.0, 1.5]

# Run at both stochasticities. They ask different questions: eta = 0 re-asks
# whether truncation error matters, on a model that — unlike the one this sweep
# originally ran against — can actually use its prefix. eta = 1 asks whether the
# promoted stochastic sampler still wants more budget.
ETAS = [0.0, 1.0]


@torch.no_grad()
def sample_and_score(
    denoiser, decoder, tokeniser, arch, scale, nfe, shift, samples, device, seed,
    eta=0.0,
):
    """Draft `samples` traces at (nfe, shift) and classify every block's decode.

    The seed is reset per cell so the initial noise is identical across the sweep —
    otherwise a difference between two cells could just be a lucky draw.

    Returns:
        (well-formed, true, per-block [(wf, true)], mean relative shell departure)
    """
    torch.manual_seed(seed)
    blocks = arch["blocks"]

    # One recorder on a single-sample run, purely for the shell statistic. The
    # bulk run below is batched and does not record.
    probe = TrajectoryRecorder(decoder, tokeniser, arch["latents_per_block"], scale)
    draft(
        denoiser, None, blocks, arch["latents_per_block"], arch["latent_dim"], nfe,
        device=str(device), batch=1, recorder=probe, shift=shift, eta=eta,
    )
    shell = probe.shell()
    # The departure that matters is at the END of each block's window, where
    # truncation error has finished accumulating and the latent is handed to the
    # decoder. Averaging the whole window would dilute it with the easy high-t part.
    tail = [abs(obs - exp) / exp for _, _, t, obs, exp in shell if t < 0.25]
    shell_err = sum(tail) / len(tail) if tail else float("nan")

    torch.manual_seed(seed)
    latents = draft(
        denoiser, None, blocks, arch["latents_per_block"], arch["latent_dim"], nfe,
        device=str(device), batch=samples, shift=shift, eta=eta,
    )
    per_block_latents = (latents * scale).view(
        samples * blocks, arch["latents_per_block"], -1
    )
    texts = [tokeniser.decode(row) for row in decoder(per_block_latents).argmax(dim=-1)]

    formed, true = rates(texts)
    grid = [texts[i * blocks : (i + 1) * blocks] for i in range(samples)]
    per_block = [rates([row[b] for row in grid]) for b in range(blocks)]
    return formed, true, per_block, shell_err


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--denoiser", type=Path, default=Path("checkpoints/denoiser_debug_d16.pt"))
    p.add_argument("--vae", type=Path, default=Path("checkpoints/vae_debug_d16.pt"))
    p.add_argument("--config", type=Path, default=Path("configs/debug.yaml"))
    p.add_argument("--samples", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    config = load_config(args.config)
    device = torch.device(config.device)
    tokeniser = CharTokeniser()

    vae, _ = load_vae(args.vae, device)
    denoiser, arch, scale = load_denoiser(args.denoiser, device)
    if scale is None:
        scale = LatentCache(
            CACHE_DIR / f"latents_{config.name}_d{arch['latent_dim']}.pt"
        ).scale

    blocks = arch["blocks"]
    print(f"denoiser      {args.denoiser}  {arch['n_layers']}L x {arch['d_model']}w")
    print(f"vae           {args.vae}  D={arch['latent_dim']}")
    print(f"latent scale  {scale:.4f}")
    print(f"sweep         eta {ETAS} x nfe {NFE_VALUES} x shift {SHIFTS}, "
          f"{args.samples} traces x B={blocks} blocks per cell")
    print()

    # Measured once: the floor is a property of the decoder and does not move with
    # nfe or with the schedule.
    floor, floor_true, examples = noise_floor(
        vae.decoder, tokeniser,
        (arch["latents_per_block"], arch["latent_dim"]), 1000, device,
    )
    print("=" * 78)
    print("NOISE FLOOR — random Gaussian latents through this decoder")
    print("=" * 78)
    print(f"  well-formed          {floor:7.1%}")
    print(f"  arithmetically true  {floor_true:7.1%}   <- the bar that matters")
    for text in examples:
        print(f"    e.g. {text!r}")
    print()

    rows = []
    print("=" * 78)
    print("SWEEP")
    print("=" * 78)
    print(f"  {'nfe':>4} {'shift':>6} {'well-formed':>12} {'d floor':>9} "
          f"{'true':>8} {'d floor':>9} {'shell err':>10} {'secs':>7}")
    for eta in ETAS:
        print(f"  --- eta = {eta:.1f} " + "-" * 55)
        for shift in SHIFTS:
            for nfe in NFE_VALUES:
                t0 = time.perf_counter()
                formed, true, per_block, shell_err = sample_and_score(
                    denoiser, vae.decoder, tokeniser, arch, scale, nfe, shift,
                    args.samples, device, args.seed, eta=eta,
                )
                secs = time.perf_counter() - t0
                rows.append((eta, nfe, shift, formed, true, per_block, shell_err))
                # `shell err` is only meaningful at eta = 0: the stochastic step
                # deliberately moves the point off the noise shell, so a departure
                # under eta > 0 measures the sampler doing its job, not error.
                shell = f"{shell_err:>10.3f}" if eta == 0.0 else f"{'n/a':>10}"
                print(f"  {nfe:>4} {shift:>6.1f} {formed:>12.1%} "
                      f"{formed - floor:>+9.1%} {true:>8.1%} "
                      f"{true - floor_true:>+9.1%} {shell} {secs:>7.1f}", flush=True)

    print()
    print("=" * 78)
    print("PER-BLOCK ARITHMETIC TRUTH")
    print("=" * 78)
    header = "  ".join(f"b{b}".rjust(6) for b in range(blocks))
    print(f"  {'eta':>4} {'nfe':>4} {'shift':>6}  {header}")
    for eta, nfe, shift, _, _, per_block, _ in rows:
        cells = "  ".join(f"{tr:>6.1%}" for _, tr in per_block)
        print(f"  {eta:>4.1f} {nfe:>4} {shift:>6.1f}  {cells}")

    print()
    print(f"  Read the `true` column against the {floor_true:.1%} floor, not against zero.")
    print("  Truth climbing with nfe means the fault was integration error.")
    print("  Flat across both schedules exonerates integration entirely.")
    print("  Shifted beating uniform at LOWER nfe is train/inference schedule")
    print("  mismatch rather than budget — a different and more useful finding.")
    print("  `shell err` is mean |RMS - sqrt((1-t)^2+t^2)| / expected over t < 0.25:")
    print("  how far off the noise shell the trajectory ends up. It should fall as")
    print("  nfe rises if truncation error is what is walking the sampler off-manifold.")


if __name__ == "__main__":
    main()
