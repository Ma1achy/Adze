"""Unconditional probes: the real ceiling, the jitter ablation, and SDE vs ODE.

The unconditional single-block model is the clean setting — no prefix, no
positional structure, no per-block budget split — so the two surviving hypotheses
about why generated latents decode badly are tested here first, where nothing is
confounded.

  ceiling  Decoding real cached latents gives ~97.5% truth, but that measures
           reconstruction of steps the VAE memorised. Anything generated is novel,
           so the honest ceiling is the decode rate on steps never seen.
  jitter   Latent clustering (NN/mean 0.280 vs a 0.744 Gaussian null) is currently
           exactly where live-dimension shrinkage was before it was ablated: a
           measured discrepancy with a plausible mechanism. Shrinkage looked just
           as convincing and turned out to be a symptom, so clustering gets its own
           ablation before it earns any conclusion.
  SDE      A deterministic ODE is a bijection; hitting a spiky multimodal target
           needs a contorted map. Stochastic sampling keeps trajectories in
           regions the model saw. On the §8 not-in-v0 list — measured here to
           inform a promotion decision, and kept out of draft().

Usage:
    python scripts/m4_uncond_probes.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from adze.config import load_config
from adze.data.corrupt import make_pair
from adze.data.dataset import LatentCache, TraceDataset
from adze.data.generate import generate_dataset
from adze.data.tokeniser import CharTokeniser
from adze.eval.checks import unseen_decode_ceiling
from adze.eval.load import load_vae
from adze.model.denoiser import Denoiser
from adze.sample.draft import draft
from adze.sample.stochastic import sample_stochastic
from adze.sample.trajectory import noise_floor
from adze.train.unconditional import decode_rates, train_unconditional

CACHE_DIR = Path("data/cache")
CHECKPOINT_DIR = Path("checkpoints")

# Validity we require the jitter to preserve. Stated up front so sigma is chosen
# against a fixed criterion rather than picked to make a result look good.
JITTER_VALIDITY_FLOOR = 0.95


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--vae", type=Path, default=Path("checkpoints/vae_debug_d16.pt"))
    p.add_argument("--config", type=Path, default=Path("configs/debug.yaml"))
    p.add_argument("--baseline", type=Path,
                   default=Path("checkpoints/uncond_debug_d16_shift1.5.pt"))
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--samples", type=int, default=2000)
    p.add_argument("--nfe", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    config = load_config(args.config)
    device = torch.device(config.device)
    tokeniser = CharTokeniser()

    vae, vae_arch = load_vae(args.vae, device)
    d, k = vae_arch["latent_dim"], vae_arch["latents_per_block"]
    cache = LatentCache(CACHE_DIR / f"latents_{config.name}_d{d}.pt")
    scale = cache.scale
    latents = cache.load().to(device)
    block_mask = cache.load_block_mask().to(device)
    z0 = latents.view(-1, k, d)[block_mask.reshape(-1)]

    print(f"vae           {args.vae}  D={d} K={k}")
    print(f"latent scale  {scale:.4f}")
    print(f"data          {z0.shape[0]} real block latents")
    print()

    # ---- 1. The ceiling -------------------------------------------------------
    print("=" * 74)
    print("CEILING — decode rate on steps the VAE has NEVER SEEN")
    print("=" * 74)
    train_traces = generate_dataset(
        n=60_000, seed=config.data.seed,
        max_depth=config.data.max_depth, operand_max=config.data.operand_max,
    )
    train_texts = {s.render() for t in train_traces for s in t.steps}
    held_traces = generate_dataset(
        n=8_000, seed=config.data.seed + 909_091,
        max_depth=config.data.max_depth, operand_max=config.data.operand_max,
    )
    held_texts = [s.render() for t in held_traces for s in t.steps]
    ceil = unseen_decode_ceiling(vae, tokeniser, train_texts, held_texts)
    floor, floor_true, _ = noise_floor(vae.decoder, tokeniser, (k, d), 2000, device)
    print(f"  held-out steps also seen verbatim in training  {ceil['seen_share']:.1%}")
    print(f"  decode truth, SEEN steps                       {ceil['seen_true']:.1%}")
    print(f"  decode truth, UNSEEN steps ({ceil['n_unseen']} of them)"
          f"{'':>7}{ceil['unseen_true']:.1%}   <- THE CEILING")
    print(f"  decode well-formed, UNSEEN steps               {ceil['unseen_formed']:.1%}")
    print(f"  noise floor                                    {floor_true:.1%} true, "
          f"{floor:.1%} well-formed")
    print()

    # ---- 2. Isotropic noise tolerance, then jitter ---------------------------
    print("=" * 74)
    print("JITTER — decoder tolerance to isotropic noise, then the ablation")
    print("=" * 74)
    print("  Existing curves covered dead-dims-only noise and live-dim scaling.")
    print("  Neither is isotropic additive noise, which is what jitter applies.")
    print()
    probe = z0[:8000]
    print(f"  {'sigma':>7} {'well-formed':>12} {'true':>8}")
    tolerance = []
    for sd in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6):
        pert = probe + torch.randn_like(probe) * sd
        wf, tr = decode_rates(vae.decoder, tokeniser, pert * scale)
        tolerance.append((sd, wf, tr))
        print(f"  {sd:>7.2f} {wf:>12.1%} {tr:>8.1%}")

    base_true = tolerance[0][2]
    ok = [sd for sd, _, tr in tolerance if tr >= JITTER_VALIDITY_FLOOR * base_true]
    sigma = max(ok) if ok else 0.0
    print(f"\n  chosen sigma = {sigma:.2f} — the largest keeping truth within "
          f"{JITTER_VALIDITY_FLOOR:.0%} of unperturbed ({base_true:.1%}).")
    print("  Any lift at this sigma cannot be the jitter destroying content.")

    print("\n  training unconditional model on jittered latents")
    jit_model = train_unconditional(
        z0, k, d, config.model.denoiser.d_model, config.model.denoiser.n_layers,
        config.model.denoiser.n_heads, args.steps, args.batch, args.lr,
        1.5, device, seed=args.seed, jitter=sigma,
    )
    torch.manual_seed(args.seed)
    gen = draft(jit_model, None, 1, k, d, args.nfe,
                device=str(device), batch=args.samples)
    jit_wf, jit_true = decode_rates(vae.decoder, tokeniser, gen.view(-1, k, d) * scale)

    # ---- 3. SDE vs ODE, on the unjittered baseline ---------------------------
    state = torch.load(args.baseline, map_location=device, weights_only=False)
    base = Denoiser(**state["arch"]).to(device)
    base.load_state_dict(state["model"])
    base.eval()

    torch.manual_seed(args.seed)
    ode = draft(base, None, 1, k, d, args.nfe, device=str(device), batch=args.samples)
    ode_wf, ode_true = decode_rates(vae.decoder, tokeniser, ode.view(-1, k, d) * scale)

    print()
    print("=" * 74)
    print("JITTER RESULT")
    print("=" * 74)
    print(f"  {'model':>22} {'well-formed':>12} {'true':>8}")
    print(f"  {'baseline (no jitter)':>22} {ode_wf:>12.1%} {ode_true:>8.1%}")
    print(f"  {f'jittered (sd {sigma:.2f})':>22} {jit_wf:>12.1%} {jit_true:>8.1%}")
    print(f"  {'ceiling':>22} {ceil['unseen_formed']:>12.1%} "
          f"{ceil['unseen_true']:>8.1%}")
    print(f"  {'floor':>22} {floor:>12.1%} {floor_true:>8.1%}")

    print()
    print("=" * 74)
    print("SDE vs ODE — unconditional, on the baseline model")
    print("=" * 74)
    print("  eta = 0 must reproduce the ODE exactly; it is the correctness check.")
    print()
    print(f"  {'nfe':>5} {'eta':>6} {'well-formed':>12} {'true':>8}  note")
    for nfe in (args.nfe, 8):
        for eta in (0.0, 0.25, 0.5, 0.75, 1.0):
            torch.manual_seed(args.seed)
            z = sample_stochastic(base, 1, k, d, nfe, eta=eta,
                                  device=str(device), batch=args.samples)
            wf, tr = decode_rates(vae.decoder, tokeniser, z.view(-1, k, d) * scale)
            note = ""
            if eta == 0.0:
                torch.manual_seed(args.seed)
                z_ode = draft(base, None, 1, k, d, nfe,
                              device=str(device), batch=args.samples)
                delta = (z - z_ode).abs().max().item()
                note = f"identity check vs draft: max|diff| = {delta:.2e}"
            print(f"  {nfe:>5} {eta:>6.2f} {wf:>12.1%} {tr:>8.1%}  {note}")


if __name__ == "__main__":
    main()
