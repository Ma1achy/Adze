"""Does the sampler reproduce the MARGINAL distribution of valid steps?

The training data contains only true arithmetic steps, so the marginal over
blocks is a distribution over valid steps. A correctly trained generative model
sampling that marginal produces valid steps whether or not it can predict WHICH
one. Low prefix information explains why the model cannot know block 3 should be
'11 + 20 = 31'. It cannot explain '11 + 20 = 5'. Real block latents decode valid
at ~96%, generated ones at 6% — so the marginal is not being reproduced, and that
is a defect in the diffusion rather than an absence of conditioning.

MSE on velocity gives E[v | z_t, t], which IS the marginal velocity field.
Integrating it transports noise onto the data DISTRIBUTION, not onto its mean —
otherwise unconditional diffusion could not work. So an under-normed endpoint is
not an expected consequence of the loss; it is evidence the learned field is
wrong.

Sections:
  1  per-dimension variance, generated vs real. Separates uniform shrinkage from
     a few collapsed dimensions — different faults, and a scalar shell statistic
     averages them, potentially cancelling two errors of opposite sign.
  1b two ablations on REAL latents at the magnitudes section 1 measures. Section
     1 shows that generated latents differ; only these show whether the
     difference is what breaks decoding.
  1c is the target distribution reachable by a flow at all? kl_beta is 1e-3, so
     nothing pressures the aggregate posterior into shape. Clustered latent means
     would force the flow to transport an isotropic Gaussian onto a spiky
     multimodal target.
  2  per-dimension variance along the trajectory, against the true interpolant's
     (1-t)^2 * sigma_d^2 + t^2. Localises in t where the distributions separate.

Nothing here filters, snaps or retries. What is measured is what the model
produced.

Usage:
    python scripts/m4_marginal.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from adze.config import load_config
from adze.data.dataset import LatentCache
from adze.data.tokeniser import CharTokeniser
from adze.eval.load import load_denoiser, load_vae
from adze.pad import real_positions
from adze.sample.draft import draft
from adze.sample.trajectory import noise_floor, rates

CACHE_DIR = Path("data/cache")

# Fraction of cumulative variance defining the "live" dimensions. Stated here
# rather than tuned per run; build_cache.py uses the same 99% cut.
LIVE_CUT = 0.99


def live_dimensions(var: torch.Tensor) -> torch.Tensor:
    """Bool mask over dims inside the LIVE_CUT cumulative-variance cut."""
    order = var.argsort(descending=True)
    cum = var[order].cumsum(0) / var.sum()
    n_live = int((cum < LIVE_CUT).sum()) + 1
    mask = torch.zeros_like(var, dtype=torch.bool)
    mask[order[:n_live]] = True
    return mask


@torch.no_grad()
def decode_rates(decoder, tokeniser, latents, k, batch=4096):
    """(well-formed, true) for [M, K, D] latents, decoded in the UNSCALED space."""
    texts = []
    for i in range(0, latents.shape[0], batch):
        chunk = latents[i : i + batch]
        texts += [tokeniser.decode(row) for row in decoder(chunk).argmax(dim=-1)]
    return rates(texts)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--denoiser", type=Path, default=Path("checkpoints/denoiser_debug_d16.pt"))
    p.add_argument("--vae", type=Path, default=Path("checkpoints/vae_debug_d16.pt"))
    p.add_argument("--config", type=Path, default=Path("configs/debug.yaml"))
    p.add_argument("--samples", type=int, default=400)
    p.add_argument("--nfe", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    config = load_config(args.config)
    device = torch.device(config.device)
    tokeniser = CharTokeniser()
    torch.manual_seed(args.seed)

    vae, _ = load_vae(args.vae, device)
    denoiser, arch, scale = load_denoiser(args.denoiser, device)
    blocks, k, d = arch["blocks"], arch["latents_per_block"], arch["latent_dim"]
    cache = LatentCache(CACHE_DIR / f"latents_{config.name}_d{d}.pt")
    if scale is None:
        scale = cache.scale

    print(f"denoiser      {args.denoiser}  {arch['n_layers']}L x {arch['d_model']}w")
    print(f"vae           {args.vae}  D={d}  K={k}  B={blocks}")
    print(f"latent scale  {scale:.4f}")
    print()

    # ---- Real block latents, the reference distribution -----------------------
    all_latents = cache.load().to(device)                    # scaled
    block_mask = cache.load_block_mask().to(device)
    real_pos = real_positions(block_mask, k).squeeze(-1)     # [n, N]
    real_blocks = all_latents.view(-1, k, d)[
        block_mask.reshape(-1)
    ]                                                        # [M, K, D] scaled
    real_flat = all_latents[real_pos]                        # [P, D] scaled

    # ---- Generated block latents ---------------------------------------------
    torch.manual_seed(args.seed)
    # eta=0 pinned: this script's recorded numbers predate the eta=1 default.
    gen = draft(denoiser, None, blocks, k, d, args.nfe, eta=0.0,
                device=str(device), batch=args.samples)
    gen_blocks = gen.reshape(-1, k, d)
    gen_flat = gen.reshape(-1, d)

    real_var = real_flat.var(0, unbiased=False)
    gen_var = gen_flat.var(0, unbiased=False)
    live = live_dimensions(real_var)

    print("=" * 78)
    print("1 — PER-DIMENSION VARIANCE, generated vs real")
    print("=" * 78)
    print(f"  {'dim':>4} {'real var':>10} {'gen var':>10} {'ratio':>8}  live")
    for i in real_var.argsort(descending=True).tolist():
        flag = "yes" if live[i] else " no"
        print(f"  {i:>4} {real_var[i]:>10.4f} {gen_var[i]:>10.4f} "
              f"{(gen_var[i] / real_var[i]):>8.3f}  {flag}")

    ratio = gen_var / real_var
    live_ratio = ratio[live].mean().item()
    dead_ratio = ratio[~live].mean().item() if (~live).any() else float("nan")
    print()
    print(f"  live dims          {int(live.sum())} of {d}")
    print(f"  mean ratio, live   {live_ratio:.3f}")
    print(f"  mean ratio, dead   {dead_ratio:.3f}")
    print(f"  mean ratio, all    {ratio.mean().item():.3f}")
    print(f"  total variance     real {real_var.sum():.3f}  gen {gen_var.sum():.3f}"
          f"  ({gen_var.sum() / real_var.sum():.3f}x)")
    print()
    print("  Opposite-signed live and dead ratios would mean the scalar shell")
    print("  statistic was averaging two errors that partly cancel.")

    # ---- Decode rates for both ------------------------------------------------
    floor, floor_true, _ = noise_floor(vae.decoder, tokeniser, (k, d), 2000, device)
    real_wf, real_true = decode_rates(vae.decoder, tokeniser, real_blocks[:8000] * scale, k)
    gen_wf, gen_true = decode_rates(vae.decoder, tokeniser, gen_blocks * scale, k)
    print()
    print("=" * 78)
    print("DECODE RATES — the gap this session exists to explain")
    print("=" * 78)
    print(f"  real latents         well-formed {real_wf:6.1%}   true {real_true:6.1%}"
          f"   <- the ceiling")
    print(f"  generated latents    well-formed {gen_wf:6.1%}   true {gen_true:6.1%}")
    print(f"  random latents       well-formed {floor:6.1%}   true {floor_true:6.1%}"
          f"   <- the floor")

    # ---- 1b: ablations on REAL latents ---------------------------------------
    # Section 1 shows generated latents DIFFER. These show whether the difference
    # is what breaks decoding. Sweeping a range with the measured value marked
    # beats a single yes/no: the question is how steep the cliff is and which side
    # of it the sampler sits on.
    print()
    print("=" * 78)
    print("1b — ABLATIONS ON REAL LATENTS, at the magnitudes section 1 measured")
    print("=" * 78)

    # Each sweep decodes the pool once per magnitude, so the pool is capped. 8000
    # blocks puts the standard error on a rate near 6% at ~0.3pp, far below any
    # difference worth reading — the full 250k set would cost 30 minutes to
    # measure the same thing.
    abl = real_blocks[:8000]

    print("\n  dead-dimension energy: real latents + noise in dead dims only")
    print(f"  {'noise sd':>9} {'well-formed':>12} {'true':>8}")
    dead_sd_measured = (gen_var[~live].mean().sqrt().item() if (~live).any() else 0.0)
    for sd in sorted({0.0, 0.1, 0.25, 0.5, 1.0, 2.0, round(dead_sd_measured, 3)}):
        pert = abl.clone()
        noise = torch.randn_like(pert) * sd
        pert[..., ~live] += noise[..., ~live]
        wf, tr = decode_rates(vae.decoder, tokeniser, pert * scale, k)
        mark = "  <- sampler's measured dead-dim sd" if abs(
            sd - dead_sd_measured) < 1e-6 else ""
        print(f"  {sd:>9.3f} {wf:>12.1%} {tr:>8.1%}{mark}")

    print("\n  live-dimension shrinkage: real latents with live dims scaled")
    print(f"  {'scale':>9} {'well-formed':>12} {'true':>8}")
    live_scale_measured = live_ratio**0.5     # variance ratio -> amplitude ratio
    for s in sorted({1.0, 0.9, 0.8, 0.7, 0.5, 0.3, round(live_scale_measured, 3)}):
        pert = abl.clone()
        pert[..., live] *= s
        wf, tr = decode_rates(vae.decoder, tokeniser, pert * scale, k)
        mark = "  <- sampler's measured live-dim amplitude" if abs(
            s - live_scale_measured) < 1e-6 else ""
        print(f"  {s:>9.3f} {wf:>12.1%} {tr:>8.1%}{mark}")

    print()
    print("  If neither reproduces ~6% truth at the measured magnitude, the variance")
    print("  discrepancy is a SYMPTOM and not the cause, and a shift sweep would be")
    print("  chasing it.")

    # ---- 1c: is the target reachable by a flow? -------------------------------
    # kl_beta is 1e-3, so almost nothing pressures the aggregate posterior into
    # shape. Clustered latent means force the flow to transport an isotropic
    # Gaussian onto a spiky multimodal target — which predicts exactly the observed
    # pattern: right at t=1 where the answer is the mean, wrong at intermediate t
    # where the field must encode multimodality, and under-normed at the end
    # because landing BETWEEN modes is nearer the origin than landing IN one.
    print()
    print("=" * 78)
    print("1c — IS THE TARGET DISTRIBUTION REACHABLE BY A FLOW?")
    print("=" * 78)
    n_pool = 2000
    pool = real_blocks[:n_pool].flatten(1)
    dim = pool.shape[1]

    def concentration(x: torch.Tensor) -> float:
        dist = torch.cdist(x, x)
        dist.fill_diagonal_(float("inf"))
        nn = dist.min(dim=1).values.mean().item()
        mean = dist[dist.isfinite()].mean().item()
        return nn / mean

    observed = concentration(pool)
    # The null: isotropic Gaussian, same pool size and same dimension. Without it
    # the ratio has no scale — the same discipline as the noise floor and the
    # cosine null.
    null = concentration(torch.randn(n_pool, dim, device=device))
    print(f"  pool {n_pool} points in {dim} dims (K*D)")
    print(f"  nearest-neighbour / mean pairwise distance")
    print(f"    real latents   {observed:.3f}")
    print(f"    gaussian null  {null:.3f}   <- what an unclustered cloud looks like")
    print(f"    ratio          {observed / null:.3f}")
    print()
    print("  Well below the null means the latents are clustered — roughly one mode")
    print("  per distinct step — and the flow must transport an isotropic Gaussian")
    print("  onto a spiky multimodal target. If so the fix is upstream of the")
    print("  denoiser entirely: raise kl_beta, or add explicit prior-matching.")

    # ---- 2: per-dimension variance along the trajectory -----------------------
    print()
    print("=" * 78)
    print("2 — PER-DIMENSION VARIANCE ALONG THE TRAJECTORY")
    print("=" * 78)
    print("  observed Var(z_t)_d vs the true interpolant's (1-t)^2*sigma_d^2 + t^2.")
    print("  sigma_d^2 is the REAL per-dimension variance, not 1.0 — the pooled")
    print("  scaling constant makes the total unit, not each dimension.")
    print()

    curve: list[tuple[float, float, float]] = []

    def collect(step, t, active, latents):
        lo, hi = active * k, (active + 1) * k
        obs = latents[:, lo:hi].reshape(-1, d).var(0, unbiased=False)
        exp = (1 - t) ** 2 * real_var + t**2
        curve.append((
            t,
            ((obs[live] - exp[live]).abs() / exp[live]).mean().item(),
            ((obs[~live] - exp[~live]).abs() / exp[~live]).mean().item()
            if (~live).any() else float("nan"),
        ))

    torch.manual_seed(args.seed)
    draft(denoiser, None, blocks, k, d, args.nfe, eta=0.0,
          device=str(device), batch=args.samples, on_step=collect)

    # Average across blocks at each t: the row is then a property of schedule
    # position rather than of whichever block happened to be active.
    by_t: dict[float, list[tuple[float, float]]] = {}
    for t, lv, dv in curve:
        by_t.setdefault(round(t, 4), []).append((lv, dv))

    print(f"  {'t':>6} {'live dev':>10} {'dead dev':>10}")
    departure = None
    for t in sorted(by_t, reverse=True):
        vals = by_t[t]
        lv = sum(a for a, _ in vals) / len(vals)
        dv = sum(b for _, b in vals) / len(vals)
        # Threshold stated, not tuned: 10% mean relative deviation over live dims.
        if departure is None and lv > 0.10:
            departure = t
        print(f"  {t:>6.3f} {lv:>10.1%} {dv:>10.1%}"
              f"{'   <- departure begins' if departure == t else ''}")
    print()
    print(f"  departure (live dims exceed 10% mean relative deviation) at t = "
          f"{departure if departure is not None else 'never'}")


if __name__ == "__main__":
    main()
