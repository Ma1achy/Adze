"""Localise the sampler failure — is it the denoiser, the prefix, or the manifold?

The brief asked for "the training loss on the sampler's own z_t". That is not
well-posed: the regime-A loss is ||u(z_t,t) - (eps - z0)||^2, and for a GENERATED
block there is no ground-truth z0 or eps. The only recovery available is
z0 = z_t - t*u and eps = z_t + (1-t)*u, which is the model's own prediction and
makes the loss identically zero. So this measures the well-posed decomposition of
the same question instead, everything paired on the same t and the same eps:

  A   real z0, real prefix from the SAME trace.
      What training and the M3 gate see. Its absolute value is also a result:
      held-out A against train A is the underfit check, and it runs first because
      it is the most boring explanation and the cheapest to rule out. The overfit
      gate proves the model CAN memorise 8 examples; it says nothing about whether
      it learned the distribution.

  A'  real z0, real prefix from a DIFFERENT trace.
      A control A did not have. Swapping the prefix for a generated one changes two
      things at once — the prefix stops being informative about the target, and it
      leaves the training distribution. A' is uninformative but in-distribution, so
      A' - A is the information term and B - A' is the distribution-shift term.
      Without it, B - A conflates the two.

  B   real z0, the sampler's GENERATED prefix.
      Uninformative and off-distribution. B - A' is exposure bias proper, measured
      rather than inferred.

  C   the sampler's own finished block latent, re-noised and scored as if it were
      data, under the generated prefix. A denoiser cannot denoise toward a point it
      does not recognise, so C - A is the off-manifold readout, and the closest
      well-posed reading of the literal ask.
      CAVEAT: C is also high for on-manifold-but-RARE points, so on its own it
      conflates "off the manifold" with "in a low-density region of it". Read it
      with the shell statistic, which is geometric and does not care about density.

  D   the shell statistic, from TrajectoryRecorder.shell(). E[z_t^2] = (1-t)^2 + t^2
      per dimension on the true path; the departure from that curve is truncation
      error, and it separates geometry from density in a way C cannot.

Usage:
    python scripts/m4_loss_gap.py
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
from adze.eval.load import load_denoiser, load_vae
from adze.invariants import MaskMode
from adze.model.flow import interpolate, schedule, velocity_target
from adze.model.masks import regime_a_mask
from adze.pad import masked_mean, real_positions
from adze.sample.draft import draft
from adze.sample.trajectory import TrajectoryRecorder

CACHE_DIR = Path("data/cache")


@torch.no_grad()
def encode_split(vae, config, arch, n: int, seed_offset: int, device) -> tuple:
    """Encode fresh traces with the frozen VAE. Returns (latents [n,N,D], mask [n,B]).

    Unscaled — the caller applies the TRAINING scale, not one recomputed here. A
    per-split scale would silently renormalise held-out data to a different radius
    from the one the denoiser was trained on, and the gap would read as a model
    failure rather than as an arithmetic one.
    """
    traces = generate_dataset(
        n=n,
        seed=config.data.seed + seed_offset,
        max_depth=config.data.max_depth,
        operand_max=config.data.operand_max,
    )
    pairs = [make_pair(t, rng_seed=i) for i, t in enumerate(traces) if len(t.steps) >= 2]
    dataset = TraceDataset(
        pairs,
        blocks=arch["blocks"],
        latents_per_block=arch["latents_per_block"],
    )
    latents = torch.empty(len(dataset), dataset.n_positions, arch["latent_dim"])
    masks = torch.zeros(len(dataset), arch["blocks"], dtype=torch.bool)
    for i in range(len(dataset)):
        item = dataset[i]
        mu, _ = vae.encoder(item["tokens"].to(device))
        mu[~item["block_mask"].to(device)] = vae.pad_latent.to(mu.dtype)
        latents[i] = mu.reshape(dataset.n_positions, -1).cpu()
        masks[i] = item["block_mask"]
    return latents.to(device), masks.to(device)


@torch.no_grad()
def cell_loss(denoiser, z0_target, z0_prefix, block_ids, blocks, b, t_val, eps, keep):
    """Regime-A loss on block `b` at timestep `t_val`, for one (target, prefix) pair.

    Args:
        z0_target: [batch, N, D] supplies block b's clean value — the regression
            target's source.
        z0_prefix: [batch, N, D] supplies blocks < b, the conditioning context.
        eps:       [batch, N, D] the noise, held fixed across cells so A/A'/B/C are
            paired rather than independently sampled.
        keep:      [batch, N, 1] bool, real (non-pad) positions of block b.

    Splicing the two sources is what makes the decomposition work: every cell holds
    the same target and the same noise, and varies only where the context came from.
    """
    is_b = (block_ids == b).view(1, -1, 1)
    z0 = torch.where(is_b, z0_target, z0_prefix)

    t = torch.zeros(z0.shape[0], blocks, device=z0.device)
    t[:, b] = t_val
    t[:, b + 1 :] = 1.0

    z_t = interpolate(z0, eps, t)
    target = velocity_target(z0, eps)
    pred = denoiser(z_t, t, block_ids, MaskMode.CAUSAL, mask=regime_a_mask(block_ids, b))
    return masked_mean((pred - target) ** 2, keep).item()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--denoiser", type=Path, default=Path("checkpoints/denoiser_debug_d16.pt"))
    p.add_argument("--vae", type=Path, default=Path("checkpoints/vae_debug_d16.pt"))
    p.add_argument("--config", type=Path, default=Path("configs/debug.yaml"))
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--draws", type=int, default=8, help="eps draws averaged per cell")
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

    block_ids = torch.repeat_interleave(torch.arange(blocks), k).to(device)

    print(f"denoiser      {args.denoiser}  {arch['n_layers']}L x {arch['d_model']}w")
    print(f"vae           {args.vae}  D={d}")
    print(f"latent scale  {scale:.4f}")
    print()

    # ---- Splits ---------------------------------------------------------------
    train_latents = cache.load().to(device)
    train_mask = cache.load_block_mask().to(device)
    idx = torch.randperm(train_latents.shape[0], device=device)[: args.batch]
    train_z0, train_bm = train_latents[idx], train_mask[idx]

    held_raw, held_bm = encode_split(vae, config, arch, args.batch * 2, 909_091, device)
    held_z0 = (held_raw / scale)[: args.batch]
    held_bm = held_bm[: args.batch]

    knots = schedule(args.nfe, 1.0, device=device)[:-1]

    # ---- A absolute: the underfit check, run first ----------------------------
    print("=" * 78)
    print("UNDERFIT CHECK — regime-A loss, train vs held-out at matched t")
    print("=" * 78)
    print("  the cheapest explanation: if held-out is far above train, the denoiser")
    print("  never learned the distribution and everything below is a consequence")
    print()
    print(f"  {'t':>6} {'train':>10} {'held-out':>10} {'ratio':>8}")
    underfit = []
    for t_val in knots.tolist():
        tr, ho = [], []
        for b in range(blocks):
            for _ in range(args.draws):
                eps = torch.randn(args.batch, blocks * k, d, device=device)
                keep_tr = real_positions(train_bm, k) & (block_ids == b).view(1, -1, 1)
                keep_ho = real_positions(held_bm, k) & (block_ids == b).view(1, -1, 1)
                if keep_tr.sum() and keep_ho.sum():
                    tr.append(cell_loss(denoiser, train_z0, train_z0, block_ids,
                                        blocks, b, t_val, eps, keep_tr))
                    ho.append(cell_loss(denoiser, held_z0, held_z0, block_ids,
                                        blocks, b, t_val, eps, keep_ho))
        a, c = sum(tr) / len(tr), sum(ho) / len(ho)
        underfit.append((t_val, a, c))
        print(f"  {t_val:>6.3f} {a:>10.4f} {c:>10.4f} {c / a:>8.2f}x")
    mean_ratio = sum(c / a for _, a, c in underfit) / len(underfit)
    print(f"\n  mean held-out / train  {mean_ratio:.2f}x")

    # ---- The sampler's own output --------------------------------------------
    torch.manual_seed(args.seed)
    gen = draft(denoiser, None, blocks, k, d, args.nfe,
                device=str(device), batch=args.batch)

    # A': a real prefix from a different trace. Uninformative but in-distribution.
    shuffled = held_z0[torch.randperm(held_z0.shape[0], device=device)]

    print()
    print("=" * 78)
    print("LOSS GAP — same target, same eps, varying the prefix and the target source")
    print("=" * 78)
    print("  A  real z0, real prefix (same trace)     <- the reference")
    print("  A' real z0, real prefix (OTHER trace)    <- uninformative, in-distribution")
    print("  B  real z0, generated prefix            <- uninformative, off-distribution")
    print("  C  generated z0, generated prefix       <- the sampler's own output as data")
    print()

    grid: dict[int, list[tuple]] = {}
    for b in range(1, blocks):          # block 0 has no prefix; nothing to vary
        rows = []
        for t_val in knots.tolist():
            acc = [0.0, 0.0, 0.0, 0.0]
            keep = real_positions(held_bm, k) & (block_ids == b).view(1, -1, 1)
            if keep.sum() == 0:
                continue
            keep_gen = (block_ids == b).view(1, -1, 1).expand(args.batch, -1, 1)
            for _ in range(args.draws):
                eps = torch.randn(args.batch, blocks * k, d, device=device)
                acc[0] += cell_loss(denoiser, held_z0, held_z0, block_ids,
                                    blocks, b, t_val, eps, keep)
                acc[1] += cell_loss(denoiser, held_z0, shuffled, block_ids,
                                    blocks, b, t_val, eps, keep)
                acc[2] += cell_loss(denoiser, held_z0, gen, block_ids,
                                    blocks, b, t_val, eps, keep)
                acc[3] += cell_loss(denoiser, gen, gen, block_ids,
                                    blocks, b, t_val, eps, keep_gen)
            rows.append((t_val, *[v / args.draws for v in acc]))
        grid[b] = rows

    print(f"  {'block':>5} {'t':>6} {'A':>9} {'A-prime':>9} {'B':>9} {'C':>9} "
          f"{'B-A':>8} {'C-A':>8}")
    for b, rows in grid.items():
        for t_val, a, ap, bb, c in rows:
            print(f"  {b:>5} {t_val:>6.3f} {a:>9.4f} {ap:>9.4f} {bb:>9.4f} {c:>9.4f} "
                  f"{bb - a:>+8.4f} {c - a:>+8.4f}")

    print()
    print("  by block, averaged over t:")
    print(f"  {'block':>5} {'A':>9} {'A-prime':>9} {'B':>9} {'C':>9} "
          f"{'info':>8} {'shift':>8} {'manifold':>9}")
    for b, rows in grid.items():
        n = len(rows)
        a, ap, bb, c = (sum(r[i] for r in rows) / n for i in range(1, 5))
        print(f"  {b:>5} {a:>9.4f} {ap:>9.4f} {bb:>9.4f} {c:>9.4f} "
              f"{ap - a:>+8.4f} {bb - ap:>+8.4f} {c - a:>+9.4f}")
    print("    info     = A' - A   prefix stops being informative")
    print("    shift    = B  - A'  prefix leaves the training distribution (exposure bias)")
    print("    manifold = C  - A   the sampler's own output scored as data")

    # ---- D: the shell statistic ----------------------------------------------
    print()
    print("=" * 78)
    print("SHELL — trajectory RMS against sqrt((1-t)^2 + t^2)")
    print("=" * 78)
    print("  geometric, not density-dependent: this is what separates 'off the")
    print("  manifold' from 'in a low-density part of it', which C alone cannot.")
    print()
    print(f"  {'nfe':>4}  {'t':>6} {'observed':>10} {'expected':>10} {'rel err':>9}")
    for nfe in (args.nfe, 50):
        torch.manual_seed(args.seed)
        rec = TrajectoryRecorder(vae.decoder, tokeniser, k, scale)
        draft(denoiser, None, blocks, k, d, nfe,
              device=str(device), batch=1, recorder=rec)
        # Average over blocks at each t, so the row is a property of the schedule
        # position rather than of whichever block happened to be active.
        by_t: dict[float, list[tuple[float, float]]] = {}
        for _, _, t_val, obs, exp in rec.shell():
            by_t.setdefault(round(t_val, 4), []).append((obs, exp))
        for t_val in sorted(by_t, reverse=True):
            vals = by_t[t_val]
            obs = sum(o for o, _ in vals) / len(vals)
            exp = sum(e for _, e in vals) / len(vals)
            print(f"  {nfe:>4}  {t_val:>6.3f} {obs:>10.3f} {exp:>10.3f} "
                  f"{(obs - exp) / exp:>+9.1%}")

    main_ceiling(denoiser, train_z0, train_bm, held_z0, held_bm, block_ids,
                 blocks, k, d, knots, args.draws, device)


@torch.no_grad()
def conditional_variance(pool, pool_bm, query, query_bm, k_lat, b, kk: int = 10):
    """Estimate Var(z0_b | prefix) non-parametrically, by nearest neighbour in prefix.

    `A`'s absolute value only means something against a ceiling. The ceiling is set
    by how much blocks < b actually determine block b IN THE DATA — if the prefix
    barely constrains the next step, a loss near the marginal variance is the task
    being hard, not the model being bad, and no amount of training moves it.

    Two estimators with opposite biases, reported together:
      k=1  : E||z_b - z_b(nn)||^2 = 2*sigma_c^2 if the neighbour is conditionally
             an independent draw, so sigma_c^2 ~ residual / 2.
      k=10 : averaging 10 neighbours shrinks the prediction, giving
             sigma_c^2 ~ residual * k/(k+1).
    They bracket the truth. Blocks fill in order, so block b being real implies
    every block before it is real — the prefix needs no separate mask.
    """
    sel_p = pool_bm[:, b]
    sel_q = query_bm[:, b]
    if sel_p.sum() < kk + 1 or sel_q.sum() == 0:
        return None
    pref_p = pool[sel_p][:, : b * k_lat].flatten(1)
    pref_q = query[sel_q][:, : b * k_lat].flatten(1)
    tgt_p = pool[sel_p][:, b * k_lat : (b + 1) * k_lat].flatten(1)
    tgt_q = query[sel_q][:, b * k_lat : (b + 1) * k_lat].flatten(1)

    dist = torch.cdist(pref_q, pref_p)
    nn = dist.topk(kk, largest=False).indices

    total = tgt_q.var(0, unbiased=False).mean().item()
    res1 = (tgt_q - tgt_p[nn[:, 0]]).pow(2).mean().item()
    resk = (tgt_q - tgt_p[nn].mean(1)).pow(2).mean().item()
    return total, res1 / 2, resk * kk / (kk + 1)


def achievable(sigma_c: float, t: float) -> float:
    """Lowest regime-A loss attainable at timestep t given Var(z0|prefix) = sigma_c.

    Write z0 = m + r with m = E[z0|prefix] known and Var(r) = sigma_c. The predictor
    sees w = (1-t)r + t*eps and must estimate eps - r. Under a Gaussian assumption
    the residual is

        (sigma_c + 1) - (t - (1-t)*sigma_c)^2 / ((1-t)^2*sigma_c + t^2)

    At sigma_c = 1 (prefix tells you nothing) this is the marginal case; at
    sigma_c = 0 (prefix determines the block) it collapses to 0. Gaussian is an
    approximation — the latents are not — so read the curve as a scale, not a bound.
    """
    num = (t - (1 - t) * sigma_c) ** 2
    den = (1 - t) ** 2 * sigma_c + t**2
    return (sigma_c + 1) - num / max(den, 1e-9)


def main_ceiling(denoiser, train_z0, train_bm, held_z0, held_bm, block_ids,
                 blocks, k, d, knots, draws, device):
    """Section E — is A good or bad in absolute terms?"""
    print()
    print("=" * 78)
    print("CEILING — measured A against what the data allows")
    print("=" * 78)
    print("  A's absolute value is only meaningful against Var(z0_b | prefix). If the")
    print("  prefix barely determines the next block, a high loss is the task, not the")
    print("  model, and more training will not move it.")
    print()
    print(f"  {'block':>5} {'Var(z0_b)':>10} {'sig_c k=1':>10} {'sig_c k=10':>11} "
          f"{'explained':>10}")
    sigmas = {}
    for b in range(1, blocks):
        out = conditional_variance(train_z0, train_bm, held_z0, held_bm, k, b)
        if out is None:
            continue
        total, s1, sk = out
        sigmas[b] = (s1 + sk) / 2
        print(f"  {b:>5} {total:>10.4f} {s1:>10.4f} {sk:>11.4f} "
              f"{1 - sigmas[b] / total:>10.1%}")

    print()
    print("  measured A vs the achievable curve at that sigma_c:")
    print(f"  {'block':>5} {'t':>6} {'A':>9} {'floor':>9} {'excess':>9}")
    for b, sigma in sigmas.items():
        for t_val in knots.tolist():
            keep = real_positions(held_bm, k) & (block_ids == b).view(1, -1, 1)
            if keep.sum() == 0:
                continue
            acc = 0.0
            for _ in range(draws):
                eps = torch.randn(held_z0.shape[0], blocks * k, d, device=device)
                acc += cell_loss(denoiser, held_z0, held_z0, block_ids,
                                 blocks, b, t_val, eps, keep)
            a = acc / draws
            fl = achievable(sigma, t_val)
            print(f"  {b:>5} {t_val:>6.3f} {a:>9.4f} {fl:>9.4f} {a - fl:>+9.4f}")

    print()
    print("  `excess` near zero means the denoiser is at the information limit the")
    print("  data allows and the loss cannot be trained away. Large positive excess,")
    print("  especially at high t, means skill is being left on the table.")


if __name__ == "__main__":
    main()
