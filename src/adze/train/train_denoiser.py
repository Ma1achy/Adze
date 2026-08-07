"""M3/M6 — denoiser training.

M3 is regime A only. M6 adds regime B and the 90/10 mix.

Regime A (draft): sample block b; noise block b only; blocks < b clean;
blocks > b absent; causal mask; loss on b.

Regime B (refine): select subset S; t_i = 1 for i in S (complete erasure);
blocks outside S clean; global mask; loss on S.

M6 requires retraining from scratch — mixing changes what the model learns, not
just how it is used. Acceptance for M6 is that pass-one quality does NOT regress
against the M5 baseline.

Usage:
    python -m adze.train.train_denoiser configs/debug.yaml
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from adze.config import load_config
from adze.data.dataset import LatentCache
from adze.eval.checks import overfit_one_batch
from adze.invariants import MaskMode
from adze.model.denoiser import Denoiser
from adze.model.flow import interpolate, sample_timesteps, velocity_target
from adze.model.masks import regime_a_mask, vectorised_regime_a_mask
from adze.pad import masked_mean, real_positions
from adze.train.regime_b import regime_b_batch, regime_b_loss

CHECKPOINT_DIR = Path("checkpoints")
CACHE_DIR = Path("data/cache")


def regime_a_batch(
    z0: torch.Tensor,
    block_ids: torch.Tensor,
    blocks: int,
    generator: torch.Generator | None = None,
    force_block: int | None = None,
    block_mask: torch.Tensor | None = None,
    t_shift: float | None = 1.5,
) -> dict[str, torch.Tensor]:
    """Build one regime A (draft) training batch.

    Sample a block `b`, noise it, leave blocks < b clean, remove blocks > b, and
    score only block b.

    One `b` is drawn for the whole batch rather than per example. The mask is an
    [N, N] function of `b`, so a per-example `b` would need a per-example mask and
    a batched attention mask; drawing per batch keeps a single mask and costs only
    that consecutive steps see one block each. Over training every block is
    sampled uniformly often.

    Args:
        z0:        [batch, N, D] clean latents.
        block_ids: [N]
        blocks:    B.
        t_shift:   shift for logit-normal timestep sampling; None samples t
                   uniformly, which is the worse-conditioned baseline kept only
                   so the two can be compared at matched budget.

    Returns:
        dict with z_t, t, target, mask, loss_mask, and the sampled block index.
    """
    batch, n_positions, _ = z0.shape
    device = z0.device

    if force_block is not None:
        b = force_block
    elif block_mask is None:
        b = int(torch.randint(0, blocks, (1,), generator=generator).item())
    else:
        # Sample only among blocks that are real for at least one example. A block
        # that is padding across the whole batch has no target: its loss is zero,
        # its gradient is zero, and the step is wasted. With B=7 and traces of 3-7
        # steps that is ~61% of block-6 draws and ~11% of all steps, which also
        # starves the last blocks of the gradient they do need.
        available = torch.nonzero(block_mask.any(dim=0), as_tuple=False).flatten()
        pick = int(torch.randint(0, len(available), (1,), generator=generator).item())
        b = int(available[pick].item())

    # t = 0 clean everywhere, then noise block b only. Blocks after b are absent,
    # so their timestep is never read — set to 1 to make that explicit rather than
    # leaving a value that looks meaningful.
    t = torch.zeros(batch, blocks, device=device)
    if t_shift is None:
        t[:, b] = torch.rand(batch, generator=generator, device=device)
    else:
        t[:, b] = sample_timesteps(
            (batch,), shift=t_shift, device=device, generator=generator
        )
    t[:, b + 1 :] = 1.0

    eps = torch.randn(z0.shape, generator=generator, device=device)
    z_t = interpolate(z0, eps, t)
    target = velocity_target(z0, eps)

    # The same function `draft` samples under. Shared by construction so the two
    # cannot drift apart — see adze.model.masks.regime_a_mask.
    mask = regime_a_mask(block_ids, b)

    # Score block b, but only for examples where block b holds a real step. A
    # padded block carries the same constant vector every time, so it is trivially
    # predictable and scoring it deflates the loss toward zero without the model
    # having learned anything. Pad blocks are masked out, not merely represented.
    loss_mask = (block_ids == b).view(1, n_positions, 1).expand(batch, -1, 1).clone()
    if block_mask is not None:
        loss_mask = loss_mask & real_positions(block_mask, n_positions // blocks)

    return {
        "z_t": z_t,
        "t": t,
        "target": target,
        "mask": mask,
        "loss_mask": loss_mask,
        "block": torch.tensor(b),
    }


def vectorised_regime_a_batch(
    z0: torch.Tensor,
    block_ids: torch.Tensor,
    blocks: int,
    generator: torch.Generator | None = None,
    block_mask: torch.Tensor | None = None,
    t_shift: float | None = 1.5,
    t: torch.Tensor | None = None,
    eps: torch.Tensor | None = None,
    zero_prefix: bool = False,
) -> dict[str, torch.Tensor]:
    """Regime A with EVERY block denoised in one forward pass.

    `regime_a_batch` is the naive algorithm: it samples one block index and scores
    that block alone, because the clean context differs per block. With B=7 that
    gives each block position 1/7th of the gradient steps — measured at ~46k draws
    per position against the unconditional model's 5.1M, a ~110x starvation.

    This is BD3-LM's fix (arXiv 2503.09573 §3.2). Run the model over the
    concatenation [z_t ; z0] of length 2N under `vectorised_regime_a_mask`, and all
    B conditionals are computed at once. Every block gets gradient every step.

    Two index spaces, and conflating them is the failure mode to watch for:
      - `block_ids` (real, [N]) builds the MASK. M_OBC fires on block(j) < block(i),
        which is meaningless if the clean half's ids are offset past B.
      - `block_ids_full` (offset, [2N]) drives the TIMESTEP GATHER only, so the
        clean half can be handed t=0 through the same [batch, 2B] tensor.

    Args:
        t, eps: supply these to make the batch deterministic. Used by the
            equivalence test, which must give both paths identical noise.

    Returns:
        dict with z_full, t_full, block_ids_full, target, mask, loss_mask.
        `loss_mask` covers the NOISED half only; the clean half is context.
    """
    batch, n_positions, _ = z0.shape
    device = z0.device
    k = n_positions // blocks

    if t is None:
        if t_shift is None:
            t = torch.rand(batch, blocks, generator=generator, device=device)
        else:
            t = sample_timesteps(
                (batch, blocks), shift=t_shift, device=device, generator=generator
            )
    if eps is None:
        eps = torch.randn(z0.shape, generator=generator, device=device)

    z_t = interpolate(z0, eps, t)
    target = velocity_target(z0, eps)

    # Clean copy at t=0 — the same value the naive path gives prefix blocks, and
    # the same value `draft` gives already-generated blocks. All three agree.
    t_full = torch.cat([t, torch.zeros(batch, blocks, device=device)], dim=1)
    block_ids_full = torch.cat([block_ids, block_ids + blocks])
    clean = torch.zeros_like(z0) if zero_prefix else z0
    z_full = torch.cat([z_t, clean], dim=1)

    mask = vectorised_regime_a_mask(block_ids)

    loss_mask = torch.ones(batch, n_positions, 1, dtype=torch.bool, device=device)
    if block_mask is not None:
        loss_mask = real_positions(block_mask, k)

    return {
        "z_full": z_full,
        "t_full": t_full,
        "block_ids_full": block_ids_full,
        "target": target,
        "mask": mask,
        "loss_mask": loss_mask,
        "t": t,
        "eps": eps,
    }


def vectorised_regime_a_loss(model: Denoiser, batch: dict[str, torch.Tensor]):
    """Velocity MSE over every block's noised half, real positions only."""
    pred = model(
        batch["z_full"],
        batch["t_full"],
        batch["block_ids_full"],
        MaskMode.CAUSAL,
        mask=batch["mask"],
    )
    n = batch["target"].shape[1]
    if batch["loss_mask"].sum() == 0:
        return (pred * 0).sum()
    return masked_mean((pred[:, :n] - batch["target"]) ** 2, batch["loss_mask"])


def regime_a_loss(model: Denoiser, batch: dict[str, torch.Tensor], block_ids: torch.Tensor):
    """Velocity MSE over the noised block only, real positions only.

    Returns NaN-free zero if the sampled block is padding for every example in the
    batch; callers should skip such steps rather than average them in.
    """
    pred = model(
        batch["z_t"],
        batch["t"],
        block_ids,
        MaskMode.CAUSAL,
        mask=batch["mask"],
    )
    if batch["loss_mask"].sum() == 0:
        # Every example's block b is padding. masked_mean raises on this by design;
        # callers skip such steps, and this keeps the graph intact for the ones that
        # do not check first.
        return (pred * 0).sum()
    return masked_mean((pred - batch["target"]) ** 2, batch["loss_mask"])


def resolve_b_prob(configured: float, override: float | None,
                   mixed: bool) -> float:
    """The regime-B share this run will actually use.

    A CLI override, not a config edit — same pattern as --batch / --lr /
    --denoiser-layers. M7's crossing table showed the model is specialised to
    whichever configuration it saw most, so this share is the knob under test
    rather than a constant inherited from DiD.

    Validated at the TOP of training rather than where it is first used: the gate
    runs first and costs 6000 steps, and a rejected argument should not cost them.
    """
    if not mixed:
        return 0.0
    b_prob = configured if override is None else override
    if not 0.0 < b_prob <= 1.0:
        raise ValueError(
            f"regime_b_prob must be in (0, 1], got {b_prob}. A share of 0 with "
            f"--mixed trains regime A only while claiming otherwise; drop "
            f"--mixed instead."
        )
    return b_prob


def train_denoiser(
    config_path: Path,
    mixed: bool = False,
    steps: int | None = None,
    gate_steps: int = 6000,
    latent_dim: int | None = None,
    seed: int = 0,
    t_shift: float | None = 1.5,
    batch_size: int | None = None,
    lr: float | None = None,
    vectorised: bool = True,
    zero_prefix: bool = False,
    n_layers: int | None = None,
    regime_b_prob: float | None = None,
    b_structure: str = "random",
    tag: str | None = None,
) -> dict[str, float]:
    """Args:
        config_path: yaml config.
        mixed: False for M3 (regime A only), True for M6 — the 90/10 mix of
            regime A (draft) and regime B (refine). The split comes from
            `config.train.regime_b_prob`. DiD's bimodal finding is the reason it
            is a mix rather than two separate models: the two regimes share a
            denoiser, and refine needs draft's representation to stay put.
        t_shift: shift for logit-normal timestep sampling. 1.5 concentrates draws
            at high t (SD3's setting, and M3's); 1.0 is plain logit-normal; values
            BELOW 1 invert the transform and concentrate on small t, which is where
            final sharpening happens. None samples t uniformly. The gate is run at
            the same shift so the trade between gate loss and sample quality is
            visible rather than hidden.
        batch_size, lr: override the config's training budget WITHOUT editing the
            config. debug.yaml is sized for fast iteration (batch 8, 500 steps),
            which is a different quantity from the compute a result needs — and
            resizing the YAML to make a number move is what CLAUDE.md forbids.
            Overriding here keeps the config honest and the run explicit.
        vectorised: compute every block's loss in one pass (BD3-LM's algorithm).
            The default, because the naive per-block form starves each position by
            ~110x. False keeps the naive path, retained for the equivalence test
            and for reproducing pre-vectorisation results.
        zero_prefix: zero the clean prefix blocks instead of letting them carry
            content. Tests whether the model fits spurious correlations in a
            prefix that carries only 2-7% of the information about the next block.
            Threaded into `draft` as well, since a sampler whose prefix differs
            from training's measures something the model was never taught.
        n_layers: override the config's denoiser depth. Every result so far is at
            debug.yaml's 2 layers, so "the model cannot do arithmetic above 30"
            and "a 2-layer model cannot" are not yet distinguishable. A CLI
            override rather than a config edit, for the same reason as batch/lr,
            and the depth goes in the checkpoint name so a deeper run cannot
            overwrite the 2-layer one it is being compared against.
    """
    config = load_config(config_path)
    # Before the gate, which costs 6000 steps.
    b_prob = resolve_b_prob(config.train.regime_b_prob, regime_b_prob, mixed)
    torch.manual_seed(seed)
    device = torch.device(config.device)

    d = latent_dim if latent_dim is not None else config.model.latent_dim
    cache = LatentCache(CACHE_DIR / f"latents_{config.name}_d{d}.pt")
    latents = cache.load().to(device)          # already scaled to ~unit variance
    block_mask = cache.load_block_mask().to(device)   # [n, B] real vs padded

    blocks = config.data.blocks_per_sequence
    k = config.model.latents_per_block
    block_ids = torch.repeat_interleave(torch.arange(blocks), k).to(device)
    depth = n_layers if n_layers is not None else config.model.denoiser.n_layers

    model = Denoiser(
        latent_dim=latents.shape[-1],
        d_model=config.model.denoiser.d_model,
        n_layers=depth,
        n_heads=config.model.denoiser.n_heads,
        latents_per_block=k,
        blocks=blocks,
    ).to(device)

    print(f"config        {config_path}  ({config.name})")
    print(f"device        {device}")
    # Report on real positions only. Averaging in pad blocks understates both
    # figures — they hold one constant vector — and makes the scaling look broken
    # when it is exact.
    real = real_positions(block_mask, k).squeeze(-1)
    real_latents = latents[real]
    root_d = latents.shape[-1] ** 0.5
    print(f"latents       {tuple(latents.shape)}  [n, N=B*K, D]  scale {cache.scale:.4f}")
    print(f"              real positions {real.float().mean().item():.1%}, "
          f"std {real_latents.std().item():.4f}, RMS norm "
          f"{real_latents.pow(2).sum(-1).mean().sqrt().item():.3f} "
          f"vs sqrt(D)={root_d:.3f}")
    print(f"params        {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"shapes        B={blocks} K={k} D={latents.shape[-1]} N={blocks * k}")
    print()

    # ---- HARD GATE: overfit one batch ---------------------------------------
    print("=" * 74)
    print("GATE — overfit one batch (8 examples, loss to near zero)")
    print("=" * 74)
    # config.train.steps is the debug loop's fast-iteration budget, not an
    # optimisation budget for this gate. At 500 steps / lr 3e-4 the model reaches
    # only 53% of initial and the gate reads as miswiring; at 6000 / 1e-3 it
    # overfits properly. Giving the gate enough steps is not lowering the bar —
    # the threshold below is unchanged.
    gate = overfit_one_batch(
        model,
        {
            "z0": latents[:8],
            "block_ids": block_ids,
            "blocks": torch.tensor(blocks),
            "block_mask": block_mask[:8],
        },
        steps=gate_steps,
        lr=1e-3,
        t_shift=t_shift,
    )
    print(f"  initial loss  {gate['initial_loss']:.6f}")
    print(f"  final loss    {gate['final_loss']:.6f}")
    print("  per block (final / initial):")
    for b, ratio in sorted(gate["per_block"].items()):
        note = "   <- no prefix; unconditional, floors above the rest" if b == 0 else ""
        print(f"    b={b}  {ratio:6.2%}{note}")
    # Block 0 is scored but excluded from the threshold: with no preceding context
    # it cannot be driven to zero until question conditioning arrives at M5.
    conditioned = [r for b, r in gate["per_block"].items() if b > 0]
    worst = max(conditioned)
    passed = worst < 0.02
    print(f"  {'PASS' if passed else 'FAIL'} — worst conditioned block "
          f"{worst:.2%} of initial (threshold 2%)")
    if not passed:
        print("\n  Something is miswired. Suspect adze.model.flow.broadcast_t and")
        print("  adze.model.masks.build_mask before suspecting anything conceptual.")
        return gate

    # ---- Full regime A training ---------------------------------------------
    n_steps = steps if steps is not None else config.train.steps
    n_batch = batch_size if batch_size is not None else config.train.batch_size
    n_lr = lr if lr is not None else config.train.lr
    model = Denoiser(
        latent_dim=latents.shape[-1],
        d_model=config.model.denoiser.d_model,
        n_layers=depth,
        n_heads=config.model.denoiser.n_heads,
        latents_per_block=k,
        blocks=blocks,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=n_lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)

    print()
    # A CLI override, not a config edit — same pattern as --batch / --lr /
    # --denoiser-layers. M7's crossing table showed the model is specialised to
    # whichever configuration it saw most, so this share is the knob under test
    # rather than a fixed constant inherited from DiD.
    label = (f"regime A + B mix ({1 - b_prob:.0%}/{b_prob:.0%})" if mixed
             else "regime A training")
    print(f"{label} — {n_steps} steps, batch {n_batch}, lr {n_lr}")
    t0 = time.perf_counter()
    model.train()
    n_b_steps = 0
    for step in range(1, n_steps + 1):
        idx = torch.randint(0, latents.shape[0], (n_batch,), device=device)
        # Regime B on a b_prob share of steps. Drawn per step rather than per
        # example: the two regimes use DIFFERENT masks and different sequence
        # lengths, so they cannot share a forward pass.
        if mixed and torch.rand(1, device=device).item() < b_prob:
            n_b_steps += 1
            batch = regime_b_batch(
                latents[idx], block_ids, blocks, block_mask=block_mask[idx],
                structure=b_structure,
            )
            loss = regime_b_loss(model, batch, block_ids)
        elif vectorised:
            batch = vectorised_regime_a_batch(
                latents[idx], block_ids, blocks, block_mask=block_mask[idx],
                t_shift=t_shift, zero_prefix=zero_prefix,
            )
            loss = vectorised_regime_a_loss(model, batch)
        else:
            batch = regime_a_batch(
                latents[idx], block_ids, blocks, block_mask=block_mask[idx],
                t_shift=t_shift,
            )
            loss = regime_a_loss(model, batch, block_ids)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % max(1, n_steps // 10) == 0 or step == 1:
            note = "all" if vectorised else batch.get("block", "all")
            extra = f"  regime B steps {n_b_steps}" if mixed else ""
            print(f"  step {step:>6}  loss {loss.item():.6f}  block {note}{extra}")

    print(f"\ntrained in {time.perf_counter() - t0:.1f}s")
    if mixed:
        # The REALISED share, not the requested one. The mix is a per-step
        # Bernoulli draw, so they differ, and the realised share is what the model
        # actually saw — which is the number any comparison across mixes rests on.
        print(f"regime B      {n_b_steps}/{n_steps} steps "
              f"({n_b_steps / n_steps:.1%} realised, {b_prob:.0%} requested)")
        print(f"regime A      {n_steps - n_b_steps} steps")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    # The shift goes in the filename: a sweep over shifts would otherwise
    # overwrite its own earlier runs, and the comparison needs all of them.
    suffix = "" if t_shift == 1.5 else f"_shift{t_shift}"
    suffix += "" if vectorised else "_naive"
    suffix += "" if seed == 0 else f"_seed{seed}"
    suffix += "_zeroprefix" if zero_prefix else ""
    suffix += "" if n_layers is None else f"_L{depth}"
    # The mix share goes in the filename for the same reason the shift and the
    # layer count do: a sweep over it would otherwise overwrite its own earlier
    # arms, and the comparison needs all of them.
    suffix += f"_mixedP{round(b_prob * 100)}" if mixed else ""
    # Distinguishes otherwise-identical runs WITHOUT touching a knob that
    # changes the run. Borrowing --seed for this cost a confound once:
    # matched-A differed from its comparison in step count AND seed.
    suffix += "" if b_structure == "random" else f"_S{b_structure}"
    suffix += "" if tag is None else f"_{tag}"
    ckpt = CHECKPOINT_DIR / f"denoiser_{config.name}_d{latents.shape[-1]}{suffix}.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": str(config_path),
            "arch": {
                "latent_dim": latents.shape[-1],
                "d_model": config.model.denoiser.d_model,
                "n_layers": depth,
                "n_heads": config.model.denoiser.n_heads,
                "latents_per_block": k,
                "blocks": blocks,
            },
            "latent_scale": cache.scale,
            "t_shift": t_shift,
            "zero_prefix": zero_prefix,
            "mixed": mixed,
            # THE TRAINING RECIPE, recorded. `denoiser_cap100_d16_L4_mixed.pt`
            # saved none of this, so nothing could be compared against it without
            # guessing at its batch size, learning rate and step count — which
            # cost a retrain of the reference arm. The same lesson as `arch`:
            # a checkpoint that does not record how it was made cannot be a
            # baseline for anything.
            "regime_b_prob": b_prob,
            "b_structure": b_structure,
            "regime_b_steps": n_b_steps,
            "steps": n_steps,
            "batch_size": n_batch,
            "lr": n_lr,
            "seed": seed,
            "vectorised": vectorised,
        },
        ckpt,
    )
    print(f"checkpoint    {ckpt}")

    return gate


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path, nargs="?", default=Path("configs/debug.yaml"))
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--gate-steps", type=int, default=6000)
    p.add_argument("--latent-dim", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--t-shift", type=float, default=1.5,
                   help="logit-normal shift; <1 concentrates on small t")
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--zero-prefix", action="store_true",
                   help="zero the clean prefix; control for spurious prefix fitting")
    p.add_argument("--naive", action="store_true",
                   help="use the pre-vectorisation per-block algorithm")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--denoiser-layers", type=int, default=None,
                   help="override denoiser depth; checkpoint is suffixed _L{n}")
    p.add_argument("--b-structure", choices=["random", "single", "contiguous"],
                   default="random",
                   help="regime B erasure SHAPE. `random` damages the prefix on a "
                        "typical step; `single` and `contiguous` leave it clean")
    p.add_argument("--tag", type=str, default=None,
                   help="arbitrary suffix to distinguish otherwise-identical runs. "
                        "Use this rather than --seed to avoid a checkpoint-name "
                        "collision: --seed changes the RUN, so borrowing it to "
                        "rename a file confounds the comparison it was made for")
    p.add_argument("--regime-b-prob", type=float, default=None,
                   help="override config.train.regime_b_prob. The knob M7's "
                        "crossing table pointed at; goes in the checkpoint name")
    p.add_argument("--mixed", action="store_true",
                   help="M6: mix regime B (refine) in at config.train.regime_b_prob")
    args = p.parse_args()
    train_denoiser(args.config, steps=args.steps, gate_steps=args.gate_steps,
                   latent_dim=args.latent_dim, seed=args.seed, t_shift=args.t_shift,
                   batch_size=args.batch, lr=args.lr, vectorised=not args.naive,
                   zero_prefix=args.zero_prefix, n_layers=args.denoiser_layers,
                   mixed=args.mixed, regime_b_prob=args.regime_b_prob,
                   b_structure=args.b_structure, tag=args.tag)


if __name__ == "__main__":
    main()
