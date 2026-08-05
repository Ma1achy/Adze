"""M2 — VAE training loop.

Ends with two hard gates, both in adze.eval.checks:
  1. reconstruction > 95% exact token match on held-out steps
  2. latent-use check — decoding from shuffled latents must degrade badly

Gate 2 is the important one. A VAE that reconstructs perfectly while ignoring its
latent makes everything downstream meaningless.

Usage:
    python -m adze.train.train_vae configs/debug.yaml
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from adze.config import Config, load_config
from adze.data.corrupt import make_pair
from adze.data.dataset import LatentCache, TraceDataset
from adze.data.generate import Trace, generate_dataset
from adze.data.tokeniser import MAX_STEP_LEN, CharTokeniser
from adze.eval.checks import latent_use_check, reconstruction_accuracy
from adze.model.vae import build_vae

CHECKPOINT_DIR = Path("checkpoints")
CACHE_DIR = Path("data/cache")


def _step_texts(traces: list[Trace]) -> list[str]:
    """Flatten traces to individual rendered steps — the VAE's training unit."""
    return [step.render() for trace in traces for step in trace.steps]


def _split(
    config: Config, n_train: int | None = None, n_val: int | None = None
) -> tuple[list[Trace], list[Trace]]:
    """Train and held-out traces from disjoint seeds."""
    train = generate_dataset(
        n=n_train if n_train is not None else config.data.n_train,
        seed=config.data.seed,
        max_depth=config.data.max_depth,
        operand_max=config.data.operand_max,
    )
    val = generate_dataset(
        n=n_val if n_val is not None else config.data.n_val,
        seed=config.data.seed + 999_983,
        max_depth=config.data.max_depth,
        operand_max=config.data.operand_max,
    )
    return train, val


def train_vae(
    config_path: Path,
    steps: int | None = None,
    seed: int = 0,
    n_train: int | None = None,
    n_val: int | None = None,
    batch_size: int | None = None,
    vae_layers: int | None = None,
    vae_width: int | None = None,
    latent_dim: int | None = None,
    resume: bool = False,
) -> dict[str, float]:
    """Train the step VAE and run both M2 gates. Returns the gate numbers.

    `n_train` and `batch_size` override the config's *data* and batch size only.
    The model stays exactly the size the config specifies — these exist because
    debug.yaml's 100 traces are sized for the M3 overfit gate, which wants a set
    small enough to memorise, whereas gate 1 here measures held-out generalisation
    and wants the opposite.

    `vae_layers` / `vae_width` override encoder and decoder capacity. A plateau
    below the gate with training loss already near zero is a capacity limit, not an
    optimisation one, and more steps will not move it. Overriding here keeps
    debug.yaml at its stated 2x128 identity.
    """
    config = load_config(config_path)
    torch.manual_seed(seed)

    device = torch.device(config.device)
    tokeniser = CharTokeniser()

    train_traces, val_traces = _split(config, n_train, n_val)
    train_texts = _step_texts(train_traces)
    val_texts = _step_texts(val_traces)

    # How much of the held-out set is genuinely unseen. With a small operand range
    # many val steps are literally identical to training steps, so a headline
    # held-out number can be most of the way to a training number. Report the
    # unseen subset separately rather than letting that pass unnoticed.
    seen = set(train_texts)
    unseen_idx = [i for i, t in enumerate(val_texts) if t not in seen]

    train_tokens = tokeniser.encode_batch(train_texts).to(device)
    val_tokens = tokeniser.encode_batch(val_texts).to(device)

    # Resolved architecture, recorded so the checkpoint is self-describing. A run
    # with --vae-layers saves a model that does NOT match its config file, and
    # rebuilding from the config alone would then silently construct the wrong
    # shape. Anything loading this checkpoint should read `arch`, not the YAML.
    arch = {
        "vocab_size": tokeniser.vocab_size,
        "d_model": vae_width if vae_width is not None else config.model.vae.d_model,
        "n_layers": vae_layers if vae_layers is not None else config.model.vae.n_layers,
        "n_heads": config.model.vae.n_heads,
        "latents_per_block": config.model.latents_per_block,
        "latent_dim": latent_dim if latent_dim is not None else config.model.latent_dim,
        "kl_beta": config.train.kl_beta,
        "max_len": MAX_STEP_LEN,
    }
    vae = build_vae(**arch).to(device)

    n_steps = steps if steps is not None else config.train.steps
    batch = batch_size if batch_size is not None else config.train.batch_size
    opt = torch.optim.AdamW(vae.parameters(), lr=config.train.lr)
    # Cosine decay to zero. Exact-match reconstruction is a sharp target and the
    # last few points come from annealing rather than from more steps at full lr.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)

    print(f"config        {config_path}  ({config.name})")
    print(f"device        {device}")
    print(f"vocab         {tokeniser.vocab_size} symbols, max_len {MAX_STEP_LEN}")
    print(f"params        {sum(p.numel() for p in vae.parameters()) / 1e6:.2f}M")
    print(f"train steps   {len(train_texts)} from {len(train_traces)} traces")
    print(f"held-out      {len(val_texts)} steps, {len(unseen_idx)} unseen "
          f"({len(unseen_idx) / len(val_texts):.1%})")
    print(f"optimiser     {n_steps} steps, batch {batch}, "
          f"lr {config.train.lr}, kl_beta {config.train.kl_beta}")
    print()

    # Resume support. A sustained run on this box gets interrupted — thermal
    # throttling, or simply being killed — and losing an hour to that is avoidable.
    # The mid-run checkpoint is separate from the final one so an interrupted run
    # can never be mistaken for a finished one.
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    resume_path = CHECKPOINT_DIR / f"vae_{config.name}_d{arch['latent_dim']}_resume.pt"
    start_step = 1
    if resume and resume_path.exists():
        state = torch.load(resume_path, map_location=device, weights_only=False)
        vae.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        start_step = state["step"] + 1
        print(f"resumed       from {resume_path} at step {state['step']}\n")

    report_every = max(1, n_steps // 10)

    vae.train()
    t0 = time.perf_counter()
    for step in range(start_step, n_steps + 1):
        idx = torch.randint(0, len(train_texts), (batch,), device=device)
        out = vae(train_tokens[idx])

        opt.zero_grad()
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % report_every == 0 or step == start_step:
            recon = reconstruction_accuracy(vae, val_tokens)
            vae.train()
            print(f"  step {step:>6}  loss {out['loss'].item():7.4f}  "
                  f"recon {out['recon_loss'].item():7.4f}  kl {out['kl_loss'].item():8.3f}  "
                  f"held-out exact {recon['exact_match']:6.1%}")
            torch.save(
                {
                    "model": vae.state_dict(),
                    "opt": opt.state_dict(),
                    "sched": sched.state_dict(),
                    "step": step,
                },
                resume_path,
            )

    elapsed = time.perf_counter() - t0
    print(f"\ntrained in {elapsed:.1f}s\n")

    # ---- Gate 1: held-out reconstruction -------------------------------------
    gate1 = reconstruction_accuracy(vae, val_tokens)
    gate1_unseen = reconstruction_accuracy(vae, val_tokens[unseen_idx])

    print("=" * 74)
    print("GATE 1 — held-out reconstruction (need > 95% exact match)")
    print("=" * 74)
    print(f"  token accuracy         {gate1['token_acc']:7.2%}")
    print(f"  exact match            {gate1['exact_match']:7.2%}   <- the gate")
    print(f"  exact match, unseen    {gate1_unseen['exact_match']:7.2%}   "
          f"({len(unseen_idx)} steps not present in training)")
    gate1_pass = gate1["exact_match"] > 0.95
    print(f"  {'PASS' if gate1_pass else 'FAIL'}")

    # ---- Gate 2: latent use --------------------------------------------------
    gate2 = latent_use_check(vae, val_tokens, n_shuffles=8)

    print()
    print("=" * 74)
    print("GATE 2 — latent use (need a LARGE gap)")
    print("=" * 74)
    print(f"  clean exact match      {gate2['clean_acc']:7.2%}")
    print(f"  shuffled latents       {gate2['shuffled_acc']:7.2%}")
    print(f"  gap                    {gate2['gap']:7.2%}   <- the gate")
    print(f"  random latents         {gate2['random_acc']:7.2%}   (cross-check)")
    print()
    if gate2["clean_acc"] < 0.5:
        # Zero clean accuracy gives a zero gap, which the ratio test below would
        # wave through. An undertrained model must read as inconclusive, not pass.
        print("  INCONCLUSIVE — clean accuracy is too low for the gap to mean "
              "anything. Fix gate 1 first.")
    elif gate2["gap"] < 0.5 * gate2["clean_acc"]:
        print("  FAIL — the decoder is largely ignoring the latent. STOP.")
    else:
        print("  PASS — shuffling the latent destroys reconstruction.")

    # ---- Checkpoint and latent cache ----------------------------------------
    tag = f"{config.name}_d{arch['latent_dim']}"
    ckpt = CHECKPOINT_DIR / f"vae_{tag}.pt"
    torch.save({"model": vae.state_dict(), "config": str(config_path),
                "arch": arch}, ckpt)

    pairs = [make_pair(t, rng_seed=i) for i, t in enumerate(train_traces)
             if len(t.steps) >= 2]
    dataset = TraceDataset(
        pairs,
        blocks=config.data.blocks_per_sequence,
        latents_per_block=config.model.latents_per_block,
    )
    cache_path = CACHE_DIR / f"latents_{tag}.pt"
    LatentCache(cache_path).build(dataset, vae)
    cached = LatentCache(cache_path).load()

    print()
    print(f"checkpoint    {ckpt}")
    cache = LatentCache(cache_path)
    print(f"latent cache  {cache_path}  {tuple(cached.shape)}  [n, N=B*K, D]")
    print(f"scale         {cache.scale:.4f}  "
          f"(scaled latent std {cached.std().item():.4f}, "
          f"per-position norm {cached.flatten(0, 1).norm(dim=-1).mean().item():.3f} "
          f"vs sqrt(D)={arch['latent_dim'] ** 0.5:.3f})")
    if dataset.n_dropped:
        print(f"              {dataset.n_dropped} traces dropped — more steps than "
              f"B={config.data.blocks_per_sequence}")

    return {
        "gate1_exact_match": gate1["exact_match"],
        "gate1_token_acc": gate1["token_acc"],
        "gate2_clean_acc": gate2["clean_acc"],
        "gate2_shuffled_acc": gate2["shuffled_acc"],
        "gate2_gap": gate2["gap"],
        "gate2_random_acc": gate2["random_acc"],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path, nargs="?", default=Path("configs/debug.yaml"))
    p.add_argument("--steps", type=int, default=None,
                   help="override train.steps from the config")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-train", type=int, default=None,
                   help="override data.n_train; the model size is untouched")
    p.add_argument("--n-val", type=int, default=None,
                   help="override data.n_val; a >95%% gate needs enough held-out steps")
    p.add_argument("--batch", type=int, default=None,
                   help="override train.batch_size")
    p.add_argument("--vae-layers", type=int, default=None,
                   help="override model.vae.n_layers")
    p.add_argument("--vae-width", type=int, default=None,
                   help="override model.vae.d_model")
    p.add_argument("--latent-dim", type=int, default=None,
                   help="override model.latent_dim (D)")
    p.add_argument("--resume", action="store_true",
                   help="continue from checkpoints/vae_<name>_resume.pt if present")
    args = p.parse_args()
    train_vae(args.config, steps=args.steps, seed=args.seed,
              n_train=args.n_train, n_val=args.n_val, batch_size=args.batch,
              vae_layers=args.vae_layers, vae_width=args.vae_width,
              latent_dim=args.latent_dim, resume=args.resume)


if __name__ == "__main__":
    main()
