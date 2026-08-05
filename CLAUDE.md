# Adze — agent instructions

Read this before touching anything. Then read `docs/design.md` for the *what* and `docs/build-plan.md` for the *sequence*.

## What this is

A latent diffusion reasoner. Reasoning happens in continuous latent space; text exists only at the interfaces. Draft block-causally, then refine globally so earlier reasoning steps can be revised in light of later ones.

The first result the whole repo is aimed at:

> Given the corrupted block's location, does global regeneration repair it more reliably than causal regeneration?

## How to work

**One milestone per session.** They're in `docs/build-plan.md`, M0 through M7. Do not start M(n+1) until M(n)'s acceptance test passes.

**Tests are the spec.** `tests/` contains failing tests that encode each milestone's acceptance criteria. Make them pass. Don't rewrite a test to match an implementation — if a test seems wrong, stop and flag it.

**Report back per milestone** with: what passed, what you changed, anything in the design that turned out to be underspecified.

**Commit per milestone.** `git commit -m "M3: denoiser, regime A"`.

## Invariants — hold these everywhere

```
B  = blocks per sequence      (fixed for v0)
K  = 4                        latents per block
D  = 64                       latent channel dim
N  = B * K                    total latent positions

latents    : [batch, N, D]
timesteps  : [batch, B]       PER BLOCK, broadcast to K within the block
block_ids  : [N]              which block each position belongs to
mask       : [N, N]           bool, True = attend
```

**The per-block timestep is not standard DiT.** Vanilla DiT takes one scalar `t` per sample and broadcasts it globally. Regime B needs different blocks at different noise levels in the same forward pass. Build it in from the start — retrofitting means touching every conditioning path.

**Two mask modes**, both pure functions of `block_ids`:
- `causal` — bidirectional within a block, causal across blocks
- `global` — fully bidirectional

**Convention:** `t = 0` is clean, `t = 1` is pure noise.

## Training formulation (design §3.1)

Rectified flow, **velocity prediction**, ODE sampler.

```
z_t = (1 - t) * z0 + t * eps           eps ~ N(0, I)
target = eps - z0
loss = || u_theta(z_t, t, ctx) - (eps - z0) ||^2
z_{t-dt} = z_t - dt * u_theta(z_t, t, ctx)      # integrate t: 1 -> 0
```

**Regime A — draft (90% of steps):** sample block `b`; noise block `b` only; blocks `< b` clean; blocks `> b` absent; causal mask; loss on `b`.

**Regime B — refine (10% of steps):** select subset `S`; set `t_i = 1` for `i in S` (complete erasure); blocks outside `S` clean; global mask; loss on `S`.

## Hard gates — do not proceed past these

**M2: latent-use check.** Decode from shuffled latents. If quality barely drops, the decoder is ignoring the latent (posterior collapse) and everything downstream is meaningless. Stop and fix.

**M3: overfit one batch.** On `configs/debug.yaml`, 8 examples must drive loss to near zero. If they can't, something is miswired — most likely the per-block timestep broadcast or the mask. Do not proceed on a model that can't overfit 8 examples.

## Not in v0 — if you're building any of these, you've drifted

- termination head / variable reasoning length
- learned block segmentation
- partial re-noising (`t < 1` in pass two)
- uncertainty-based block selection
- rollout adaptation, semantic-correction objective
- looped refinement, adaptive compute
- byte-level input, learned input tokenisation
- distillation, RL
- x-prediction, SDE sampler
- KV caching (not until after M7, and only against a verified uncached baseline)

These are all in design §8 as deliberate future work. They are excluded on purpose, not forgotten.

## Environment

Apple Silicon, MPS backend. Two things bite:

- **Silent CPU fallbacks.** `PYTORCH_ENABLE_MPS_FALLBACK=1` masks them and tanks throughput without erroring. M0 exists to catch this.
- **Thermal throttling** on sustained runs. Checkpoint and resume from the start.

Keep `configs/debug.yaml` working at every milestone — 2 layers, 128 wide, 8 NFE, 100 examples. It's the difference between a 40-second iteration and a 40-minute one.

**`debug.yaml`'s `n_train` cannot pass a generalisation gate, and shouldn't.** Its 100 traces are sized for M3's overfit gate, which wants a set small enough to memorise. M2's gate 1 measures held-out reconstruction and wants the opposite — at `n_train: 100` the VAE drives training loss to 0.005 and reaches 16% held-out. Override the *data* from the CLI (`--n-train`, `--n-val`, `--batch`) and leave the model size alone. Do not resize the config to make a gate pass.

**M2's gate 1 needs 4 VAE layers, not debug.yaml's 2.** At 2×128 the VAE plateaus at 94.70% held-out exact match — 0.3pp short, with training loss already at 0.017, so it is a capacity ceiling and more steps do not move it. 4×128 reaches 95.65%. This is a real tension with "keep debug.yaml working": resolve it deliberately rather than by editing the config to make a gate pass. Pass `--vae-layers 4`.

**Checkpoints record their own architecture** in an `arch` key. A run trained with `--vae-layers` does not match its config file, so rebuild from `arch`, never from the YAML.

**Reconstruction fidelity and representation quality are close to unrelated.** A VAE can reconstruct at >97% while its latent space is spiky and awkward, and the symptom arrives at M3 disguised as "the denoiser won't train". `scripts/m2_interpolate.py` decodes latent midpoints as an early warning — it is a diagnostic, not a gate.

## Style

- Type hints on public functions
- Docstrings state shapes
- No config magic — explicit dataclasses, one YAML load at entry
- Prefer boring code; this is a research repo where debuggability beats elegance
