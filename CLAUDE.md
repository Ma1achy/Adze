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

**Regime A — draft (90% of steps):** VECTORISED — every block noised to its own `t_b` and scored in one pass over `[z_t ; z0]`, under the four-quadrant mask in design §3.1. The naive form (sample one `b`, loss on `b` alone) starved each block position ~110x and converges ~5x slower; it is kept only as the equivalence-test reference. Prefix blocks carry `t = 0` in training and sampling alike.

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
- x-prediction
- KV caching (not until after M7, and only against a verified uncached baseline)

These are all in design §8 as deliberate future work. They are excluded on purpose, not forgotten.

**Promoted off this list at M4 — do not re-add them:**

- **SDE sampler.** `DEFAULT_ETA = 1.0` in `adze.sample.draft`. Promoted on 38.6% → 70.7% unconditional truth (95% of the 74.5% ceiling), with `eta = 0` reducing to the Euler step algebraically and `tests/test_m4_stochastic.py` holding that permanently. EDM churn is implemented alongside it and defaults **off** — it helps at `eta = 0` and harms at `eta = 1`, so `eta = 1, S_churn = 0` is a measured optimum.
(Rollout adaptation was promoted here at M4 and **un-promoted** in the same milestone — see the retired findings below. It is back on the not-in-v0 list above.)

## Retired findings — measured, believed, and since disproved

Keep these. Each was a real measurement that looked like the thing and wasn't, and
each cost sessions.

- **"Live-dimension shrinkage explains the sampler failure."** Variance ratio
  0.831 looked causal; ablating it on real latents cost 1.7pp. Retired.
- **"The sampler reproduces the marginal."** Rested on the per-dimension latent
  variance ratio reaching 1.000. But **matching second moments in a 16-dim latent
  space places almost no constraint on the semantic distribution the decoder
  produces.** The same model generates the easy operand bin at 47.5% against 4.5%
  in real data — a gross distributional failure a variance check is structurally
  blind to. Retired for the same reason as shrinkage.
- **"The generated prefix is worse than none, so exposure bias."** True premise,
  wrong diagnosis: `correct ≈ generated` means there are no own-errors to be
  robust to. Rollout adaptation was promoted and un-promoted on this.
- **"The prefix carries 2-7% of the information."** kNN artifact in 64-dim prefix
  space. Destroying the prefix actually costs +1.42 nats.

**A distribution can match on every moment you check and still be wrong where it
matters.** The general form: a statistic computed in latent space does not
constrain the semantic distribution downstream of a decoder, because the decoder
is not required to be an isometry. Check the thing you care about in the space
where you care about it. This is why **distribution-matched truth**, not latent
statistics, is the standard readout.

**Every M4 number recorded before the promotion was measured at `eta = 0`.** Pass `eta=0.0` explicitly to reproduce them; the historical diagnostic scripts pin it.

## Environment

Apple Silicon, MPS backend. Two things bite:

- **Silent CPU fallbacks.** `PYTORCH_ENABLE_MPS_FALLBACK=1` masks them and tanks throughput without erroring. M0 exists to catch this.
- **Thermal throttling** on sustained runs. Checkpoint and resume from the start.

Keep `configs/debug.yaml` working at every milestone — 2 layers, 128 wide, 8 NFE, 100 examples. It's the difference between a 40-second iteration and a 40-minute one.

**`debug.yaml`'s `n_train` cannot pass a generalisation gate, and shouldn't.** Its 100 traces are sized for M3's overfit gate, which wants a set small enough to memorise. M2's gate 1 measures held-out reconstruction and wants the opposite — at `n_train: 100` the VAE drives training loss to 0.005 and reaches 16% held-out. Override the *data* from the CLI (`--n-train`, `--n-val`, `--batch`) and leave the model size alone. Do not resize the config to make a gate pass.

**M2's gate 1 needs 4 VAE layers, not debug.yaml's 2.** At 2×128 the VAE plateaus at 94.70% held-out exact match — 0.3pp short, with training loss already at 0.017, so it is a capacity ceiling and more steps do not move it. 4×128 reaches 95.65%. This is a real tension with "keep debug.yaml working": resolve it deliberately rather than by editing the config to make a gate pass. Pass `--vae-layers 4`.

**Checkpoints record their own architecture** in an `arch` key. A run trained with `--vae-layers` does not match its config file, so rebuild from `arch`, never from the YAML.

**Reconstruction fidelity and representation quality are close to unrelated.** A VAE can reconstruct at >97% while its latent space is spiky and awkward, and the symptom arrives at M3 disguised as "the denoiser won't train". `scripts/m2_interpolate.py` decodes latent midpoints as an early warning — it is a diagnostic, not a gate.

**Read its nearest-neighbour rank, not its round-trip cosine.** Cosine asks whether a point lies in the encoder's *stable set*; it does not ask whether the point means anything. The two come apart exactly when the encoder is degenerate — a collapsed space round-trips everything well because everything is near everything. Measured: D=8 has the *best* chance-normalised cosine (0.769 vs 0.673 at D=16) and the *worst* NN rank by a factor of four (median 29 vs 6, both-endpoints-in-top-10 25% vs 56%), matching its 29% unseen reconstruction. Cosine said D=8 was the smoothest space; it was the most collapsed one. Never compare cosine across latent dimensionality.

## Standing findings

**Loss is not a proxy for sample quality here.** Three independent instances:
block 0 floors highest on gate loss (12.9%) while sampling *strongest* (61% true);
timestep shift 0.5 nearly halved training loss (0.39 vs 0.71) while losing 8pp of
truth (30.3% vs 38.6%); and after vectorisation the per-block gate losses are flat
(0.56-0.82%) while per-block truth spans 5.6-61.0%. Three cases makes it a
property, not a coincidence. A loss number that looks encouraging is not evidence
about samples — decode and classify.

**Vectorised regime A converges ~5x faster than the naive form.** The naive
algorithm scores one block per step and needs ~30k gate steps to cross the 2%
threshold; vectorised crosses it in 6k and lands lower (0.8% vs 1.14%). Budget
accordingly: a 6k naive run is under-converged, and several M4 diagnostics taken
on one turned out to be measuring undertraining rather than the thing they named.

**Distribution-matched truth is the standard readout.** `adze.eval.readout` —
matched as the headline, raw pooled truth kept and labelled, generated vs real
magnitude histograms printed alongside, per-bin ceilings rather than one pooled
one. Do not report a bare pooled truth figure.

**A pooled rate is not a rate unless the two sides are distribution-matched.**
The model's generated steps land in the 10-29 operand bin 51% of the time; real
steps do so 4.5% of the time. So a pooled truth figure is partly a measure of
which problems the model *chose*. Reweighting the per-bin rates by the data's
magnitude shares turns 46.3% into ~10.4%. The 74.5% ceiling is weighted by the
real distribution, so pooled-truth-against-ceiling was never apples to apples.
Report per-bin, or report the reweighted number, or say which one it is.

**Any number that gates a decision gets reported with its budget and its spread.**
An M3 gate figure of 1.20% was reported as a single draw without its step count;
later 6k runs read 2.2-2.8% and were mistaken for a regression, then for
nondeterminism. It cost several sessions to establish that the number came from a
30k run. Actual MPS run-to-run drift on this gate is ~0.06pp. State the budget,
state the spread over seeds, or the number is not a result.

## Style

- Type hints on public functions
- Docstrings state shapes
- No config magic — explicit dataclasses, one YAML load at entry
- Prefer boring code; this is a research repo where debuggability beats elegance
