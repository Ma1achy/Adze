# Adze

A latent diffusion reasoner. Reasoning happens in continuous latent space; text
exists only at the interfaces. A normal LLM writes like a typewriter — one token
at a time, left to right, every keystroke permanent, and an error made in step
two is load-bearing by step five. Adze drafts reasoning steps as continuous
vectors block-causally, then refines them globally so that earlier steps can be
revised in light of later ones, and only converts to text once the thinking is
done. Thinking is not typing; this separates the two.

---

## The result

**The control is what stands. The effect size is still moving.**

The experiment erases one step of a corrupted reasoning trace and regenerates it
under two attention masks that differ *only* in whether the model can see later
steps. The corruption is inconsistent by construction: one intermediate result is
changed and nothing downstream is recomputed, so later steps still consume the
original correct value. That contradiction is the only evidence the error exists,
and it lies downstream.

The decisive control does not remove the mechanism — it **moves** it, and the
effect follows:

| condition | what downstream evidence points at | gap vs clean | gap vs corrupted |
|---|---|---|---|
| root corrupted | nothing — no consumer, no pin | −2.5% \*\*\* | — |
| early corrupted | the **clean** step | **+2.3%** \*\*\* | — |
| consistent corruption | the **corrupted** step | −0.5% n.s. | **+1.9%** \*\*\* |

Same model, same erased block, same noise seed. The third row perturbs a step and
recomputes everything downstream from it, so downstream now agrees with the
corruption instead of contradicting it. Global's advantage against the clean step
vanishes and reappears, significantly, against the corrupted step instead.
χ² = 21.25, global-only 49 against causal-only 12.

An effect that *tracks* the mechanism when the mechanism is relocated is a
stronger claim than one that merely disappears without it.

Four supporting stratifications, **all predicted before the data**:

- **operand provenance is the axis** — the gap is +3.00% ± 0.29 where the prefix
  determines least (both-leaves), +2.10% ± 0.57 with one leaf, and **−2.86% ±
  1.29** where the prefix determines most (both-from-earlier, where causal
  recomputes the step at 9.1% against 0.6% chance).
- **no measured reach limit — over a range of four.** The advantage holds at
  every distance the tree generator produces: one-leaf at **d = 4, the furthest
  cell, is +2.59% (z = 3.00)**. That cell holds ~117 traces per seed, and d = 4 is
  the maximum, so this is *not measured beyond four* rather than *shown absent*.
- **distance to consumer was a proxy for provenance, not a separate finding.**
  Pooled, the profile reads +2.9 / +2.8 / +0.3 / +0.3 at d = 1…4 and looks like a
  window of two. But the near cells are 76–92% both-leaves and the far cells are
  ~30% both-from-earlier, and **composition with no distance term at all predicts
  +2.79 / +2.82 / +0.86 / +0.75** — leaving residuals of +0.12 / +0.03 / −0.57 /
  −0.47, every one inside its error bar. Within a fixed class the advantage
  survives at every distance measured: one-leaf at **d = 4 is +2.59% (z = 3.00)**,
  the cell that read null when pooled. See RESULTS, session 22.
- **operand provenance** — largest where the prefix determines least
  (both-leaves +3.1), inverted where it determines most (both-from-earlier:
  causal recovers the clean step at 9.1% against 0.6% chance).
- **pad-free subset** — the gap is *larger* without padding (+2.9 vs +2.3), so
  the softmax-normalisation confound is not producing it.
- **chance** — measured as a permutation null: RESULT 0.6–0.7%, operands 0.0%.

## The effect size, with its spread

> **+2.5pp ± 0.6pp raw. +4.4pp ± 0.7pp recalibrated. Six seeds, one configuration,
> against a permutation chance rate of 0.6pp, under oracle block selection.**

| seed | causal | global | effect | handicap | recalibrated |
|---|---|---|---|---|---|
| 0 | 1.1% | 4.6% | +3.45% | −2.25% | +5.70% |
| 1 | 0.9% | 2.4% | +1.50% | −2.50% | +4.00% |
| 2 | 0.9% | 3.2% | +2.35% | −1.70% | +4.05% |
| 3 | 1.2% | 3.8% | +2.55% | −1.65% | +4.20% |
| 4 | 1.0% | 3.6% | +2.60% | −1.30% | +3.90% |
| 5 | 1.1% | 3.7% | +2.60% | −2.10% | +4.70% |

All six positive, minimum +1.50pp. Causal is stable at 0.9–1.2%; global carries
all the variance. Three qualifications, stated rather than buried:

1. **Oracle block selection.** The corrupted index is known and erased, so this is
   an *upper bound* on what uncertainty-steered selection could achieve.
2. **The recalibration is a fairness correction, and only that.** The global arm
   carries a condition-level penalty unrelated to downstream evidence: −1.9pp ±
   0.4pp measured where no pin exists. It is diagnosed rather than assumed —
   shielding the erased block from its own context changes nothing, and crossing
   conditioning against mask shows either departure from the training
   configuration costs ~5pp alone and both together cost no more. It also survived
   a direct test: across a 5× change in refine-mode training exposure it did not
   move, so it behaves as a constant independent of the mechanism.

   What it *is* turned out to be the reverse of the obvious guess. Global's rate
   is flat in prefix length while causal's climbs 1.3% → 7.8%, so it is not a
   noise penalty from a wider receptive field — **global does not use the prefix
   at all**, in either condition. The two arms exploit disjoint sources.

   A tempting further claim is **withdrawn**: at n=2 the recalibration appeared to
   halve the variance, which would have made it a control variate. At n=6,
   corr(handicap, effect) = +0.21 and sd(recalibrated)/sd(raw) = 1.10 — it
   slightly *increases* variance. Subtraction only reduces variance when corr >
   sd(handicap)/(2·sd(effect)) = 0.355, and 0.21 is below it.
3. **One configuration.** All six seeds share p=0.50, 20k steps, 4L×128. A sweep
   over the regime A/B mix was run and **withdrawn**: the two points independent
   of this distribution, p=0.10 (+1.9pp, −0.98σ) and p=0.75 (+2.3pp, −0.34σ), both
   fall inside it, so the sweep could not be distinguished from scatter. Its
   p=0.50 point is seed 0 itself and so carries no evidence either way. Regime B's
   cost at matched compute is small and separately measured — 1.5pp of draft
   quality.

## Settled, provisional, and weak

**Settled.** The mechanism. Downstream latent state causally supports
reconstruction of an earlier erased block, the support is concentrated where the
prefix underdetermines the step, and it reverses direction when the downstream
evidence is redirected. The harness is validated against a stub denoiser, the null is measured rather than assumed, and
chance is a permutation null rather than an analytic guess.

**Settled, as of six seeds.** The magnitude, at this configuration: +2.5pp ± 0.6pp
raw, +4.4pp ± 0.7pp recalibrated, every seed positive. It is not settled that this
is the *best* configuration — one mix sweep was withdrawn for being inside seed
noise — but the number itself now has a spread attached.

**Weak, and now diagnosed.** Absolute performance. Global reconstructs the correct
result on ~3.6% of traces against a 0.6% chance rate — about 6× chance, not 60×.
The mechanism exists and is exploited poorly.

The regime-B training share is **eliminated** as the cause: a 5× change in it left
the handicap untouched. What the measurements point at instead is that **the two
arms use disjoint information sources**. Causal's rate scales with prefix length
and ignores downstream evidence; global's scales with downstream evidence and is
flat in prefix length. Neither combines them, and an ideal refine mode would.

A candidate mechanism sits in the training signal rather than in capacity: regime
B erases a mean of 2.24 blocks from traces holding ~4.4, so roughly half the
prefix is itself erased on a typical refine step and the model never sees a
reliable clean-prefix-to-erased-block mapping. That points at changing *what*
regime B erases rather than how often it fires. Untested.

The other standing suspect is the absence of question conditioning, without which
nothing determines the operands — and they sit at chance in every arm.

Nothing here is snapped, filtered, retried or cleaned. A regenerated block that
decodes to garbage is counted as garbage.

### One thing the baseline is not

The `none` condition scores 21.7%, far above either treatment arm, and it is
**not** a no-revision floor. The VAE was trained on valid arithmetic only, so it
projects an off-distribution corrupted step onto the nearest valid one and
silently repairs 19.6% of corruptions on round-trip. The treatment arms erase the
block and discard that free repair. `none` measures the round-trip; the reference
for the treatment arms is chance.

---

## Reproducing the M7 result

Checkpoints are not in the repo. From a clean clone the full path is roughly
three hours on an Apple-silicon laptop, most of it the VAE.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
PYTHONPATH=src python3 -m pytest           # 97 passed, 2 skipped

# 1. VAE — 4 layers, not debug.yaml's 2, and 60k steps. Both are budget
#    lessons paid for once already; see CLAUDE.md.
PYTHONPATH=src python3 -m adze.train.train_vae configs/debug.yaml \
    --steps 60000 --n-train 60000 --n-val 2000 --batch 256 --vae-layers 4

# 2. Latent cache (the VAE writes one; rebuild explicitly if needed)
PYTHONPATH=src python3 scripts/build_cache.py \
    --checkpoint checkpoints/vae_cap100_d16.pt --n-train 60000

# 3. Denoiser — 4 layers, 20k steps, the headline configuration.
#    --seed 0..5 gives the six runs the headline averages over.
PYTHONPATH=src python3 -m adze.train.train_denoiser configs/debug.yaml \
    --steps 20000 --batch 256 --lr 1e-3 --denoiser-layers 4 \
    --mixed --regime-b-prob 0.50 --seed 0

# 4. THE CENTRAL EXPERIMENT — effect and handicap
PYTHONPATH=src python3 scripts/m7_central.py --corrupt early --traces 2000 \
    --denoiser checkpoints/denoiser_cap100_d16_L4_mixedP50.pt --out runs/seed0_early.json
PYTHONPATH=src python3 scripts/m7_central.py --corrupt final --traces 2000 \
    --denoiser checkpoints/denoiser_cap100_d16_L4_mixedP50.pt --out runs/seed0_final.json

# 5. The headline, once several seeds exist
PYTHONPATH=src python3 scripts/m7_seeds.py runs/seed*_early.json
```

**The reproduction gate.** A single seed should land inside the measured
distribution — **effect +2.5pp ± 0.6pp, handicap −1.9pp ± 0.4pp** — with causal
near 1.0% and global near 3.5%. Across the six recorded seeds the effect ranged
+1.50 to +3.45pp, so a single run landing anywhere in that band reproduces.

This is a distribution, not an identity. A run outside it by more than the effect
itself means nothing downstream is readable, and that — rather than any particular
number — is what the gate is for. Seed spread here is a quarter of the effect, so
**do not compare two single-seed runs and conclude anything.** This repo did
exactly that once and had to withdraw a four-point sweep.

**The controls**, once the gate holds:

```bash
# The null, and why it is not zero: shielding, then conditioning x mask
PYTHONPATH=src python3 scripts/m7_shield.py

# Every stratification. Sampling dumps records; all re-cutting is CPU-only.
PYTHONPATH=src python3 scripts/m7_central.py --corrupt final      --traces 2000 --out runs/final.json
PYTHONPATH=src python3 scripts/m7_central.py --corrupt early      --traces 2000 --out runs/early.json
PYTHONPATH=src python3 scripts/m7_central.py --corrupt consistent --traces 2000 --out runs/consistent.json
PYTHONPATH=src python3 scripts/m7_strata.py runs/*.json
```

`--corrupt consistent` is the redirected pin. `m7_strata.py` prints the distance
profile, the provenance table, the pad-free subset, every rate against its
permutation chance rate, and the recalibration against the measured handicap.

---

## Repo map

| | |
|---|---|
| `docs/design.md` | the frozen v0 design — what and why. The reference. |
| `docs/build-plan.md` | the milestone sequence, M0 through M7. |
| `docs/endpoint.md` | v2. Learned segmentation, byte-level input, adaptive compute. **Plan only, gated on M7** — nothing in it is in scope. |
| `docs/positioning.md` | prior art and claim wording for the writeup. Read §4 before describing this work. |
| `RESULTS-*.md` | the running record, per session. Numbers live here; this README points at them rather than restating them. |
| `CLAUDE.md` | invariants, the not-in-v0 list, and the retired findings. |

```
src/adze/
  data/     synthetic traces, corruption, the M7 controls, dataset
  model/    VAE, denoiser, attention masks
  train/    regime A, regime B, the A/B mix
  sample/   draft (pass one), the SDE sampler
  eval/     gates, the standard readout, central.py, strata.py
configs/    debug.yaml (fast iteration), v0.yaml (sized for another machine)
scripts/    diagnostics and experiment drivers, one per question
tests/      acceptance criteria, one module per milestone
```

---

## On reading the commit history

The plan documents in this repository **predate their results**, and that is
visible only from the history. Every stratification above was named as a
prediction before the data existed: support falling off with distance (it does,
though the decomposition later showed provenance carrying it),
provenance reversing where the prefix alone determines the step, the
both-from-earlier cell going to causal, and the redirected pin switching the
advantage to the corrupted target. All four held.

Ten stratifications found after the fact would be p-hacking. Ten stated in
advance that then hold is the opposite, and it is part of why the result is worth
believing. The commit sequence is the record of which came first.

`CLAUDE.md` also carries a **retired findings** section — measurements that
looked like the thing and were not, each kept with the reason it failed. Live
dimension shrinkage, the sampler reproducing the marginal, the exposure-bias
story, the prefix information estimate. Four confident wrong turns, recorded so
they are not taken again.

## Related work

This is not framed as a first. The nearest published result — Speculative
Correction (arXiv 2608.02625) — reports that local refinement captures much of
the gain on some benchmarks while global helps clearly on others, task-dependent
in a way it does not explain. An earlier draft of this README offered the
distance window as the explanation. **That explanation is withdrawn with the
window.** A provenance-based account is available as a conjecture — task
dependence would track how much of a task's reasoning each step's own prefix
determines — but testing it needs their data cut by dependency type, and nobody
has done that.

## A methodological finding, transferable beyond this work

**Where consumer distance is induced by a fixed evaluation order over a
hierarchical structure, distance and dependency type can be strongly coupled.**
The coupling is in the *linearisation*, not in the tree — a tree fixes ancestry,
and distance only appears once it is laid out in an order. Post-order over an
expression tree is a case where the coupling is tight: a step whose operands both
come from earlier steps roots a larger subtree, so its parent cannot be emitted
until that subtree has been, and its consumer is further away.

Anyone reporting a distance profile on such data without decomposing by
provenance can find a spurious window. This repository found one, at z = 6.06 and
z = 4.21 over six seeds, made it the headline, and believed it for several days.
Six seeds fixed the sampling noise and left the confound untouched.

The concrete measure of how badly the two are welded: **far + both-leaves is 43.1%
of a decorrelated generator's records and ~69 per seed here.** A both-leaves node
has a minimal subtree, so under post-order its parent follows almost immediately.
That cell is nearly absent by construction, not by sampling, so more data from
this generator would not have recovered it.

See `docs/positioning.md` §3 for the prior-art survey and §4 for the claim
wording, including the claims deliberately not made.
