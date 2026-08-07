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

- **distance to consumer** — +3.7 / +1.2 / −0.6 / −2.6 at d = 1…4. Decays.
- **operand provenance** — largest where the prefix determines least
  (both-leaves +3.1), inverted where it determines most (both-from-earlier:
  causal recovers the clean step at 9.1% against 0.6% chance).
- **pad-free subset** — the gap is *larger* without padding (+2.9 vs +2.3), so
  the softmax-normalisation confound is not producing it.
- **chance** — measured as a permutation null: RESULT 0.6–0.7%, operands 0.0%.

## The effect size, with all three qualifications

**Raw measured: +2.3pp. Recalibrated: +4.8pp. Single seed.**

Every one of those needs its qualification stated rather than buried:

1. **+2.3pp is what was measured** — 2000 held-out corrupted traces, oracle block
   selection (the corrupted index is known and erased, so this is an *upper
   bound* on what uncertainty-steered selection could achieve).
2. **+4.8pp assumes the handicap is additive.** The global arm carries a
   condition-level penalty that has nothing to do with downstream evidence: it
   is −2.5pp in the no-pin null. Diagnosed, not assumed — shielding the erased
   block from its own context changes nothing, and crossing conditioning against
   mask shows either departure from the training configuration costs ~5pp alone
   and both together cost no more. Subtracting it gives +4.8pp, and a second
   independent estimate from the redirected pin gives +4.4pp.

   **A regime-mix sweep has since tested the additive assumption and it held
   up.** Across a 5× change in refine-mode training exposure the handicap did not
   move (−2.3 → −2.2) while the raw effect nearly doubled (+1.9 → +3.5): it
   behaves as a constant penalty independent of the mechanism. It also scales
   with trace length in a way that is near-identical across mixes — structural,
   not trained. What it *is* turned out to be the reverse of the guess: global
   sits flat at chance while causal climbs with more prefix, so it is a
   prefix-exploitation deficit rather than an irrelevant-context penalty.
3. **Seed spread is large, and is the binding limitation.** Two p=0.50 runs give
   +3.5pp and +1.6pp raw. That spread is as wide as a four-point mix sweep's
   entire range, so **no mix comparison in this repo is currently distinguishable
   from scatter**. Multi-seed runs at a fixed mix are the next measurement, and
   until they exist the effect size should be read as "a few points, direction
   secure, magnitude not".

   One encouraging detail: the same two runs give +5.7pp and +4.8pp *recalibrated*
   — half the raw spread, because the handicap moved with the effect. If that
   survives more seeds, the recalibrated figure is a lower-variance estimator and
   not merely a fairer one.

## Settled, provisional, and weak

**Settled.** The mechanism. Downstream latent state causally supports
reconstruction of an earlier erased block, the support decays with distance, and
it reverses direction when the downstream evidence is redirected. The harness is
validated against a stub denoiser, the null is measured rather than assumed, and
chance is a permutation null rather than an analytic guess.

**Provisional.** The magnitude. The additivity assumption behind the recalibration
has survived a direct test, but seed-to-seed spread on the raw effect is ±1pp or
more and no configuration has been run at multiple seeds. Read the magnitude as
"a few points" and nothing finer.

**Weak, and under investigation.** Absolute performance. Global reconstructs the
correct result on 3.6% of traces against a 0.7% chance rate — 5× chance, not 50×.
The mechanism exists and is exploited poorly. Two suspects: the 9.7% regime-B
training share (the mix sweep is testing this now), and the absence of question
conditioning, without which nothing determines the operands and they sit at
chance in every arm.

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

# 3. Denoiser — 4 layers, 20k steps, the 90/10 regime A/B mix
PYTHONPATH=src python3 -m adze.train.train_denoiser configs/debug.yaml \
    --steps 20000 --batch 256 --lr 1e-3 --denoiser-layers 4 \
    --mixed --regime-b-prob 0.10

# 4. THE CENTRAL EXPERIMENT
PYTHONPATH=src python3 scripts/m7_central.py --traces 500
```

**The reproduction gate.** Step 4 at `--traces 500` should print:

```
   condition    exact   RESULT  operands
        none    19.6%    21.2%     86.6%
      causal     0.2%     1.2%      1.8%
      global     0.0%     4.0%      0.0%

  GAP (global - causal)   RESULT +2.8%
  PAIRED (McNemar) on RESULT: global-only 20, causal-only 6, chi2 = 6.50, p < 0.05
```

A fresh training run has its own seed-level drift, so read this as agreement
within a spread rather than as an identity. If the two disagree by more than the
effect being measured, nothing downstream is readable — and that, rather than the
number, is what the gate is for.

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
prediction before the data existed: distance decaying toward the handicap,
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
in a way it does not explain. The distance decay measured here is a candidate
explanation: the benefit of seeing downstream is real but reaches only a step or
two, so where dependency structure is shallow a local window captures everything.

See `docs/positioning.md` §3 for the prior-art survey and §4 for the claim
wording, including the claims deliberately not made.
