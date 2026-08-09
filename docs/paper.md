# Downstream latent state causally supports reconstruction of an earlier reasoning step, where the prefix underdetermines it

**Draft.** Markdown, not LaTeX. Numbers are from the committed record in
`RESULTS-2026-08-07.md`; every figure here is traceable to a run in `runs/`.

---

## 1. The question

Refinement architectures for diffusion language models share an assumption:
regenerating a token or a block with *later* context in view can provide
corrective evidence unavailable to the prefix. It is the reason global refinement
passes exist, and the reason "revision" is thought to differ from ordinary
generation.

The assumption is rarely tested directly. It is normally evidenced at the outcome
level — a system with a global refinement pass scores better on a benchmark than
one without. That comparison changes the scope, the mask and often the training
configuration together, so it establishes that the *configuration* helps without
identifying what carries the benefit.

The question here is narrower and mechanistic:

> Hold an erased block's direct inputs and its corruption fixed. Intervene only
> on whether the downstream latent state was computed from the clean or the
> corrupted predecessor. Does recovery of the erased block follow?

## 2. Setup

A latent diffusion reasoner. Reasoning happens in continuous latent space; text
exists only at the interfaces. A VAE encodes each reasoning step into K = 4
latents of D = 16; a DiT-style denoiser drafts block-causally and refines
globally, under rectified flow with velocity prediction.

The data is synthetic arithmetic traces — a chain of steps in which each step's
operands are either literals or the *results of named earlier steps*. That
provenance is recorded, not inferred, so the dependency structure of every trace
is exactly known. This is what makes the intervention possible: for any step we
know precisely which later steps consume its result, and at what distance.

The experiment corrupts one step's result, erases that step's latents entirely,
and regenerates them under two masks that differ **only** in whether positions
after the erased block are visible. Everything else is held identical — the same
traces, the same corrupted indices, the same erasure noise, the same schedule —
so the comparison is paired within each example.

Block selection is **oracle**: the corrupted index is known. Every figure is
therefore an upper bound on what uncertainty-steered selection could achieve, and
is reported as one.

## 3. The control: a redirected pin

The mechanism, stated so it can fail: a later step that consumes the erased
step's result *states that result*, so downstream context identifies it. Call
that later step the **pin**.

If the pin is what carries the benefit, then moving the pin should move the
effect. Three conditions, on the same architecture and the same checkpoint:

| condition | what downstream points at | gap vs clean | gap vs corrupted |
|---|---|---|---|
| **root corrupted** — last step, nothing consumes it | nothing; no pin | −2.5% | — |
| **early corrupted** — a step with consumers | the **clean** value | **+2.3%** | — |
| **consistent corruption** — corrupt, then recompute everything downstream | the **corrupted** value | −0.5% n.s. | **+1.9%** |

**Which sample those statistics come from:** χ² = 21.25 with global-only 49
against causal-only 12 is McNemar with continuity correction on the paired RESULT
outcomes of the **`early` condition** — row two of the table — over n = 2000
held-out corrupted traces from a single checkpoint, comparing global against
causal on the same traces with the same erasure noise. It is not pooled across
conditions and not pooled across the six seeds; the six-seed figure is §4's.

**What consistent corruption does.** It changes the target step's *result* while
leaving its *operands* untouched, so that step is deliberately made locally
inconsistent — its own arithmetic no longer checks — and every descendant is then
recomputed from the changed value so the rest of the trace is internally
consistent with it. `is_valid()` is therefore False on the result, and must be:
leaving block b's operands alone is exactly what lets a causal regeneration
recompute the clean value from the prefix while the pin points at the corrupted
one.

The third row is the load-bearing one. Consistent corruption does not remove the
pin — it **redirects** it. Global regeneration then loses against the clean value
and wins against the corrupted one, which is the direction the mechanism predicts
and the opposite of what a general "global is better" account predicts.

**An effect that tracks the mechanism when the mechanism is relocated is a
stronger claim than one that merely disappears without it.**

## 4. Effect size

Six independently trained seeds, identical configuration, 2000 held-out traces
each, paired within checkpoint:

- **the headline is the RAW effect: +2.51pp ± 0.62**, every seed positive
- a **baseline-adjusted estimate** of +4.42pp ± 0.69 is reported alongside it
- chance **0.63%**, measured as a permutation null grouped by block index — not
  an analytic guess

The baseline adjustment subtracts the handicap measured in the root condition,
where no pin exists and global carries a condition-level penalty unrelated to the
mechanism. It is an estimate under the assumption that the handicap is additive
and condition-level, which is an assumption and not a measurement, so the raw
figure is primary and the adjusted one is secondary. A test across seeds for
whether the adjustment is also a lower-variance estimator — a control variate —
**failed**: corr(handicap, effect) = +0.210 against a required 0.355, so that
claim was dropped rather than softened.

**The absolute rates are weak and this is stated plainly.** Global reaches ~3.6%
against 0.63% chance. That is 5× chance, not 50×. The mechanism exists and is
exploited weakly.

## 5. Where the advantage lives: provenance, not distance

Each step's operands are either literals or results of named earlier steps, so
every step carries a known **provenance class** — how much of it the preceding
context already determines. Over six seeds:

| class | share | gap | ± SE |
|---|---|---|---|
| both-leaves — prefix determines least | 72% | **+3.00%** | 0.29 |
| one-leaf | 23% | **+2.10%** | 0.57 |
| both-from-earlier — prefix determines most | 5% | **−2.86%** | 1.29 |

**The advantage is largest where the prefix is least informative and reverses
where it is most informative.** In the both-from-earlier class a causal
regeneration has everything it needs to recompute the step, and does so at 9.1%
against 0.6% chance — global's extra view is not merely useless there, it costs.

### The apparent distance window is explained by provenance composition

Stratifying instead by the distance from the erased block to the step that
consumes its result gives, over the same six seeds, +2.90 / +2.84 / +0.29 / +0.28
at d = 1…4 — two strong distances and then nothing, which reads as a **window of
two**. That was this work's headline for several days.

It does not survive decomposition. The near and far cells have very different
provenance mixes: d = 1 is 76% both-leaves and 0% both-from-earlier, while d = 4
is 20% both-leaves and 31% both-from-earlier. Predicting each distance from its
class mix alone, **with no distance term of any kind**:

| d | observed | predicted from composition | residual |
|---|---|---|---|
| 1 | +2.90% ± 0.48 | +2.79% | +0.12% ± 0.48 |
| 2 | +2.84% ± 0.68 | +2.82% | +0.03% ± 0.68 |
| 3 | +0.29% ± 0.39 | +0.86% | −0.57% ± 0.39 |
| 4 | +0.28% ± 0.42 | +0.75% | −0.47% ± 0.42 |

Every residual is inside its error bar — though those bars omit uncertainty in
the estimated class gaps, so they are narrower than a formal test would allow.

And within the **one-leaf** class the advantage remains significant at the maximum
measured distance, **d = 4: +2.59% (z = 3.00)** — the very cell that reads null
when pooled. That result rejects a hard two-step horizon. It is stated for
one-leaf specifically because the other two classes do not have adequate support
at d = 4: both-leaves holds ~23 records per seed there and both-from-earlier ~36.

**No hard two-step horizon survives.** The window was a marginal profile over
cells whose provenance composition changes with distance.

**And d = 4 is the maximum this generator produces.** "Every distance" means every distance
the tree generator produces, and its furthest cell holds ~117 traces per seed. The
statement is true and narrower than it sounds: nothing here rules out a reach
limit at d = 6, or d = 10. **"No measured reach limit" means *not measured*, not
*shown absent*.** A decorrelating generator exists and extends the clean range to
d = 1..5, but has produced no interpretable measurement yet — §12.

This is a consolidation rather than a loss. It collapses what looked like two
findings — a distance window and a source split — into one: global wins where the
prefix determines least, causal wins where it determines most, and the "distance
decay" is what that looks like plotted against a variable correlated with
provenance. One mechanism, two views.

**Caveat on the decomposition's strength.** The class gaps and the distance
profile come from the same records, so the inputs are not independent, and the
thin cells (both-leaves at d = 4 is ~23 traces per seed) are not printed as rates.
Worse, the class effects are estimated from distance-imbalanced cells — at d = 1,
both-from-earlier has **zero support** — so no estimator can separate the two
variables there; see §12.

The decomposition shows that changing provenance composition is sufficient to
account descriptively for the pooled distance profile. Because the same records
estimate the class effects and the distance outcomes, that table alone is not a
formal test that the conditional distance effect is zero. **Both tests that would
make it one have now been run**, on the same six seeds.

**Conditional regression.** A linear probability model on the paired difference
`1[global correct] − 1[causal correct]`, fitted per seed and averaged, so the
uncertainty is between-seed:

| model | slope in distance, per block | 95% CI (t, 5 df) | z |
|---|---|---|---|
| `y ~ 1 + d` (marginal) | **−0.905% ± 0.245** | [−1.54, −0.28] | −3.69 |
| `y ~ 1 + d + provenance` (conditional) | **−0.291% ± 0.391** | [−1.30, +0.72] | −0.74 |

Holding provenance fixed removes **68%** of the marginal slope and takes it from
significant to not distinguishable from zero.

**Leave-one-seed-out prediction.** Class gaps estimated on five seeds, applied to
the held-out seed's own composition, so no record contributes to both sides. The
out-of-sample residuals are +0.12% ± 0.52, +0.03% ± 0.70, −0.57% ± 0.43 and
−0.47% ± 0.50 at d = 1…4 — every one inside its bar. **The decomposition survives
out of sample**, which the in-sample table could not establish.

**What this does and does not license.** The conditional slope's interval
**contains the marginal slope**. So this is a failure to reject, not an
equivalence result: the design cannot exclude a conditional distance effect as
large as the one originally claimed. What it establishes is that provenance
composition predicts the profile out of sample, and that once provenance is in the
model the distance term is no longer supported by the data. Those are the two
claims §5 makes.

**This decomposition is POST HOC.** It was not among the predictions registered
before the data existed. It is reported as a correction to a claim this work made,
which is a different epistemic status from the pre-registered cuts in §9.

## 6. Disjoint sources

Two routes could in principle reconstruct an erased step: the **prefix**, from
which the step can be recomputed, and the **pin**, which states its result
outright. A conflict design separates them — consistent corruption makes the
prefix imply the clean value while the pin states the corrupted one, and on 0 of
2000 records do the two coincide.

Scoring the same regenerated block against both targets:

| arm | vs clean | vs corrupt | χ² |
|---|---|---|---|
| causal | 1.4% | 0.6% | **5.03** |
| global | 0.9% | **2.5%** | **14.56** |

**Each arm follows its own source, in opposite directions, on the same records,
both significant.** Causal recomputes from the prefix; global reads the pin.

This is corroborated from the other side. Causal's accuracy climbs with prefix
length (+1.06 to +1.47pp per block across four independently trained
configurations) while global's is flat in prefix length in every one of them.
Where the pin is removed entirely — erasing everything after the target, so both
arms see identical information — global falls to chance while causal is
unchanged.

## 7. Related work, and a claim withdrawn

**Catruna & Radoi** (arXiv 2607.15893) study masked diffusion language models
mechanistically and find previous-token and next-token pathways as separate
layer-0 circuits, either able to steer the output alone, with an effective window
of about ±2 tokens and a conflict prompt that splits near-equally between the two
sources.

**An earlier draft of this work cited that as convergent evidence — two settings
independently finding a ±2 window. That claim is withdrawn.** The window here was
not a distance effect, so there is no shared phenomenon for the two numbers to be
evidence of, and their agreement is coincidence unless something else connects
them. The paper stays in related work for the finding that does still touch this
one: **separate pathways rather than one merged mechanism**, which is the same
shape as the source split in §6, reached by a different method. Their circuit
*copies* from a matching position where using the prefix here requires
*recomputing* arithmetic, so even that is an analogy and not a replication.

**Speculative Correction** (arXiv 2608.02625) reports that local refinement
captures much of the gain on some benchmarks while global helps clearly on
others, task-dependent in a way it does not explain. **This work no longer offers
an explanation of that result.** The window-based account — shallow dependencies
fit inside a local view, deep ones fall outside it — was the natural reading of a
distance profile that turned out to be provenance.

A provenance-based account is *available* and is stated here as a conjecture
rather than a finding: task-dependence would track how much of a task's reasoning
each step's own prefix already determines, not how far its dependencies reach.
Testing it needs their data cut by dependency type, which has not been done.

**Residual Context Diffusion** (arXiv 2601.22954) and **Learned Relay
Representations** (arXiv 2605.22967) preserve continuous information across
successive denoising iterations, demonstrating the utility of temporal latent
persistence. Our intervention concerns a different axis: later positions within
the generated reasoning trace. Neither work holds an erased position's direct
state fixed while manipulating whether visible later-position latents derive from
a clean or corrupted predecessor.

They are, however, direct prior art for a *fix* this work independently proposed.
§1a of `docs/scratchpad-reach.md` measured that our refinement loop **shreds**
context rather than relaying it: a pass regenerates a neighbour and replaces it
with something usually wrong, `preserved` collapses from 0.10% to 0.00% over three
passes, and the effect falls from +2.51% to +0.93%. Relay names that problem the
**"hard reset"** between denoising rounds and fixes it by carrying last-layer
hidden states forward through a differentiable per-token channel — which is the
content/carrier separation this repository had proposed for its own relay
experiment, already implemented and trained.

## 8. Negative results

Absolute exploitation is weak. Five attempts to improve it by changing the
training signal produced **four experiments that found no detectable improvement**
and one false positive. They are reported because they constrain the explanation.

**A noisy null does not establish invariance.** Each row below reports a failure
to detect, at the power available; none of them shows that the quantity is
unaffected.

| hypothesis | outcome |
|---|---|
| regime A/B mix share | no detected change — a 5× change produced no detected change in the handicap |
| \|S\| mismatch between training and inference | no detected improvement — accuracy declines monotonically with \|S\|, best at \|S\| = 1 |
| prefix reliability during refinement training | no detected improvement — structured erasure left the prefix slope within 1σ of zero |
| mode partition (regime A always prefix-only, regime B sometimes pinned) | no detected transfer — removing the pin from half of refinement training gave convergence at chance, not transfer |
| shortcut learning | **never cleanly tested** — its premise was measured false first: the pin is already absent on 59.5% of erased blocks |

One arm appeared to work: a positive prefix slope at +1.27pp, z = 1.42 on one
seed. It was reported as suggestive rather than established, seeded, and
**withdrawn** — three seeds gave +1.27, −0.96, −1.52, mean −0.40pp, z = −0.47.

The methodological lesson is recorded as a standing rule: **between-seed variance
is the error bar for any comparison between training runs.** Within-seed binomial
SE understates run-to-run uncertainty by ~2.2× here, and two single-seed runs can
read as individually significant in opposite directions around a null mean.

## 9. Pre-registration

Every stratification in §3 and §5 was named as a prediction before the data
existed, and the plan documents in the repository **predate their results** —
visible in the commit history:

- support falling off with distance (it does — though the decomposition
  later showed provenance carrying it, and the registered prediction was
  therefore right about the plot and wrong about the cause)
- provenance reversing where the prefix alone determines the step
- the both-from-earlier cell going to causal
- the redirected pin switching the advantage to the corrupted target

All four held as plots. The first did not hold as a *cause*: the composition
decomposition in §5 showed provenance carrying it.

**That decomposition is POST HOC and is marked as such wherever it appears.** It
was not registered in advance; it is a correction to a claim this work made after
seeing the data, and it carries the weaker epistemic status that implies.

Four confident wrong turns are also kept in the record with the reason each
failed: live-dimension shrinkage, the sampler "reproducing the marginal", an
exposure-bias story, and a prefix information estimate that was a kNN artifact.

## 10. A methodological finding

In this generator, distance-to-consumer is strongly associated with dependency
type. Its construction and linearisation place different provenance classes into
different distance distributions. A marginal distance profile therefore combines
any genuine reach effect with changing provenance composition. Stratification can
separate the variables where the design has overlapping support; combinations
with structurally zero support require a redesigned generator.

An earlier draft of this section claimed the confound holds "in tree-structured
reasoning tasks, by construction", and that a consumer is further away "in any
evaluation order". Both are false. **Tree structure does not determine linear
consumer distance — scheduling and sibling order do.** The result belongs to this
generator and this linearisation, not to tree-structured reasoning generally.

The consequence is specific and checkable: **a distance profile computed on such
data can show a spurious window**, because the near cells are dominated by the
dependency type where downstream evidence helps most and the far cells by the type
where it helps least. This work found that window at z = 6.06 and z = 4.21 over
six seeds and led with it.

More data does help wherever cells overlap — the one-leaf d = 4 cell is a
demonstration of exactly that. A redesigned generator is required only for
combinations with structurally zero support.

**But "the design contains this contrast" and "the design contains enough of it to
measure at reasonable cost" are different statements**, and the same cell shows
the gap: one-leaf d = 4 holds ~134 records per seed and needed all six seeds to
reach z = 4.04. It is barely estimable rather than comfortably so. Where overlap
is that thin, more data is the correct remedy in principle and an expensive one in
practice, which is why redesign was the route taken here.

**Replication reduces sampling and training-run uncertainty, but does not remove
design confounding; stratification or redesign is also required.**

We suggest that any distance or reach profile reported on structured reasoning
data be accompanied by the same decomposition, and — where a genuine reach claim
is intended — a generator whose consumer assignment does not inherit the layout.

## 11. Method note: the interior band

We restrict the distance estimand to a pre-specified interior band in which the
intended provenance and distance levels are jointly supportable. At boundary
positions, some combinations have structurally zero support; regression adjustment
cannot recover those cells without extrapolating beyond the observed design.
Boundary steps remain in training and aggregate evaluation but are excluded from
the conditional distance analysis.

Concretely: a step at index 0 or 1 cannot have two operands from earlier steps, so
its provenance class is restricted by position; a step within `DISTANCE_MAX` of
the end cannot have a distant consumer, so its distance is restricted by position.

**The interior band is one construction among several that avoid boundary
contamination** — padding, burn-in, restricting eligible target positions, or
generating longer traces and analysing an interior window would each do it. An
earlier draft said "no construction avoids it", which was both too strong and in
contradiction with the design it was defending. The band is defined before the
data is generated, from the generator's own parameters rather than from anything
measured, so it cannot be tuned to a result, and the count of excluded steps is
reported.

The same treatment applies to distances past `DISTANCE_MAX`, which occur only
where capacity forced an overshoot. They are printed, flagged, and excluded from
the swing.

## 12. Limitations

- **Scale.** GPT-2 scale at most; the reported configuration is 4 layers × 128
  wide with D = 16 latents. Nothing here establishes how the effect scales.
- **Data.** Synthetic arithmetic with exactly-known provenance. That is what makes
  the intervention possible and it is also the main threat to external validity.
- **Oracle selection.** The corrupted index is known. All figures are upper
  bounds on what uncertainty-steered selection could reach.
- **Weak absolute performance.** ~5× chance. The mechanism is real and thinly
  exploited, and five training-signal interventions failed to move it.
- **The measured distance range is 1 to 4, and the furthest cell is thin**
  (~117 traces per seed). "The advantage holds at every distance measured" is a
  statement about a short range, and is not evidence that no limit exists further
  out.
- **The current generator has SPARSE AND UNBALANCED CONDITIONAL COVERAGE.** Near
  cells are 76-92% both-leaves and far cells ~30% both-from-earlier. Some
  provenance-by-distance combinations are well supported — the one-leaf class runs
  the full range — and others are not.

- **Thin overlap is not the same as no overlap, but it costs like it.** The
  one-leaf class spans the full distance range, so the contrast exists — yet
  d = 4 holds ~134 records per seed and took six seeds to reach z = 4.04. More
  data recovers such cells in principle; the cost is why this work redesigned the
  generator instead.

- **At d = 1, one class has ZERO SUPPORT, and the class effects are themselves
  estimated from distance-imbalanced cells.** both-from-earlier is 0.0% of the
  d = 1 cell and ~30% of d = 3 and d = 4, so the provenance effect is estimated
  almost entirely from far records and then used to explain near ones. A
  conditional regression of the effect on distance and provenance jointly cannot
  separate them at d = 1 — not because the cell is noisy, but because **the design
  has no overlap there.** No estimator recovers a contrast the data does not
  contain. This is why a redesigned generator is load-bearing rather than tidy.

- **The old generator is STRUCTURALLY INCAPABLE of the decorrelated regime's
  dominant cell, not merely underpowered for it.** A both-leaves node has both
  children as leaves, so its subtree is minimal, and under post-order its parent
  follows immediately unless a large sibling subtree intervenes — right-children
  land at d = 1 by construction. So *far + both-leaves* barely exists: ~69 records
  per seed. In the decorrelated generator that same cell is **43.1%** of records,
  and it is why a within-class reweighting from the old data to the new covers
  only **52%** of them. More traces from the old generator would not help; the
  cell is nearly absent by construction rather than by sampling.

- **The decorrelated generator exists and is verified, and no model trained on it
  repairs well enough to measure.** Three now exist: +0.43% at 20k steps, +0.73%
  at 60k, and −0.20% on a length-matched B = 7 variant, against a permutation
  chance of ~0.63% and an old-regime effect of +2.51% ± 0.62. Causal sits at
  chance in all three. **Every distance claim in this paper therefore still rests
  on the confounded generator**, with the decomposition in §5 as the correction.

  Four alternatives were checked and eliminated: undertraining (3× budget bought
  +0.30pp), trace length (old-generator 7-step traces are the *strongest* cell at
  +3.50% ± 0.89, so length runs the opposite way), a degenerate checkpoint (the
  decorrelated models *draft* better than the old one — 29.8% and 31.3% matched
  truth against 23.9%), and eval-side erasure size (flat at chance across |S| 1–7).

  **The surviving explanation is the dependency structure itself, and it is
  coherent with §5.** The old generator's pin was mostly adjacent — d = 1 was 60.2%
  of records, mean distance 1.60 — and §5 shows the old effect is carried by
  d = 1–2. Post-order emission placed consumers next to their producers, which is
  simultaneously what welded distance to provenance *and* what made the pin easy to
  exploit. **The confound and the effect share a cause**, so removing the weld
  removes most of the signal.

  The consequence for the paper's scope is direct: this work measures repair **when
  the pin is adjacent**, and cannot yet extend that to when it is far.

## Claim wording

We found no prior later-position latent-state provenance control for refinement.
We are not aware of an intervention that holds an erased position's direct state
and corruption fixed while changing only whether visible later-position latents
derive from the clean or corrupted predecessor.

Deliberately not claimed: any "first" — not the first mechanism test of diffusion
refinement, not the first causal ablation of refinement, not the first causal
intervention in a diffusion model. Activation patching, interchange interventions
and causal tracing are standard; the contribution is the application to
refinement, and it is narrow.
