# Downstream latent state causally supports reconstruction of an earlier reasoning step, where the prefix underdetermines it

**Draft.** Markdown, not LaTeX. Numbers are from the committed record in
`RESULTS-2026-08-07.md`; every figure here is traceable to a run in `runs/`.

---

## 1. The question

Refinement architectures for diffusion language models share an assumption:
regenerating a token or a block with *later* context in view is better than
regenerating it with only earlier context. It is the reason global refinement
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

χ² = 21.25 on the paired outcomes; global-only 49 against causal-only 12.

The third row is the load-bearing one. Consistent corruption does not remove the
pin — it **redirects** it. Global regeneration then loses against the clean value
and wins against the corrupted one, which is the direction the mechanism predicts
and the opposite of what a general "global is better" account predicts.

**An effect that tracks the mechanism when the mechanism is relocated is a
stronger claim than one that merely disappears without it.**

## 4. Effect size

Six independently trained seeds, identical configuration, 2000 held-out traces
each, paired within checkpoint:

- **raw +2.51pp ± 0.62**, every seed positive
- **recalibrated +4.42pp ± 0.69**, against the measured no-pin handicap
- chance **0.63%**, measured as a permutation null grouped by block index — not
  an analytic guess

The recalibration subtracts the handicap measured in the root condition, where no
pin exists: global carries a condition-level penalty there unrelated to the
mechanism. It is reported as a fairness correction and nothing more. A test
across seeds for whether it is also a lower-variance estimator — a control
variate — **failed**: corr(handicap, effect) = +0.210 against a required 0.355,
so the variance claim was dropped rather than softened.

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

### The distance profile was this effect in disguise

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

Every residual is inside its error bar. And within a fixed class the advantage
persists at every distance measured — one-leaf at **d = 4 is +2.59% (z = 3.00)**,
the very cell that reads null when pooled.

**There is no measured distance horizon out to d = 4.** Distance was a proxy for
provenance.

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

What the decomposition establishes is that distance adds nothing once provenance
is known **on this generator**. It does not establish that no horizon exists. That
requires a generator whose consumer assignment does not inherit the evaluation
order, and the status of that work is stated once, in §12: the generator is built
and verified, a model has been trained on it, and its repair effect is at chance,
so it has produced no interpretable distance measurement yet.

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

**Two results carry state across denoising steps, which is the mechanism this
work's refinement loop lacks.** Both are prior art for the fix rather than for the
measurement.

**RCD** (arXiv 2601.22954) converts discarded token representations into
contextual residuals and injects them into the next denoising step, so information
that would otherwise be thrown away when a token is committed survives into the
following round.

**Relay** (arXiv 2605.22967) carries last-layer hidden states forward through a
differentiable per-token channel, trained with truncated backpropagation through
time, and frames the problem as the **"hard reset"** between denoising rounds.

That framing is the same problem this work measured from the other side. §1a of
`docs/scratchpad-reach.md` found the refinement loop **shreds** context rather
than relaying it: a pass regenerates a neighbour and replaces it with something
usually wrong, so `preserved` collapses from 0.10% to 0.00% over three passes and
the effect falls from +2.51% to +0.93%. Nothing is carried forward because each
pass restarts from the committed discrete output.

Relay's fix — carry the continuous state forward rather than recompute it from the
discrete output — is the content/carrier separation this repository had proposed
for its own relay experiment, already implemented and trained. Any future version
of that experiment should build on it rather than reinvent it.

## 8. Negative results

Absolute exploitation is weak, and five attempts to improve it by changing the
training signal produced four nulls and one false positive. They are reported
because they constrain the explanation.

| hypothesis | outcome |
|---|---|
| regime A/B mix share | null — a 5× change left the handicap unmoved |
| \|S\| mismatch between training and inference | null — accuracy declines monotonically with \|S\|, best at \|S\| = 1 |
| prefix reliability during refinement training | null — structured erasure left the prefix slope within 1σ of zero |
| mode partition (regime A always prefix-only, regime B sometimes pinned) | null — removing the pin from half of refinement training gave convergence at chance, not transfer |
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

All four held. Ten stratifications found after the fact would be p-hacking; four
stated in advance that then hold is the opposite.

Four confident wrong turns are also kept in the record with the reason each
failed: live-dimension shrinkage, the sampler "reproducing the marginal", an
exposure-bias story, and a prefix information estimate that was a kNN artifact.

## 10. A methodological finding

**The confound is in the LINEARISATION, not in tree structure itself.** An earlier
draft of this section said "in tree-structured reasoning tasks, distance-to-
consumer is confounded with dependency type by construction". That is an
overclaim, and the correction matters because it says what a fix must change.

A tree does not determine linear consumer distance. It determines *ancestry*.
Distance only appears once the tree is laid out in an evaluation order, and it is
that layout which couples the two variables. Under **post-order** emission the
coupling is tight and predictable: a step whose operands both come from earlier
steps roots a larger subtree, and its parent cannot be emitted until that whole
subtree has been, so its consumer is further away. A step with two literal
operands roots a minimal subtree and its parent often follows immediately.

The same tree under a different linearisation gives a different distance profile
from identical dependency structure. So the claim is:

> Where consumer distance is induced by a fixed evaluation order over a
> hierarchical structure, distance and dependency type can be strongly coupled,
> and post-order over an expression tree is a case where they are.

The consequence is specific and checkable: **a distance profile computed on such
data can show a spurious window**, because the near cells are dominated by the
dependency type where downstream evidence helps most and the far cells by the type
where it helps least. This work found that window at z = 6.06 and z = 4.21 over
six seeds and led with it.

The correction is not more seeds. Six seeds is what made it look unassailable.
**Replication fixes sampling noise; only decomposition fixes a confound.**

We suggest that any distance or reach profile reported on structured reasoning
data be accompanied by the same decomposition, and — where a genuine reach claim
is intended — a generator whose consumer assignment does not inherit the layout.

## 11. Method note: why the interior band excludes the ends

The decorrelating generator's cross-tab is reported over an **interior band** —
steps at index >= 2 whose consumer draw has the full bounded range available. A
reader is entitled to ask why the ends are dropped.

Position bounds both variables near the ends, from opposite directions. A step at
index 0 or 1 cannot have two operands from earlier steps, so its provenance class
is constrained by where it sits. A step within `DISTANCE_MAX` of the end cannot
have a distant consumer, so its distance is constrained by where it sits. In those
regions the two variables are both partly functions of position.

**An earlier draft said "no construction avoids it". That was wrong, and it
contradicted the design it was defending.** The interior band IS a construction
that avoids boundary contamination — that is precisely what it does. Other
constructions exist too: a cyclic or wrap-around dependency structure would remove
the boundary entirely, and padding the chain beyond the analysed region would push
the bounded region outside it. What is true is narrower:

> Given a linear chain with hard ends and a bounded consumer draw, positions
> within `DISTANCE_MAX` of either end have a restricted range, so a decorrelated
> estimate requires either excluding them or a different chain topology. This
> generator excludes them.

Excluding is honest; including them and adjusting would be fabrication. The band
is defined before the data is generated, by the generator's own parameters rather
than by anything measured, so it cannot be tuned to a result. Steps outside it are
still generated, still trained on, and still valid — they are excluded from the
*distance analysis* only, and their count is reported.

The same applies to distances past `DISTANCE_MAX`, which occur only where capacity
forced an overshoot. They are printed, flagged, and excluded from the swing.

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
- **Distance and provenance are correlated in this generator**, and that is the
  main limitation. Near cells are 76-92% both-leaves and far cells ~30%
  both-from-earlier, so no distance range is available at fixed provenance.

- **The class effects are themselves estimated from distance-imbalanced cells,
  and at d = 1 one class has ZERO SUPPORT.** both-from-earlier is 0.0% of the
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

- **The decorrelated regime has no interpretable measurement yet.** A model has
  been trained on it and its repair effect is +0.43% against a 0.64% permutation
  chance, with causal at chance — so the per-distance profile it yields is noise
  and no reach conclusion is drawn from it. The generator is verified (swing
  0.05pp) and the pipeline passes its gates; what is missing is a model that
  performs the task well enough on that data to be measured. Every distance claim
  in this paper therefore still rests on the confounded generator, with the
  decomposition above as the correction.

## Claim wording

We found no prior downstream-state provenance control for latent refinement.
Prior work evaluates correction through end-task comparisons, refinement-scope
ablations, or descriptive intermediate trajectories. We are not aware of a prior
intervention that holds an erased block's direct inputs fixed while manipulating
only the provenance of downstream latent state.

Deliberately not claimed: any "first" — not the first mechanism test of diffusion
refinement, not the first causal ablation of refinement, not the first causal
intervention in a diffusion model. Activation patching, interchange interventions
and causal tracing are standard; the contribution is the application to
refinement, and it is narrow.
