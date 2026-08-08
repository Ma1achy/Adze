# Scratchpad — extending the reach

**Status:** research queue, partly executed. §1 and §5 have results (see §1a, §5a). Order below is fixed — work through it in sequence.

---

## What exists right now (v0), and what doesn't

This note is written in endpoint vocabulary. **Most of that vocabulary describes components that do not exist yet.** What is actually training:

| exists in v0 | does not exist |
|---|---|
| VAE — step → K=4 latents of D=16, frozen | byte frontend, router₁ |
| 4-layer DiT denoiser, d=128, over 7×4 = 28 positions | router₂, learned segmentation |
| velocity prediction, rectified flow | the carrier lattice, `b` and `ℓ` channels |
| block-causal and global masks, 50/50 | pack / unpool, Mamba anywhere |
| SDE sampler, η=1 | adaptive ρ, adaptive R, weight sharing, hierarchy |

~400k parameters. Blocks are **newline-delimited**, extent is fixed, segmentation is a heuristic.

**Translation for reading the items below:**

- **§1 "refresh carrier states"** — there is no carrier. The v0 analogue is running the denoiser over all *block latents* each pass while re-noising only the selected ones. Same idea, different object, simpler.
- **§2 hierarchy** — needs router₂ and sections. **Not buildable now.**
- **§4 dilated heads** — buildable; an attention-mask change over 28 positions.
- **§0, §3, §7** — data and eval work, all buildable now.

Near-term queue is therefore **§0, §1, §3, §4**. Everything else waits on components that do not exist.

---

**Context:** the window is a **cliff**, not a gradient — d=1 z=6.06, d=2 z=4.21, d=3 z=0.75, d=4 z=0.67 over six seeds. Better supported than the effect size itself. Everything below is about whether that window can be widened, and whether it is a property of the model or of the task.

---

## 0. The prior question — routing horizon or structural horizon?

**This may make most of the rest unnecessary, and it should be settled first.**

In the current generator every step's result is consumed by **exactly one** parent — post-order emission over an expression tree. So a step three positions away is related only *transitively*, through a value the model would have to recompute rather than read.

So: is ±2 a **routing horizon** (the model cannot transport information further) or a **structural horizon** (there is no useful information further, and reach 2 is the correct policy)?

**Free check, existing dumps.** Condition the d=3 and d=4 nulls on the *consumer* being at that distance rather than any downstream block. A genuine consumer at d=3 giving nothing → routing. Distant blocks being mostly non-consumers → structural, and reach 2 is a correct read of the task.

**The decisive version, needs a generator change.** Build traces where a value is consumed at **multiple depths** — used by its parent *and* again five steps later. Genuine long-range evidence then exists at d=5.

- window still stops at 2 → routing horizon. The interventions below are the right response.
- window extends → the cliff was the tree structure showing through, and the current result should be restated.

This reframes what is currently established. Not *"refinement reaches two steps"* but **"refinement reaches as far as the dependency structure carries information, measured at two on a task where two is roughly all there is."** Weaker as a limitation, more honest, and it makes the multi-consumer generator the most informative next build.

---

## 1. Does reach compound with refinement passes?

**Prediction:** each pass has reach 2, but after pass one block *b* has absorbed *b+1* and *b+2*. On pass two, regenerating *b−1* reads a *b* that already contains *b+2*. Effective **reach ≈ 2R**.

**The blocker, and it is probably why it does not compound now:** unerased blocks are currently *read but never updated*. Information flows **into** erased blocks from clean ones and never **through** them. No relay, so no compounding — which means the 2R prediction is wrong about the current implementation regardless of what the seeds say.

**The fix — separate content editing from carrier refreshing.** Run the denoiser over all blocks so every `h_i` updates each pass, but only re-noise the selected ones. Unerased blocks sit at low noise, so their content barely moves while their representation absorbs context. This turns "use global context" into message passing the model cannot route around.

### The frozen-target design (supersedes the naive R-sweep)

Regenerating *b* on every pass conflates *information propagating to b* with *more attempts at b*. The frozen-target version removes that by construction:

1. corrupt target *b* identically in both arms
2. for R−1 passes refresh **only downstream** carrier states, not *b*
3. on the final pass regenerate *b* **once**, from the same direct input and noise in both arms

The final target's inputs then differ only through **relay provenance** — the same isolation discipline as the redirected pin, one level up.

### Three schedules, predictions stated in advance

| schedule | what it does | prediction |
|---|---|---|
| **target-only** | refresh nothing but *b* | stays capped at d=2 |
| **Jacobi** | refresh every carrier state in parallel each pass | frontier near d = 2R |
| **wavefront** | update relay shells far-to-near before touching *b* | tests whether **stale** parallel states are the bottleneck rather than representation |

- Jacobi extends reach → the ordinary outer loop suffices
- only wavefront extends → the fix is a reach-aware update scheduler, not training
- neither extends → downstream states are not learning to act as couriers; training or architecture must change

Six seeds, between-seed error bars. Shape claims wait for seeds.

Adjacent precedent: repeated partial regeneration is effective elsewhere in diffusion (arXiv 2605.19317), but its expansion of *causal reach* has not been isolated this way.

---

## 1a. RESULT — the loop is a shredder, not a relay

**Measured, six checkpoints, R = 1…3.** Reach does not compound. But the reading is not "flat, therefore per-model" — it is a fourth thing.

`preserved` is the whole story:

| arm | R=1 | R=2 | R=3 |
|---|---|---|---|
| only-b (control) | 98.8% | 98.8% | 98.8% |
| one (b + 1 neighbour) | 0.10% | 0.05% | 0.00% |
| p50 | 9.95% | 1.75% | 0.40% |

p50's 9.95% is not partial survival — it is the ~9.5% of draws where the subset happened to select nothing beyond `b`, i.e. the control by accident.

**Propagation requires `b` to absorb `b+2` *and carry it correctly*, and carrying means being regenerated at ~3.5% accuracy.** A pass does not transport information through a neighbour; it **replaces that neighbour with something usually wrong**.

Aggregate confirms it: +2.51% → +1.09% → +0.93% over three passes in the `one` arm; the control holds +2.51 / +2.52 / +2.46.

**This is not evidence the reach is architecturally fixed.** It is evidence that propagation-by-regeneration cannot be tested at this accuracy — the same wall the five training-signal experiments hit.

### The precondition — belongs in endpoint.md §3

**Iterative refinement only pays once a regenerated block is more often right than wrong.** Below that threshold more passes destroy more than they fix.

Nothing in §3's adaptive-`R` machinery currently says this. It should: **`R > 1` is gated on per-block accuracy**, and the stopping criterion in §3.3 is not the binding constraint when accuracy is this low — the *starting* condition is.

### Harness validation

`only-b` at R=1 reproduced the committed six-seed profile to the digit: +2.90 / +2.84 / +0.29 / +0.28, aggregate +2.51%. The multi-pass path perturbs nothing.

---

## 5a. RESULT — depth deepens the window, it does not widen it

**L8, one seed, against the L4 six-seed reference:**

| | d=1 | d=2 | d=3 | d=4 | effect |
|---|---|---|---|---|---|
| L4 (6 seeds) | +2.90 ± 0.48 | +2.84 ± 0.68 | +0.29 ± 0.39 | +0.28 ± 0.42 | +2.51 ± 0.62 |
| L8 (1 seed) | +4.15 | +5.56 | +0.57 | −4.27 | +3.70 |

**The multiplicative prediction failed.** d=3 moved +0.29 → +0.57, which is +0.7σ against L4's between-seed spread. The cliff stayed at two.

What doubling depth bought was **more exploitation inside the existing window** — d=2 at +5.56 against +2.84 ± 0.68.

**Which is the additive-in-graph-distance picture, and it says the cliff is structural at the mask level, not capacity-limited.** Doubling parameters buys better use of ±2 and nothing at ±3.

**Consequence for §4.** If depth cannot reach past 2, the constraint is that **nothing writes a cue at offset 3**. So the fix must be giving layer 0 an explicit wider offset, not more layers matching on the same local cues. That raises §4's prior considerably.

**Consequence for §8.** If depth deepens the window without widening it, banding at w=2 should cost nothing **at any depth** — which makes the efficiency claim stronger, not weaker.

**Caveat, and check it before reading the seeds:** L8's d=4 = −4.27% has causal at an anomalous 5.98% in the thinnest cell (n≈117). If causal is that far above its usual ~1%, the gap is being driven by the *causal* arm rather than the global one, which is a different fact. Report causal and global separately there, not only the gap.

This is a one-seed shape claim — exactly what the standing rule says to distrust. Two more seeds pending.

---

## 2. Hierarchy — only if it is a two-way path

**Correction to `endpoint.md` §4.3.** I claimed hierarchy compounds reach multiplicatively. That is only true if the hierarchy performs

```
chunks → blocks → sections → blocks → chunks
```

with coarse states refreshed **bottom-up** and then broadcast **top-down** before the fine target update. **Pooling alone gives a larger window, not a shorter path.** With branching factor *g* the two-way version gives logarithmic path length; grouping alone gives none of that.

Design notes:

- section states should be **refinement-only** carrier latents, with no decoding length
- add a light auxiliary loss requiring each coarse state to retain recoverable information about its constituents — otherwise the hierarchy pools away exactly the provenance needed for correction
- simpler variant: a small bank of refinement-only **mailbox** slots that blocks write to in one sublayer and read from in the next. Precedent in Longformer (2004.05150) and BigBird (2007.14062), though indexed hierarchical summaries are probably better here because provenance matters

---

## 3. Distance curriculum — cheap, targets a real imbalance

The consumer-distance distribution is **304 / 130 / 38 / 28**. The model sees d=3 and d=4 in ~13% of cases, so there is almost no gradient pressure to learn long-range matching. Rebalance the **corruption** distribution to sample distant consumers equally.

**Caveat:** this changes the eval distribution too. Train on rebalanced, evaluate on **both**, or the comparison is meaningless.

### Train for necessity, not merely distance

Uniform *d* fixes exposure but does not force global-context use — cross-entropy still takes a local shortcut whenever one exists. Construct **courier** cases where:

- the target **and its local evidence** are erased
- a valid descendant at distance *d* remains
- intermediate carrier positions are refreshable
- *d* is sampled ~uniformly
- training is unrolled for enough passes to span *d*

Provenance permits a sharper paired objective: give matched clean-descendant and corrupted-descendant states and require the target to prefer the clean provenance. A contrastive *"which downstream state belongs to this target?"* auxiliary head is probably safer than forcing distant attention mass — **attention mass can be gamed and is not evidence of causal use.**

Mix forced-courier examples with the natural distribution. Evaluate on: untouched natural frequencies, a balanced-*d* diagnostic, and the provenance intervention itself.

---

## 4. Dilated carrier heads — before adding depth

If recurrence fails, give layer-0 somewhere to write long-range cues:

- local heads at offsets 1, 2
- one or two heads at 4, 8, …
- **rotate or randomise the exact offsets during training** to prevent a new offset-specific cliff

Full global attention theoretically already reaches everywhere, but the model empirically learns a local code. Sparse *mandatory* hops may be easier to learn than asking heads to discover useful positions inside an unrestricted global field. Precedent: LongNet (2307.02486), DiNAT (2209.15001) — exponentially expanding receptive fields, logarithmic dependency paths.

---

## 5. Depth — demoted, and my prediction was wrong

I predicted depth would extend reach **multiplicatively**, on the reasoning that composition lets layer 2 match on layer-1 outputs.

**That is not the default.** With a fixed local radius *w*, stacked layers expand the theoretical receptive field roughly as **L·w — additively in graph distance.** Exponential growth requires dilated edges, hierarchy, or learned jumps.

More depth could still improve the cue-writing and matching computation. But "reach doubles" is not the prediction, and depth belongs *after* the transport interventions rather than before them.

---

## 6. Relative-position bias at wider offsets — last

An explicit learned bias making ±3, ±4 cheaper to write cues for.

Correctly last: a bias can make attending to d=3,4 cheaper, but **it cannot ensure those positions encode a useful reconstructive cue.** The cliff says to address transport and training incentive first.

---

## 7. Other task structures — the variable is dependency shape

The reach measured is as much a property of the task as of the model. Different dependency structures should give different reach, and that is the thing to manipulate.

**Factual recall in isolation is the degenerate case — and it has already been run.** *"The capital of France is X"* has no chain; X is determined by the question, not by other steps. No downstream pin, so global refinement has nothing to exploit. That is the **root-corruption null**, and global scored −2.5pp there. The no-dependency control exists and behaves as the mechanism requires.

**Recall in discourse is the interesting case.** *"The capital of France is X. X is famous for the Eiffel Tower. Tourists visit X every year."* One value, pinned three times, at three distances. That is the **multi-consumer** structure expression trees lack — arriving naturally rather than by construction.

### Chess is the best natural fit

- algebraic notation is short character strings — `Nf3`, `exd5`, `O-O` — nearly drop-in for arithmetic steps. Same length scale, one per block, natural boundaries, character-level tokenisation already fits
- PGN databases have millions of games
- **dependency structure is dense and long-range**: corrupt move 12 and if move 30 captures a piece that only exists given move 12, then move 12 is pinned at distance 18
- verification is exact — replay and check legality

Arithmetic gives one consumer at distance 1–4. Chess gives many, at arbitrary distance. **It is the first domain where evidence beyond d=4 exists**, and therefore the first where reach beyond 4 can be measured at all.

### Ordering note

Before a new domain, make **dependency range a knob in the existing generator** — values consumed at controllable distances, multi-consumer where wanted. Same verification, same comparability with six seeds of existing results, and it separates routing horizon from structural horizon directly (§0).

If reach still stops at 2 when evidence exists at 5, it is the model. If it extends, the cliff was tree structure, and chess becomes the natural place to see how far it goes.

---

## 8. Band the refine mask — the window as a design parameter

**The inverse framing, and the only item here that can succeed rather than merely rule something out.** If global attention over a whole sequence buys almost nothing beyond ±2, that is not only a limitation — it is a **measured setting for a design parameter**.

**What is new is not "use local attention".** Sliding-window attention is thoroughly established — Longformer (2004.05150), BigBird (2007.14062), Mistral. Window size is normally a compute-budget guess. What the reach measurement gives is a **principled way to set it**.

**And it explains a published result.** Speculative Correction (2608.02625) reports that local refinement captures much of the gain on MBPP and MATH while global adds on GSM8K — observed, not explained. The window is the explanation, and it predicts *when*: it depends on whether the domain's dependency structure fits inside the reach.

### The consequence for the architecture is larger than a saving

`endpoint.md` §4.3 justifies pack/unpool partly because global attention over the carrier is expensive. But **drafting is already block-causal**, so the only place `O(M²)` bites is the refine pass.

**Band the refine mask and the architecture is sub-quadratic with no hierarchy at all.**

This does not kill pooling — compression still reduces the `L·d²` term, not just the attention term — but the *urgency* drops sharply, and hierarchy moves from load-bearing to optional. A real simplification of the endpoint, arrived at from a measurement rather than a preference.

### The experiment — cheap, existing checkpoints, no training

Run refinement with a **banded** global mask at widths w = 1, 2, 3, 5, and full global. Measure the effect, the distance profile and cost at each.

- **w=2 matches global** → the efficiency result, measured directly. Band the refine mask, drop the hierarchy urgency.
- **banded falls short on a minority of cases** → there is a long-range tail. *"Contributes nothing on average"* is not *"never contributes"*, and that distinction matters for code and prose where a variable's uses can be arbitrarily distant.

Report the *distribution*, not just the mean, for exactly that reason.

### Caveat — §0 again

If the window is 2 because the **task's** dependencies reach 2, the right rule is not *"use w=5"*. It is **set the window from the domain's dependency structure**, which is a better claim anyway — and it is the same experiment: measure reach, set the band.

---

## Ordering

1. **§0 free check** — condition the d=3/d=4 nulls on genuine consumers
2. **§8 banded refine mask** — free, existing checkpoints, and the only item that can produce a positive result
3. **§1 frozen-target relay**, R × d, three schedules, six seeds
4. **§0 multi-consumer generator** — routing vs structural, decisively
5. **§3 distance curriculum / forced courier**
6. **§4 dilated carrier heads**
7. **§2 two-way hierarchy**
8. **§5 depth**
9. **§6 relative-position bias**
10. **§7 chess** — once reach beyond 4 is worth measuring

**Progress:** §1 done (§1a — reach does not compound; the loop shreds context). §5 done, pending seeds (§5a — depth deepens the window, does not widen it).

Remaining order stands as listed. §8 stays where it is rather than being promoted — it is the item most likely to produce a positive result, and it gets stronger the more §5a holds.

**And none of it blocks the writeup.** The finished result — six seeds, the redirected pin, the window at z>4, the conflict cut — has been ready for several sessions. Every session since has added qualifications rather than substance.
