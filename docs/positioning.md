# Adze — result positioning and prior art

**Date:** 7 August 2026
**Status:** for the writeup. Not a design document.

---

## 1. What the result is

A latent diffusion reasoner regenerates a corrupted reasoning step under two masks that differ only in whether the model can see *later* steps. Global beats causal. The decisive control moves the mechanism and the effect follows.

| condition | downstream points at | gap vs clean | gap vs corrupted |
|---|---|---|---|
| root corrupted | nothing — no pin | −2.5% *** | — |
| early corrupted | the **clean** step | +2.3% *** | — |
| consistent corruption | the **corrupted** step | −0.5% n.s. | **+1.9% ***** |

χ² = 21.25, global-only 49 vs causal-only 12. Recalibrated against the measured condition handicap: **+4.8pp and +4.4pp** from two independent estimates.

Supporting cuts, all predicted in advance:

- **operand provenance is the axis** — +3.00% ± 0.29 both-leaves (prefix determines least), +2.10% ± 0.57 one-leaf, **−2.86% ± 1.29** both-from-earlier (prefix determines most, causal recomputes at 9.1% vs 0.6% chance).
- **distance to consumer is a PROXY for provenance, not a second finding.** Pooled it reads +2.9 / +2.8 / +0.3 / +0.3 at d = 1…4 and looks like a window of two; composition with no distance term predicts +2.79 / +2.82 / +0.86 / +0.75, residuals +0.12 / +0.03 / −0.57 / −0.47, all inside their bars. Within a fixed class the advantage holds at every distance — one-leaf d = 4 is +2.59% (z = 3.00).
- **operand provenance** — both-leaves +3.1\*\*\* (prefix determines least) → both-from-earlier −6.1, where causal recomputes at 9.1% against 0.6% chance.
- **pad-free subset** — gap is *larger* without padding (+2.9 vs +2.3). The confound is dead, retroactively too.
- **chance** — RESULT 0.6–0.7%, operands 0.0%. Causal ~2× chance, global ~5×.

**The honest caveat:** absolute numbers are weak. Global at 3.6% is 5× chance, not 50×. The mechanism exists and is exploited weakly. Regime B fired on 9.7% of training steps and is the obvious suspect for both that and the handicap.

---

## 2. The positioning — mechanism, not novelty

**Do not lead with novelty.** Lead with this:

> Here is the mechanism behind an effect the field has already observed at the outcome level, and it explains why the answer is task-dependent.

**Speculative Correction** (arXiv 2608.02625, July 2026) reports that *local refinement captures much of the gain on MBPP and MATH, while global provides a clear additional gain on GSM8K*. Scope-dependence, observed at the outcome level, task-dependent in a way they do not explain.

**The distance-window explanation of their result is WITHDRAWN.** It read: shallow dependency structure fits inside a local window, deep structure falls outside it. The window was provenance composition, so the explanation goes with it.

**A prefix-determination account is available as a CONJECTURE, and should be written as one.** Task-dependence would track how much of a task's reasoning each step's own prefix already determines — global carries a step the prefix underdetermines, and can subtract where the step is recomputable. Testing it needs their data cut by dependency type. Until then this work explains its own mechanism and does not explain theirs.

This framing is better than a priority claim on three counts: it connects to published results rather than standing alone, it survives someone finding a precedent tomorrow, and it is a more useful contribution than a first.

**Structure for the writeup:** lead with the mechanism and the provenance axis. Cite Speculative Correction as an outcome-level observation this work does NOT explain — the explanation was withdrawn with the window. Put the provenance-control novelty in **one hedged sentence in related work**.

**And carry the methodological finding, which is transferable and may outlast the mechanism.** In any tree-structured reasoning task, distance-to-consumer is confounded with dependency type BY CONSTRUCTION: a step whose operands both come from earlier steps sits higher in the tree, so its own consumer is further away. Distance and provenance are not independent and cannot be made so by collecting more data. **Anyone reporting a distance profile on such data without decomposing by provenance will find a spurious window** — this work found one at z = 6.06 and z = 4.21 over six seeds and believed it.

---

## 3. Prior art — checked 7 Aug 2026

No exact precedent found for the control:

> Hold the erased block's direct inputs and corruption fixed; intervene only on whether downstream latent state was computed from the clean or corrupted predecessor; measure recovery of the erased block.

### Closest work

| Work | What it controls | Why it differs |
|---|---|---|
| **Speculative Correction** — arXiv 2608.02625 | Same model as drafter and refiner. Compares baseline / draft-before-refinement / mask-only / local-64-token / full-global. Calls it a "causal draft/refine ablation". | **Verified against the paper.** Interventions change the target's *initialisation* or *available scope*. They never equalise the target's direct inputs while varying downstream-state provenance. Outcome is final-answer accuracy and pass@1. |
| **Iterative Partial Refinement** — arXiv 2605.19317 | Corrupts image regions and measures recovery; fixed-vs-fresh noise, fixed-vs-changing region selection. | Closest cross-domain analogue and a genuine controlled recovery experiment. Tests whether repeated regional regeneration helps, not whether downstream information causally reconstructs an earlier erased latent. |
| **Diffusion in Diffusion** — arXiv 2601.13599 | Revision block size, global receptive field, refinement ratio. ~90/10 small-block/global training exposure. | Evidence is output perplexity under different refinement scopes. Scope, mask and training config move together; no internal-state intervention. |
| **LaDiR** — arXiv 2510.04573 | Decodes intermediate latents, varies refinement steps and block size, visualises trajectories. | Descriptive trajectory evidence plus end-task ablations. No causal intervention on a refinement step's latent state. |
| **ProSeCo** — arXiv 2602.11590 | Corrector use, correction frequency, compute allocation. | Establishes correction improves outputs, not which internal path produces it. |
| **Catruna & Radoi** — arXiv 2607.15893 | Mechanistic study of masked diffusion LMs. Separates previous-token and next-token pathways as distinct layer-0 circuits, either able to steer the output alone. Left-only 2.23 nats vs a matched AR model's 3.46; both-sides 4.39 ≈ the sum. Effective window ~5 tokens. Conflict prompt gives log p(A) − log p(B) of −0.01 to +0.32 across six models against +2.1 to +2.8 when both windows agree; ablating either layer-0 head tips it. | **Verified against the paper** (read 8 Aug 2026). **The ±2 convergence claim is WITHDRAWN.** It rested on both settings finding a distance window; the window here was provenance composition, so there is no shared phenomenon for the two numbers to be evidence of. What still touches this work is the SEPARATE-PATHWAYS finding, which is the same shape as the source split reached by a different method — and even that is analogy, since their circuit *copies* from a matching position where the prefix here must be *recomputed*. |

### Method precedent exists, and should be cited

Activation patching, interchange interventions and causal tracing are standard causal-localisation methods — e.g. Zhang and Nanda, arXiv 2309.16042. There are causal interventions inside diffusion models aimed at identifying concepts or circuits.

**So the novelty is not "inventing causal intervention on neural latents."** It is the application to refinement, and it is narrow.

---

## 4. Claim wording

**Use:**

> We found no prior downstream-state provenance control for latent refinement.

Or the longer form:

> Prior diffusion and latent-refinement work evaluates correction primarily through end-task comparisons, refinement-scope ablations, or descriptive intermediate trajectories. We are not aware of a prior intervention that holds an erased block's direct inputs fixed while manipulating only the provenance of downstream latent state, thereby testing whether later latents causally support reconstruction of the earlier block.

**Do not use:**

- "the first mechanism test of diffusion refinement"
- "the first causal ablation of refinement"
- "the first causal intervention in a diffusion model"

Speculative Correction now occupies the first two in ordinary usage despite a weaker identification design. Contesting that costs more than it wins.

---

## 5. Why the result is credible

Worth stating explicitly in the writeup, because ten stratifications after the fact would be p-hacking and ten *registered* predictions that hold is the opposite.

Every cut was called before the data:

- support falling off with distance — it does, though the decomposition later showed provenance carrying it: right about the plot, wrong about the cause
- provenance reversing where the prefix alone determines the step
- the both-from-earlier cell going to causal
- the redirected pin switching the advantage to the corrupted target

All four held as PLOTS. The first did not hold as a CAUSE — decomposition showed provenance carrying it — and that is recorded rather than counted as a hit.

The pre-registration is part of the contribution, not just hygiene. So is the correction: a six-seed result at z = 6.06 was withdrawn because replication does not repair a confound.

---

## 6. Open items before writing

- **Regime A/B mix sweep** — the handicap and the weak exploitation both point at 9.7% B exposure. This number may change the effect size, so it comes before seeds.
- **Seeds at the winning mix** — three, with spread. That is the number the result stands on.
- **M5 (question conditioning)** — currently nothing determines the operands, which is why they sit at chance in every arm. With a question the whole step is determined, so the ceiling on recoverable content rises sharply. 4% may be an underdetermined task rather than a capability limit.
- **Re-run the redirected pin at the winning mix** — the redirection *is* the result; confirm it survives the config change.
