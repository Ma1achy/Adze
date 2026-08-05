# Adze — Design

**Machine:** MacBook Pro M5 Max, 64 GB
**Status:** v0 design frozen; implementation not started
**Framing:** built for interest, not for novelty.

**One line:** Reasoning as a denoising process — continuous latent thought blocks, diffusion within a step, autoregressive across steps, then a second global pass that lets earlier steps be revised in light of later ones.

---

## 1. Why this shape

Autoregressive CoT commits left to right. Once step 3 is written it's fixed, even if step 7 reveals it was wrong. Diffusion refines globally — the whole thing moves at once — which is closer to drafting than to typing, and there's evidence models actually do this: denoising trajectories show syntax and entity structure settling early while later steps only correct content.

But pure full-sequence diffusion has no natural termination and a fixed compute budget regardless of problem difficulty. Block-causal generation fixes that and reintroduces the commitment problem at block granularity.

So: **draft block-causally, then refine globally.** Each pass supplies what the other lacks.

### The organising principle

The model thinks in continuous space. Tokens exist only at the two interfaces — where training text is read in, and where latents are made readable on the way out. Discreteness is a human-readability format, not part of the reasoning.

This has a consequence that decides several design questions below: **anything that is part of the thinking should be learned, and anything that is only an interface can stay conventional.**

- Input tokenisation (text → tokens → VAE) is where the principle bites hardest. BPE is a **starting convenience, not a considered choice** — see §9. It shapes what the encoder can perceive before any reasoning happens, which makes it less of a pure interface than it first appears.
- Output decoding is an interface, but a *learned* one — a small network, not a tied softmax — because it has to resolve continuous latents into discrete text using context.
- **Block segmentation is not an interface.** Blocks are the unit of reasoning. Their boundaries are part of the thinking and should be learned.

---

## 2. The stack

### 2.1 Representation

A VAE encodes CoT reasoning into blocks of continuous thought tokens. Latent geometry is learned; don't diffuse over a frozen token-embedding space.

### 2.1.1 Block boundaries — heuristic start, learned target

A chain of thought arrives pre-split into steps, so newline-delimited segmentation is available for free. **Use it as a warm start, not as the answer.**

The heuristic is weak in a specific way: a "step" in GSM8K is whatever the person writing it put on a line. Some are one trivial substitution, others pack three operations. That's SpaceByte one level up — trusting the annotator's newlines the way SpaceByte trusts spaces. The model may want a different granularity than the annotator chose, and under the organising principle above, that choice belongs to the model.

**The learned version** is H-Net's dynamic chunking ported to block granularity:

- **Routing module** — predicts boundaries from cosine similarity between projected adjacent encoder outputs. Similarity rather than a learned classifier; that's what keeps it stable.
- **Smoothing module** — interpolates representations weighted by boundary probability. Soft where the router is unsure, hard where it's confident. This is the gradient path through a discrete decision.
- **Ratio loss** — anti-collapse. Without it, segmentation degenerates to every-position-a-boundary or one-giant-block. Target a compression ratio near the heuristic's.

Two things make this far more tractable here than in the general text case:

1. **Scale.** Segmenting ~150 tokens into ~10 blocks, not thousands of bytes into hundreds of chunks.
2. **Warm start.** The router initialises from heuristic boundaries, so it's learning a *correction to a decent prior* rather than learning segmentation from scratch against a diffusion target that's itself still moving. This directly defuses the non-stationary-target problem that makes learned chunking plus diffusion look nasty in general.

Keep the heuristic version as a running baseline — "did the router find better boundaries than the newlines" is a question with a clean answer, and inspecting where it disagrees is the interesting part.

### 2.2 Pass one — draft (block-causal)

- Bidirectional attention **within** a block, causal attention **across** blocks
- Flow matching, linear interpolant, **velocity prediction** (see §3.1)
- Each block denoised conditioned on all previous blocks

Gives open-ended length and stable training. Costs global revision — that's what pass two is for.

### 2.3 Termination

A binary head fires whenever an end-of-thought token is emitted: **continue thinking** or **start answering**.

The model chooses its own reasoning depth, so test-time compute scales with difficulty rather than being fixed. This is also the clean answer to the variable-length problem that recurs everywhere else in diffusion LMs.

### 2.4 Pass two — refine (global)

The distinctive part.

1. Take the completed sequence of blocks from pass one
2. **Partially** re-noise — back to intermediate `t`, not to pure noise (SDEdit-style)
3. Denoise again with **full bidirectional attention** across all blocks

Now step 3 can change because step 7 exists.

#### Weight sharing and mixed-scale training

**One denoiser, shared between both passes** — but only with explicit mixed training. DiD's ablation: a model fine-tuned only on small blocks *degrades* under global refinement, 27.95 → 31.97 PPL. Their conclusion is that standard block diffusion cannot generalise to global scope zero-shot.

**Mixing attention masks is necessary but not sufficient.** The two passes differ in *corruption regime*, not just receptive field:

| | Draft | Refinement |
|---|---|---|
| Noised | current block only | several or all blocks simultaneously |
| Context | previous blocks clean | neighbouring blocks partially noised |
| Attention | causal across blocks | fully bidirectional |

Train on both mask *and* corruption regime, or the denoiser never sees pass two's joint corruption distribution and arrives at inference out of distribution. Exact formulation in §3.1.

**Starting mix: 90/10**, following DiD's bimodal finding (a uniform spread over intermediate scales was worse at drafting, 31.98 vs 27.36). Treat this as a starting point, not an established optimum — DiD studied discrete token remasking on open-ended text, with a 110M model initialised from an 850k-step full-scale checkpoint and tuned over a very large token budget. None of that transfers automatically to continuous reasoning blocks at this scale.

#### Two separate controls, not one

**Correction worth flagging.** DiD's revision ratio γ is the fraction of discrete tokens *completely remasked*. It is not a diffusion timestep. Their U-shaped result says nothing directly about how strongly to Gaussian-noise a continuous latent, and treating γ as `t` conflates two different quantities.

In the continuous setting there are two independent knobs:

- **ρ** — what fraction of blocks to revise at all
- **t** — how strongly to re-noise each selected block

**For v0, `t` is fixed at 1** (complete erasure — see §3.1). That matches the *selected block's* source distribution to training: both start from pure noise. **Its context does not automatically match.** Regime B trains with clean neighbouring blocks; end-to-end inference supplies generated ones. The synthetic central experiment is matched because you deliberately retain valid later blocks — real pass-one → pass-two operation stays off-policy until rollout adaptation (§3.1 stage 2).

Erasure also leaves ρ as the only knob. DiD's 0.25–0.5 region is a reasonable prior for it.

Once that works, grid them independently — moderate noise on every block may behave nothing like erasure of the least-confident quarter:

- ρ ∈ {0.25, 0.5, 1.0}
- t ∈ {0.15, 0.3, 0.5, 1.0}

Nothing published constrains **t** in this setting. That's yours to find, and it's a measured change rather than a starting assumption.

What does transfer: their gains only appeared at large refinement receptive fields — small revision scope *hurt* relative to no revision at all. So pass two should be genuinely global regardless of ρ.

#### Which blocks to re-noise — and what NOT to use

**What's established, and what isn't.** The discrete result is solid: in DiD, post-hoc confidence scored 29.85 against a 27.36 baseline — *worse than not refining* — because a completed sequence gets reaffirmed by the model that produced it. Random was worse still (30.26). Only snapshot confidence, recorded at the moment each token was decided, worked (21.85).

**But continuous diffusion has no "moment the token was decided."** There's no discrete commitment event to snapshot. The analogue is unsettled, and my earlier suggestions were hypotheses dressed as recommendations. Candidates, all unvalidated:

- final-step x-prediction disagreement
- stabilisation time — when the block's trajectory stopped moving
- trajectory curvature or path length
- decoded-block consistency across sampler seeds
- an explicitly trained error/uncertainty head

**The experiment that decides it** is not whether a score varies across blocks — it will. It's whether high-scoring blocks are disproportionately the ones containing wrong operations or values. Rank blocks by each candidate score and measure hit rate against known error locations, benchmarked against four baselines: random, post-hoc likelihood, oracle (true corruption location), and no-revision.

This is the single strongest argument for building the synthetic arithmetic set first — with programmatically generated expression trees, corrupted locations are exactly known, so the oracle baseline is available and the whole comparison becomes clean.

#### What counts as an error

Pass two must be trained on things that are wrong in the way real pass-one outputs are wrong. Four distinct sources, and they are not interchangeable:

| Source | What it teaches | Use? |
|---|---|---|
| Gaussian noise / early-stopped latents | Denoising back to the manifold | Necessary but insufficient — teaches recovery from *noise*, not from plausible mistakes |
| Synthetically corrupted operations and intermediate values | Repairing arithmetic and factual slips | Yes — cheap, exactly verifiable, targets the real failure mode |
| Genuine incorrect pass-one rollouts | The actual error distribution | Yes, most valuable, but needs rollout generation and correctness labelling |
| **Valid alternative reasoning paths** | **Leaving correct reasoning alone** | **Yes — as preservation examples, not correction targets** |

That last row is the subtle one. A refiner trained to map every pass-one output onto the single annotated CoT learns *"rewrite toward the annotator's wording"* rather than *"repair inconsistent reasoning."* It'll look like it's working — outputs converge on the reference — while destroying correct reasoning that took a different route.

But excluding those chains only lets you *test* for that degradation; it never trains against it. Better: pair a valid alternative chain **with itself** under light corruption, so the target is its own reasoning restored rather than the annotator's. Or apply answer supervision alone without forcing the chain toward the reference. Preservation is a behaviour worth training, not just a failure to avoid.

**Answer correctness is an imperfect filter.** An invalid chain can land on the right answer by luck, and would then be treated as a preservation example when it should be a correction target. Another reason to start with programmatically generated arithmetic: operations, dependencies and corruption sites are all exactly verifiable, so "is this chain valid" isn't a proxy question.

### 2.5 Decode

Learned projection back to text — a small network, not a tied linear unembedding. The answer is generated autoregressively conditioned on the final latent blocks.

---

## 3. Design decisions, with reasons

### 3.1 Training formulation

The one component that was consciously unresolved. Fixing it as the v0 baseline — rectified flow, velocity prediction, ODE sampler, matching LaDiR's published formulation.

**Convention:** `t = 0` clean, `t = 1` pure noise.

**Forward (linear interpolant):**
```
z_t = (1 − t)·z₀ + t·ε        ε ~ N(0, I),  t ~ U(0, 1)
```

**Target (velocity):**
```
u* = dz_t/dt = ε − z₀
L = E_{t,ε} ‖ u_θ(z_t, t, c) − (ε − z₀) ‖²
```

**Reverse (Euler ODE, t = 1 → 0):**
```
z_{t−Δt} = z_t − Δt · u_θ(z_t, t, c)
```

**Regime A — draft (~90% of steps)**
1. Sample block index `b`
2. Sample `t ~ U(0,1)`; noise block `b` only
3. Blocks `< b` clean; blocks `> b` absent
4. Attention: bidirectional within `b`, causal to `< b`
5. Loss on block `b`

**Regime B — refinement (~10% of steps)**
1. Sample `ρ`; select a subset `S` of blocks
2. Sample `t_i` **per selected block** (independently)
3. Noise each `i ∈ S` to its own `t_i`; blocks outside `S` clean
4. Attention: fully bidirectional across all blocks
5. Loss on blocks in `S`

#### Clean-data training is not correction training

**The mismatch that has to be stated plainly.** The equations above interpolate between *clean* latents and noise, so the velocity target `ε − z_clean` is correct for that path. But pass-two inference starts from `(1−t)·z_draft + t·ε`, where `z_draft` may contain a plausible semantic error. That is a different distribution, and the target is not automatically valid on it.

The same exposure gap exists in Regime A: training conditions on clean preceding blocks, inference conditions on generated ones. LaDiR adds rollout training precisely for this.

So three stages, in order, not one:

| Stage | Inputs | Teaches |
|---|---|---|
| **1. Oracle mixed training** | Clean latent chains, Regimes A and B exactly as above | Drafting and global denoising |
| **2. Rollout adaptation** | Clean prefixes and refinement inputs replaced by cached or live pass-one outputs | Operating on its own distribution |
| **3. Semantic-correction adaptation** | Paired corrupted/valid chains, under a **separately specified** correction objective | Repairing plausible mistakes |

Stage 3 is not a data swap. Substituting `z_bad` into the clean-data flow equation is not the same objective and needs its own path or conditioning mechanism — specify it when you get there, don't assume the existing loss covers it.

#### v0 pass two: complete erasure, not partial noise

Given the above, the honest v0 is smaller than SDEdit-style partial re-noising:

**Set `t_i = 1` for selected blocks — erase them entirely — and regenerate globally.**

This matches Regime B training exactly (the flow path from pure noise is what stage 1 learns), it mirrors DiD's complete remasking, and it tests the central claim directly: does regenerating a block conditioned on *later* blocks beat generating it causally?

Partial re-noising then becomes a measured change, not a starting assumption. If you do keep it in v0, label it in the code and the writeup as an **inference-time generalisation** — Regime B has not learned it.

**Implementation consequence:** regime B requires **per-block timestep conditioning** — different blocks carry different noise levels simultaneously, so the timestep embedding is per-position, not global. Easy to build in now, painful to retrofit, and it's what makes adaptive per-block re-noising possible once you get there.

**Deferred as measured changes, not v0 defaults:**
- **x-prediction.** ELF's finding that v-prediction breaks was specific to sharing weights with a *discretisation* step. Here the sharing is between two denoising regimes, with a separate VAE decoder — so the finding doesn't obviously transfer. LaDiR uses velocity. Start there, measure x-prediction as a change.
- **SDE sampler.** Better in few-step regimes elsewhere; unvalidated here. ODE first.

### 3.2 Other decisions

| Decision | Why |
|---|---|
| **Velocity prediction, ODE sampler** | v0 baseline — see §3.1. x-prediction and SDE are measured changes, not starting assumptions. |
| **Contextual encoder representations as the diffusion target** | From-scratch learnable embeddings were the *worst* variant in ELF's ablation. Joint optimisation of embeddings and denoiser is hard. |
| **Rollout conditioning — deferred** | Replace oracle contexts with cached or live pass-one outputs during stage-two adaptation (§3.1). **Not in v0**, which trains on clean data only. The trick has been the right answer three times in this design; it's postponed, not dismissed. |
| **Self-conditioning** | Small change, disproportionate quality gain. |

---

## 4. Day-one checks

Build these before the model works, not after.

**Latent-use check.** Decode from a shuffled or random latent. If output quality barely drops, the decoder is ignoring the latent and nothing downstream means anything. This is the single most important early test — VAE posterior collapse is the standard failure here.

**Trajectory printer.** Decode and print at every denoising step. This is both the main debugging tool and the actual pleasure of the architecture — watching noise resolve into structure into content. If step 40 of 50 is still gibberish you know long before a loss curve tells you.

**v0 metrics.** Note the pass-two delta log is *not* useful yet: under complete erasure with oracle-selected blocks, selected blocks necessarily move fully and unselected blocks don't move at all, so the log is mechanically determined by ρ. It becomes informative again with partial re-noising and learned selection. For v0 measure instead:

- exact repaired-operation accuracy
- answer accuracy before vs after refinement
- preservation of *unselected* decoded blocks (did refinement disturb what it shouldn't have?)
- global vs causal regeneration, matched otherwise

---

## 5. Build order

Nothing here is a controlled experiment. Get each stage generating something before adding the next.

1. **VAE.** Encode/decode CoT steps to latent blocks and back. Verify reconstruction, run the latent-use check.
2. **Pass one only.** Block-causal diffusion, **fixed** number of blocks, no termination head. Get it producing coherent reasoning steps. Train with the bimodal mask mix from the start — pass two needs it and retrofitting means retraining.
3. **Answer decoding.** End-to-end question → fixed-length latent reasoning → answer.
4. **Pass two, complete erasure.** `t = 1` on selected blocks, grid ρ. Read decoded CoTs. **This is the experiment.**
5. **Pass two, uncertainty-steered.** Per-block selection from a validated uncertainty score; grid ρ and t independently as measured changes.
6. **Termination head.** Variable reasoning depth.
7. **Learned block segmentation.** Router + smoothing + ratio loss, warm-started from the heuristic boundaries.

**Why termination moved after pass two.** Variable reasoning depth is useful but not necessary to test the central idea. Fixed-length draft → answer → global revision reaches the interesting experiment several weeks sooner, and termination is then added to a system whose distinctive component already works.

**Honest prior-art status of steps 1–3.** Not "reproduces a known-working design." LaDiR validates this architecture at 8B; CODI validates a *different* continuous-reasoning mechanism at GPT-2 scale. Nobody has shown this design works at this size. That's a real risk in steps 1–3, not just in step 4.

### The central experiment

**Oracle block selection.** You know which block was corrupted and erase that one. This answers *"can global regeneration repair a known-bad block using future context?"* — not *"can the system detect which block needs repair?"* That's correct for v0, but label it so the result can't be mistaken for end-to-end uncertainty steering.

Construction:

1. Corrupt an **early** intermediate value in a generated arithmetic trace
2. Retain the **later** steps, computed from the *correct* value — the chain is now internally inconsistent, and the inconsistency is only visible downstream
3. Erase the corrupted block and regenerate

**Three otherwise-identical conditions:**

| Condition | Mask | Tests |
|---|---|---|
| No revision | — | Baseline |
| Erase + regenerate **causally** | block-causal | Repair from the question and preceding steps alone |
| Erase + regenerate **globally** | full bidirectional | Repair using later valid reasoning |

The causal-vs-global comparison is the result. Use the **causal mask** rather than physically deleting later blocks — deletion also changes sequence length and positional context, confounding the comparison. "Remove future evidence entirely" can stay as an additional ablation.

Without the causal condition you've only shown correction from the question, which an autoregressive model does too. The gap between causal and global is what demonstrates revision *in light of later reasoning* — the one property justifying the architecture.

Programmatic generation is what makes this work: operations, dependencies and corruption sites are all exactly known, so success is verifiable rather than judged.

### The v0 freeze

Design is closed. If a first implementation started today:

- fixed newline-delimited blocks
- four latent tokens per block
- fixed reasoning length
- shared denoiser, two corruption regimes, **oracle clean-data training only** (§3.1 stage 1)
- pass two by **complete erasure** (`t = 1`), grid over ρ alone
- velocity prediction, ODE sampler
- programmatically generated arithmetic before natural GSM8K
- **no** rollout adaptation, semantic-correction objective, partial re-noising, termination head, learned segmentation, or adaptive compute

Everything beyond this list is §8. Further conceptual refinement will produce less value than the first reconstruction curve and trajectory printout.

**Why segmentation comes last despite being conceptually central.** It's the highest-variance component: if the router and the denoiser are both unproven, a bad sample doesn't tell you which one is broken. Running it against a working heuristic baseline means any change is attributable. Late in the order for debugging reasons, not because it's optional.

---

## 6. Scale and practicalities

- LaDiR is 8B LLaMA — out of reach and not the point. **CODI matched explicit CoT at GPT-2 scale**, so that's the target size. **GSM8K is the evaluation family, not the training source:** plain GSM8K's ~7.5k is very likely too small for a VAE, a denoiser and a refiner, and CODI's GPT-2 result used an augmented set. Train on programmatically generated arithmetic traces first — exactly verifiable, and they give the oracle needed for the central experiment — then augmented GSM8k. Settle this before step 1.
- Check throughput early: PyTorch MPS has silent op fallbacks to CPU that tank speed without erroring. Profile a few hundred steps before committing to a long run.
- Checkpoint and resume from the start — a laptop will thermally throttle on sustained runs.

### 6.1 Development loop

The binding constraint is how fast you can run an experiment and read the answer, not FLOPs. At L ≈ 150 positions the attention term is roughly a fifth of per-step cost — FlashAttention-class optimisations and quantisation are solving problems you don't have. These are the things that actually matter:

**Cache the VAE latents.** Once the VAE is trained and frozen for a stage, encode the whole dataset once and store the latents to disk. Every subsequent denoiser experiment then skips encoding entirely. One afternoon's work, permanently deletes a large fraction of per-step cost. Biggest practical win available.

**Keep a debug config that overfits in under a minute.** 2 layers, 128 wide, 100 examples, 8 denoising steps. Wrong mask, wrong timestep broadcast, wrong loss reduction — all surface here in seconds instead of hours. Scale up only once the loop is provably correct.

**Overfit-one-batch as a hard gate.** If loss won't go to near zero on a single batch, nothing downstream matters. Specifically catches per-block timestep conditioning being wired wrong, which otherwise presents as "diffusion is hard."

**Drop NFE during development.** 50 steps is for measuring quality. 8 is fine for "does the pipeline run."

**Cheap wins worth taking anyway:**

- `F.scaled_dot_product_attention` rather than hand-rolled attention — free fusion from whatever the backend supports
- bf16 mixed precision; watch for instability in the flow-matching loss and fall back to fp32 for the loss itself if it's noisy
- Benchmark `torch.compile` rather than assuming — MPS support has been uneven

**KV caching across blocks in pass one — after correctness, not from the start.** Previous blocks are clean and fixed during drafting, so their keys and values are reusable across every denoising step of the current block; this is the standard efficiency argument for block diffusion, and BD3-LM implements it. But at L ≈ 150 it's unlikely to justify complicating the first working denoiser. Get uncached generation passing deterministic equivalence tests first, then add caching and verify cached vs uncached outputs numerically.

**Note the asymmetry: pass two can't use it.** Full bidirectional attention means nothing is fixed, so refinement pays full cost every step. That's a real reason to watch ρ.

**If you need to go genuinely faster, the lever is NFE** — a better sampler, consistency distillation, or a tuned schedule. Halving 50 steps to 25 beats any attention optimisation available at this size.

See §4 for the diagnostics themselves — at this scale the trajectory printer and block-delta log are higher-value than any throughput work.

---

## 7. Prior art position

| Component | Status |
|---|---|
| Latent thought blocks via VAE, block-causal diffusion, termination head | **LaDiR** (Apple, arXiv 2510.04573). Read it first; it's the base. |
| Draft-then-refine two-pass structure | **Diffusion-in-Diffusion** (arXiv 2601.13599) — validated, but in *discrete masked* diffusion over tokens. Supplies three discrete-domain findings to reproduce or retest: mixed training exposure (most likely to transfer), revision ratio optimum, and confidence policy (both may not). |
| Continuous-embedding refinement before finalising | **LRD** (arXiv 2510.11052) — token-level, not reasoning blocks. |
| GPT-2-scale latent reasoning | **CODI** (arXiv 2502.21074, code at github.com/zhenyi4/codi). |
| Two-pass refinement over *continuous latent reasoning blocks*; uncertainty-steered re-noising | **Open.** |

Useful that draft-then-refine is already validated against exactly the myopia problem block-causal generation creates — the structure is known to work, just not in this space.

---

## 8. Parked

**Soft existence probabilities over blocks.** Each block carries an existence probability; attention is soft-masked by `log(p)`; parameterise as a survival function so it decays monotonically. Would let pass two *add or remove* a reasoning step rather than only revising in place.

Much more tractable at ~10 blocks than at thousands of bytes, which is why it's worth keeping on the list. Watch for: vanishing gradients as `p → 0` (positions turn off easily, back on almost never — needs a floor and a ratio regulariser), and existence probably needs a faster denoising schedule than content.

**Related to the item below.** Soft existence relaxes a discrete choice — how many blocks exist — so that the *answer loss* can influence reasoning extent directly, rather than extent being fitted to annotated lengths. (Note the termination head is not gradient-free; it trains fine. Its problem is the *objective* — it imitates how long a human took — not an absence of gradients. And soft existence doesn't escape that on its own: without a compute penalty or regulariser it can settle into the same annotated-length imitation.) The two items also govern different axes: soft existence controls **extent** (how many steps), halting controls **depth** (how much reconsideration per step). Not interchangeable — refining harder won't supply a missing step, and adding steps won't fix a wrong one. They share a failure mode (degenerate optimum in differentiable resource allocation), so the anti-collapse machinery transfers even though the axes don't.

**Learned compute allocation.** The natural successor to the termination head.

What's already adaptive: *which* blocks get revised (snapshot confidence makes this data-dependent) and *how many* reasoning blocks exist (termination head). What isn't: denoising steps per block, number of refinement passes, and the confidence→noise mapping — all inference-time hyperparameters.

The honest limitation of the termination head as specified: it's trained on annotated CoT lengths, so it learns *how many steps a human used on a problem like this*, not how much thinking this model needs. A genuinely uncertain model gets the same budget as a confident one on a superficially similar problem. That's imitation of someone else's allocation, not self-assessment.

Two routes to the real thing, both established:

- **ACT / PonderNet-style halting** — a head emitting halting probability per refinement pass, trained against a ponder cost. PonderNet's reformulation gives better gradients than Graves' original. Known to be twitchy: the standard failure is collapse to always-minimum or always-maximum passes, and the cost coefficient is notoriously sensitive.
- **RL on correctness minus compute.** Direct precedent exists — LaDi-RL runs RL on top of LaDiR, using it as a cold start.

**Why this architecture suits it unusually well.** In an AR model, "how long to think" is entangled with "how much to write" — more thinking means more tokens. Here they're separate axes: denoising steps and refinement passes are compute spent without changing output length at all. That's a much cleaner substrate for a halting decision than AR offers, and arguably the strongest form of the test-time-compute claim.

The plumbing is already present: snapshot confidence is exactly the state a halting head would read. "Blocks still uncertain → run another pass" is a two-line heuristic and a research project to make learned. Do the heuristic version first; a learned halting head stacked on three unproven components is the configuration where nothing is diagnosable.

**Looped refinement — dynamic N.** The two-pass design is a loop unrolled twice. Making N a variable is the natural generalisation, and it sits on the same **depth** axis as the halting head above.

**Corrected architecture** — the loop runs in latent space, encode and decode stay outside it:

```
                    bytes
                      │
                      ▼
    ┌─────────────────────────────────────┐
    │  ENCODE  (once)                     │
    │  bytes → router₁ → chunks           │
    │        → router₂ → latent blocks    │
    └─────────────────────────────────────┘
                      │
                      ▼
    ┌─────────────────────────────────────┐
    │  SHARED DENOISER                    │
    │  conditioned on (t, mode, iter)     │
    │                                     │
    │  iter 0 : causal mask, from noise   │  ← draft
    │  iter 1+: global mask, erase ρ      │  ← refine
    │                                     │
    │  inner loop: denoising steps (t)    │
    └─────────────────────────────────────┘
                    │   ▲
                    └───┘  × N, latents only
                      │
                      ▼
    ┌─────────────────────────────────────┐
    │  DECODE  (once)                     │
    │  latent blocks → dechunk → chunks   │
    │                → dechunk → bytes    │
    └─────────────────────────────────────┘
                      │
                      ▼
                    bytes
```

**Do not loop through bytes.** Discretising each iteration breaks gradient flow between passes, destroys the uncertainty information refinement depends on (a block that was 60/40 between two values becomes a committed string), pays full encode/decode per loop, and makes tokens an internal representation rather than an interface — contradicting §1.

**Diffusion is already a loop.** Shared weights applied repeatedly, conditioned on `t`. So `t` is already the depth conditioning that looped-transformer work has had to invent — one ASR paper found that reapplying a block unconditioned was insufficient and needed FiLM depth conditioning for iterations to specialise.

**But there are two nested loops and they need separate signals.** Inner: denoising steps, conditioned on `t`. Outer: draft/refine iterations, needing its own `iter`/`mode` conditioning. Reusing `t` for both means the model can't distinguish "first draft" from "fourth revision." This matches the two-scale finding in recurrent-depth models — small local refinements within a looped block, larger drift across iterations.

**Two versions, very different costs:**

| | What | Cost |
|---|---|---|
| **Free version** | After v0 trains, run pass two 3 or 5 times instead of once. Pure inference change. | An afternoon. Gives a quality-vs-N curve and reveals whether iteration 2 does anything at all. |
| **Real version** | Sample N during training so the model operates at arbitrary depth; add `iter` conditioning; add halting. | A redesign. Unjustifiable before the free version shows something. |

**Do the free version.** It's the two-line experiment that's easy to forget to try, and it's the evidence that decides whether the real version is worth building.

**Known failure mode:** overthinking. Recurrent-depth models degrade at extreme recursion depths. DiD's γ→1 explosion is plausibly the same pathology in a different guise, so expect the curve to turn over somewhere. **Useful diagnostic:** in healthy looped models, updates get smaller and increasingly orthogonal across iterations — local refinement rather than pushing further in one direction. Log per-iteration update magnitude and direction; drift rather than refinement is what failure looks like.

**Note on positioning.** Three parked items now sit on the depth axis — halting, learned compute allocation, and this. "Diffusion as CoT" is LaDiR's ground; "looped latent refinement with learned halting" is much less occupied. If v0 works, that's probably the more interesting identity for the project.

**Distillation for data, and RL on top.** Distinct from the three depth-axis items above — this is about the training story, not the architecture.

**Distilled CoT traces give volume and diversity — not automatic supervision.** Sampling many traces per problem is genuinely useful, but three things don't follow:

- **A correct final answer doesn't prove every step is valid.** Chains reach right answers by luck or by compensating errors, so "correct answer → preservation example" is unsound.
- **A wrong trace is not yet a correction pair.** It needs a verified valid target and, for block-level training, an alignment between the two chains.
- **Free-form traces aren't exactly verifiable** just because their final arithmetic is checkable. You'd need parsing, execution, or a judge — all less reliable than the programmatic route.

Programmatically structured traces *are* exactly verifiable, which is why they stay first in the data plan. Distillation supplements them with volume, natural chain-length variation, and stylistic diversity — it doesn't replace them for stage-three supervision.

**What it does not give you at all: stage 2.** External traces are off-policy. Rollout adaptation is by definition about the model seeing its *own* output distribution, and no volume of teacher data substitutes.

**Practical:** several large distilled reasoning-trace sets already exist on HuggingFace; check before spending GPU hours generating. Caveat: most are built for large models and may carry longer, more elaborate chains than a GPT-2-scale model can absorb — chain length distribution is worth checking against the fixed block count.

**RL as the second half.** The framing is sound: rejection-sampling fine-tuning is roughly offline RL with a verifiable reward, and CODI is already a distillation method (teacher explicit CoT, student continuous). LaDi-RL is the direct precedent — latent diffusion reasoner as cold start, RL on top, reported to prevent entropy collapse.

**The claim worth testing, though, is about exploration.** RL on reasoning needs diverse rollouts. AR models buy diversity with temperature, trading directly against per-sample quality. LaDiR gets it structurally instead — explicit diversity guidance repels batch members in latent space during denoising, producing genuinely distinct solution paths rather than degraded ones.

If that holds, **a diffusion reasoner is a better substrate for RL than an AR model**, not merely a different one. Testable directly: measure rollout diversity at matched per-sample quality against a temperature-sampled AR baseline. That's a sharper claim than "apply GRPO to a new architecture," and it would justify the RL story rather than making it a bolt-on.

**The readability dial.** A λ-weighted loss forcing latents to reconstruct their CoT text. Legibility comes nearly free here since the VAE already reconstructs steps; the *sweep* — plotting task accuracy against recovery fidelity — is a complete result on its own if the mood ever changes. If pursued, the faithfulness interventions are mandatory: perturb a latent and check the recovered CoT changes correspondingly, or you're just scoring a fluent confabulator.

---

## 9. Endpoint — learn the input tokenisation too

BPE at the input is a convenience for getting started, not a principled stopping point. The consistent version of the organising principle removes it as well.

**The architecture:** a two-stage H-Net whose top level is a diffusion reasoner rather than an AR transformer.

```
bytes → [router 1] → word-ish chunks → [router 2] → reasoning blocks
                                                          ↓
                                              two-pass latent diffusion
                                                          ↓
      bytes ← [decoder 1] ← word-ish chunks ← [decoder 2] ← blocks
```

No hand-designed linguistic segmentation and no fixed vocabulary remain. (Plenty is still designed — hierarchy depth, routing mechanism, target compression ratios, noise schedule, attention structure, training objectives. The claim is about *linguistic* priors, not about the architecture being assumption-free.)

**Why the efficiency objection doesn't apply here.** The negative byte-diffusion results concern diffusion running *over bytes*. In this design the diffusion runs over reasoning blocks at the top of a hierarchy; byte-level input only costs anything in the small encoder and decoder networks. H-Net uses SSMs at those levels precisely because they compress well and are linear in sequence length. The expensive component never sees a byte sequence.

**The hypothesis behind it.** BPE shapes what the VAE can perceive before any reasoning happens — "3.14159" arbitrarily split means the encoder's view of the number is already fragmented. Whether byte-level *fixes* this is genuinely unknown: it removes the arbitrary split but hands the model a longer sequence from which it must learn number structure itself, which may be no easier. Treat it as a hypothesis worth testing on arithmetic specifically, not as established motivation.

Independent of that: decoding to raw bytes means no vocabulary at all, which serves the readability framing better than decoding into a token inventory someone else chose.

**Why it's an endpoint and not stage 8.** That's two nested learned segmenters — two routers, two ratio losses, two collapse modes — and H-Net's own account is that nesting hierarchies is where prior end-to-end approaches destabilised and failed to scale. Making two stages work was the hard part of their paper. Stacking that under an unproven diffusion reasoner gives you three unproven learned components sharing one loss signal, which is not a debuggable configuration.

**What to do about it now:** build stage 1's VAE so its input side is *swappable* — a small encoder network taking a sequence of embeddings, rather than something that assumes a token vocabulary throughout. Costs nothing now; the difference between a v2 and a rewrite later.

---

## 10. Dropped along the way

Cascaded text diffusion (Cola DLM); bit-stream / analog-bits input; soft existence probabilities at byte granularity.

Note that learned dynamic tokenisation (H-Net) is **not** on this list — it moved into §2.1.1 as the block segmenter, and into §9 as the input segmenter.
