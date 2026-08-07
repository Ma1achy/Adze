# Adze — Endpoint (v2)

**Status:** plan only. Gated on the v0 result (M7).
**Relationship to the other docs:** `design.md` is the frozen v0 design; `build-plan.md` is its milestone sequence. This is what v0 becomes if M7 says the idea works. Nothing here is in scope until then.

---

## The target

**Inference** — there is no input carrier to encode. The state is generated from a prior, conditioned on the prompt:

```
    input bytes ──→ CONTEXT ENCODER ──→ conditioning c
                                            │
    C query-only carrier slots              │
                      │                     │
                      ▼                     ▼
    ┌─────────────────────────────────────────────┐
    │  carrier proposal  (sub-quadratic, over C)  │
    │  → initial b, ℓ                             │
    └─────────────────────────────────────────────┘
                      │
                      ▼
    ┌─────────────────────────────────────────────┐
    │  hard pack  P(h, b, ℓ)  →  M blocks         │
    ├─────────────────────────────────────────────┤
    │  HEAVY BLOCK CORE  |  c        (over M)     │
    │  conditioned on (ν_h, ν_b, ν_ℓ, mode, iter) │
    │                                             │
    │  iter 0 : causal mask, from prior   ← draft │
    │  iter 1+: global mask, erase ρ     ← refine │
    ├─────────────────────────────────────────────┤
    │  unpool / update  (sub-quadratic, over C)   │
    │  → h, b, ℓ                                  │
    └─────────────────────────────────────────────┘
                    │   ▲
                    └───┘  × S inner steps, × R outer iterations
                      │
                      ▼
              (h₀, b₀, ℓ₀)
                      │
                      ▼
    ┌─────────────────────────────────────┐
    │  DECODE  (once)                     │
    │  carrier → dechunk → chunks         │
    │          → dechunk → bytes          │
    └─────────────────────────────────────┘
                      │
                      ▼
                 output bytes
```

At debug scale, packing is skipped and the carrier is denoised directly (§4.3) — the middle three rows collapse to one. The diagram above is the endpoint.

**Training only** — target output bytes pass through the *carrier* encoder to supply `(h₀, b₀, ℓ₀)`, which is then corrupted and denoised:

```
    target bytes ──→ CARRIER ENCODER ──→ (h₀, b₀, ℓ₀) ──→ corrupt ──→ denoise
                     bytes → router₁ → chunks → router₂ → boundaries
```

**Two encoders, two roles.** The context encoder reads the prompt and produces conditioning; the carrier encoder exists only to manufacture training targets and is never run at inference. Conflating them is what made the earlier single-path diagram wrong — it implied the thing being denoised was an encoding of the input, which is true at training and false at generation.

Everything learned except the 256 byte values. Segmentation at both levels, the latent geometry, the denoising, the expansion, the decode — and how long to think.

---

## 1. The dependency chain

The most important thing the literature review changed. **Latent geometry comes before segmentation, not after.**

COSMOS's finding: naively minimising token-reconstruction loss yields a brittle latent geometry that hampers diffusion. Adze's measured latent clustering is NN/mean 0.377 against a 0.744 Gaussian null — brittle, by their definition, from the recipe they identify as producing it (CE + tiny KL).

A brittle latent doesn't only hurt the denoiser. It hurts every component sitting on it, including a router trying to learn boundaries against it. So:

```
1. latent geometry        (COSMOS recipe)
2. block boundaries       (router₂, H-Net recipe)
3. input segmentation     (router₁, byte-level)
4. the refinement loop over learned segmentation   ← unsolved, see §4
```

Stacking a learned router on a brittle latent gives two unproven components compounding, with no way to attribute a failure. Same argument that put segmentation last in the v0 build order; it now applies to the latent space itself.

---

## 2. Component recipes — what exists

| Component | Recipe | Source |
|---|---|---|
| Smooth latent geometry | encoder-side perturbation robustness + activation MSE | COSMOS, arXiv 2506.21170 |
| Block boundaries (router₂) | similarity routing + smoothing module + ratio loss | H-Net, arXiv 2507.07955 |
| Input segmentation (router₁) | same, one level down, over bytes | H-Net (two-stage) |
| Vectorised block training | clean-context pass + all-blocks-parallel pass | BD3-LM, arXiv 2503.09573 |
| Draft-then-refine | mixed-scale training, complete erasure | DiD, arXiv 2601.13599 |
| Block-causal latent reasoning | VAE thought blocks, termination head | LaDiR, arXiv 2510.04573 |
| Adaptive stochasticity | entropy-gated Langevin corrections | arXiv 2605.07013 |

### The COSMOS correction, stated precisely

The perturbation goes in at **autoencoder training**, not at diffusion training. Train the encoder to reconstruct the *original* hidden states when given *perturbed* inputs.

Adze already tried the other thing — noise added to cached latents before flow training — and it cost 9.3pp. Different intervention, opposite result. The distinction is that COSMOS makes the space smooth by construction; jitter smears an already-brittle one.

Diagnostic already built: the interpolation probe (decode the midpoint of two latents, chance-normalise against a measured null) is COSMOS's §5.1 analysis. It will tell you directly whether the recipe worked.

---

## 3. Adaptive compute — one mechanism, two scales

Three things could be adaptive. Two resolve into the same mechanism; the third is deferred.

### 3.1 The two scales are structurally the same operation

At η = 1 the inner loop is not integrating a trajectory. Each step predicts the clean carrier state `(ĥ₀, b̂₀, ℓ̂₀)` and re-noises with fresh noise — a Markov chain. The outer loop does the same thing: predict, re-corrupt, repeat. The only differences are that the inner loop's noise follows a schedule down to zero and applies globally, while the outer loop's sits at t = 1 on selected blocks.

This resolves two measurements that looked inconsistent in v0: truth was **flat from nfe 8 to 300 at η = 0** (integration converged; more steps sample the same path more finely) and **still climbing at nfe 100 at η = 1** (more resampling rounds, more averaging, more chances to correct).

So use one convergence mechanism at both scales, **over all three channels** — the state is `(h, b, ℓ)`, so a criterion on content alone is incomplete:

```
δ_h = RMS(ĥ₀⁽ˢ⁾ − ĥ₀⁽ˢ⁻¹⁾) / latent_scale
δ_b = mean |b̂₀⁽ˢ⁾ − b̂₀⁽ˢ⁻¹⁾|
δ_ℓ = mean JS(p_ℓ⁽ˢ⁾, p_ℓ⁽ˢ⁻¹⁾)
```

- **inner:** stop when the clean carrier-state estimate `(ĥ₀, b̂₀, ℓ̂₀)` settles — smoothed `δ_h`, `δ_b`, `δ_ℓ` all below threshold
- **outer:** stop when no carrier region exceeds the uncertainty threshold

**Normalise `δ_h` by latent scale**, or encoder unfreezing (Phase E) silently changes the criterion under you.

**Both need smoothing.** Under stochastic sampling consecutive predictions fluctuate even at stationarity, so "stops moving" must be a moving average over a patience window, not a single-step comparison. Same for the outer selector, whose threshold crossings will flicker near the boundary.

### 3.2 Inner steps — adaptive, not learned

Under deterministic flow, inner NFE is primarily a **discretisation** choice once the trajectory is numerically resolved. Under η = 1 it is also the number of stochastic transition and correction rounds, so it **remains a compute axis** — which is why truth kept climbing at nfe 100 there. Both readings are true at their own η, and the earlier flat-versus-climbing pair is that difference, not a contradiction.

Standard adaptive ODE solvers (embedded error estimates, `torchdiffeq`) are **inapplicable at η = 1**: there is no single deterministic trajectory whose truncation error is being estimated. Keep them for η = 0 comparisons; they are the right tool for the case not in use.

The first control mechanism should therefore be **entropy-gated stochasticity plus a smoothed convergence criterion**, not a learned step-count head. Under SDE the first local control is not step count alone but how much noise to inject per step per position (§2), which needs no head.

### 3.3 Outer loop R — rungs 1 and 2, combined

R *is* a computation choice, because each iteration erases and regenerates rather than refining an integration. Two rungs, and they compose.

**Rung 1 — convergence criterion. Free, falls out of adaptive ρ.** (Selection is over carrier *regions*; hard blocks exist only after a commit.)

With uncertainty-gated carrier-region selection, if nothing exceeds threshold then ρ = 0 and the iteration is a no-op. So the stopping rule is just **"is any carrier region above threshold?"** — checked *before* erasing anything. No decode, no wasted pass. It is the adaptive-ρ machinery already in the design, read as a stopping rule.

**Rung 2 — distil R into a predictor. Post-hoc, and it touches nothing.**

Run the criterion across the training set and record how many iterations each example needed. That manufactures **supervised targets for R**. Train a small head to predict it from the question and initial state.

**Predict a survival curve, not a scalar.** The required iteration count depends on the *sampled* draft, so R is stochastic given the input and a predicted mean is the wrong object. Predict `P(R ≥ k | x)` and group batches by an upper quantile — conservative scheduling, with the convergence criterion still making the final stop decision. The predictor is never a cap.

This is a separate cheap run *after* the main curriculum, interfering with nothing — which matters given §4.11's complexity budget. And it produces predictive halting (knowing from the outset that a problem is easy) without a ponder cost, without RL, and without the always-minimum/always-maximum collapse that afflicts both.

**How they combine.** The criterion always wins at inference; the predictor buys two things it cannot:

- **Batch scheduling.** Under the criterion alone every sequence finishes at a different iteration and you run mostly-idle passes. Predicted R lets you group. Irrelevant at debug scale, real at deployment.
- **A free rung-3 trigger.** Systematic disagreement between predicted and actual R is exactly the evidence that the convergence criterion is the wrong target — the only thing that would justify a ponder cost or RL. The diagnostic comes free.

**Rung 3 — ponder cost or RL. Deferred.** Only if rung 2's predictor is systematically wrong in a way that correlates with final quality.

### 3.4 Adaptive ρ comes before learned R

If carrier-region selection is uncertainty-driven and the model is confident everywhere, ρ → 0 and the iteration is a no-op. **A fixed R with adaptive ρ therefore behaves like variable R** — you overpay in forward passes but not in quality.

It also does something better: it **bounds** the overthinking risk. Recurrent-depth models degrade at extreme depth, and v0's churn sweep showed harm past a point.

But **"extra iterations cannot hurt" is too strong.** Once the uncertainty selector returns an empty set, subsequent iterations are skipped exactly and cannot alter the sample. *Before* that point, false-positive uncertainty, boundary churn or selector miscalibration can still cause harmful erasure. Selector calibration and quality-by-iteration stay required measurements.

So adaptive ρ is the prerequisite, not learned R. It is already in the design, it bounds the known failure mode, and it demotes R from a quality parameter to a compute-efficiency one — which rung 2 then recovers.

### 3.5 What is learned, and what is chosen

- **learned:** content (`h`), structure (`b`), extent (`ℓ`) — all supervised by reconstruction, all properties of the output
- **adaptive, unlearned:** inner steps, ρ, R via rung 1 — runtime decisions from measurable uncertainty
- **learned, post-hoc:** R via rung 2 — targets manufactured from rung 1
- **chosen:** what compute *costs*. Nothing in the data says how much thinking a problem deserved, so without a cost term the optimal policy is always to think maximally. That price is a design decision, not something the model discovers.

---

## 4. Generated structure and refinement

The genuinely novel part, and the only component with no reference implementation. LaDiR does refinement with fixed blocks; H-Net does learned segmentation without refinement. The intersection is unclaimed.

The problem has two parts:

1. **At generation there is no text to segment.** The routers segment *input*; denoising from noise has none, so boundaries must be predicted rather than encoded.
2. **Under refinement, predicted boundaries move.** Blocks split, merge, appear, vanish between iterations.

(2) is only hard while content and segmentation share a representation. Separate them and the combinatorial problem disappears, leaving a dynamical one.

### 4.1 Persistent state

A carrier lattice of *C* chunk positions, three channels:

| Channel | Type | Role |
|---|---|---|
| `h_i` | continuous, *d*-dim | chunk content |
| `b_i` | [0,1] | probability of a block boundary after *i* |
| `ℓ_i` | integer ≥ 0 | byte expansion; `ℓ_i = 0` means the chunk produces nothing |

Content keeps its identity when boundaries move. Block count follows from the boundary field, not from the shape of the content tensor.

**`ℓ = 0` is permitted anywhere, not only as a trailing run.** A trailing run of zeros is termination; a zero mid-sequence is a deletion; a position going 0 → positive is an insertion. One channel covers insertion, deletion and termination, which is what retires the variable-length problem — and it only works if mid-sequence zeros are legal. A prefix-only constraint would force deletion to be a left-shift, i.e. exactly the reindexing the carrier exists to prevent.

**`ℓ_i = 0` means non-emitting and non-conditioning — but query-active.** The carrier coordinate stays query-active so it can observe context and *propose its own reactivation*; its key, value and pooling contributions are gated off.

**Gate hard in the forward pass, soft in the backward.** A raw probability `P(ℓ_i > 0)` is almost never exactly zero, so multiplying keys and values by it leaves an inactive position weakly influencing everything — which contradicts "non-conditioning". Use straight-through:

```
p_active = 1 − p(ℓ_i = 0)
a_hard   = 1[incoming ℓ_i > 0]
a_st     = sg(a_hard − p_active) + p_active

query_i = always enabled
key_i   = a_st · key(h_i)
value_i = a_st · value(h_i)
pool_i  = a_st · pool(h_i)
emit_i  = 1[ℓ_i > 0]
```

Forward behaviour is exactly active or inactive; gradients still flow. Note `a_hard` uses the **incoming** state, the same circularity-breaking rule the mask uses for `b` (§4.5).

**Key/value gating describes the attention core.** The carrier is an SSM and has no separate key path — the equivalent there is gating a position's contribution *into the shared recurrence* while still letting it receive the scan's output, reapplied at every layer. See §5.4.

**Masking the query would make zero absorbing.** A position with no contextual representation cannot compute anything, so it could never predict `ℓ_i > 0`, and insertion would be impossible — falsifying the claim that one channel covers all three operations. `ℓ` would support deletion and termination only.

**What the gating does and does not buy.** It prevents an inactive position acting as *shared* scratch space: it computes for itself, and nothing reads from it. But it does **not** prevent transient scratch use — a position can activate, export information to its neighbours, and deactivate again before the final decode, paying no emitted byte at all. That costs transient computation, not output length.

If that is undesirable, either commit activity **monotonically within an inner trajectory** (once active, stays active until the trajectory ends) or accept it explicitly as latent scratch capacity with its own accounting. Do not assume it away.

A categorical `UNKNOWN`/`MASK` length state stays query-active on the same reasoning.

**`C_extent`** is the index of the last position with `ℓ > 0` — deliberately not called `N_active`, since with internal holes the last occupied index and the count of emitting positions differ.

**Boundaries attach to carrier edges, including across inactive gaps** — that keeps `b`'s coordinates stable and `ℓ`-independent. But **the anti-collapse loss must measure effective separation, not raw edge mass.**

The hole: with active chunks at positions 2 and 10 and 3–9 inactive, boundary mass placed anywhere inside that gap counts toward a raw ratio — while the empty blocks it creates are discarded during packing. The content network gets near-global attention over the *emitting* chunks and the ratio loss passes anyway. This is the internal-hole analogue of edge-packing.

So with emitting indices `i₁ < i₂ < … < i_A`, define the probability that consecutive emitters are separated, and use its mean:

```
c_m       = 1 − ∏_{k=i_m}^{i_{m+1}−1} (1 − b_k)
r_eff     = (1/(A−1)) · Σ_{m=1}^{A−1} c_m
L_ratio   = (r_eff − r*)²
```

Skip the term when fewer than two chunks emit.

**Note `c_m = 1 − S(i_m, i_{m+1})`** — the same no-cut product the attention mask uses (§4.4). The loss and the mask now read the same quantity, so they cannot drift apart: whatever the mask treats as one block is exactly what the ratio counts as unseparated.

This is not a reversal of the coordinate rule. `b` still lives on carrier edges and its *meaning* never depends on `ℓ`. What depends on `ℓ` is the *loss* — deliberately, because a boundary field should be judged on whether it separates the chunks that actually participate in computation. Keep raw edge-mass ratio as a diagnostic or weak prior.

Output structure decomposes into three questions, and the boundary field answers only the middle one:

- `C_extent` — how far the carrier reaches
- `b` — how chunks group into blocks
- `ℓ` — how many bytes each chunk produces

### 4.2 Shared denoiser, typed heads

One backbone, three typed input channels, three prediction heads. The channels share contextual computation but **need not share a corruption process**:

- `h` — continuous flow matching
- `b` — Bernoulli, logit-space, or absorbing
- `ℓ` — categorical or ordinal

Draft-versus-refine, outer iteration count, erasure ratio, entropy-gated stochasticity and boundary-commit rules are **sampling policies**, not separate models. The backbone is trained across mask modes and adapted to its own rollouts; nothing else needs its own weights.

### 4.3 Compute placement

The carrier stays at chunk resolution throughout. The expensive computation runs on transient block summaries — `z = P_b(h)`, `h' = U_b(z', h)` — where `P_b` softly pools chunks by the boundary field and `U_b` projects refined summaries back. Moving a boundary changes the pooling map without changing chunk identities.

**Pooling is what makes an oversized buffer affordable, not merely faster.** `C` is *allocated*, not generated — you pick a maximum and `ℓ` decides how much emits. Without pooling the denoiser runs over the whole buffer and pays `O(C²)` regardless of how short the output turns out. With pooling, inactive chunks contribute to no block, so the block count `M` tracks actual extent and the expensive network never sees unused canvas. That is the point of blocks, and it is load-bearing at the endpoint rather than an optimisation to add later.

**Compaction requires hard packing, and soft pooling does not provide it.** Dense soft pooling over `N` candidate block slots computes everything and weights it — no FLOPs saved. Genuine compaction needs discrete IDs and a packed sequence:

```
hard boundaries + activity
        ↓
prefix-sum block IDs
        ↓
scatter/reduce active chunks
        ↓
pack M non-empty blocks
        ↓
heavy denoiser over M
```

If a given regime uses fully dense soft pooling instead, **say so** — that regime does not receive the compaction saving and only hard inference does.

Batching note: `M` varies per example, so packing needs ragged or nested tensors, or padding to max-in-batch. Padding to batch-max still beats padding to buffer-`C`; the saving is real but not perfect.

**Straight-through does not automatically survive packing.** Straight-through works because the hard and soft tensors have the *same shape*. A ragged pack changes shape from `C` chunks to `M` blocks — autograd cannot differentiate prefix-sum indices, nor the appearance of a block slot that wasn't in the forward pass. "Hard forward, soft backward" is not free here; it needs a formulation.

**The staged answer, and it is the one §4.11 wants:**

1. At current scale, **skip compaction** and denoise the carrier directly under the soft boundary-derived mask. Cheap, and routing gradients are free.
2. At the scale transition, **compile to hard packed routing**.
3. **Freeze routing under packing** initially.
4. Add a packed-routing gradient estimator **only if** frozen packed routing leaves a measured quality gap.

The scale transition is the trigger, not a schedule. Below it there is nothing to compact; above it, measure before building the estimator.

**The fallback design, recorded so it isn't re-derived.** If step 4 fires: use a monotone chunk-to-block assignment matrix `A ∈ ℝ^{C × M_max}` with aligned slots *before* compaction —

```
A_st = sg(A_hard − A_soft) + A_soft
z_m  = Σ_i A_st,im · a_st,i · h_i  /  (Σ_i A_st,im · a_st,i + ε)
```

Forward uses hard assignments; empty hard slots are gathered out before the heavy denoiser; backward supplies a biased soft-assignment gradient to the selected slots.

**It still doesn't differentiate block birth and death cleanly** — no downstream gradient reaches a slot omitted from the packed forward pass. That part is carried by boundary supervision, the inserted/deleted-boundary regimes, and rollout training, not by the estimator.

**Three levels, three masks.**

- no pooling — the carrier denoiser uses `S_ij` directly (§4.4)
- with pooling — the heavy denoiser runs on `M` block summaries under the induced causal/global *block* mask
- a lightweight carrier updater then unpools to all `C` query positions and predicts `h, b, ℓ`

**That updater cannot be ordinary `C × M` cross-attention and also be called sub-quadratic** when `M ∝ C` — that term is `O(CM)`, which is `O(C²)` in disguise. Either use an SSM or local unpooler, or state the `O(CM)` cost honestly. The constraint on the chunk-level networks is real, and cross-attention violates it.

**At current scale, skip pooling anyway.** Roughly 21 chunk positions against 7 blocks; the compression ratio is ~3 and the buffer is small enough that quadratic on the whole thing is free. Denoise the carrier directly for the first implementation — but note this is a scale exemption, not evidence that pooling is optional.

### 4.4 Soft block masks

Probability that no boundary lies between *i* and *j*:

```
S_ij = ∏_{k = min(i,j)}^{max(i,j)-1} (1 - b_k)
log S_ij = Σ_k log(1 - b_k + ε)        # prefix sum
```

Draft mask:

```
M_ij = 1        if j <= i
     = S_ij     if j >  i
```

With binary boundaries this reduces **exactly** to bidirectional attention within the current block and causal attention across blocks — so the v0 masks are the limiting case and the equivalence is testable. Applied as an additive bias: `softmax(qᵀk/√d + log(M + ε))`.

**Refinement must keep global communication.** Either reserve a fraction of heads for unrestricted global attention, or use a floor `M^refine = ε_g + (1 - ε_g)·S_ij`. Without an escape path a same-block affinity makes refinement block-local and deletes the property the refinement pass exists for.

### 4.5 Circularity and bootstrap

The mask depends on `b`; `b` is predicted from representations computed under the mask. Break it by building each mask from the *incoming* boundary state with gradients stopped through mask construction — the newly predicted field affects the following step.

At maximum noise neither channel carries enough information to determine structure, so the **first function evaluation uses a fallback**: strict chunk-causal attention, or a fixed prior at the target block ratio. Subsequent evaluations use the predicted field.

**Extent needs the same bootstrap, for a sharper reason.** At `t = T` every `ℓ_i` is unknown, so the model does not yet know which positions are active — and therefore cannot pack (§4.3). Two options:

1. Treat all `C` positions as active on the first pass, losing the oversized-buffer saving exactly once.
2. Start every slot query-only and run a **cheap context-conditioned proposal pass** that predicts initial `ℓ` and `b` before the heavy denoiser is invoked at all.

```
prompt conditioning c
        +
C query-only carrier slots
        ↓
linear / sub-quadratic proposal pass
        ↓
initial activity and boundaries
        ↓
pack active blocks
        ↓
heavy denoising
```

Option 2 is cleaner and costs one sub-quadratic pass. It is the extent analogue of the boundary bootstrap above, and the two should run together.

### 4.6 Training — warm-start, then one curriculum

Three options, and the middle one is a trap:

- **Joint denoiser training** across all three channels — desirable, and the eventual regime.
- **Monolithic end-to-end from random init** — possible, unnecessarily unstable, and full of degenerate solutions (see below).
- **Warm-start, then joint fine-tuning** — the best bet, and what follows.

The five phases below are **initialisation and curriculum, not five separate models**. Only A and B are genuinely separate runs.

#### Phase A — carrier autoencoder (warm-start)

Bytes → carrier → bytes, no diffusion. Byte CE; `ℓ` supervised from the encoder's known expansion trace; router₁ with its ratio loss; and the COSMOS intervention — encode *perturbed* inputs, reconstruct *clean* targets.

*Gates:* byte reconstruction, latent-use check, interpolation smoothness, clustering geometry.

Diffusion needs a reasonably stationary target. Without this warm-start the encoder can keep moving or rescaling the space to make the denoising loss easier, and the whole thing chases itself.

#### Phase B — supervised boundary head (warm-start)

Freeze the carrier; train router₂ to predict `b` from clean states. Synthetic data with deliberately non-trivial rendering — multiple operations per line, operations split across lines, stray whitespace — gives genuine semantic boundary labels, so the mechanism is verifiable before it has to work unsupervised.

*Gate — two parts:* boundary recovery against ground truth, **and** that the recovered boundaries improve downstream regeneration. A router that nails the labels without helping is a null result wearing a pass.

For natural text, GeoBlock-style dependency geometry is a weak prior or pseudo-target, not ground truth — it selects local update granularity, not hierarchical codec segmentation.

#### Two encoders — what shares, what doesn't

The corrected diagram introduces a **context encoder** (reads the prompt, produces conditioning `c`) alongside the **carrier encoder** (manufactures training targets, never runs at inference). They need a stated relationship or the context path has no training story at all.

**Share the byte/router₁ frontend.** Then:

- context path — a conditioning projection on top of the shared frontend
- carrier path — the `h, b, ℓ` target heads and the decoder

The shared frontend is initialised in Phase A. The context projection trains from Phase C onward through the conditional denoising objective — it has no separate loss. **EMA applies only to the carrier target encoder in Phase E**, not to the context encoder, which isn't producing a moving diffusion target.

If they end up as fully separate networks instead, the context encoder needs its own initialisation, objective, unfreezing rule and gate — four things the shared version gets for free.

#### Phase C onward — one joint run with sampled regimes

From here it is a single continuous training run. Every batch samples a regime; the mixture shifts over time.

One training step:

```
E_target = E_frozen              # phases C–D
E_target = EMA(E_trainable)      # phase E onward

c = E_context(prompt)
(h₀, b₀, ℓ₀) = E_target(output)

τ ~ per example or carrier region                 # [B, C] — carrier coords
ν_h[i] = τ[i]
ν_ℓ[i] = τ[i]^β
ν_b[k] = edge_lift(τ[k], τ[k+1])^α                # [B, C−1]

(h_t, b_t, ℓ_t) = corrupt(h₀, b₀, ℓ₀; ν_h, ν_b, ν_ℓ)      # typed kernels

b_mask = sg(b_t) + γ_mask · (b_t − sg(b_t))

if direct_carrier:
    routing_t = soft_mask(b_mask, activity_st(ℓ_t))
    cond_t    = carrier_noise_embedding(ν_h, ν_b, ν_ℓ)
else:
    routing_t = hard_pack(b_t, ℓ_t)          # γ_pack / A_st path, if enabled
    cond_t    = resample_noise(ν_h, ν_b, ν_ℓ, routing_t)

(ĥ₀, b̂₀, ℓ̂₀) = D_θ(h_t, b_t, ℓ_t, routing_t, c, cond_t)
```

**The two branches must be written separately or `γ_mask` is dead code.** In the direct-carrier path `routing_t` is built from `b_mask`, so `γ_mask` genuinely controls whether content gradients reach `b`. In the packed path `routing_t` is built from `b_t` through a hard prefix sum — `γ_mask` cannot touch it, and only `γ_pack` could. A single `mask_or_pack(b_t, …)` call would silently disconnect `γ_mask` while appearing to use it, which is exactly the conflation §4.6 rules out.

**`edge_lift` is a policy that has to be stated.** `τ` is position-indexed `[C]` and `ν_b` is edge-indexed `[C−1]`, so a lift is required — maximum, mean, or an independently sampled edge field. **Maximum** is the natural default for local erasure: if either adjacent chunk is being touched, the edge between them is structurally uncertain.

All four losses applied together. **Pass all three noise levels explicitly** — a single scalar `t` is ambiguous once the channels run on different schedules (§4.8).

**Regimes sampled per batch:**

```
bootstrap proposal          ← the inference path starts here; train it
clean carrier reconstruction
fixed-boundary denoising
boundary jitter (fixed count)
inserted / deleted boundary
extent-only corruption
fully joint corruption
model-rollout refinement
unrolled (random outer depth)
```

**The bootstrap regime, spelled out.** §4.5's inference path begins with a cheap proposal pass over unknown `b` and `ℓ` — a state no other regime produces. Without it the proposal network only ever sees an inference-only input distribution.

```
input   h_T sampled from the prior
        b_T = UNKNOWN
        ℓ_T = UNKNOWN
        C query-only slots
        prompt conditioning c
        no packed block sequence
target  initial b₀ and ℓ₀  (optionally a coarse h₀)
loss    L_boundary + L_length  (+ L_denoise(h) if h is predicted)
```

Weight it **early and heavily**, before fixed-boundary denoising — everything downstream consumes its output.

*Gate:* emitting-position precision **and recall**; `C_extent` error; effective block-count error; fraction of target chunks included in the initial pack; and downstream draft quality under proposed versus teacher structure. Recall matters more than precision here — a false positive costs compute, a false negative excludes useful state from the heavy core entirely.

Early training weights clean and fixed-boundary heavily; joint corruption and rollout regimes grow over time. Attribution is preserved because every regime has its own metric, measurable at any point in the run.

**Two exposure anneals, annealed independently.** Scheduled sampling for the receptive field:

```
clean teacher state → corrupted ground truth → one-step model prediction
    → state from a complete draft → state from an outer refinement rollout
```

`ℓ` has exactly the same exposure gap as `b` — teacher-forced from the encoder at training, predicted at inference. Annealing one and not the other fixes the structure gap and leaves the extent one.

**Anneal them on separate schedules**, so boundary prediction can move off teacher forcing without simultaneously exposing the model to unreliable extent predictions. Attribution survives; a simultaneous anneal makes a regression un-attributable.

#### Internal deletion — three things it requires

`ℓ_i = 0` is legal at any position (§4.1), and decode concatenates non-empty expansions in carrier order. Deleting an internal chunk leaves a **hole** rather than shifting later positions, which is the whole point of the carrier. Three consequences that need deciding rather than discovering:

**1. Rollout targets must preserve carrier alignment.** When a clean target deletes or inserts content relative to the draft, align its chunks to the persistent carrier and supervise the unmatched positions with `ℓ_i = 0`. Without this, training teaches the model to **refill** internal holes rather than use them as deletions — which silently removes the deletion mechanism while every metric looks fine.

**2. Inactive positions are key/value/pool-gated by straight-through, not by a soft probability.** Per §4.1, `ℓ_i = 0` leaves the position query-active so it can propose reactivation, while `a_st` gates its key, value and pooling contributions to *exactly* zero in the forward pass. Monitor for leaks: check whether an inactive position's hidden state influences any *other* position's output. It should be exactly zero, not small — if it isn't, the straight-through has been implemented as a soft multiply.

**Interaction with §4.3.** The query-active rule is implemented by an `O(C)` **masked carrier scan** (§5.4), not by attention — the carrier path has no Q/K/V. The quadratic operation runs only over the `M·K` packed summary positions, where hard packing has already dropped inactive chunks.

**3. Dense-prefix preference is a diagnostic, not a term.** Free placement of zeros gives more degrees of freedom than strictly needed. Measure whether unnecessary internal sparsity harms quality, compute or optimisation; add a regulariser only if it does.

**Boundary supervision.** Targets come from the Phase-B router on clean data, which is distillation and caps quality at the router's. Acceptable *because* Phase B's gate already requires those boundaries to help downstream. The router unfreezes in Phase E so the loop can exceed it. On synthetic data with ground truth, use ground truth.

**Coupled trajectories.** If diffusion times are sampled independently per channel, the boundary state used to build the mask must come from the **same coupled forward trajectory** as the prediction target — otherwise the training mask corresponds to no point on any inference trajectory.

#### The re-segmentation ladder

Within the joint run, the *inference-time* re-segmentation policy climbs a ladder. Each rung gates the next:

1. **Frozen** draft segmentation — the required baseline
2. **Recompute without committing** — measure how much the proposed segmentation *would* change
3. **Shift-only** — boundaries move locally, block count fixed
4. **Local split/merge** — variable count inside anchored regions
5. **Unrestricted** — only if 4 shows clear gains

Rung 2 is the cheapest and most informative: if boundaries barely move, or movement doesn't correlate with recoverable errors, freezing is the correct engineering answer and most of this section is unnecessary. Rung 3 tests whether segmentation/content co-adaptation matters without introducing block birth and death.

Whatever constrained-commit rule inference uses must also appear in the rollout regime — otherwise the model never trains on the boundaries it will actually see.

#### Phase E — unfreeze the encoder

Lower learning rate, **EMA copy as the diffusion target**. EMA matters here and only here: while the encoder is frozen the target is already stationary and EMA buys nothing; once it moves, the denoiser needs a slowly-moving target.

Retain clean autoencoder batches throughout, or the latent space and decoder drift together while the diffusion loss looks healthy.

#### Safeguards

1. **Warm-start the carrier** (Phase A) — a moving latent target is the main instability.
2. **EMA encoder targets** — from Phase E, not before.
3. **Stop gradients through mask construction** — initially. See below for when to open it.
4. **Hard-forward, soft-backward routing** — teacher boundaries or predicted hard boundaries determine the packed forward sequence; soft monotone assignments or a straight-through surrogate supply routing gradients. Regimes using dense soft pooling receive no compaction saving and are labelled as such (§4.3).
5. **Retain clean autoencoder batches** — byte reconstruction stays part of the content objective rather than becoming a separate conceptual term.

**On opening the mask gradient path.** Safeguard 3 is necessary but cannot be permanent: while it holds, `b` is trained only by imitation of the Phase-B router and can never learn boundaries that are good *for the denoiser*, which caps the system at the router's quality forever.

Make it a dial rather than a switch:

```
b_mask = sg(b) + γ_mask · (b − sg(b))
```

`γ_mask = 0` is complete gradient isolation, `1` is full end-to-end gradients through the **dense soft carrier mask**. The forward mask is identical for every value — only the backward path changes — so opening it cannot alter behaviour except through learning.

**`γ_mask` and `γ_pack` are different dials.** `γ_mask` opens content gradients through the dense soft carrier mask. It does *nothing* for hard prefix-sum packing: once the model runs the packed block core, the induced block mask is hard and no value of `γ_mask` makes an omitted block differentiable. That needs `γ_pack` — content gradients through the monotone assignment surrogate `A_st` (§4.3) — which does not exist until that estimator is built.

```
debug scale :  γ_mask 0 → 1 under the gates below ;  γ_pack nonexistent
packed scale:  routing initially frozen           ;  γ_pack = 0
```

Raise `γ_pack` only if frozen packed routing leaves a measured quality gap, and give it its own prerequisites in the curriculum controller.

**The ratio loss is not a sufficient safety condition on its own.** The danger in opening the path is that the content loss rewards whatever `b` makes reconstruction easiest, i.e. near-global attention. Low boundary mass is *one* way to get that and the ratio loss does penalise it — but the ratio constrains a **scalar**, and the same total mass can be edge-packed or smeared so that every `S_ij` sits mid-range. Placement is unconstrained by a mass budget.

So the conditions for raising `γ_mask` are all of:

- supervised boundary metric has cleared its gate
- boundary mass matches a target calibrated from clean data
- ratio loss active
- diffuse-boundary and edge-concentration diagnostics acceptable

And **boundary supervision stays active as an anchor** throughout, rather than being retired once the gradient path opens. If content gradients start trading against the ratio penalty, replace the penalty with an exact or dual-constrained boundary budget rather than raising its weight indefinitely — a penalty you keep increasing is a constraint you should have stated.

#### Degenerate solutions — each with its measurement

Not hoped against. Monitored, throughout, with the specific check named:

| Failure | Measurement |
|---|---|
| Encoder collapse or rescaling | latent covariance, effective rank, interpolation probe, decoder sensitivity |
| Diffuse boundary mass at correct ratio | `E[b(1−b)]`, boundary entropy, mass concentration |
| Edge-packing at correct ratio | boundary position histogram; mass in the first/last few positions |
| `b` optimising receptive-field convenience | boundary quality vs. performance under fixed-boundary-count interventions |
| Decoder ignoring `b` | perturb or replace boundaries, hold `h, ℓ` fixed, measure decode change |
| Decoder ignoring `ℓ` | perturb expansion lengths, hold `h, b` fixed |
| Length leaking into `h` | remove or shuffle `ℓ`, measure retained reconstruction |
| Zero-length scratch positions | ablate their hidden states and attention participation |
| Moving-target instability | distance between trainable and EMA encoder, after unfreezing |
| Boundary mass hidden in inactive gaps | raw boundary mass vs. effective cut rate `r_eff` |
| Redundant cuts inside one gap | cuts per consecutive-emitter interval |
| Packed routing defeating the ratio constraint | effective block count vs. target block count |
| Exposure gap | teacher vs. predicted `b` and `ℓ` at matched noise (→ the anneals above) |

Monitoring is cheap; loss terms are expensive. Per §4.11, each stays a measurement until it actually fires.

### 4.7 Objective

Four terms:

```
L = L_denoise(h) + λ_ℓ·L_length(ℓ) + λ_b·L_boundary(b) + λ_r·L_ratio(b)
```

Content, extent, structure, and the two boundary-collapse extremes. Ratio measured as effective separation between consecutive emitting chunks, not raw edge mass (§4.1).

Sharpness, cycle consistency and inter-iteration stability start as **diagnostics**. Add a term only when the corresponding measurement exposes a real failure. The likely addition is `L_sharp = E[b(1-b)]`, since ratio loss constrains total boundary mass without distinguishing one confident cut from mass smeared across several positions.

### 4.8 Structure-first schedule

Structure and extent should resolve faster than content. With `τ ∈ [0,1]` the master level on carrier coordinates (§4.6):

```
ν_h[i] = τ[i]
ν_ℓ[i] = τ[i]^β
ν_b[k] = edge_lift(τ[k], τ[k+1])^α          α, β > 1
```

`ν_b` is edge-indexed `[C−1]` while `τ` is position-indexed `[C]`, hence the lift — see §4.6 for the policy.

The mask *is* the receptive field, so a wrong skeleton corrupts everything downstream in that trajectory. This doesn't fix the maximally-noisy first evaluation — hence the bootstrap in §4.5 — and it must be ablated against equal schedules, since resolving boundaries too aggressively could lock in a structure the emerging content doesn't support. The global refinement heads are the escape path from that failure.

### 4.9 Inference

1. Initialise `C` query-only carrier slots and prompt conditioning `c`
2. Run the **cheap carrier proposal pass** — predicts initial `b` and `ℓ` (§4.5)
3. **Hard-pack** active chunks into `M` block summaries
4. Chunk-causal draft through the heavy block denoiser
5. **Unpool** and predict updated `h, b, ℓ`
6. Select uncertain **carrier regions** by content, length and boundary entropy
7. Erase, **repack**, and refine globally with mixed global and boundary-aware attention
8. Commit structural changes according to the current ladder rung
9. Repeat for the chosen outer depth, then decode once

Steps 6–8 are policy. Only learned halting adds a head and an objective, and it is last.

The heavy core is one stage of a four-part inner cycle, not the whole thing:

```
carrier proposal / update  (sub-quadratic, over C)
            ↓
      hard pack  P(h, b, ℓ)
            ↓
   heavy block core  (over M·K)
            ↓
 sub-quadratic unpool / update  (over C)
```

### 4.10 What would falsify this

Stated explicitly, because this project has twice lost sessions to a metric measuring the wrong thing.

- **Learned segmentation beats the heuristic on downstream quality at matched compute** — not on boundary F1. A router can win on boundary agreement while making generation worse.
- **Mutable boundaries beat frozen boundaries across refinement iterations.** This is the re-segmentation ladder's rung 2 promoted from diagnostic to gate: refine under frozen boundaries, re-run the router on the result, and test whether boundary disagreement predicts where refinement still fails. If it doesn't, freeze and stop.

Track throughout: task accuracy at matched NFE; boundary precision/recall with positional tolerance; boundary churn between iterations; split/merge rate; mean boundary displacement; fraction of content changes triggering distant boundary changes; quality after *each* iteration rather than only the last; and performance separated by unchanged / shifted / split / merged cases.

### 4.11 The complexity rule

Three channels, three corruption processes, five stages, four-plus loss terms, two schedule exponents, a bootstrap rule and constrained commits. Each is individually justified; collectively it is a system where a bad result has thirty candidate causes.

**No loss term is added until a measurement shows the failure it prevents.** And since §4.6 is one continuous run rather than gated stages, the gate discipline survives as **curriculum control**: a regime's share of the mixture does not increase until the regime beneath it clears its bar.

- bootstrap-proposal weight is high from the start; everything downstream consumes its output
- boundary-corruption weight rises only after supervised routing passes
- extent-corruption weight rises only after fixed-structure length prediction passes
- joint corruption rises only after fixed-structure, boundary-only and extent-only pass **simultaneously**
- rollout weight rises only after joint denoising *improves* its input rather than degrading it
- unrolled depth rises only after one-step rollout improvement is reliable
- `γ_mask` rises only under the four conditions in §4.6; `γ_pack` only after frozen packed routing shows a measured gap
- the encoder unfreezes only after the whole frozen-encoder loop is stable
- **and any weight comes back down if its prerequisites regress**

That last clause is what makes it a controller rather than a schedule. Same attribution, same refusal to build on a failing foundation, one training run.

The rule has already earned its place — live-dimension shrinkage, the clustering hypothesis and the exposure-bias story were all convincing measurements that ablation showed were not causal.

### 4.12 Scope

Router₁ stays fixed throughout the latent refinement loop. Bytes aren't regenerated until the final decode, so changing byte-level segmentation mid-loop would need a partial decode/re-encode cycle. Router₂ is where the mechanism gets established.

---

## 5. Layer architecture

Notation, fixed here and used throughout:

| | |
|---|---|
| `C` | carrier capacity — allocated chunk positions |
| `M` | packed block count |
| `K` | latents per block summary (4) |
| `S` | inner denoising steps |
| `R` | outer refinement iterations |
| `d` | carrier channel width |

### 5.1 Shape pipeline

```
  bytes                          [B, C_bytes]
    │
    │  Embedding(259) → Mamba-2 ×6 → router₁
    ▼
  carrier                        [B, C, d]          C ≈ C_bytes / 4.5
    │
    │  channel fusion + carrier proposal (SSM)
    ▼
  carrier state                  [B, C, d_model]
    │
    │  activity gate → prefix-sum IDs → pool
    ▼
  block summaries                [B, M, K, d_model]
    │
    │  ═══ HEAVY DiT CORE ═══  ×S inner steps
    ▼
  refined blocks                 [B, M, K, d_model]
    │
    │  gather by block ID + residual (SSM)
    ▼
  carrier                        [B, C, d_model]
    │
    ├──→ h-head    [B, C, d]              velocity
    ├──→ b-head    [B, C−1]               boundary
    └──→ ℓ-head    [B, C, L_max+1]        extent
```

**Pool only when `C/M > K`.** At debug scale a reasoning step is ~3 chunks against `K = 4`, so pooling would *expand* — which is why §4.3 skips it there. At endpoint scale a step is ~11 chunks, giving ~2.75:1.

### 5.2 Byte frontend — SSM, not attention

Shared by the context and carrier encoders (§4.6).

```
  byte_ids
    │
    ▼
  ┌──────────────────────────────┐
  │ Embedding(259, d_small)      │   256 bytes + PAD/BOS/EOS
  └──────────────────────────────┘
    │
    ▼
  ┌──────────────────────────────┐
  │  ┌────────────────────────┐  │
  │  │ RMSNorm → Mamba-2      │──┼─⊕   ×6
  │  │ RMSNorm → SwiGLU MLP   │──┼─⊕
  │  └────────────────────────┘  │
  └──────────────────────────────┘
    │
    ▼
  ┌──────────────────────────────┐
  │ router₁                      │
  │   q = W_q h ,  k = W_k h     │
  │   p_t = (1 − cos(q_t,k_{t−1}))/2
  │   downsample where p > 0.5   │
  └──────────────────────────────┘
    │
    ▼
  carrier chunks
```

Mamba-2 rather than transformer blocks: this runs over raw bytes at full length, so it must be linear, and SSMs carry the compression inductive bias H-Net relies on. `p_t` is high when adjacent representations are *dissimilar* — a boundary is a discontinuity.

### 5.3 Channel fusion — three channels, one stream

```
  h_t  [d]        ──→ W_h ──────────┐
  b_t  [1]        ──→ fourier → W_b ─┼──→ ⊕ ──→ x_t  [d_model]
  ℓ_t  [categorical] ──→ Embed_ℓ ────┘
```

Continuous gets a linear map, scalar-in-[0,1] gets Fourier features then projection, categorical gets a table lookup. **Sum, don't concatenate** — `d_model` stays fixed as channels are added.

### 5.4 Carrier proposal / update — sub-quadratic over C

```
  x  [B, C, d_model]  +  learned carrier-position embedding
    │
    ▼
  ┌──────────────────────────────┐
  │  ┌────────────────────────┐  │
  │  │ adaLN(cond)            │  │
  │  │ u_i = a_st_i · RMSNorm(x_i)
  │  │ y   = SSM(u)  (scan)   │──┼─⊕   ×n
  │  │ RMSNorm → SwiGLU MLP   │──┼─⊕
  │  └────────────────────────┘  │
  └──────────────────────────────┘
```

**The activity gate has no Q/K/V here.** §4.1 states it as key/value gating, which describes the attention core; the carrier is Mamba and has no separate key path. The SSM realisation is: gate a position's contribution **into the shared recurrence**, while letting it still *receive* the scan's output.

- `u_i = a_st_i · RMSNorm(x_i)` — inactive positions inject nothing into the recurrent state. Use `a_st`, not `a_hard`: they are identical in the forward pass, and writing `a_hard` throws away the length-routing gradient
- `x_i ← x_i + y_i` — inactive positions still receive context, which is what lets them propose reactivation
- **reapply the gate at every carrier SSM layer**, not once at the input
- positionwise MLPs may update inactive slots freely; they cannot reach other positions

`a_st` is exactly hard in the forward pass, so non-conditioning is exact; the surrogate only affects the backward path.

**Scan direction follows mode — and a `mode` embedding cannot do this on its own.** A Mamba layer is causal by construction; switching to bidirectional requires a second scan, not a conditioning signal. Specify both:

```
y_f = SSM_forward(u)

draft :  y = y_f
refine:  y_r = reverse(SSM_reverse(reverse(u)))
         y   = y_f + y_r
```

State whether the two scans share parameters. The activity gate is reapplied to the recurrence input in **both** directions. This also makes the cost asymmetry explicit: refinement's carrier updater is two linear scans, drafting is one.

Runs over all `C` positions, so SSM by the §4.3 constraint. Used for both the bootstrap proposal (§4.5) and the post-unpool update. Position information is a learned or sinusoidal embedding, **not RoPE** — Mamba has no Q/K vectors to rotate.

### 5.5 Pack

```
  activity      a_st = sg(a_hard − p_active) + p_active     [B, C]
  boundaries    b                                            [B, C−1]
       │
       ▼
  block_id[i] = Σ_{k<i} b_hard[k]      # EXCLUSIVE prefix sum, O(C)
       │
       ▼
  within-block resample to K summaries                       O(CK)
       │
       ▼
  drop empty blocks, pack ragged → [B, M, K, d_model]
```

**The prefix sum must be exclusive.** `b_i` means "boundary after position `i`", so position `i` belongs to the block determined by boundaries strictly *before* it. An inclusive `cumsum(b)[i]` puts the position immediately before a boundary into the *following* block — an off-by-one that silently misassigns one chunk per block. Unit-test it against singletons, adjacent boundaries, inactive gaps and the terminal edge.

**A scatter-reduce gives `[B, M, d]`, not `[B, M, K, d]`.** One reduction cannot manufacture `K` distinct summaries per block. Use a fixed-`K` within-block resampler — `K` learned queries with a relative coordinate inside the block:

```
α_ik = softmax_{i ∈ I_m} ( q_kᵀ W h_i  +  φ_k(r_i) )
z_mk = Σ_{i ∈ I_m} α_ik · h_i
```

`r_i` is the chunk's relative position within its block. Because `K` is fixed this is `O(CK)`, not `O(C²)` — the softmax is over each block's own members, not the whole carrier.

**If that machinery isn't wanted, set `K = 1`** and a plain scatter-reduce is exactly correct. Worth trying first: `K = 1` is the honest baseline and `K > 1` has to earn itself.

Hard forward, soft backward (§4.3). No attention over the full carrier at this stage.

### 5.6 Heavy DiT core — the only quadratic component

Over `M·K` positions, not `C`.

```
  z  [B, M·K, d_model]
    │
    ▼
  ┌───────────────────────────────────────────────┐
  │                                               │
  │   shift₁,scale₁,gate₁ ← adaLN-Zero(cond_{m,k}) │
  │            │                                  │
  │            ▼                                  │
  │   RMSNorm ─ ×(1+scale₁) + shift₁              │
  │            │                                  │
  │            ▼                                  │
  │   Attention ── QK-norm ── block mask, mode    │
  │            │                                  │
  │            ▼                                  │
  │   × gate₁ ──────────────────────────────── ⊕ │
  │                                               │
  │   shift₂,scale₂,gate₂ ← adaLN-Zero(cond_{m,k}) │
  │            │                                  │
  │            ▼                                  │
  │   RMSNorm ─ ×(1+scale₂) + shift₂              │
  │            │                                  │
  │            ▼                                  │
  │   SwiGLU MLP (4× expansion)                   │
  │            │                                  │
  │            ▼                                  │
  │   × gate₂ ──────────────────────────────── ⊕ │
  │                                               │
  └───────────────────────────────────────────────┘
                        ×12
```

Three things worth stating:

- **adaLN-Zero makes the heavy core an identity map at initialisation** — not a zero-velocity denoiser. Zero velocity additionally requires the `h`-head to be zero-initialised (§5.10). Boundary and extent heads instead start at their empirical priors. With both zeroed the *content* velocity is zero at init — but `b` and `ℓ` start at their empirical priors, so the full `(h, b, ℓ)` trajectory is not static. Content sitting still while structure moves is correct behaviour at init, and looks like a broken sampler.
- **QK-norm.** RMSNorm on Q and K before the dot product. Diffusion training destabilises without it at depth.
- **The mask is over flattened `(m,k)` positions, not carrier `S_ij`.** Draft: `M_{(m,k),(n,l)} = 1[n ≤ m]` — unrestricted among the `K` summaries within a block, plus all earlier blocks. Refinement: global, with the §4.4 floor. `S_ij` is for direct-carrier mode and soft pooling only.
- **RoPE here, and only here** — this is the sole attention stage. Use the block's carrier *span* or centre as its coordinate, not its packed ordinal, since blocks have variable width.
- **`cond_{m,k}` is derived from carrier-coordinate noise** after the incoming partition is packed — not sampled per block. See §5.7.
- **Boundary-edge noise needs its own pooling rule.** The chunk resampler weights cover `[C]` positions; `ν_b` is `[C−1]` edges and is not directly covered by them. Define an edge-to-summary rule (which edges belong to block `m`, and how they pool) rather than assuming the content weights apply. Open item.

### 5.6a Weight sharing — `L` distinct blocks, `Q = 12/L` cycles

**Two separate ideas, and only the first is in scope.**

1. **Weight sharing at fixed compute** — fewer physical blocks, still twelve block applications, achieved as `Q` cycles of the physical stack. This is the experiment.
2. **Variable recurrent depth** — train for arbitrary loop counts, exit early. Deferred; see below.

Conflating them imports machinery the first doesn't need.

#### The experiment: fixed compute

```
core = (B₁, …, B_L)^Q      Q = 12 / L
```

Train through all twelve applications. Backprop through all of them; use gradient checkpointing if memory demands it. **No randomised unrolling, no truncated backpropagation, no new stopping mechanism.** `L = 12` is the reference and sharing has to earn itself against it.

At `d = 768`, ~13M params per block:

| `L` | `Q` | params | optimiser state (~16 B/param) |
|---|---|---|---|
| 12 | 1 | 156M | ~2.5 GB |
| 4 | 3 | 52M | ~0.8 GB |
| 3 | 4 | 39M | ~0.6 GB |

Parameter and optimiser memory only. Activation memory is unchanged at fixed compute — that saving belongs to truncated backprop, which belongs to variable depth.

#### What the literature actually says

Sharing is a *reallocation*, not a free improvement. Pretraining at 250B tokens found looped models of the same effective depth carry a stronger inductive bias toward **reasoning at the cost of memorisation and perplexity**; FLOP-matched comparisons confirm the same trade. Length generalisation improves substantially. Theoretically, looped transformers with padding solve parallelisable problems in ways fixed-depth ones cannot — the same TC⁰ boundary as §4.

The trade points the right way for the synthetic and CoT stages, where fact recall is nearly irrelevant. **It inverts at the general-text endpoint.** Mitigation if needed: a regulariser encouraging layers to stay *close* rather than fully tied, making the trade continuous.

#### Depth conditioning — unresolved, test it

I previously listed a layer-index embedding as required. That is **not settled**. There is a reasonable argument against: repeating the same function on a changing state is what makes a model recurrent, and conditioning on the recurrence index may undercut the path independence that recurrent-depth models exhibit.

At fixed `Q` it is cheap to test both ways — one embedding into a conditioning sum that already exists. Run it as an ablation rather than assuming either answer.

#### Deferred: variable `Q`

If fixed-compute sharing succeeds, variable depth becomes a separate gated experiment. It needs randomised unrolling — with truncated backpropagation only once sampled depths exceed the full-BPTT memory budget — and its own state estimate and gate — and note that **`Q` is not `R`**. Core-loop depth and outer refinement depth are different loops; giving `Q` an early exit creates a *third* adaptive-compute mechanism alongside §3.3's.

**Two claims I overstated and am withdrawing:**

- *"Monotonic depth-wise loss is possible only with weight sharing."* The result behind that is a theorem about restricted in-context linear regression, not a guarantee about denoising quality here. Separately, one 2026 study on compositional reasoning found final-only supervision preferable to intermediate losses, which it associated with heuristic shortcuts — three domains, not a general result about depth-recurrent models. **Treat intermediate supervision as an ablation, not a default**, and don't build the early-exit story on it either way.
- *"Depth conditioning is required."* See above.

#### Practical

**Don't sweep this at debug scale** — two layers at 128 wide is ~400k parameters and sharing saves nothing measurable. Sweep at `v0.yaml` scale, and try gradient checkpointing and gradient accumulation first since neither costs quality.

### 5.7 Conditioning


```
  ν_h ──→ fourier ─┐
  ν_b ──→ fourier ─┼──→ concat ──→ MLP ──┐
  ν_ℓ ──→ fourier ─┘                     │
  mode  ──→ Embed ───────────────────────┼──→ ⊕ ──→ cond
  iter  ──→ Embed ───────────────────────┤
  layer ──→ Embed ───────────────────────┤   ← optional ablation when L < 12
  c     ──→ mean-pool ───────────────────┘
```

**Three separate noise levels**, because the channels run different schedules (§4.8) — a single scalar `t` is ambiguous.

**Noise lives on carrier coordinates, not on blocks.** Blocks split and merge under corruption, so a noise tensor indexed by the clean block count has no stable mapping to `h_t`, `b_t`, `ℓ_t`, or the current packed membership. A clean block that splits — or two differently-noised blocks that merge — would have no well-defined `cond_{m,k}`.

```
ν_h  [B, C]        ν_ℓ  [B, C]        ν_b  [B, C−1]
```

A master level may still be sampled **per clean block or per corruption region** and broadcast onto its carrier positions; the channel schedules (§4.8) apply elementwise. After packing, derive `cond_{m,k}` by resampling the member noise embeddings **with the same assignment weights that produce `z_{m,k}`**. Pooling Fourier embeddings preserves a block's mixed noise levels better than pretending it has one exact scalar.

Packed-summary conditioning remains block-local, but it is no longer per-block in the strict sense: different summaries `k` within one block may receive different pooled noise embeddings. The *sampling* happens in a coordinate system that survives boundary changes. (The simplest starting alternative is one scalar triplet per example. Either is coherent — `[B, M]` before mutable segmentation is not.)

`c` mean-pooled is the simple option. **Flag it as a suspected bottleneck**: for exact arithmetic and code, a single pooled vector is a severe constraint on prompt detail. Keep it as the baseline, but gate it against cross-attention or a small set of learned context summaries on tasks that need exact prompt content. Empirical, not a consistency issue.

### 5.8 Unpool — gather plus residual

```
  z'  [B, M, K, d_model]
    │
    │  within-block resample back, O(CK)
    ▼
  ┌──────────────────────────────────────────┐
  │  u_i  = Σ_k β_ik · W_u · z'_{m(i),k}     │
  │  h'_i = h_i + SSM(u_i)   ← carrier residual
  └──────────────────────────────────────────┘
    │
    ▼
  [B, C, d_model]
```

`β_ik` depends on the chunk's relative position within its block — **symmetric with the pack resampler**. A flat `z'[block_id[i]]` broadcast would hand every chunk in a block the same vector, discarding the `K` summaries the core just produced. (With `K = 1` the broadcast is correct and this collapses to a gather.)

**Not `C × M` cross-attention** — that would be `O(CM)`, i.e. `O(C²)` when `M ∝ C` (§4.3). The resampler is `O(CK)` with `K` fixed.

The **residual** matters independently: the sub-quadratic stages *refine* the carrier rather than replacing it, so gradients reach `h` without passing through the pack/unpool bottleneck.

### 5.9 Heads and decoder

```
  h-head:  RMSNorm → Linear(d_model → d)                  velocity target
  b-head:  RMSNorm → Linear(d_model → 1) → sigmoid        per carrier edge
  ℓ-head:  RMSNorm → Linear(d_model → L_max+1) → softmax  categorical
```

`ℓ` is categorical rather than regression: the 0-versus-1 distinction is a discrete decision about *existence*, and a regression head sits awkwardly across it.

```
  h_i ──→ broadcast to ℓ_i positions + byte-offset embedding
       ──→ Mamba-2 ×4
       ──→ Linear(d → 259) ──→ byte logits
```

Ragged across chunks — pack by `ℓ` and run one batched pass.

### 5.10 Conventions, stated once

| | |
|---|---|
| Norm | RMSNorm everywhere, pre-norm |
| MLP | SwiGLU, 4× expansion |
| Position | RoPE in the heavy core only; learned/sinusoidal embeddings in every SSM stage |
| Precision | bf16, fp32 loss accumulation |
| Init | see below — **never zero both multiplicative paths in one branch** |
| Attention | *quadratic* attention only in the heavy core; pooling uses fixed-`K` within-block attention at `O(CK)`; everything else over `C` is SSM |
| Depth | 12 layers of compute from `L` distinct blocks — see §5.6a |

**Initialisation, stated carefully because getting it wrong kills the core.** For a gated branch `y = x + g · W_o · f(x)`, zero-initialising *both* `g` and `W_o` gives `∂y/∂g = W_o f(x) = 0` and `∂y/∂W_o = g f(x) = 0`. Neither parameter receives gradient. The branch is **dead**, not merely quiet.

- **heavy DiT** — adaLN-Zero gates zeroed; attention and MLP output projections initialised **normally**
- **ungated carrier residual blocks** — zero-init the residual output projection alone (valid, since there is no second zero)
- **h-velocity head** — optionally zero-init, if an initial zero field is wanted
- **b- and ℓ-head biases** — initialise from empirical boundary and extent priors, not zero

---

## 6. Build order

Each stage ends with a measurement that decides whether the next is worth starting.

| # | Stage | Gate |
|---|---|---|
| 0 | **v0 result (M7)** | Does global regeneration beat causal? If not, none of the below matters. |
| 1 | COSMOS latent recipe | Interpolation probe improves; clustering statistic moves toward the null. |
| 2 | Entropy-gated sampling | Beats uniform η at matched NFE. |
| 3 | Scale — proper `v0.yaml` run, or rented compute | Do the small-scale conclusions survive? This is where the compute threshold for compaction is established, using existing fixed/heuristic blocks — learned hard packing only becomes available once stage 4 supplies router₂. |
| 4 | Natural CoT data **and** learned block boundaries (router₂) | Coupled, one stage. On synthetic arithmetic the newline heuristic is already optimal, so a router trained there learns "newline means boundary" and discovers nothing. It only earns its keep once step boundaries stop being obvious — which is what natural CoT supplies. Gate: does the router beat the heuristic on downstream quality at matched compute? |
| 5 | Byte-level input (router₁) | Two-stage hierarchy trains stably. |
| 6 | Refinement over learned segmentation | §4, climbed as a ladder. Rung 2 (recompute without committing) may end it early — if boundaries barely move, freeze and stop. |
| 7 | Adaptive ρ, then R via rungs 1 and 2 | §3. Adaptive ρ first — it bounds the overthinking risk. Rung 1 is free; rung 2 is a post-hoc head that interferes with nothing. |

Note on stage 6: it reuses the carrier and router warm-starts from §4.6 phases A and B. The novel work begins at joint boundary corruption, rollout alignment, and the re-segmentation ladder — and the ladder's rung 2 may terminate the branch before mutable segmentation is implemented at all.

---

## 7. What this does not change

The organising principle from `design.md` §1 stands unaltered: the model thinks in continuous space, and tokens exist only at the two interfaces. Everything above is that principle applied consistently — the endpoint is the version where no linguistic prior is hand-designed anywhere.

The freeze also stands. Nothing in this document is in scope until M7 returns a number.

---

## 8. Reading list, in build order

1. **COSMOS** — arXiv 2506.21170. The autoencoder recipe. Read §5 (latent-space properties) closely.
2. **Entropy-gated bitstream diffusion** — arXiv 2605.07013. The sampler; ignore the bitstream representation for now.
3. **H-Net** — arXiv 2507.07955. Dynamic chunking, the ratio loss, and why nesting two routers is the hard part.
4. **CCDD** — arXiv 2510.03206. Continuous diffusion is more expressive and less trainable; useful framing for every trainability wall hit so far.
5. **Perfect diffusion is TC⁰** — arXiv 2507.12469. Why deep sequential computation inside a block does not work, and why interpolating sequential with parallel is the recommended answer.

For §4 specifically:

6. **Edit Flows** — arXiv 2506.09018 — and **FlexMDM** — arXiv 2509.01025. Generation defined over variable-length sequence space, with insertion and deletion as primitives. The cleanest existing treatments of what `ℓ` is doing.
7. **Align-Refine** — arXiv 2010.14233. From ASR, and the source of the central move: treating length-*L* sequences as the latent space makes insertions and deletions hard, so refine a fixed-resolution alignment instead.
8. **GeoBlock** — arXiv 2603.26675. Attention-geometry boundary inference, training-free. A diagnostic and a possible initialiser for `b`, not a router₂ substitute.
9. **DAEDAL** — arXiv 2508.00819. Training-free length expansion at inference; the cheap version of what `ℓ` learns.

For §5.6a specifically:

10. **Reasoning with latent thoughts: on the power of looped transformers** — Saunshi et al. 2025. The reasoning-versus-memorisation trade, measured at pretraining scale, plus the closeness regulariser that makes it continuous.
11. **Scaling up test-time compute with latent reasoning** — Geiping et al. 2025. Randomised unrolling and truncated backpropagation; the recipe that makes looping trainable.
12. **Ouro / LoopLM** — Zhu et al. 2025. Looped pretraining through the full modern pipeline, ~2–3× parameter efficiency.
13. **Thinking deeper, not longer** — arXiv 2603.21676. Final-only versus intermediate supervision in depth-recurrent transformers, on three compositional domains. The source of the shortcut-learning caveat in §5.6a.

Trackers: `VILA-Lab/Awesome-DLMs`, `AIDASLab/Awesome-Diffusion-LLM`. The continuous-DLM literature is moving fast enough that this document has a shelf life of months.
