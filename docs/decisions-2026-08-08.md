# Specification decisions — 8 August 2026

External input (ChatGPT), recorded verbatim as the source for the decisions now
folded into `docs/endpoint.md`. Where `endpoint.md` and this file disagree,
`endpoint.md` is authoritative — three items were amended on the way in, and the
amendments are recorded in the section they belong to and in the session-26 entry
of `RESULTS-2026-08-07.md`.

**Amended before adoption:** item 4 (one justification dropped), item 5 (a
required diagnostic added), item 7 (default reversed), item 8 (one addition).

---

I think five specification questions can be closed now. The clean defaults are:
**max edge lift, span-aware edge pooling, absorbing categorical structure kernels,
hysteretic atomic commits, and per-example ratio targets**.

## Decisions to record

### 1. `edge_lift`: use max

```text
q_i    = τ_i^α
ν_b[k] = max(q_k, q_{k+1})
```

Why:

- If either endpoint is touched, their relationship is uncertain.
- `max(x,x)=x`, so a uniformly noised region does not acquire artificially higher edge noise.
- Mean incorrectly gives only 0.5 noise when one endpoint is completely erased.
- Independent edge sampling breaks the intended coupling between content corruption and structural uncertainty.

For realised local-erasure masks, use the corresponding hard rule:

```text
edge_touched[k] = position_touched[k] OR position_touched[k+1]
```

Bootstrap needs no special case: if every position has maximal noise, max and mean both return maximal edge noise. The explicit `UNKNOWN` state is doing the real bootstrap work.

### 2. Pool boundary noise by structural role

Do not convert `ν_b` to chunk values and silently reuse the content weights. Each packed block has three relevant edge sets:

- the interval separating it from the previous emitting block;
- all internal carrier edges, including edges across inactive holes;
- the interval separating it from the next emitting block.

Embed the edge noise first, then pool:

```text
e_left[m]     = mean φ_b(ν_b[e]), e in left interface interval
e_internal[m] = resample_k φ_b(ν_b[e]), e internal to block m
e_right[m]    = mean φ_b(ν_b[e]), e in right interface interval

cond_b[m,k] =
    W_left e_left[m]
  + W_internal e_internal[m,k]
  + W_right e_right[m]
```

The interface interval belongs to both adjacent blocks, once as `right` and once as `left`. This is desirable: uncertainty about their separation affects both summaries.

For `K=1`, `resample_k` is simply a mean. Missing outer interfaces get learned BOS/EOS-edge embeddings. Pool the Fourier embeddings, not the raw scalar noise values.

### 4. Use absorbing categorical corruption for both `b` and `ℓ`

Start with:

```text
b_t = b₀       with probability 1 − ν_b
      UNKNOWN  with probability ν_b

ℓ_t = ℓ₀       with probability 1 − ν_ℓ
      UNKNOWN  with probability ν_ℓ
```

Then predict clean `b₀` and `ℓ₀` categorically.

This matches generation, which begins with unknown structure, and avoids inventing a metric over boundary states. Absorbing-state kernels are a standard discrete-diffusion construction ([D3PM](https://arxiv.org/abs/2107.03006)) and underpin modern masked diffusion ([FlexMDM discussion](https://arxiv.org/html/2509.01025)).

Teach incorrect-but-known structural states through separate rollout regimes:

- boundary insertion, deletion and shift;
- zero/nonzero extent errors;
- incorrect positive lengths;
- model-generated `b` and `ℓ`.

That separation is useful: the forward kernel teaches inference from uncertainty, while structured corruption teaches correction of committed mistakes. Edit-based models similarly treat insertion and deletion as explicit operations rather than pretending they are ordinary categorical noise ([Edit Flows](https://arxiv.org/abs/2506.09018)).

I would not introduce an ordinal `ℓ` kernel initially. Although positive lengths are ordered, a one-byte error can change alignment discontinuously. Absorbing categorical is the assumption-free baseline. Factor `P(ℓ=0)` from `P(ℓ>0)` later only if existence errors dominate.

### 5. Commit with hysteresis, once per outer iteration

Maintain two objects:

- continuous proposed probabilities `p_b`;
- committed hard boundaries `c_b`, used for packing.

Never alter the hard partition during an inner flow trajectory. At its end:

```text
if c_b[k] = 0: add boundary only when p_b[k] ≥ θ_on
if c_b[k] = 1: remove boundary only when p_b[k] ≤ θ_off
otherwise: hold
```

Start with `θ_on=0.7`, `θ_off=0.3`; calibrate them on held-out downstream quality, not boundary F1.

Apply ladder constraints atomically:

- **Shift-only:** pair one removal with one addition within radius `δ`; block count remains fixed.
- **Local split/merge:** permit at most one split or merge inside each selected carrier region, enforce minimum block width, and freeze its outer anchors.
- **Unrestricted:** remove the locality restriction but retain hysteresis, minimum width and non-conflicting edits.

When several candidates compete, choose the non-conflicting set with the largest summed log-odds improvement minus movement/split/merge penalties. Only edges inside the uncertainty-selected region plus a small halo are eligible; everything else stays fixed.

If churn remains high, require the same proposal for two outer iterations. That persistence rule should be added only if the measurement fires. Training rollouts must use exactly the same commit function.

### 6. Make `r*` per-example, not a global guessed scalar

During training the clean structural target already exists. Use it:

```text
r*_n = stopgrad(r_eff(b₀_n, ℓ₀_n))

L_ratio =
    mean_n (r_eff(b̂₀_n, stopgrad(ℓ₀_n)) − r*_n)²
```

Sources for the clean target:

- synthetic stage: ground-truth boundaries;
- natural text: the frozen Phase-B router or accepted pseudo-target;
- after unfreezing: the EMA target encoder/router.

Use target activity in this loss so `ℓ` cannot compensate for bad `b` by hiding boundary mass in inactive positions. Continue measuring the fully predicted `r_eff(b̂,ℓ̂)` as the deployment diagnostic.

This preserves natural variation in block count instead of forcing every example toward the same ratio. If deployment requires a fixed compute budget, add that as a separate batch-level constraint; do not conflate a compute target with semantic segmentation.

H-Net instead specifies a desired downsampling factor `N` and drives the selected fraction toward `1/N` ([ratio loss](https://arxiv.org/html/2507.07955)). Adze can use the stronger per-example target because Phase B already supplies clean boundaries.

## Defaults for the remaining forks

- **3. Depth conditioning:** no cycle-index embedding by default. Ablate it at `L=4,Q=3` and `L=3,Q=4`, over multiple seeds. A physical-layer identity is already encoded by its weights; the disputed signal is the recurrence-cycle index.

- **7. Activity:** allow non-monotone activity initially and account for transient active-slot compute. Compare against **monotone routing**, where routing activity can only grow during the inner trajectory but the final `ℓ` prediction may still commit a deletion afterward. This avoids accidentally removing deletion capability.

- **8. Scan parameters:** start with a shared forward/reverse SSM core but separate directional norms and learned output gates. Untie the reverse scan only if refinement quality shows a gap; report the extra parameters or parameter-match the comparison.

- **9. Pooling:** `K=1`, as written.

- **10. Encoder sharing:** keep the shared byte/router₁ frontend, with separate norms/projections or small adapters for context and carrier roles. Freeze the shared frontend through C–D; in E, the context path uses online weights while the carrier target uses their EMA copy. Split the networks only if gradient conflict or validation quality demonstrates negative transfer.

- **11. Batching:** bucket examples by packed block count `M`, then pad to maximum `M` within each batch. It is simple and reliable. Move to true ragged/variable-length kernels only when profiling shows padding is material.

- **12–14:** remain correctly parked.

The main specification simplification is that `r*` need not be “calibrated from data” as a mysterious global hyperparameter: the clean target already tells you the appropriate compression for each training example.
