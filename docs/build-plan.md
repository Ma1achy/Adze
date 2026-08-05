# Adze — Build Plan

Companion to `design.md`, which is the **design reference**. This document is the **build order**. Where they disagree, the design doc wins on *what*, this one wins on *sequence*.

---

## How to run this with an agentic coding tool

- **One milestone per session.** They're sized so each fits comfortably in one context.
- **Put the design doc in the repo** and reference it by section rather than re-explaining. `See §3.1 for the training formulation` beats pasting equations.
- **The acceptance criteria are the contract.** A milestone is done when its test passes, not when the code looks right. Write the test first where practical.
- **Scope creep is the main failure mode.** §8 of the design doc is a list of things that must not appear in v0. If a session starts adding a termination head, stop it.
- **Commit per milestone** so you can bisect when M6 breaks something M3 established.

---

## Invariants — state these once, hold them everywhere

These shapes are the thing most likely to get silently wrong. Put them in a docstring somewhere central and refer back.

```
B  = blocks per sequence      (fixed for v0)
K  = 4                        latents per block
D  = 64                       latent channel dim
N  = B * K                    total latent positions

latents    : [batch, N, D]
timesteps  : [batch, B]       PER BLOCK, broadcast to K within the block
block_ids  : [N]              which block each position belongs to
mask       : [N, N]           built from block_ids + mode
```

**The per-block timestep is not standard DiT.** Vanilla DiT takes a scalar `t` per sample and broadcasts globally. Regime B needs different blocks at different noise levels in the same forward pass. Build this from the start (design doc §3.1) — retrofitting it means touching every conditioning path.

**Two mask modes:**
- `causal` — bidirectional within block, causal across blocks
- `global` — fully bidirectional everywhere

Both are pure functions of `block_ids`. One function, one switch.

---

## Repo shape

```
ldr/
  data/
    generate.py      # synthetic arithmetic traces
    corrupt.py       # corruption + matched pair construction
    dataset.py       # torch Dataset; latent caching
  model/
    vae.py           # encoder/decoder (Perceiver queries)
    denoiser.py      # DiT + per-block t conditioning
    masks.py         # causal / global mask construction
  train/
    train_vae.py
    train_denoiser.py
  sample/
    draft.py         # pass one
    refine.py        # pass two (erasure)
    trajectory.py    # step-by-step decode printer
  eval/
    checks.py        # latent-use, overfit-one-batch gates
    central.py       # the three-condition experiment
  configs/
    debug.yaml       # 2 layers, 128 wide, 100 examples, 8 NFE
    v0.yaml          # 12 layers, 768 wide, 50 NFE
```

Keep `debug.yaml` working at every milestone. It's the difference between a 40-second iteration and a 40-minute one.

---

## Milestones

### M0 — Scaffold and throughput probe

**Goal:** know your numbers before writing anything that depends on them.

- Repo skeleton, config loading, seeding
- A throughput script: dummy transformer, forward+backward, measure tokens/sec on MPS
- **Check for silent CPU fallbacks.** `PYTORCH_ENABLE_MPS_FALLBACK=1` will mask them; profile op placement explicitly
- Benchmark `torch.compile` on/off, bf16 vs fp32

**Acceptance:** a measured steps/sec figure for a GPT-2-small-shaped model, and a decision on whether MPS is viable or you need MLX / rented GPUs.

**Do not:** build any real components yet.

---

### M1 — Synthetic trace generator

**Goal:** the dataset. Pure Python, no ML, no GPU. This is the only component with zero dependencies and zero uncertainty, which is why it's first.

- Expression trees with configurable depth and operand ranges
- Render to a step-per-line textual trace
- Every intermediate value recorded structurally, not just in the text
- `corrupt.py`: corrupt one intermediate value at a chosen block index, **recompute nothing downstream** — later steps keep using the correct value, so the chain is internally inconsistent and the inconsistency is only visible downstream (design doc, central experiment)
- Emit `(clean_trace, corrupted_trace, corrupted_block_index)` triples

**Decisions this forces, currently unmade:**
- What a step looks like as text — determines tokens per step, and whether K=4 is remotely right
- Step count distribution — determines B
- Operand magnitude — determines how much tokenisation mangles the numbers

**Acceptance:** generate 10k traces; assert every clean trace evaluates to its stated answer; assert every corrupted trace does not; assert the corrupted block index round-trips. Print length statistics and **revisit K and B against them**.

---

### M2 — VAE

**Goal:** text ↔ latent blocks.

- Encoder: small bidirectional transformer over a step's tokens → K=4 learned Perceiver query tokens cross-attending → `[K, D]`
- Decoder: mirror — K latents → token sequence for that step
- Loss: cross-entropy reconstruction + small KL (β ≈ 1e-3)
- Latent caching: once trained, encode the whole dataset to disk once (design doc §6.1)

**Acceptance — two gates, both hard:**
1. **Reconstruction:** >95% exact token match on held-out steps
2. **Latent-use check:** decode from shuffled/random latents. If quality barely drops, the decoder is ignoring the latent — posterior collapse — and everything downstream is meaningless. **Stop and fix before M3.**

---

### M3 — Denoiser, Regime A only

**Goal:** block-causal drafting.

- DiT backbone; adaLN conditioning extended to per-block `t`
- `masks.py`: causal and global, both from `block_ids`
- Rectified flow, velocity prediction (design doc §3.1)
- Regime A training loop: sample block `b`, noise it, previous blocks clean, causal mask, loss on `b`

**Acceptance — overfit-one-batch first.** On `debug.yaml`, a single batch of 8 examples must drive loss to near zero. If it can't, something is miswired — most likely the per-block timestep broadcast or the mask. **Do not proceed on a model that can't overfit 8 examples.**

Then: training loss decreasing on the full set.

---

### M4 — Sampling and the trajectory printer

**Goal:** see it work.

- Euler ODE sampler, `t: 1 → 0`, block by block
- `trajectory.py`: decode and print every denoising step

**Acceptance:** generated traces are syntactically well-formed arithmetic steps. Correctness not required yet. The trajectory printer is the main debugging instrument for everything after this — build it properly, not as a `print` in a loop.

---

### M5 — Answer decoding

**Goal:** end to end.

- Question conditioning by prefix concatenation
- Answer head: autoregressive over answer tokens, conditioned on the latent blocks

**Acceptance:** non-trivial answer accuracy on held-out problems. This is your **pass-one baseline** — every later number is measured against it.

---

### M6 — Regime B and mixed training

**Goal:** teach the denoiser to operate globally.

- Regime B: select subset `S` of blocks, `t_i = 1` for those (complete erasure), others clean, global mask, loss on `S`
- Mixed sampling: 90% Regime A / 10% Regime B per step
- Retrain from scratch — this changes what the model learns, not just how it's used

**Acceptance:** pass-one quality does not regress against M5. Design doc warns of the opposite failure (drafting degraded by global exposure); if it appears, adjust the mix ratio before anything else.

---

### M7 — Pass two and the central experiment

**Goal:** the result.

- `refine.py`: erase selected blocks, regenerate with global mask
- `eval/central.py`: three conditions, otherwise identical —

| Condition | Mask |
|---|---|
| No revision | — |
| Erase + regenerate causally | `causal` |
| Erase + regenerate globally | `global` |

Oracle block selection: you know the corrupted index and erase that one.

**Metrics** (design doc §4 — note the delta log is *not* useful under erasure + oracle selection):
- exact repaired-operation accuracy
- answer accuracy before vs after refinement
- preservation of unselected decoded blocks
- global vs causal, the headline comparison

**Acceptance:** a number for each condition, and an answer to:

> Given the corrupted block's location, does global regeneration repair it more reliably than causal?

That's the project's first real result. Everything in design doc §8 is downstream of it.

---

## Not in v0

If a coding session starts building any of these, it has drifted:

- termination head / variable reasoning length
- learned block segmentation
- partial re-noising (`t < 1`)
- uncertainty-based block selection
- rollout adaptation, semantic-correction objective
- looped refinement, adaptive compute
- byte-level input, learned input tokenisation
- distillation, RL
- x-prediction, SDE sampler
- KV caching (until after M7, and only against a verified uncached baseline)

---

## Suggested first session

M0 and M1 together. Neither touches the model, both are independently useful, and M1's length statistics feed directly into the K and B choices that M2 depends on.
