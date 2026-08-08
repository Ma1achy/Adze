# Scratchpad — other domains

**Status:** idle speculation. Not gated on anything, not planned, not scoped. Written down so it doesn't evaporate.

---

## Why time series is a natural fit

The organising principle is *"the model thinks in continuous space; tokens are only a human-readability interface."*

For time series **there is no interface**. The data is already continuous. So you delete:

- the byte frontend
- router₁ (bytes → chunks)
- the byte decoder
- `ℓ` as *byte* expansion
- the whole discretisation apparatus, including the `ℓ = 0` query-active machinery that exists to make insertion possible in a discrete output space

Roughly a third of the architecture, gone. What's left is the part that was interesting anyway: a carrier of continuous latents, learned segmentation into blocks, block-causal drafting, global refinement, adaptive compute.

**The architecture is arguably *more* natural here than in text.** Text needed an elaborate interface layer to make a continuous method work on discrete data. Time series doesn't.

---

## The deflating part, stated up front

**The M7 mechanism becomes obvious.** "Does downstream context help reconstruct an erased chunk?" is gap-filling, and *interpolation beats extrapolation* is not news to anyone working on series.

The finding is interesting for **reasoning** precisely because it wasn't obvious there — people build refinement architectures on that assumption without testing it. Move to a domain where everyone already assumes it correctly and the contribution evaporates.

So: **do not port the experiment. Port the machinery.**

---

## What actually transfers, and maps onto real open problems

| Adze component | Time-series analogue | Why it's interesting there |
|---|---|---|
| **router₂** — learned block boundaries | **learned changepoint / regime segmentation** | Currently done with heuristics, fixed windows, or separate changepoint detectors. Learning boundaries end-to-end against a downstream generative objective is a genuine contribution. |
| **`ℓ`** — chunk → byte expansion | **variable temporal resolution** — how many timesteps a block covers | The model learns to spend fine resolution on volatile stretches and coarse on quiet ones. Adaptive sampling *learned* rather than imposed. Real problem for irregular and multi-scale data. |
| **adaptive `ρ` / `R`** | **more refinement passes on hard segments** | Same idea one level up: spend inference compute where the series is hard. |
| **draft → refine** | **forecast → revise given later observations** | Closer to smoothing than filtering, which is a well-defined and well-studied setting to be measured against. |
| **§3.5 forward/backward disagreement** | **the same signal, and possibly stronger** | Two independent paths to the same value. In a series with genuine temporal structure this is closer to the round-trip consistency setting it came from (arXiv 2608.00675) than text ever was. |

---

## What breaks — the one real design gap

**Multivariate.** Text is one dimension of bytes. Series are usually `D` channels × `T` steps.

So the carrier needs a channel axis, and that raises a question the current spec has no answer to:

> Do channels share block boundaries, or does each channel get its own segmentation?

Both are defensible. Shared boundaries assume regimes are global to the system; per-channel boundaries allow sensors to change behaviour independently. Probably a per-dataset property rather than a design constant — which means it wants to be a config axis, not a decision.

Neither is hard. But it's a genuine addition, not a reskin.

**Secondary:** irregular sampling. If timestamps aren't evenly spaced, position encoding has to carry actual time rather than index — a small change, but it touches every position-encoding site.

---

## Code is the better adjacent domain

Not really a fallback — it's already on the endpoint path, and the mechanism stays genuinely non-obvious there.

A variable's declaration is constrained by its later uses. A function's signature is constrained by its call sites. That is **exactly the downstream-pin structure** the M7 result is about, and unlike interpolation it is *not* obvious that a model exploits it.

It's also the domain where the distance-decay result becomes practically load-bearing: if the reach extends far enough with training, *"regenerate this line using the rest of the file"* is a real capability rather than a demo. The current profile reaches ~2 steps; the open question is whether that's a property of the architecture or of how much refinement training it's had.

---

## Honest risk note

"Time series as a fallback if text fails" is the wrong framing, for the reason at the top: it's the domain where the finding is *least* surprising. If text fails, a series result isn't much of a landing.

The actual safety net is that **the M7 result already stands independently.** "Global refinement uses downstream evidence, here is its reach, here is the control that establishes it" is a finding about a class of models the field is already building. It does not need Adze to succeed.

So the risk isn't the idea being wrong. It's the project running out of evenings before the interesting version gets built.

---

## If it ever gets picked up

Rough order, none of it scoped:

1. Strip the byte layer. Carrier of continuous observations, no router₁, no byte decoder.
2. Keep `ℓ` as temporal extent. This is the interesting one and it's free — the mechanism is already specified.
3. Univariate first. Multivariate is the design gap, not the starting point.
4. Measure against smoothing baselines, not forecasting ones. Draft-then-refine is a smoother.
5. The contribution is **learned segmentation + adaptive resolution**, not "diffusion for time series", which exists.
