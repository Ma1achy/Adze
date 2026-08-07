"""M6 — regime B: refine. Erase a subset of blocks, regenerate them globally.

Design §3.1. Select a subset S of blocks, set `t_i = 1` for `i in S` — COMPLETE
erasure, not partial re-noising — leave every block outside S clean at `t = 0`,
attend under the GLOBAL mask, and take the loss on S only.

## Why this needs no 2N concatenation, unlike regime A

Regime A's vectorised form exists because each block's clean context differs from
every other block's, so scoring all B conditionals at once needs both a noised and
a clean copy of the sequence. Regime B has no such problem. At `t = 1` the erased
block IS pure noise — `z_t = (1-1)*z0 + 1*eps = eps` — and the unselected blocks
are already clean, in place, at their real positions. One sequence of length N
carries both roles.

**This is still the vectorised form, not the naive one.** Every block in S is
scored in the same forward pass. The naive analogue would be erasing and scoring
one block at a time, which is what starved regime A by ~110x, and it is not
reintroduced here.

## Why S may vary per example, where regime A's block index could not

`regime_a_batch` draws ONE block index for the whole batch because its mask is a
function of that index, and a per-example index would need a per-example [N, N]
attention mask. Regime B's mask is GLOBAL — fully bidirectional, identical for
every example — so only the timesteps and the loss mask vary with S, and both are
already per-example tensors. Each example therefore gets its own independently
drawn subset, which is strictly more signal per step at no cost.

## Padding

Only REAL blocks are eligible for S. Erasing a pad block would ask the model to
reconstruct a constant from noise and would count that as refinement skill.
"""

from __future__ import annotations

import torch

from adze.invariants import MaskMode
from adze.model.denoiser import Denoiser
from adze.model.flow import interpolate, velocity_target
from adze.model.masks import build_mask
from adze.pad import masked_mean


STRUCTURES = ("random", "single", "contiguous", "suffix", "pinmix")


def sample_subset_structured(
    block_mask: torch.Tensor,
    structure: str = "random",
    p: float = 0.5,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Choose which blocks to erase, under a given SUBSET STRUCTURE.

    ## Why the structure matters

    M7 measured that the two arms exploit disjoint information sources. Causal's
    accuracy climbs with prefix length (1.3% -> 7.8%) and ignores downstream
    evidence; global's is set by whether downstream evidence exists (~3-5% with,
    ~1% without) and is FLAT in prefix length. Global does not use the prefix at
    all, even where the prefix is informative.

    The candidate cause is in this function. Under `random`, each real block is
    erased independently at p = 0.5, so mean |S| = 2.24 out of ~4.4 real blocks —
    **roughly half the prefix is itself erased on a typical refine step.** A model
    trained that way never sees a reliable clean-prefix-to-erased-block mapping,
    so learning not to depend on one is the correct response to its training
    distribution.

    ## The three structures, and what each preserves

    | structure | prefix (blocks < min S) | pin (blocks > max S) | |S| |
    |---|---|---|---|
    | `random` | damaged — holes throughout | damaged | Binom(n, 0.5) |
    | `single` | clean | clean | 1 |
    | `contiguous` | clean | clean | Binom(n, 0.5) |

    `single` matches M7's inference condition exactly. `contiguous` erases a run
    `b .. b+k-1`, so everything before `b` is clean and reliable and everything
    after `b+k-1` is clean and still carries the pin — both sources intact at
    |S| > 1. Together they separate "small |S| matches inference" from "structured
    erasure makes the prefix reliable".

    A contiguous SUFFIX would be the wrong test and is deliberately not offered:
    erasing everything after `b` removes the downstream evidence, which is the
    very thing the global arm depends on. It would test the opposite hypothesis.

    Args:
        block_mask: [batch, B] bool, True where the block is real.
        structure: one of STRUCTURES.
        p: per-block selection probability, and the size distribution for
           `contiguous` — kept identical to `random`'s so the two differ in
           SHAPE only, not in how much is erased.
    """
    if structure not in STRUCTURES:
        raise ValueError(f"structure must be one of {STRUCTURES}, got {structure!r}")
    if structure == "random":
        return sample_subset(block_mask, p=p, generator=generator)
    if structure in ("suffix", "pinmix"):
        return _suffix_or_mix(block_mask, structure, p, generator)

    batch, blocks = block_mask.shape
    device = block_mask.device
    n_real = block_mask.sum(dim=1)                                  # [batch]

    if structure == "single":
        sizes = torch.ones(batch, dtype=torch.long, device=device)
    else:
        # Same size distribution as `random`: Binomial(n_real, p), floored at 1.
        draw = torch.rand(block_mask.shape, generator=generator, device=device)
        sizes = ((draw < p) & block_mask).sum(dim=1).clamp(min=1)

    # Uniform start among the runs that fit inside the real blocks. Real blocks
    # are a prefix 0..n_real-1, so a run of length s fits at starts 0..n_real-s.
    span = (n_real - sizes + 1).clamp(min=1)
    start = (torch.rand(batch, generator=generator, device=device) * span).long()
    start = torch.min(start, span - 1)

    idx = torch.arange(blocks, device=device).unsqueeze(0)          # [1, B]
    selected = (idx >= start.unsqueeze(1)) & (idx < (start + sizes).unsqueeze(1))
    return selected & block_mask


def _suffix_or_mix(block_mask, structure, p, generator):
    """`suffix` erases b..end; `pinmix` is half interior, half suffix.

    THE POINT OF SUFFIX ERASURE. With everything after `b` gone, the downstream
    consumer that states block b's result exactly no longer exists, so the clean
    PREFIX is the only route to it. The pin is a lookup; the prefix requires
    recomputation. If the model ignores the prefix because the lookup is easier
    whenever both are available, removing the lookup for half of regime B should
    force the harder route to be learned.

    `b` is drawn from 1..n_real-1, never 0: erasing from block 0 leaves neither a
    prefix nor a pin, which trains reconstruction from nothing.

    NOTE, measured before this existed: the `random` baseline ALREADY has no pin
    on 59.5% of erased blocks, because a block's consumer is frequently erased
    alongside it. So this is not "adding a pin-free condition that did not exist"
    — it is making pin-absence structural and total rather than incidental and
    partial, and removing the indirect downstream constraint that survives when
    only the direct consumer is erased.
    """
    batch, blocks = block_mask.shape
    device = block_mask.device
    n_real = block_mask.sum(dim=1)

    start = 1 + (torch.rand(batch, generator=generator, device=device)
                 * (n_real - 1).clamp(min=1)).long()
    start = torch.min(start, (n_real - 1).clamp(min=0))
    idx = torch.arange(blocks, device=device).unsqueeze(0)
    suffix = (idx >= start.unsqueeze(1)) & block_mask

    if structure == "suffix":
        return suffix

    interior = sample_subset(block_mask, p=p, generator=generator)
    take_suffix = (torch.rand(batch, generator=generator, device=device) < 0.5)
    return torch.where(take_suffix.unsqueeze(1), suffix, interior)


def sample_subset(
    block_mask: torch.Tensor,
    p: float = 0.5,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Choose which blocks to erase. Returns [batch, B] bool, True = erase.

    Each real block is selected independently with probability `p`. If an example
    draws the empty set, one real block is forced in — an example with nothing
    erased contributes no loss and would silently dilute the batch.

    Args:
        block_mask: [batch, B] bool, True where the block is real (not padding).
        p: per-block selection probability. 0.5 spans the whole range of subset
           sizes rather than fixing |S|, so the model sees single-block repair
           (what M7 measures) and multi-block repair alike.
    """
    if not 0.0 < p <= 1.0:
        raise ValueError(f"p must be in (0, 1], got {p}")
    draw = torch.rand(block_mask.shape, generator=generator, device=block_mask.device)
    selected = (draw < p) & block_mask

    # Force one real block for any example that drew nothing.
    empty = ~selected.any(dim=1)
    if empty.any():
        # The first real block of each such example. Deterministic given the
        # padding layout, which is fine — this is a rare correction, not the
        # sampling mechanism.
        first_real = block_mask.float().argmax(dim=1)
        selected[empty, first_real[empty]] = True
    return selected


def regime_b_batch(
    z0: torch.Tensor,
    block_ids: torch.Tensor,
    blocks: int,
    block_mask: torch.Tensor | None = None,
    p: float = 0.5,
    generator: torch.Generator | None = None,
    eps: torch.Tensor | None = None,
    selected: torch.Tensor | None = None,
    structure: str = "random",
) -> dict[str, torch.Tensor]:
    """Build one regime B (refine) training batch.

    Args:
        z0: [batch, N, D] clean latents.
        block_ids: [N]
        blocks: B.
        block_mask: [batch, B] bool, True where real. All-real if omitted.
        eps, selected: supply to make the batch deterministic. The equivalence
            test needs both paths to see identical noise and identical subsets.

    Returns:
        dict with z_t, t, target, mask, loss_mask, selected.
        `t` is exactly 1.0 on selected blocks and exactly 0.0 elsewhere — complete
        erasure, matching what M7's sampler will do. Partial re-noising (`t < 1`)
        is deliberately NOT in v0; see design §8.
    """
    batch, n_positions, _ = z0.shape
    device = z0.device
    k = n_positions // blocks

    if block_mask is None:
        block_mask = torch.ones(batch, blocks, dtype=torch.bool, device=device)
    if selected is None:
        selected = sample_subset_structured(
            block_mask, structure=structure, p=p, generator=generator)
    if eps is None:
        eps = torch.randn(z0.shape, generator=generator, device=device)

    # t = 1 on selected, 0 elsewhere. interpolate() then yields eps on selected
    # blocks and z0 everywhere else, exactly.
    t = selected.to(z0.dtype)
    z_t = interpolate(z0, eps, t)
    target = velocity_target(z0, eps)

    # Loss on the erased blocks only, and only where they are real.
    loss_mask = (selected & block_mask).repeat_interleave(k, dim=1).unsqueeze(-1)

    return {
        "z_t": z_t,
        "t": t,
        "target": target,
        "mask": build_mask(block_ids, MaskMode.GLOBAL),
        "loss_mask": loss_mask,
        "selected": selected,
        "eps": eps,
    }


def regime_b_loss(
    model: Denoiser, batch: dict[str, torch.Tensor], block_ids: torch.Tensor
) -> torch.Tensor:
    """Velocity MSE over the erased blocks only, real positions only."""
    pred = model(
        batch["z_t"],
        batch["t"],
        block_ids,
        MaskMode.GLOBAL,
        mask=batch["mask"],
    )
    if batch["loss_mask"].sum() == 0:
        return (pred * 0).sum()
    return masked_mean((pred - batch["target"]) ** 2, batch["loss_mask"])
