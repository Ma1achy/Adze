"""M7 — the central experiment.

> Given the corrupted block's location, does global regeneration repair it more
> reliably than causal regeneration?

The whole repo is aimed at this one comparison. Three conditions on the SAME
corrupted traces and the SAME checkpoint:

| condition | what it does |
|---|---|
| `none`    | no revision — the corrupted trace, untouched. The floor. |
| `causal`  | erase the corrupted block, regenerate under the CAUSAL mask |
| `global`  | erase the corrupted block, regenerate under the GLOBAL mask |

`causal` sees only blocks BEFORE the corrupted one. `global` sees the whole chain,
including the later steps that consumed the original correct value and therefore
contradict the corrupted one. That contradiction is the only evidence the
corruption exists, and it lies downstream — so the gap between these two
conditions is the experiment.

## This is a PAIRED comparison and that is what makes it sensitive

Same checkpoint, same traces, same corrupted indices, same erasure, same noise
seed. Only the attention mask differs. Everything that varies between training
runs — and this project has measured that variation to be large — cancels.
Reporting `global - causal` per trace, rather than two independently-noisy means,
is why one model can give a usable first signal.

## Oracle block selection, and what it means

The corrupted index is KNOWN and erased. Nothing here infers *which* block is
wrong. So every number this produces is an **upper bound** on what a system with
uncertainty-steered selection could achieve, and must be labelled that way.
Uncertainty-based selection is on design §8's not-in-v0 list.

## What is measured

  repaired      the regenerated block decodes to EXACTLY the clean step
  answer        the trace's final answer is correct after revision
  preserved     unselected blocks decode unchanged — a method that repairs the
                target by scrambling everything else has not repaired anything
  gap           global - causal on `repaired`. The headline.

Nothing is snapped, filtered or retried. A regenerated block that decodes to
garbage counts as garbage.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from adze.invariants import MaskMode
from adze.model.denoiser import Denoiser
from adze.model.flow import schedule
from adze.model.masks import build_mask
from adze.sample.stochastic import renoise_step

CONDITIONS = ("none", "causal", "global")


@dataclass(frozen=True)
class Repair:
    """One condition's outcome over a set of corrupted traces."""

    condition: str
    repaired: float          # regenerated block decodes exactly to the clean step
    answer: float            # final answer correct after revision
    preserved: float         # unselected blocks decode unchanged
    n: int
    texts: list[list[str]]   # decoded trace per example, raw


def revision_mask(block_ids: torch.Tensor, block: int, mode: MaskMode) -> torch.Tensor:
    """The [N, N] mask a revision pass attends under.

    GLOBAL is fully bidirectional — the erased block sees the whole chain.

    CAUSAL is the point of comparison and needs stating precisely: the erased
    block attends to itself and to STRICTLY EARLIER blocks, and nothing may attend
    forward. This is `build_mask(..., CAUSAL)` unchanged — block-causal across
    blocks, bidirectional within one — which is exactly the evidence a
    left-to-right drafter had when it produced the block the first time. That
    equivalence is the reason `causal` is the right control: it isolates the
    direction of attention and changes nothing else.
    """
    if mode is MaskMode.GLOBAL:
        return build_mask(block_ids, MaskMode.GLOBAL)
    return build_mask(block_ids, MaskMode.CAUSAL)


@torch.no_grad()
def regenerate(
    denoiser: Denoiser,
    latents: torch.Tensor,
    block_ids: torch.Tensor,
    target_block: torch.Tensor,
    blocks: int,
    nfe: int,
    mode: MaskMode,
    eta: float = 1.0,
    shift: float = 1.0,
    seed: int | None = None,
) -> torch.Tensor:
    """Erase `target_block` in each example and regenerate it. Returns [batch, N, D].

    Complete erasure: the target starts from pure noise and `t` runs 1 -> 0 on it
    while every other block is held at t = 0. That matches regime B training
    exactly. Partial re-noising is not in v0.

    `target_block` is [batch] — a per-example index. That is possible here for the
    same reason it is in regime B training: the mask is a function of `mode` and
    `block_ids` only, not of which block is erased. The per-example part lives in
    the timesteps and the write-back mask.

    Args:
        seed: set to make the erasure noise identical across conditions. The
            paired comparison depends on it — `causal` and `global` must face the
            same noise or the difference between them includes a noise draw.
    """
    device = latents.device
    batch, n_positions, _ = latents.shape
    k = n_positions // blocks

    if seed is not None:
        torch.manual_seed(seed)

    is_target = (
        torch.nn.functional.one_hot(target_block, blocks).bool()
        .repeat_interleave(k, dim=1).unsqueeze(-1)
    )
    z = torch.where(is_target, torch.randn_like(latents), latents)
    mask = revision_mask(block_ids, 0, mode)
    knots = schedule(nfe, shift, device=device)

    for i in range(nfe):
        t_val, s_val = float(knots[i]), float(knots[i + 1])
        # t = t_val on the erased block, 0 everywhere else. Clean blocks are
        # context and must be presented exactly as training presented them.
        t = torch.zeros(batch, blocks, device=device)
        t.scatter_(1, target_block.unsqueeze(1), t_val)

        v = denoiser(z, t, block_ids, mode, mask=mask)
        stepped = renoise_step(z, v, t_val, s_val, eta)
        z = torch.where(is_target, stepped, z)

    return z


def _decode(vae, tokeniser, latents: torch.Tensor, scale: float,
            blocks: int, k: int) -> list[list[str]]:
    """[batch, N, D] scaled latents -> one decoded string per block per example."""
    batch = latents.shape[0]
    flat = (latents * scale).view(batch * blocks, k, -1)
    with torch.no_grad():
        texts = [tokeniser.decode(r) for r in vae.decoder(flat).argmax(dim=-1)]
    return [texts[i * blocks : (i + 1) * blocks] for i in range(batch)]


def score(
    condition: str,
    decoded: list[list[str]],
    clean_steps: list[list[str]],
    target_block: torch.Tensor,
    n_steps: list[int],
) -> Repair:
    """Compare decoded traces against the clean reference.

    Args:
        decoded: [batch][blocks] raw decoded text.
        clean_steps: [batch][n_steps_i] the CLEAN trace's rendered steps.
        target_block: [batch] which block was corrupted and erased.
        n_steps: [batch] real step count, so padding blocks are not scored.

    `preserved` is measured over the real, non-target blocks only, and asks whether
    they decode to the CORRUPTED trace's text — which for every block but the
    target is the clean text. A revision pass must leave them alone.
    """
    repaired = answered = preserved = 0
    for i, row in enumerate(decoded):
        b = int(target_block[i])
        clean = clean_steps[i]
        if row[b] == clean[b]:
            repaired += 1

        # Answer: the last real block's result, read off the decoded text.
        last = n_steps[i] - 1
        answered += int(row[last] == clean[last])

        others = [j for j in range(n_steps[i]) if j != b]
        if others and all(row[j] == clean[j] for j in others):
            preserved += 1

    n = max(len(decoded), 1)
    return Repair(
        condition=condition,
        repaired=repaired / n,
        answer=answered / n,
        preserved=preserved / n,
        n=len(decoded),
        texts=decoded,
    )
