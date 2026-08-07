"""M7 — the central experiment.

> Given the corrupted block's location, does global regeneration repair it more
> reliably than causal regeneration?

The whole repo is aimed at this one comparison. Three conditions on the SAME
corrupted traces and the SAME checkpoint:

| condition | what it does |
|---|---|
| `none`    | the corrupted trace, round-tripped through the VAE. NOT a floor. |
| `causal`  | erase the corrupted block, regenerate under the CAUSAL mask |
| `global`  | erase the corrupted block, regenerate under the GLOBAL mask |

## `none` is not the baseline, and chance is

`none` was reported as "the floor — the corrupted block, unrevised". It is not.
The VAE was trained on valid arithmetic only, so a corrupted step is
off-distribution and the encoder projects it onto the nearest valid one: measured,
it silently REPAIRS 19.6% of corruptions and preserves them only 34.2% of the
time. `'7 - 48 = -31'` round-trips to `'7 - 48 = -41'`. So `none` scores on a
metric that should floor at zero, and what it measures is the round-trip, not
no-revision. The regeneration arms erase the block and discard that free repair,
which is why they sit an order of magnitude below it.

The reference is CHANCE, as a permutation null — see `adze.eval.strata`.

Note the direction this cuts: the valid-only prior probably HELPS the treatment
arms, and the 96-100% well-formedness is consistent with that. Training the VAE
on corrupted steps is a baseline correction, downstream of these controls rather
than upstream of them.

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
  result        it gets the RESULT right, whatever it does with the operands.
                This is the metric the experiment's logic actually points at: the
                corruption changes a RESULT, and downstream steps consume the
                original correct value, so downstream context identifies the
                result and nothing else. The operands are identified by the
                QUESTION and by earlier steps — and question conditioning (M5) is
                not built, so for an early block the operands are underdetermined
                and exact match is unreachable in principle. Reported alongside
                `repaired`, not instead of it.
  operands      it gets the operands right, whatever it does with the result —
                the other half of the decomposition, and the arm that global
                context should NOT help with
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

_STEP = __import__("re").compile(r"^(-?\d+) ([+\-*]) (-?\d+) = (-?\d+)$")


def _parse(text: str) -> tuple[int, str, int, int] | None:
    """(lhs, op, rhs, result) or None. Applied to raw output, never to fix it."""
    m = _STEP.match(text)
    return None if m is None else (int(m[1]), m[2], int(m[3]), int(m[4]))


@dataclass(frozen=True)
class Repair:
    """One condition's outcome over a set of corrupted traces."""

    condition: str
    repaired: float          # regenerated block decodes exactly to the clean step
    result: float            # ...or at least gets the RESULT right
    operands: float          # ...or at least gets the OPERANDS right
    well_formed: float       # ...or at least parses at all
    answer: float            # final answer correct after revision
    preserved: float         # unselected blocks decode unchanged
    n: int
    texts: list[list[str]]   # decoded trace per example, raw


def shielded_mask(block_ids: torch.Tensor, block: int, mode: MaskMode) -> torch.Tensor:
    """`mode`'s mask, but nothing outside `block` may attend TO `block`.

    Isolates the two things that differ between the causal and global arms, which
    the design conflated:

      1. the ERASED block reads downstream context   <- the mechanism under test
      2. the CONTEXT blocks read the erased block    <- a side effect nobody asked for

    (2) is not symmetric between the arms. Under CAUSAL only blocks *after* the
    target can read it, so when the target is the root, nothing does. Under GLOBAL
    every block reads it — and it holds pure noise at t = 1. So global's clean
    context is computed with the erasure's noise mixed in and causal's is not, at
    every layer above the first. That is a difference between the conditions with
    nothing to do with downstream evidence.

    Shielding keeps (1) and removes (2). Read the gap under this mask against the
    gap under the plain one: what survives is the mechanism.

    Single [N, N], so it is valid only where `block` is the same for every example
    in the batch. Asserted by the caller rather than silently broadcast.
    """
    mask = build_mask(block_ids, mode).clone()
    is_target = block_ids == block
    mask[~is_target.unsqueeze(1) & is_target.unsqueeze(0)] = False
    return mask


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
    mask: torch.Tensor | None = None,
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
        mask: override the [N, N] mask. `mode` still conditions the model, so this
            is for diagnostics that hold the conditioning fixed while changing what
            is visible — see `shielded_mask`.
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
    if mask is None:
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


@torch.no_grad()
def encode_traces(vae, tokens: torch.Tensor, block_mask: torch.Tensor,
                  scale: float) -> torch.Tensor:
    """[batch, B, T] tokens -> [batch, N, D] latents, in the denoiser's space.

    MUST match `LatentCache.build` exactly, and the reason is worth stating: pad
    blocks are filled with the VAE's learned `pad_latent`, NOT with whatever the
    encoder returns for a row of PAD tokens. Those are different vectors, and the
    denoiser has only ever seen the first. Feeding it the second puts the context
    off-distribution and the model returns latents that decode to empty strings —
    measured, before this helper existed: 7.4% well-formed against 99%+ for
    free-running generation.

    One function so the cache convention and the eval convention cannot drift.
    """
    batch, blocks, n_tok = tokens.shape
    mu, _ = vae.encoder(tokens.view(-1, n_tok))
    mu = mu.view(batch, blocks, *mu.shape[1:])
    mu[~block_mask] = vae.pad_latent.to(mu.dtype)
    return mu.reshape(batch, -1, mu.shape[-1]) / scale


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
    n_result = n_operands = n_formed = 0
    for i, row in enumerate(decoded):
        b = int(target_block[i])
        clean = clean_steps[i]
        if row[b] == clean[b]:
            repaired += 1

        got, want = _parse(row[b]), _parse(clean[b])
        if got is not None:
            n_formed += 1
            if want is not None:
                n_result += int(got[3] == want[3])
                n_operands += int(got[:3] == want[:3])

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
        result=n_result / n,
        operands=n_operands / n,
        well_formed=n_formed / n,
        answer=answered / n,
        preserved=preserved / n,
        n=len(decoded),
        texts=decoded,
    )
