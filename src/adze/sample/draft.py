"""M4 — pass one. Block-causal drafting.

Generate blocks in order. Each block is denoised from pure noise over `nfe` Euler
steps, conditioned on the question and all previous (clean, already-generated)
blocks, under the causal mask.

Block 0 has no prefix to condition on, so at high `t` the best available
prediction is the mean over the training distribution. That floor is structural,
not a bug, and it lifts at M5 when question conditioning arrives.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from adze.invariants import MaskMode
from adze.model.denoiser import Denoiser
from adze.model.flow import euler_step, schedule
from adze.model.masks import regime_a_mask
from adze.sample.trajectory import TrajectoryRecorder


@torch.no_grad()
def draft(
    denoiser: Denoiser,
    context: torch.Tensor,
    blocks: int,
    latents_per_block: int,
    latent_dim: int,
    nfe: int,
    device: str = "mps",
    batch: int = 1,
    recorder: TrajectoryRecorder | None = None,
    shift: float = 1.0,
    mask_for_block: Callable[[torch.Tensor, int], torch.Tensor] = regime_a_mask,
) -> torch.Tensor:
    """Returns:
        latents: [batch, N, D] the drafted reasoning chain.

    Args:
        denoiser: trained denoiser, in eval mode.
        context: question conditioning. M5; pass None until then.
        blocks: B.
        nfe: Euler steps per block. Total forward passes is nfe * B — each block is
            integrated separately because a block must be complete before it can
            serve as clean context for the next.
        shift: schedule shape. 1.0 spaces the knots uniformly in t; >1 concentrates
            them the way shifted logit-normal training concentrates its draws.
            Training uses shift 1.5, so a sampler at 1.0 spends equal budget in
            regions where the model is unequally reliable.
        mask_for_block: (block_ids, b) -> [N, N]. Defaults to `regime_a_mask`, the
            same function `regime_a_batch` trains under. Stated explicitly rather
            than implied by a mode flag: a sampler using a different mask from
            training measures something the model was never taught, and this makes
            the choice visible at the call site instead of buried in a branch.
        recorder: optional TrajectoryRecorder, fed at every step of every block.
    """
    if context is not None:
        raise NotImplementedError("question conditioning is M5, not M4")
    if nfe < 1:
        raise ValueError(f"nfe must be >= 1, got {nfe}")

    denoiser.eval()
    dev = torch.device(device)
    n_positions = blocks * latents_per_block
    block_ids = torch.repeat_interleave(torch.arange(blocks), latents_per_block).to(dev)

    # Start from pure noise everywhere. Blocks after the active one are masked out
    # in both directions, so their contents never reach the prediction; blocks
    # before it are overwritten with their generated values as we go.
    latents = torch.randn(batch, n_positions, latent_dim, device=dev)

    # Knots, not a step size. Under a shift the spacing is non-uniform, so dt is
    # read off the interval each step actually spans. Hardcoding 1/nfe here would
    # silently integrate the wrong distance on every step of a shifted schedule.
    knots = schedule(nfe, shift, device=dev)
    global_step = 0

    for b in range(blocks):
        mask = mask_for_block(block_ids, b)
        is_active = (block_ids == b).view(1, n_positions, 1)

        for i in range(nfe):
            # t runs 1 -> 0 for the active block. Earlier blocks are already clean
            # (t = 0); later blocks are absent, and their t is never read.
            t_active = float(knots[i])
            dt = float(knots[i] - knots[i + 1])
            t = torch.zeros(batch, blocks, device=dev)
            t[:, b] = t_active
            t[:, b + 1 :] = 1.0

            velocity = denoiser(latents, t, block_ids, MaskMode.CAUSAL, mask=mask)

            # Integrate the active block only. Applying the step everywhere would
            # walk the finished blocks off their values.
            stepped = euler_step(latents, velocity, dt)
            latents = torch.where(is_active, stepped, latents)

            global_step += 1
            if recorder is not None:
                recorder.record(
                    global_step, float(knots[i + 1]), latents[0], active_block=b
                )

    return latents
