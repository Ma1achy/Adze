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
from adze.model.flow import schedule
from adze.model.masks import regime_a_mask
from adze.sample.stochastic import churn_t, churn_up, renoise_step
from adze.sample.trajectory import TrajectoryRecorder

# Default stochasticity. SDE sampling is PROMOTED out of design §8 — see
# adze.sample.stochastic for the evidence. This default is a real choice about
# what the sampler does, so it is stated here rather than left implicit at every
# call site.
#
# 1.0 is the measured optimum, not a boundary picked by default: the eta x S_churn
# grid (scripts/m4_churn.py) found eta = 1, S_churn = 0 best on both the
# block-conditional model (45.9%) and the unconditional one (70.7%), with churn
# HARMING both at eta = 1. Truth does not climb past this edge of the
# parameterisation.
#
# NOTE FOR READING OLD RESULTS: every M4 number recorded before this default
# changed was measured at eta = 0. Pass eta=0.0 explicitly to reproduce them.
DEFAULT_ETA = 1.0


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
    on_step: Callable[[int, float, int, torch.Tensor], None] | None = None,
    zero_prefix: bool = False,
    clean_prefix: torch.Tensor | None = None,
    eta: float = DEFAULT_ETA,
    s_churn: float = 0.0,
    s_noise: float = 1.0,
    churn_window: tuple[float, float] = (0.0, 1.0),
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
        zero_prefix: zero already-generated blocks instead of conditioning on
            them. Must match how the model was TRAINED — a sampler whose prefix
            differs from training's measures something never taught.
        clean_prefix: [batch, N, D] real cached latents to condition on instead of
            the generated prefix — teacher forcing. This is the upper arm of the
            exposure-bias comparison: `clean_prefix` given is the CORRECT prefix,
            `zero_prefix` is NO prefix, and neither given is the GENERATED prefix
            the model actually has to live with at inference. Takes precedence over
            `zero_prefix`; the active block and everything after it are untouched,
            so only the conditioning changes.
        eta: stochasticity of the update. 0 is the Euler ODE, exactly — the step
            reduces to it algebraically, not approximately, which is what makes
            `tests/test_m4_stochastic.py`'s identity check a real guard. 1
            resamples the noise component completely at every step. See
            `adze.sample.stochastic`.
        s_churn, s_noise, churn_window: EDM-style churn — noise UP past the
            schedule, then integrate back down. 0 disables it and leaves the
            trajectory bit-identical. `churn_window` is the (t_min, t_max) band
            churn applies in; EDM leaves the ends of the schedule alone.
        on_step: optional callback (global_step, t, active_block, latents) receiving
            the FULL batch at every step. `recorder` only ever sees sample 0, so a
            variance across the batch cannot be recovered from it; rather than
            distort the recorder into carrying batches it never decodes, statistics
            that need the batch take this hook.
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

    # Knots, not a step size. Under a shift the spacing is non-uniform, so each
    # step is taken between the two knots it actually spans. Hardcoding 1/nfe here
    # would silently integrate the wrong distance on every step of a shifted
    # schedule — and churn moves the interval's upper end anyway.
    knots = schedule(nfe, shift, device=dev)
    global_step = 0

    for b in range(blocks):
        mask = mask_for_block(block_ids, b)
        is_active = (block_ids == b).view(1, n_positions, 1)

        for i in range(nfe):
            # t runs 1 -> 0 for the active block. Earlier blocks are already clean
            # (t = 0); later blocks are absent, and their t is never read.
            t_active, s_active = float(knots[i]), float(knots[i + 1])

            # Churn first: noise the active block UP to t_hat, then integrate from
            # there. Costs no function evaluation, and is the identity at
            # s_churn = 0 — t_hat is t and churn_up returns its input unchanged.
            t_hat, _gamma = churn_t(
                t_active, s_churn, nfe, churn_window[0], churn_window[1]
            )
            if t_hat > t_active:
                latents = torch.where(
                    is_active, churn_up(latents, t_active, t_hat, s_noise), latents
                )
                t_active = t_hat

            t = torch.zeros(batch, blocks, device=dev)
            t[:, b] = t_active
            t[:, b + 1 :] = 1.0

            model_in = latents
            if clean_prefix is not None:
                model_in = torch.where(
                    (block_ids < b).view(1, n_positions, 1), clean_prefix, latents
                )
            elif zero_prefix:
                model_in = torch.where(
                    (block_ids < b).view(1, n_positions, 1),
                    torch.zeros_like(latents), latents,
                )
            velocity = denoiser(model_in, t, block_ids, MaskMode.CAUSAL, mask=mask)

            # Integrate the active block only. Applying the step everywhere would
            # walk the finished blocks off their values.
            stepped = renoise_step(latents, velocity, t_active, s_active, eta)
            latents = torch.where(is_active, stepped, latents)

            global_step += 1
            if on_step is not None:
                on_step(global_step, float(knots[i + 1]), b, latents)
            if recorder is not None:
                recorder.record(
                    global_step, float(knots[i + 1]), latents[0], active_block=b
                )

    return latents
