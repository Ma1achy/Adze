"""Free-running draft sampling, shared between the readouts that need it.

Promoted out of `scripts/m4_saturation.py` when the mix sweep needed the same
thing at a fixed nfe across several checkpoints. Two copies of a sampler that
must stay identical is the joint that gives way: a difference in seeding or in
the unscale step would show up as a difference between checkpoints and be read as
a difference between training mixes.
"""

from __future__ import annotations

import torch

from adze.sample.draft import draft


@torch.no_grad()
def sample_draft(denoiser, vae, tokeniser, arch: dict, scale: float, nfe: int,
                 eta: float, samples: int, device, seed: int) -> list[str]:
    """Free-run `samples` traces and decode every block. Returns raw text.

    Nothing is filtered or retried — a block that decodes to garbage is returned
    as garbage, and the readout counts it.
    """
    torch.manual_seed(seed)
    blocks = arch["blocks"]
    latents = draft(
        denoiser, None, blocks, arch["latents_per_block"], arch["latent_dim"], nfe,
        device=str(device), batch=samples, eta=eta,
    )
    per = (latents * scale).view(samples * blocks, arch["latents_per_block"], -1)
    return [tokeniser.decode(r) for r in vae.decoder(per).argmax(dim=-1)]
