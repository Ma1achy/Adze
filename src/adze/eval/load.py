"""Checkpoint loading, shared by every diagnostic script.

Checkpoints record their own architecture under an `arch` key, and a run trained
with a CLI override (`--vae-layers 4`, `--latent-dim 16`) does not match its YAML.
Rebuilding from the config would construct the wrong shape and load_state_dict
would either error or, worse, succeed against a coincidentally-compatible size.
So: always rebuild from `arch`, never from the config. One place, so no script can
get it wrong on its own.
"""

from __future__ import annotations

from pathlib import Path

import torch

from adze.model.denoiser import Denoiser
from adze.model.vae import build_vae


def load_vae(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    """Returns (vae in eval mode, its recorded arch)."""
    state = torch.load(path, map_location=device, weights_only=False)
    vae = build_vae(**state["arch"]).to(device)
    vae.load_state_dict(state["model"])
    vae.eval()
    return vae, state["arch"]


def load_denoiser(path: Path, device: torch.device) -> tuple[Denoiser, dict, float | None]:
    """Returns (denoiser in eval mode, its recorded arch, the latent scale if stored).

    The latent scale is carried on the checkpoint because decoding a scaled latent
    produces garbage that reads as a broken sampler. It may be absent on older
    checkpoints, in which case the caller falls back to the LatentCache.
    """
    state = torch.load(path, map_location=device, weights_only=False)
    arch = state["arch"]
    denoiser = Denoiser(**arch).to(device)
    denoiser.load_state_dict(state["model"])
    denoiser.eval()
    return denoiser, arch, state.get("latent_scale")
