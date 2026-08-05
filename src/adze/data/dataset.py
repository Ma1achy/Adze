"""M1/M2 — torch Dataset and latent caching.

Latent caching (design §6.1) is the single biggest practical speedup available:
once the VAE is trained and frozen, encode the whole dataset to disk once and
every subsequent denoiser experiment skips encoding entirely.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from adze.data.corrupt import CorruptedPair


class TraceDataset(Dataset):
    """Tokenised traces. Yields per-step token sequences plus block structure."""

    def __init__(self, pairs: list[CorruptedPair], blocks: int) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Returns:
        {
            "tokens":       [B, T]   token ids per step
            "block_ids":    [N]
            "corrupted_idx": scalar  which block is corrupted, -1 if clean
        }
        """
        raise NotImplementedError


class LatentCache:
    """Encode a dataset once with a frozen VAE, store latents to disk, reload fast."""

    def __init__(self, path: Path) -> None:
        raise NotImplementedError

    def build(self, dataset: TraceDataset, encoder: torch.nn.Module) -> None:
        """Encode every example and write [n, N, D] to disk."""
        raise NotImplementedError

    def load(self) -> torch.Tensor:
        """Return cached latents, shape [n, N, D]."""
        raise NotImplementedError
