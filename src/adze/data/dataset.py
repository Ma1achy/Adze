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
from adze.data.tokeniser import MAX_STEP_LEN, PAD_ID, CharTokeniser


class TraceDataset(Dataset):
    """Tokenised traces. Yields per-step token sequences plus block structure.

    A trace has 3-7 steps against a fixed B. Blocks past the trace's length are
    padding: their token row is all PAD and `block_mask` is False there, so they
    can be masked out of any loss. Nothing is encoded in a sentinel value and no
    zero-latent convention is implied — the pad *representation* lives in
    `StepVAE.pad_latent` and is applied at cache time.

    Traces with more steps than `blocks` cannot be represented and are dropped at
    construction. The count is exposed as `n_dropped` rather than swallowed.
    """

    def __init__(
        self,
        pairs: list[CorruptedPair],
        blocks: int,
        latents_per_block: int = 4,
        use_corrupted: bool = False,
        max_len: int = MAX_STEP_LEN,
    ) -> None:
        self.blocks = blocks
        self.latents_per_block = latents_per_block
        self.use_corrupted = use_corrupted
        self.max_len = max_len
        self.tokeniser = CharTokeniser()

        self.pairs = [p for p in pairs if len(p.clean.steps) <= blocks]
        self.n_dropped = len(pairs) - len(self.pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    @property
    def n_positions(self) -> int:
        """N = B * K."""
        return self.blocks * self.latents_per_block

    def block_ids(self) -> torch.Tensor:
        """[N] — which block each latent position belongs to."""
        return torch.repeat_interleave(
            torch.arange(self.blocks), self.latents_per_block
        )

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Returns:
        {
            "tokens":        [B, T]  token ids per step, PAD rows for pad blocks
            "block_mask":    [B]     True where the block holds a real step
            "block_ids":     [N]
            "corrupted_idx": scalar  which block is corrupted, -1 if clean
        }
        """
        pair = self.pairs[idx]
        trace = pair.corrupted if self.use_corrupted else pair.clean

        tokens = torch.full((self.blocks, self.max_len), PAD_ID, dtype=torch.long)
        block_mask = torch.zeros(self.blocks, dtype=torch.bool)
        for i, step in enumerate(trace.steps):
            tokens[i] = torch.tensor(
                self.tokeniser.encode(step.render(), self.max_len), dtype=torch.long
            )
            block_mask[i] = True

        corrupted_idx = pair.block_index if self.use_corrupted else -1

        return {
            "tokens": tokens,
            "block_mask": block_mask,
            "block_ids": self.block_ids(),
            "corrupted_idx": torch.tensor(corrupted_idx, dtype=torch.long),
        }


class LatentCache:
    """Encode a dataset once with a frozen VAE, store latents to disk, reload fast."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    @torch.no_grad()
    def build(self, dataset: TraceDataset, encoder: torch.nn.Module) -> None:
        """Encode every example and write [n, N, D] to disk.

        Args:
            dataset: the dataset to encode.
            encoder: the frozen `StepVAE`. The whole VAE rather than its encoder
                alone, because pad blocks are filled with its `pad_latent` — one
                canonical pad representation, shared with the denoiser.

        The posterior *mean* is cached, not a sample. The cache is written once and
        read many times, so sampling here would freeze one arbitrary draw per step
        and pass it off as the encoding. Any noise the denoiser wants, it can add.
        """
        encoder.eval()
        device = next(encoder.parameters()).device

        latents = torch.empty(
            len(dataset), dataset.n_positions, encoder.pad_latent.shape[-1]
        )

        for i in range(len(dataset)):
            item = dataset[i]
            tokens = item["tokens"].to(device)                 # [B, T]
            block_mask = item["block_mask"].to(device)         # [B]

            mu, _ = encoder.encoder(tokens)                    # [B, K, D]
            mu[~block_mask] = encoder.pad_latent.to(mu.dtype)

            latents[i] = mu.reshape(dataset.n_positions, -1).cpu()

        self.path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(latents, self.path)

    def load(self) -> torch.Tensor:
        """Return cached latents, shape [n, N, D]."""
        if not self.path.exists():
            raise FileNotFoundError(f"no latent cache at {self.path}; run build() first")
        return torch.load(self.path, weights_only=True)
