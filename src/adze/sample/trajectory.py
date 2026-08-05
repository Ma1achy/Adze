"""M4 — the trajectory printer.

Decode and print at every denoising step. This is the main debugging instrument
for everything after M4 — build it properly, not as a print() in a loop.

Watching noise resolve into structure into content is both the fastest way to
spot a broken model and the actual pleasure of this architecture. If step 40 of
50 is still gibberish you know long before a loss curve tells you.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch


class TrajectoryRecorder:
    """Capture latents at every denoising step, decode on demand."""

    def __init__(self, decoder: torch.nn.Module) -> None:
        raise NotImplementedError

    def record(self, step: int, t: float, latents: torch.Tensor) -> None:
        raise NotImplementedError

    def decoded(self) -> Iterator[tuple[int, float, list[str]]]:
        """Yield (step, t, decoded_text_per_block) for each recorded step."""
        raise NotImplementedError

    def print(self, every: int = 1) -> None:
        """Pretty-print the trajectory to stdout."""
        raise NotImplementedError
