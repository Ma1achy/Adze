"""M2 — VAE training loop.

Ends with two hard gates, both in adze.eval.checks:
  1. reconstruction > 95% exact token match on held-out steps
  2. latent-use check — decoding from shuffled latents must degrade badly

Gate 2 is the important one. A VAE that reconstructs perfectly while ignoring its
latent makes everything downstream meaningless.
"""

from __future__ import annotations

from pathlib import Path


def train_vae(config_path: Path) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit("not implemented — see docs/build-plan.md M2")
