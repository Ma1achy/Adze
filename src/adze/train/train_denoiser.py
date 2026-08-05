"""M3/M6 — denoiser training.

M3 is regime A only. M6 adds regime B and the 90/10 mix.

Regime A (draft): sample block b; noise block b only; blocks < b clean;
blocks > b absent; causal mask; loss on b.

Regime B (refine): select subset S; t_i = 1 for i in S (complete erasure);
blocks outside S clean; global mask; loss on S.

M6 requires retraining from scratch — mixing changes what the model learns, not
just how it is used. Acceptance for M6 is that pass-one quality does NOT regress
against the M5 baseline.
"""

from __future__ import annotations

from pathlib import Path


def train_denoiser(config_path: Path, mixed: bool = False) -> None:
    """Args:
        config_path: yaml config.
        mixed: False for M3 (regime A only), True for M6 (90/10 mix).
    """
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit("not implemented — see docs/build-plan.md M3")
