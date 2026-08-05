"""Explicit config dataclasses. One YAML load at entry, no config magic.

Nothing reads the raw dict past `load_config`. If a field is not here, it is not
config — it is a constant, and it belongs next to the code that uses it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DataConfig:
    n_train: int
    n_val: int
    max_depth: int
    operand_max: int
    blocks_per_sequence: int    # B
    seed: int


@dataclass(frozen=True)
class TransformerConfig:
    n_layers: int
    d_model: int
    n_heads: int


@dataclass(frozen=True)
class ModelConfig:
    latents_per_block: int      # K
    latent_dim: int             # D
    vae: TransformerConfig
    denoiser: TransformerConfig


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int
    lr: float
    steps: int
    regime_b_prob: float
    kl_beta: float


@dataclass(frozen=True)
class SampleConfig:
    nfe: int


@dataclass(frozen=True)
class Config:
    name: str
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    sample: SampleConfig
    device: str


def load_config(path: Path) -> Config:
    """Read one YAML file into typed dataclasses. The only place YAML is parsed."""
    raw = yaml.safe_load(Path(path).read_text())
    model = raw["model"]
    return Config(
        name=raw["name"],
        data=DataConfig(**raw["data"]),
        model=ModelConfig(
            latents_per_block=model["latents_per_block"],
            latent_dim=model["latent_dim"],
            vae=TransformerConfig(**model["vae"]),
            denoiser=TransformerConfig(**model["denoiser"]),
        ),
        train=TrainConfig(**raw["train"]),
        sample=SampleConfig(**raw["sample"]),
        device=raw["device"],
    )
