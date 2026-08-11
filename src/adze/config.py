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
    # Leaf-operand distribution. "uniform" draws 1..operand_max, the original
    # sampler; "empirical" draws from the fixed point of the RESULT distribution,
    # so the model never meets a magnitude as an input it has not seen as an
    # output. See adze.data.generate.fixed_point_leaves for why raising
    # operand_max does not fix this for any value.
    leaf_distribution: str = "uniform"
    leaf_iterations: int = 5
    # WHICH GENERATOR. "tree" is the original expression-tree/post-order builder,
    # which welds distance-to-consumer to operand provenance by construction —
    # every result through session 22 was measured on it. "decorrelated" assigns
    # consumers by bounded distance so a distance profile means what it says.
    # See adze.data.build. Change the config `name` alongside it: the two produce
    # incompatible caches and checkpoints.
    generator: str = "tree"
    # decorrelated and dag. `chain_steps` is the exact step count, so B must be
    # at least this; `distance_max` bounds the consumer draw. Usable distances
    # run 1..dmax-1. dag_min/max_consumers control how many consumers each step
    # requests; ignored for all other generators.
    chain_steps: int = 10
    distance_max: int = 6
    dag_min_consumers: int = 1
    dag_max_consumers: int = 2
    # Ceiling on |result|, enforced when the operator is chosen. It sets the
    # CHARACTER of the task, not just its range: the leaf fixed point converges
    # against it, so cap 1000 produced three-digit arithmetic the denoiser could
    # not do at all (0.8% truth against a 96.6% ceiling), while cap 100 keeps
    # operands and results two-digit.
    magnitude_cap: int = 1_000


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


def trace_kwargs(config: "Config") -> dict:
    """Every argument `generate_dataset` needs to reproduce this config's data.

    One place. Threading max_depth / operand_max / leaf_values / magnitude_cap
    separately through seven call sites is how a script ends up generating traces
    that disagree with the latent cache sitting beside it — the failure would be
    silent and would look like a model problem.
    """
    return {
        "max_depth": config.data.max_depth,
        "operand_max": config.data.operand_max,
        "leaf_values": leaf_pool(config),
        "magnitude_cap": config.data.magnitude_cap,
    }


def leaf_pool(config: "Config") -> tuple[int, ...] | None:
    """The leaf-operand pool this config's data is built from.

    One place, so a script that regenerates traces cannot silently disagree with
    the one that built the latent cache. Returns None for the uniform sampler.
    """
    from adze.data.generate import configured_leaves

    return configured_leaves(
        config.data.leaf_distribution,
        config.data.leaf_iterations,
        config.data.seed,
        config.data.max_depth,
        config.data.operand_max,
        config.data.magnitude_cap,
    )
