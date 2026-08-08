"""One place that decides WHICH generator a config's data comes from.

Two generators now exist and they are not interchangeable:

- `generate.py` — expression tree, post-order. Distance-to-consumer is welded to
  operand provenance by construction, so any distance profile measured on it is
  partly a provenance profile. Everything through session 22 used this.
- `decorrelated.py` — consumers assigned by bounded distance, provenance falling
  out as emergent in-degree. Distance means what it says.

The failure this module exists to prevent is silent: a script that regenerates
traces with one generator while the latent cache beside it was built with the
other produces latents that decode to nothing and looks like a model problem. The
same reasoning already gave `trace_kwargs` a single home; this extends it to the
choice of builder.

`Config.data.generator` selects. The config `name` should differ between them so
their checkpoints and caches cannot overwrite each other.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from adze.data.decorrelated import generate_decorrelated_dataset
from adze.data.generate import Trace, configured_leaves, generate_dataset

if TYPE_CHECKING:
    from adze.config import Config

GENERATORS = ("tree", "decorrelated")


def _builder(config: "Config"):
    """A callable `(n, seed, leaf_values) -> list[Trace]` for this config."""
    g = config.data.generator
    if g == "tree":
        def build(n, seed, leaf_values):
            return generate_dataset(
                n=n, seed=seed, max_depth=config.data.max_depth,
                operand_max=config.data.operand_max, leaf_values=leaf_values,
                magnitude_cap=config.data.magnitude_cap)
        return build
    if g == "decorrelated":
        def build(n, seed, leaf_values):
            return generate_decorrelated_dataset(
                n=n, seed=seed,
                min_steps=config.data.chain_steps,
                max_steps=config.data.chain_steps,
                operand_max=config.data.operand_max, leaf_values=leaf_values,
                magnitude_cap=config.data.magnitude_cap,
                distance_max=config.data.distance_max)
        return build
    raise ValueError(f"data.generator must be one of {GENERATORS}, got {g!r}")


def leaf_pool(config: "Config") -> tuple[int, ...] | None:
    """The leaf-operand pool this config's data is built from, or None.

    The fixed point is iterated with THIS config's generator. The pool is the
    distribution of results, and the decorrelated generator produces a different
    result distribution from the tree one — same chain length everywhere, more
    steps, different operand mix — so carrying the tree's pool across would
    reintroduce exactly the input/output magnitude mismatch the fixed point exists
    to remove.
    """
    if config.data.leaf_distribution == "uniform":
        return None
    if config.data.generator == "tree":
        return configured_leaves(
            config.data.leaf_distribution, config.data.leaf_iterations,
            config.data.seed, config.data.max_depth, config.data.operand_max,
            config.data.magnitude_cap)
    return _decorrelated_leaves(
        config.data.leaf_iterations, config.data.seed,
        config.data.chain_steps, config.data.operand_max,
        config.data.magnitude_cap, config.data.distance_max)


_LEAF_CACHE: dict[tuple, tuple[int, ...]] = {}


def _decorrelated_leaves(iterations, seed, chain_steps, operand_max,
                         magnitude_cap, distance_max) -> tuple[int, ...]:
    """`fixed_point_leaves` for the decorrelated generator. Cached; ~40s."""
    key = (iterations, seed, chain_steps, operand_max, magnitude_cap, distance_max)
    if key in _LEAF_CACHE:
        return _LEAF_CACHE[key]
    leaves: tuple[int, ...] | None = None
    for i in range(iterations + 1):
        traces = generate_decorrelated_dataset(
            n=20_000, seed=seed + i, min_steps=chain_steps, max_steps=chain_steps,
            operand_max=operand_max, leaf_values=leaves,
            magnitude_cap=magnitude_cap, distance_max=distance_max)
        results = tuple(abs(s.result) for t in traces for s in t.steps
                        if abs(s.result) >= 1)
        if i < iterations:
            leaves = results
    out = leaves if leaves is not None else tuple(range(1, operand_max + 1))
    _LEAF_CACHE[key] = out
    return out


def make_dataset(config: "Config", n: int, seed: int) -> list[Trace]:
    """`n` traces for this config, from whichever generator it selects."""
    return _builder(config)(n, seed, leaf_pool(config))
