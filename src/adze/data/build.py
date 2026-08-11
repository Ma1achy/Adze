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

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

from adze.data.dag import generate_dag_dataset, generate_dag_trace
from adze.data.decorrelated import generate_decorrelated_dataset
from adze.data.generate import Trace, configured_leaves, generate_dataset

if TYPE_CHECKING:
    from adze.config import Config

GENERATORS = ("tree", "decorrelated", "dag")


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
    if g == "dag":
        def build(n, seed, leaf_values):
            return generate_dag_dataset(
                n=n, seed=seed,
                n_steps=config.data.chain_steps,
                min_consumers=config.data.dag_min_consumers,
                max_consumers=config.data.dag_max_consumers,
                distance_min=1,
                distance_max=config.data.distance_max,
                operand_max=config.data.operand_max, leaf_values=leaf_values,
                magnitude_cap=config.data.magnitude_cap)
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
    if config.data.generator == "dag":
        # Not the decorrelated pool. Fan-out means a value can be consumed twice
        # and the computed-operand mix differs, so the result distribution — which
        # IS the pool — is a different distribution. Iterating the wrong generator
        # here reintroduces the input/output magnitude mismatch the fixed point
        # exists to remove, silently.
        return _dag_leaves(
            config.data.leaf_iterations, config.data.seed,
            config.data.chain_steps, config.data.operand_max,
            config.data.magnitude_cap, config.data.distance_max,
            config.data.dag_min_consumers, config.data.dag_max_consumers)
    return _decorrelated_leaves(
        config.data.leaf_iterations, config.data.seed,
        config.data.chain_steps, config.data.operand_max,
        config.data.magnitude_cap, config.data.distance_max)


_LEAF_CACHE: dict[tuple, tuple[int, ...]] = {}

# On-disk twin of `_LEAF_CACHE`. The in-process dict is only good for the life of
# one interpreter, so every fresh launch re-paid the whole fixed point for a pool
# that is a pure function of its key. Measured at ~6s for dag10, so this is
# convenience rather than a bottleneck. Gitignored (`data/cache/`).
_POOL_DIR = Path(__file__).resolve().parents[3] / "data" / "cache" / "leafpool"


def _pool_path(key: tuple) -> Path:
    """Where the pool for `key` lives. The digest names it; the key verifies it."""
    digest = hashlib.sha256(repr(key).encode()).hexdigest()[:16]
    return _POOL_DIR / f"{digest}.json"


def _read_pool(key: tuple) -> tuple[int, ...] | None:
    """The persisted pool for `key`, or None. Any fault falls through to regenerate.

    The stored key is compared against the requested one so a digest collision or
    a hand-edited file cannot silently hand back the wrong distribution — that
    failure would present as a model problem, which is the thing this module
    exists to prevent.
    """
    path = _pool_path(key)
    try:
        blob = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if blob.get("key") != list(key):
        return None
    return tuple(blob["pool"])


def _write_pool(key: tuple, pool: tuple[int, ...]) -> None:
    """Persist `pool` under `key`. Write-and-rename, so a kill mid-write cannot
    leave a truncated file that reads as a valid pool."""
    path = _pool_path(key)
    try:
        _POOL_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"key": list(key), "pool": list(pool)}))
        tmp.replace(path)
    except OSError:
        pass    # a cache that cannot be written is slow, not wrong


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


def _dag_leaves(iterations, seed, chain_steps, operand_max, magnitude_cap,
                distance_max, min_consumers, max_consumers) -> tuple[int, ...]:
    """`fixed_point_leaves` for the DAG generator. Cached; ~40s.

    Same iteration as `_decorrelated_leaves`, run on the DAG builder so the pool
    is the fixed point of the distribution this config actually emits.
    """
    key = ("dag", iterations, seed, chain_steps, operand_max, magnitude_cap,
           distance_max, min_consumers, max_consumers)
    if key in _LEAF_CACHE:
        return _LEAF_CACHE[key]
    disk = _read_pool(key)
    if disk is not None:
        _LEAF_CACHE[key] = disk
        return disk
    leaves: tuple[int, ...] | None = None
    for i in range(iterations + 1):
        # Streamed, not `generate_dag_dataset`, so only one DagTrace is resident
        # rather than 20k per round. This was written believing the list-building
        # form was the memory offender behind a machine lock-up; it was not —
        # measured peak RSS is 0.03 GB streamed and the list form runs a round in
        # 0.9s. Kept because it is strictly cheaper and verified equivalent, not
        # because it fixed anything. Trace j of round i is seeded from
        # (seed + i) * 1_000_003 + j — the convention `generate_dag_dataset`
        # uses — so the pool is bit-identical to the list-building version.
        results: list[int] = []
        for j in range(20_000):
            trace = generate_dag_trace(
                rng_seed=(seed + i) * 1_000_003 + j, n_steps=chain_steps,
                min_consumers=min_consumers, max_consumers=max_consumers,
                distance_min=1, distance_max=distance_max,
                operand_max=operand_max, leaf_values=leaves,
                magnitude_cap=magnitude_cap)
            results.extend(abs(s.result) for s in trace.steps
                           if abs(s.result) >= 1)
        if i < iterations:
            leaves = tuple(results)
    out = leaves if leaves is not None else tuple(range(1, operand_max + 1))
    _LEAF_CACHE[key] = out
    _write_pool(key, out)
    return out


def make_dataset(config: "Config", n: int, seed: int) -> list[Trace]:
    """`n` traces for this config, from whichever generator it selects."""
    return _builder(config)(n, seed, leaf_pool(config))
