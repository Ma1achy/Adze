"""Stratification helpers for DAG traces.

Extends `adze.eval.strata` with multi-consumer distance functions. Provenance
labelling (`operand_provenance`) is unchanged — the lhs_from/rhs_from schema
is identical to the tree and decorrelated generators.
"""

from __future__ import annotations

from adze.data.dag import DagTrace
from adze.eval.strata import PROVENANCE, operand_provenance   # reused unchanged


def consumer_distances(trace: DagTrace, step_idx: int) -> tuple[int, ...]:
    """All consumer distances for step `step_idx`, sorted ascending.

    Returns an empty tuple for the root (no consumers).
    """
    return tuple(j - step_idx for j in trace.consumer_map[step_idx])


def nearest_consumer_distance(trace: DagTrace, step_idx: int) -> int | None:
    """Distance to the nearest consumer, or None if the step is the root."""
    cs = consumer_distances(trace, step_idx)
    return cs[0] if cs else None


def farthest_consumer_distance(trace: DagTrace, step_idx: int) -> int | None:
    """Distance to the farthest consumer, or None if the step is the root."""
    cs = consumer_distances(trace, step_idx)
    return cs[-1] if cs else None


def has_near_and_far(
    trace: DagTrace,
    step_idx: int,
    near_max: int = 2,
    far_min: int = 5,
) -> bool:
    """True if step has at least one near consumer (d ≤ near_max) AND at least
    one far consumer (d ≥ far_min).
    """
    cs = consumer_distances(trace, step_idx)
    return any(d <= near_max for d in cs) and any(d >= far_min for d in cs)


def exactly_one_near_one_far(
    trace: DagTrace,
    step_idx: int,
    near_max: int = 2,
    far_min: int = 5,
) -> bool:
    """True if step has EXACTLY ONE near consumer AND EXACTLY ONE far consumer.

    This is the mask-count discipline for the intervention: both arms (b) and
    (c) erase exactly one block, removing the |S| confound.
    """
    cs = consumer_distances(trace, step_idx)
    near = [d for d in cs if d <= near_max]
    far  = [d for d in cs if d >= far_min]
    return len(near) == 1 and len(far) == 1


__all__ = [
    "consumer_distances",
    "nearest_consumer_distance",
    "farthest_consumer_distance",
    "has_near_and_far",
    "exactly_one_near_one_far",
    "operand_provenance",   # re-exported for convenience
    "PROVENANCE",
]
