"""DAG generator structural diagnostic — before any model is trained.

Decision criterion: MULTI-CONSUMER COVERAGE, specifically the count of steps
with EXACTLY ONE near consumer (d ≤ 2) AND EXACTLY ONE far consumer (d ≥ 5).
That restricted cell is the intervention's actual sample size.

Chi2/dof is reported for reference but is NOT the criterion — the DAG design
does not require marginal independence between distance and provenance.

The consumer-level provenance check IS load-bearing: if far consumers are
systematically 'both-from-earlier' while near consumers are 'both-leaves', the
weld has moved to the consumer rather than the producer, and arm (b) vs arm (c)
would be confounded by what kind of step provides the evidence.

Usage:
    python scripts/m9_dag_diag.py
    python scripts/m9_dag_diag.py --traces 4000 --n-steps 10 --max-consumers 2
"""

from __future__ import annotations

import argparse
from collections import Counter

from adze.data.dag import generate_dag_dataset
from adze.eval.dag_strata import (
    consumer_distances,
    exactly_one_near_one_far,
    has_near_and_far,
    nearest_consumer_distance,
)
from adze.eval.strata import PROVENANCE, operand_provenance

NEAR_MAX = 2
FAR_MIN  = 5
MIN_CELL = 50   # fewer records → "thin", not printed

# Reference chi2/dof values from previous diagnostics.
REF_TREE = 374.8
REF_DEC  =   4.5
REF_CHESS = 604.0


# ── analysis ──────────────────────────────────────────────────────────────────

def analyse(traces):
    """Compute all distributions in one pass."""
    # consumer distance: all distances, not just nearest
    all_d: Counter = Counter()
    nearest_d: Counter = Counter()

    # provenance of the producer
    prov_counts: Counter = Counter()
    # fan-out (number of consumers) × producer provenance
    fanout_by_prov: dict[str, Counter] = {c: Counter() for c in PROVENANCE}

    # nearest consumer distance × producer provenance (for chi2 reference)
    nearest_x_prov: Counter = Counter()

    # joint (nearest_d, farthest_d) for multi-consumer steps
    joint_nd_fd: Counter = Counter()

    # near+far coverage: producer provenance
    near_far_prov: Counter = Counter()          # general (any near + any far)
    exact_one_prov: Counter = Counter()         # exactly-one-near, one-far

    # consumer-level provenance: what provenance class does the near/far consumer have?
    # For exact-one-near-one-far steps:
    near_consumer_prov: Counter = Counter()     # provenance of the near consumer step
    far_consumer_prov: Counter = Counter()      # provenance of the far consumer step
    # Joint (near_consumer_prov, far_consumer_prov) for chi2 check
    consumer_prov_joint: Counter = Counter()

    for tr in traces:
        for idx in range(len(tr.steps)):
            mv = tr.steps[idx]
            cs = consumer_distances(tr, idx)
            prov = operand_provenance(mv)
            prov_counts[prov] += 1
            fanout_by_prov[prov][len(cs)] += 1

            for d in cs:
                all_d[d] += 1

            nd = cs[0] if cs else None
            if nd is not None:
                nearest_d[nd] += 1
                nearest_x_prov[(nd, prov)] += 1

            if len(cs) >= 2:
                fd = cs[-1]
                joint_nd_fd[(nd, fd)] += 1

            if has_near_and_far(tr, idx, NEAR_MAX, FAR_MIN):
                near_far_prov[prov] += 1

            if exactly_one_near_one_far(tr, idx, NEAR_MAX, FAR_MIN):
                exact_one_prov[prov] += 1
                # Consumer-level provenance: find the single near and far consumer
                near_cs = [j for j in tr.consumer_map[idx] if j - idx <= NEAR_MAX]
                far_cs  = [j for j in tr.consumer_map[idx] if j - idx >= FAR_MIN]
                # By definition len == 1 each
                near_j = near_cs[0]
                far_j  = far_cs[0]
                n_prov = operand_provenance(tr.steps[near_j])
                f_prov = operand_provenance(tr.steps[far_j])
                near_consumer_prov[n_prov] += 1
                far_consumer_prov[f_prov] += 1
                consumer_prov_joint[(n_prov, f_prov)] += 1

    return {
        "all_d": all_d,
        "nearest_d": nearest_d,
        "prov_counts": prov_counts,
        "fanout_by_prov": fanout_by_prov,
        "nearest_x_prov": nearest_x_prov,
        "joint_nd_fd": joint_nd_fd,
        "near_far_prov": near_far_prov,
        "exact_one_prov": exact_one_prov,
        "near_consumer_prov": near_consumer_prov,
        "far_consumer_prov": far_consumer_prov,
        "consumer_prov_joint": consumer_prov_joint,
    }


# ── printing ──────────────────────────────────────────────────────────────────

def print_header(args) -> None:
    print(f"\n{'=' * 72}")
    print("DAG GENERATOR STRUCTURAL DIAGNOSTIC")
    print(f"{'=' * 72}")
    print(f"  n_steps={args.n_steps}  min_consumers={args.min_consumers}  "
          f"max_consumers={args.max_consumers}")
    print(f"  distance_min={args.distance_min}  distance_max={args.distance_max}")
    print(f"  traces={args.traces}  B={args.n_steps}  N={args.n_steps * 4}  (K=4)")


def print_fanout(traces) -> None:
    fan = Counter()
    for tr in traces:
        for idx in range(len(tr.steps) - 1):   # exclude root
            fan[len(tr.consumer_map[idx])] += 1
    total = sum(fan.values())
    print(f"\n{'=' * 72}")
    print("FAN-OUT DISTRIBUTION (non-root steps)")
    print(f"{'=' * 72}")
    for k in sorted(fan):
        print(f"  k={k}  {fan[k]:>8,}  {fan[k] / total:>6.1%}")
    print(f"  total non-root steps: {total:,}")


def print_consumer_dist(all_d: Counter) -> None:
    total = sum(all_d.values())
    print(f"\n{'=' * 72}")
    print("CONSUMER-DISTANCE DISTRIBUTION (all consumers, all fan-out levels)")
    print(f"{'=' * 72}")
    for name, pred in [
        ("d=1",      lambda d: d == 1),
        ("d=2",      lambda d: d == 2),
        ("d=3",      lambda d: d == 3),
        ("d=4",      lambda d: d == 4),
        ("d=5",      lambda d: d == 5),
        ("d=6..8",   lambda d: 6 <= d <= 8),
        ("d=9..12",  lambda d: 9 <= d <= 12),
        ("d=13+",    lambda d: d >= 13),
    ]:
        n = sum(v for k, v in all_d.items() if pred(k))
        print(f"  {name:>8}  {n:>8,}  {n / total:>6.1%}")
    print(f"  total consumer-distance records: {total:,}")


def print_near_far_coverage(data) -> None:
    print(f"\n{'=' * 72}")
    print(f"NEAR+FAR COVERAGE  (near = d ≤ {NEAR_MAX}, far = d ≥ {FAR_MIN})")
    print(f"{'=' * 72}")

    # General near+far (any number of near and far consumers)
    total_nf = sum(data["near_far_prov"].values())
    print(f"\n  General (any near + any far):  {total_nf:,} steps")
    for c in PROVENANCE:
        n = data["near_far_prov"][c]
        if n < MIN_CELL:
            print(f"    {c:<22}  thin (n={n})")
        else:
            print(f"    {c:<22}  {n:>7,}")

    # Restricted: exactly one near AND exactly one far (intervention sample size)
    total_ex = sum(data["exact_one_prov"].values())
    print(f"\n  Restricted (exactly 1 near + exactly 1 far):  {total_ex:,} steps")
    print(f"  *** THIS IS THE INTERVENTION'S SAMPLE SIZE ***")
    for c in PROVENANCE:
        n = data["exact_one_prov"][c]
        marker = " ← thin" if n < MIN_CELL else ""
        print(f"    {c:<22}  {n:>7,}{marker}")

    # Decision
    print()
    if total_ex >= 500 and all(data["exact_one_prov"][c] >= MIN_CELL for c in PROVENANCE):
        print("  COVERAGE PASSES (≥ 500 total, ≥ 50 per class)")
    elif total_ex >= 500:
        print("  COVERAGE BORDERLINE — some classes thin; consider retuning")
    else:
        print(f"  COVERAGE FAILS — {total_ex} < 500; retune generator parameters")


def print_joint_nd_fd(joint: Counter) -> None:
    """Print joint distribution of (nearest_d, farthest_d) for k>=2 steps."""
    print(f"\n{'=' * 72}")
    print("JOINT (nearest_d, farthest_d) DISTRIBUTION — multi-consumer steps")
    print(f"{'=' * 72}")
    # Summarise in buckets
    nf_buckets: Counter = Counter()
    for (nd, fd), cnt in joint.items():
        n_bucket = f"near≤2" if nd <= 2 else f"near={nd}"
        f_bucket = f"far≥5" if fd >= 5 else f"far={fd}"
        nf_buckets[(n_bucket, f_bucket)] += cnt

    total = sum(joint.values())
    print(f"  {'near':>10} {'far':>10} {'n':>8} {'%':>7}")
    for (nb, fb), n in sorted(nf_buckets.items(), key=lambda x: -x[1]):
        if n < MIN_CELL:
            continue
        print(f"  {nb:>10} {fb:>10} {n:>8,} {n / total:>6.1%}")
    print(f"  total multi-consumer steps: {total:,}")


def print_fanout_x_prov(fanout_by_prov: dict[str, Counter]) -> None:
    """Fan-out × producer provenance. Flag if coupled (weld one level up)."""
    print(f"\n{'=' * 72}")
    print("FAN-OUT × PRODUCER PROVENANCE")
    print("  If both-from-earlier has higher fan-out, the weld is one level up.")
    print(f"{'=' * 72}")
    print(f"  {'class':<22}  k=0  k=1  k=2  k=3+")
    for c in PROVENANCE:
        total = sum(fanout_by_prov[c].values())
        if total == 0:
            continue
        row = f"  {c:<22}"
        for k in (0, 1, 2):
            row += f"  {fanout_by_prov[c][k] / total:>5.1%}"
        k3plus = sum(v for ki, v in fanout_by_prov[c].items() if ki >= 3)
        row += f"  {k3plus / total:>5.1%}   n={total:,}"
        print(row)

    # Compute chi2 on fan-out (0, 1, 2, 3+) × provenance
    all_totals = {c: sum(fanout_by_prov[c].values()) for c in PROVENANCE}
    grand_total = sum(all_totals.values())
    if grand_total == 0:
        return
    fan_marg = {k: sum(fanout_by_prov[c].get(k, 0) for c in PROVENANCE) / grand_total
                for k in (0, 1, 2, 3)}
    chi2 = 0.0
    dof = 0
    for c in PROVENANCE:
        total_c = all_totals[c]
        if total_c == 0:
            continue
        for k in (0, 1, 2, 3):
            obs = fanout_by_prov[c].get(k if k < 3 else None, 0)
            if k == 3:
                obs = sum(v for ki, v in fanout_by_prov[c].items() if ki >= 3)
            exp = total_c * fan_marg[k]
            if exp > 0:
                chi2 += (obs - exp) ** 2 / exp
                dof += 1
    dof = max(dof - len(PROVENANCE) - 3, 1)
    per_dof = chi2 / dof
    print(f"\n  chi2/dof on fan-out × provenance: {per_dof:.1f}")
    if per_dof > 50:
        print("  WARNING: fan-out correlates with provenance (weld one level up)")
    else:
        print("  OK: fan-out and provenance appear independent")


def print_chi2_distance_x_prov(nearest_x_prov: Counter,
                                prov_counts: Counter) -> None:
    """Chi2 on nearest consumer distance × producer provenance. Reference only."""
    ds = sorted(set(d for d, _ in nearest_x_prov))
    usable = [d for d in ds if sum(nearest_x_prov[(d, c)] for c in PROVENANCE) >= MIN_CELL]
    if not usable:
        print("\n  (No distance bins with enough records for chi2)")
        return

    tot_usable = sum(nearest_x_prov[(d, c)] for d in usable for c in PROVENANCE)
    if tot_usable == 0:
        return
    marg = {c: sum(nearest_x_prov[(d, c)] for d in usable) / tot_usable
            for c in PROVENANCE}

    chi2 = 0.0
    for d in usable:
        n_d = sum(nearest_x_prov[(d, c)] for c in PROVENANCE)
        for c in PROVENANCE:
            obs = nearest_x_prov[(d, c)]
            exp = n_d * marg[c]
            if exp > 0:
                chi2 += (obs - exp) ** 2 / exp
    dof = max((len(usable) - 1) * (len(PROVENANCE) - 1), 1)
    per_dof = chi2 / dof
    print(f"\n{'=' * 72}")
    print("CHI2 ON NEAREST DISTANCE × PROVENANCE — reference only, not criterion")
    print(f"{'=' * 72}")
    print(f"  chi2/dof = {per_dof:.1f}  "
          f"(tree {REF_TREE}, decorrelated {REF_DEC}, chess {REF_CHESS})")
    print(f"  This is NOT the decision criterion for the DAG design.")


def print_consumer_prov_check(data) -> None:
    """Check consumer-level provenance balance. Load-bearing for the intervention."""
    print(f"\n{'=' * 72}")
    print("CONSUMER-LEVEL PROVENANCE — for exact-one-near + exact-one-far steps")
    print("  Load-bearing: if far=both-from-earlier and near=both-leaves, the")
    print("  weld has moved to the consumer; arm (b) vs (c) would be confounded.")
    print(f"{'=' * 72}")

    total = sum(data["exact_one_prov"].values())
    if total < MIN_CELL:
        print(f"  Too few exact-one steps ({total}) to assess consumer provenance.")
        return

    print(f"\n  Near consumer provenance (d ≤ {NEAR_MAX}):")
    for c in PROVENANCE:
        n = data["near_consumer_prov"][c]
        if n < MIN_CELL:
            print(f"    {c:<22}  thin (n={n})")
        else:
            print(f"    {c:<22}  {n:>7,}  ({n / total:>5.1%})")

    print(f"\n  Far consumer provenance (d ≥ {FAR_MIN}):")
    for c in PROVENANCE:
        n = data["far_consumer_prov"][c]
        if n < MIN_CELL:
            print(f"    {c:<22}  thin (n={n})")
        else:
            print(f"    {c:<22}  {n:>7,}  ({n / total:>5.1%})")

    # Chi2 test: does consumer provenance DIFFER between near and far positions?
    # Correct test: chi2 on (provenance class) × (near vs far label).
    # This answers "are near and far consumers drawn from the same distribution?"
    # rather than "are near and far consumer provenance independent of each other?"
    # The relevant confound is the MARGINAL DIFFERENCE, not the joint dependence.
    near_counts = data["near_consumer_prov"]
    far_counts  = data["far_consumer_prov"]
    n_near = sum(near_counts.values())
    n_far  = sum(far_counts.values())
    grand  = n_near + n_far
    chi2 = 0.0
    dof_cells = 0
    for c in PROVENANCE:
        n_obs = near_counts[c]
        f_obs = far_counts[c]
        row_total = n_obs + f_obs
        if row_total == 0:
            continue
        exp_near = row_total * n_near / grand if grand > 0 else 0
        exp_far  = row_total * n_far  / grand if grand > 0 else 0
        if exp_near > 0:
            chi2 += (n_obs - exp_near) ** 2 / exp_near
            dof_cells += 1
        if exp_far > 0:
            chi2 += (f_obs - exp_far) ** 2 / exp_far
            dof_cells += 1
    dof = max(dof_cells // 2 - 1, 1)   # rows - 1
    per_dof = chi2 / dof if dof > 0 else 0.0
    print(f"\n  chi2/dof (consumer provenance × near/far label): {per_dof:.1f}")
    print(f"  (Tests whether near and far consumers come from the same provenance")
    print(f"   distribution; confound if large.)")
    if per_dof > 50:
        print("  WARNING: near and far consumers have very different provenance")
        print("  distributions. Arm (b) vs (c) confounded. Include arm (d) as")
        print("  a difference-of-differences robustness check.")
    elif per_dof > 20:
        print("  INSPECT: some consumer provenance difference present.")
    else:
        print("  OK: near and far consumer provenance are from similar distributions.")


def print_novelty(traces) -> None:
    sample = traces[:1000] if len(traces) >= 1000 else traces
    unique = len(set(tuple(s.render() for s in t.steps) for t in sample))
    print(f"\n{'=' * 72}")
    print("TRACE NOVELTY")
    print(f"{'=' * 72}")
    print(f"  {unique} unique step sequences in {len(sample)} traces "
          f"({unique / len(sample):.1%})")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--traces", type=int, default=4000)
    p.add_argument("--n-steps", type=int, default=10)
    p.add_argument("--min-consumers", type=int, default=1)
    p.add_argument("--max-consumers", type=int, default=2)
    p.add_argument("--distance-min", type=int, default=1)
    p.add_argument("--distance-max", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print(f"Generating {args.traces} DAG traces …")
    traces = generate_dag_dataset(
        n=args.traces,
        seed=args.seed,
        n_steps=args.n_steps,
        min_consumers=args.min_consumers,
        max_consumers=args.max_consumers,
        distance_min=args.distance_min,
        distance_max=args.distance_max,
    )
    print(f"  done — {sum(len(t.steps) for t in traces):,} total steps")

    print_header(args)
    data = analyse(traces)

    print_fanout(traces)
    print_consumer_dist(data["all_d"])
    print_near_far_coverage(data)
    print_joint_nd_fd(data["joint_nd_fd"])
    print_consumer_prov_check(data)
    print_fanout_x_prov(data["fanout_by_prov"])
    print_chi2_distance_x_prov(data["nearest_x_prov"], data["prov_counts"])
    print_novelty(traces)


if __name__ == "__main__":
    main()
