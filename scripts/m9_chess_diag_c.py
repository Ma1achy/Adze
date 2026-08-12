"""Chess diagnostic — definition C (square-occupancy dependency).

Definition C: move j is a consumer of move i if move i placed a piece on a
square that j's legality depends on — j's destination (capture), an
intermediate path square for a sliding move (cleared line), or a check/pin
square (piece placing threat).

This gives a multi-consumer map: one ply can be a producer for many consumers
at varying distances.

## What this reports

1. chi2/dof cross-tab using the NEAREST consumer per producer — directly
   comparable to definition A's chi2/dof = 604.

2. Multi-consumer coverage: fraction with k>=2 consumers, k distribution,
   and the joint (d_near, d_far) distribution for plies with both.

3. Restricted cell (§11 within-record design sample): plies i with exactly one
   consumer at d<=2 AND exactly one consumer at d>=5 — overall and per
   provenance class.

4. Near vs far producer provenance composition: for all (i, j) dependency
   pairs split by distance, the provenance distribution of the producer i.
   The DAG's confound arm was that far consumers (d>=5) had 93% BFE producers.

5. Piece-type composition (moving and captured) per provenance class.

## Registered prediction

C should show MUCH WEAKER distance-provenance coupling than A, because
occupancy dependencies are governed by how long a piece SITS, which is roughly
independent of how it got there. If chi2/dof under C is still above 200, the
weld is deeper than the definition and chess is the fourth null for the same
reason as the others.

Decision gate:
  chi2/dof < 50 AND restricted cell >= 500 → proceed to tokeniser, VAE,
    denoiser, §11 within-record arms
  chi2/dof 50–200 → inspect conditionals; within-record design may still work
  chi2/dof > 200 → weld survives definition change; record as prediction
    failing, close chess as between-record measurement under both definitions

Usage:
    python scripts/m9_chess_diag_c.py --pgn /path/to/games.pgn
    python scripts/m9_chess_diag_c.py --pgn /path/to/games.pgn --traces 5000
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import chess

from adze.data.chess import (
    _is_classical_or_rapid,
    game_to_trace,
    generate_random_dataset,
)
from adze.eval.chess_strata import (
    chess_occupancy_consumer_map,
    chess_operand_provenance,
    PIECE_TYPE_NAMES,
)
from adze.eval.strata import PROVENANCE

import chess.pgn

# Reference baselines
REF_ORIG    = 374.8   # original arithmetic tree generator, per-dof chi2
REF_DEC     =   4.5   # decorrelated synthetic
REF_CHESS_A = 604.0   # definition A (nearest consumer, real games)

MIN_CELL = 50   # thin-cell threshold


# ── loading ───────────────────────────────────────────────────────────────────

def _categorise_tc(tc_str: str | None) -> str:
    if not tc_str or tc_str in ("-", "?"):
        return "unknown"
    try:
        if "+" in tc_str:
            base_s, inc_s = tc_str.split("+")
            base, inc = int(base_s), int(inc_s)
        else:
            base, inc = int(tc_str), 0
    except ValueError:
        return "unknown"
    total = base + 40 * inc
    if total < 30:   return "ultra-bullet"
    if total < 180:  return "bullet"
    if total < 480:  return "blitz"
    if total < 1500: return "rapid"
    return "classical"


def load_pgn_with_stats(path: Path, n: int, filter_tc: bool):
    stats = {
        "total_games": 0, "kept_games": 0, "parse_errors": 0,
        "tc_counts": Counter(), "tc_categories": Counter(),
    }
    traces = []
    game_idx = 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        while len(traces) < n:
            game = chess.pgn.read_game(fh)
            if game is None:
                break
            stats["total_games"] += 1
            tc = game.headers.get("TimeControl", "?")
            stats["tc_counts"][tc] += 1
            stats["tc_categories"][_categorise_tc(tc)] += 1
            if filter_tc and not _is_classical_or_rapid(tc):
                continue
            trace = game_to_trace(game, game_id=str(game_idx))
            game_idx += 1
            if trace is None:
                stats["parse_errors"] += 1
                continue
            traces.append(trace)
            stats["kept_games"] += 1
    return traces, stats


def print_load_stats(stats: dict) -> None:
    print(f"\n{'=' * 72}")
    print("LOADING STATS")
    print(f"{'=' * 72}")
    print(f"  total games scanned:            {stats['total_games']:>8,}")
    print(f"  kept after time-control filter: {stats['kept_games']:>8,}")
    print(f"  parse errors (skipped):         {stats['parse_errors']:>8,}")
    print()
    print("  Time-control categories:")
    for cat in ("classical", "rapid", "blitz", "bullet", "ultra-bullet", "unknown"):
        n = stats["tc_categories"][cat]
        if n > 0:
            pct = n / max(stats["total_games"], 1)
            marker = " ← kept" if cat in ("classical", "rapid") else ""
            print(f"    {cat:<15} {n:>8,}  ({pct:>5.1%}){marker}")


# ── analysis ──────────────────────────────────────────────────────────────────

def analyse(traces):
    """Build all distributions from definition-C consumer maps.

    Per ply i:
      consumers  = consumer_map[i]   (sorted tuple of j > i)
      prov       = provenance of moves[i]
      d_nearest  = min(j - i) if consumers else None
      d_farthest = max(j - i) if consumers else None
      k          = len(consumers)

    Outputs
    -------
    near_cells, near_totals    — nearest-consumer cross-tab (chi2 input)
    k_dist                     — Counter of k values (0, 1, 2, ...)
    near_far_dist              — Counter[(d_near_bin, d_far_bin)] for k>=2
    restricted                 — Counter[prov] for plies in the restricted cell
    restricted_total           — int, total restricted-cell plies
    near_prov_dist             — Counter[prov] for all (i,j) pairs with d<=2
    far_prov_dist              — Counter[prov] for all (i,j) pairs with d>=5
    pt_moving_by_class         — dict[prov -> Counter[pt_name]]
    cpt_by_class               — dict[prov -> Counter[pt_name]]  (captures only)
    """
    near_cells:  Counter = Counter()
    near_totals: Counter = Counter()
    k_dist:      Counter = Counter()
    near_far_dist: Counter = Counter()
    restricted:  Counter = Counter()
    restricted_total = 0
    near_prov_dist: Counter = Counter()   # prov of producer in near pairs
    far_prov_dist:  Counter = Counter()   # prov of producer in far pairs
    pt_moving_by_class: dict[str, Counter] = {c: Counter() for c in PROVENANCE}
    cpt_by_class:       dict[str, Counter] = {c: Counter() for c in PROVENANCE}

    total_producers = 0
    total_with_any  = 0

    for tr in traces:
        cmap = chess_occupancy_consumer_map(tr)
        n = len(tr.moves)

        for i in range(n):
            mv = tr.moves[i]
            prov = chess_operand_provenance(mv)
            pt_name = PIECE_TYPE_NAMES.get(mv.piece_type, "?")
            pt_moving_by_class[prov][pt_name] += 1
            if mv.is_capture and mv.captured_piece_type is not None:
                cpt_name = PIECE_TYPE_NAMES.get(mv.captured_piece_type, "?")
                cpt_by_class[prov][cpt_name] += 1

            consumers = cmap.get(i, ())
            k = len(consumers)
            k_dist[k] += 1
            total_producers += 1
            if k == 0:
                continue
            total_with_any += 1

            # Nearest consumer for chi2.
            d_nearest = consumers[0] - i
            near_cells[(d_nearest, prov)] += 1
            near_totals[d_nearest] += 1

            # All-pairs near/far provenance.
            for j in consumers:
                d = j - i
                if d <= 2:
                    near_prov_dist[prov] += 1
                if d >= 5:
                    far_prov_dist[prov] += 1

            # Multi-consumer: joint (d_near_bin, d_far_bin) for k>=2.
            if k >= 2:
                d_far = consumers[-1] - i

                def _bin(d: int) -> str:
                    if d <= 2:   return "d<=2"
                    if d <= 5:   return "d3-5"
                    if d <= 10:  return "d6-10"
                    return "d11+"

                near_far_dist[(_bin(d_nearest), _bin(d_far))] += 1

            # Restricted cell: exactly one consumer at d<=2 AND exactly one at d>=5.
            near_consumers = [j for j in consumers if (j - i) <= 2]
            far_consumers  = [j for j in consumers if (j - i) >= 5]
            if len(near_consumers) == 1 and len(far_consumers) == 1:
                restricted[prov] += 1
                restricted_total += 1

    return {
        "near_cells":         near_cells,
        "near_totals":        near_totals,
        "k_dist":             k_dist,
        "near_far_dist":      near_far_dist,
        "restricted":         restricted,
        "restricted_total":   restricted_total,
        "near_prov_dist":     near_prov_dist,
        "far_prov_dist":      far_prov_dist,
        "pt_moving_by_class": pt_moving_by_class,
        "cpt_by_class":       cpt_by_class,
        "total_producers":    total_producers,
        "total_with_any":     total_with_any,
    }


# ── printing helpers ──────────────────────────────────────────────────────────

def print_chi2_crosstab(near_cells, near_totals, label: str = "") -> float:
    """Print cross-tab (nearest consumer) and return per-dof chi2."""
    ds = sorted(d for d in near_totals if near_totals[d] >= MIN_CELL)
    if not ds:
        print(f"  (no distance bin with >= {MIN_CELL} records)")
        return 0.0

    tot = sum(near_totals[d] for d in ds)
    marg = {c: sum(near_cells[(d, c)] for d in ds) / tot for c in PROVENANCE}

    print(f"\n{'=' * 72}")
    print(f"NEAREST-CONSUMER CROSS-TAB (def C){' — ' + label if label else ''}")
    print(f"  (directly comparable to definition A's chi2/dof = {REF_CHESS_A})")
    print(f"{'=' * 72}")
    print(f"  {'d':>4} {'n':>8} " + " ".join(f"{c:>22}" for c in PROVENANCE))

    chi2 = 0.0
    for d in ds:
        n_d = near_totals[d]
        row = f"  {d:>4} {n_d:>8,} "
        for c in PROVENANCE:
            obs = near_cells[(d, c)]
            exp = n_d * marg[c]
            if exp > 0:
                chi2 += (obs - exp) ** 2 / exp
            if obs < MIN_CELL:
                row += f"  {'thin':>21}"
            else:
                row += f" {obs / n_d:>21.1%}"
        print(row)

    print(f"  {'ALL':>4} {tot:>8,} " + " ".join(f"{marg[c]:>21.1%}" for c in PROVENANCE))
    dof = max((len(ds) - 1) * (len(PROVENANCE) - 1), 1)
    per_dof = chi2 / dof
    print(f"\n  chi2 {chi2:.1f} on {dof} dof   per-dof {per_dof:.1f}")
    print(f"  Reference: tree-gen A {REF_CHESS_A}, arith tree {REF_ORIG}, dec-synth {REF_DEC}")
    return per_dof


def print_k_distribution(k_dist: Counter, total_producers: int) -> None:
    print(f"\n{'=' * 72}")
    print("MULTI-CONSUMER COVERAGE (definition C)")
    print(f"{'=' * 72}")
    total_with_any = sum(v for k, v in k_dist.items() if k > 0)
    print(f"  {'k':>5}  {'count':>10}  {'fraction':>10}")
    for k in sorted(k_dist):
        pct = k_dist[k] / total_producers
        print(f"  {k:>5}  {k_dist[k]:>10,}  {pct:>10.2%}")
    print(f"\n  plies with >=1 consumer:  {total_with_any:,}  "
          f"({total_with_any / total_producers:.1%})")
    multi = sum(v for k, v in k_dist.items() if k >= 2)
    print(f"  plies with >=2 consumers: {multi:,}  "
          f"({multi / total_producers:.1%})")
    print(f"  total producers:          {total_producers:,}")


def print_near_far_joint(near_far_dist: Counter) -> None:
    print(f"\n{'=' * 72}")
    print("JOINT (d_nearest, d_farthest) DISTRIBUTION — plies with k>=2 consumers")
    print(f"{'=' * 72}")
    total = sum(near_far_dist.values())
    if total == 0:
        print("  (no plies with k>=2 consumers)")
        return
    d_bins = ["d<=2", "d3-5", "d6-10", "d11+"]
    # Header
    header_lbl = "near \\ far"
    print(f"  {header_lbl:<12}" + "".join(f"  {b:>10}" for b in d_bins) + f"  {'row_tot':>10}")
    for nb in d_bins:
        row_tot = sum(near_far_dist[(nb, fb)] for fb in d_bins)
        if row_tot == 0:
            continue
        row = f"  {nb:<12}"
        for fb in d_bins:
            n = near_far_dist[(nb, fb)]
            row += f"  {n / total:>10.1%}" if n >= MIN_CELL else f"  {'thin':>10}"
        row += f"  {row_tot / total:>10.1%}"
        print(row)
    print(f"  total k>=2 plies: {total:,}")


def print_restricted_cell(restricted: Counter, restricted_total: int,
                          total_producers: int) -> None:
    print(f"\n{'=' * 72}")
    print("RESTRICTED CELL (§11 within-record design)")
    print("  Exactly one consumer at d<=2 AND exactly one consumer at d>=5.")
    print(f"{'=' * 72}")
    print(f"  Overall restricted plies: {restricted_total:,}  "
          f"({restricted_total / max(total_producers, 1):.2%} of all producers)")
    print()
    print(f"  {'provenance class':<22}  {'count':>8}  {'share':>8}")
    tot = max(restricted_total, 1)
    for cls in PROVENANCE:
        n = restricted[cls]
        print(f"  {cls:<22}  {n:>8,}  {n / tot:>8.1%}")
    print()
    if restricted_total >= 500:
        print("  GATE: PASSES (>= 500). Sample size adequate for §11 arms.")
    else:
        print(f"  GATE: FAILS ({restricted_total} < 500). Insufficient for §11 arms.")


def print_near_far_prov(near_prov: Counter, far_prov: Counter) -> None:
    print(f"\n{'=' * 72}")
    print("PROVENANCE OF PRODUCERS: NEAR (d<=2) vs FAR (d>=5) CONSUMERS")
    print("  Check: are far-consumer producers skewed toward 'both-from-earlier'?")
    print(f"  (DAG had 93% BFE at d>=5 — the confound arm that isn't fixed by design)")
    print(f"{'=' * 72}")
    n_near = sum(near_prov.values())
    n_far  = sum(far_prov.values())
    print(f"  {'class':<22}  {'near frac':>10}  {'far frac':>10}")
    for cls in PROVENANCE:
        nn = near_prov[cls]
        nf = far_prov[cls]
        pn = f"{nn / n_near:>10.1%}" if n_near else f"{'—':>10}"
        pf = f"{nf / n_far:>10.1%}"  if n_far  else f"{'—':>10}"
        print(f"  {cls:<22}  {pn}  {pf}")
    print(f"\n  n_near: {n_near:,}   n_far: {n_far:,}")


def print_piece_type_table(label: str, pt_by_class: dict[str, Counter]) -> None:
    print(f"\n{'=' * 72}")
    print(f"PIECE-TYPE COMPOSITION PER PROVENANCE CLASS — {label}")
    pt_order = ["PAWN", "KNIGHT", "BISHOP", "ROOK", "QUEEN", "KING"]
    print(f"  {'class':<22}" + "".join(f"  {p:>7}" for p in pt_order))
    for cls in PROVENANCE:
        total = sum(pt_by_class[cls].values())
        if total == 0:
            print(f"  {cls:<22}  (no records)")
            continue
        row = f"  {cls:<22}"
        for p in pt_order:
            row += f"  {pt_by_class[cls][p] / total:>6.1%}"
        row += f"   n={total:,}"
        print(row)


def print_captured_by_class(cpt_by_class: dict[str, Counter]) -> None:
    print(f"\n{'=' * 72}")
    print("CAPTURED PIECE TYPE WITHIN EACH PROVENANCE CLASS")
    pt_order = ["PAWN", "KNIGHT", "BISHOP", "ROOK", "QUEEN"]
    print(f"  {'class':<22}" + "".join(f"  {p:>7}" for p in pt_order) + "      n")
    for cls in PROVENANCE:
        total = sum(cpt_by_class[cls].values())
        if total < MIN_CELL:
            print(f"  {cls:<22}  thin (n={total} < {MIN_CELL}), not printed")
            continue
        row = f"  {cls:<22}"
        for p in pt_order:
            row += f"  {cpt_by_class[cls][p] / total:>6.1%}"
        row += f"  {total:>7,}"
        print(row)


def print_decision(per_dof: float, restricted_total: int) -> None:
    print(f"\n{'=' * 72}")
    print("DECISION GATE")
    print(f"{'=' * 72}")
    print(f"  chi2/dof (nearest consumer, def C): {per_dof:.1f}")
    print(f"  restricted cell:                    {restricted_total:,}")
    print()
    if per_dof < 50 and restricted_total >= 500:
        verdict = ("PROCEED. chi2/dof < 50 AND restricted cell >= 500. "
                   "Build tokeniser, VAE, denoiser; run §11 within-record arms.")
    elif per_dof < 200:
        verdict = ("INSPECT. chi2/dof in 50–200. Check whether coupling "
                   "concentrates in opening plies or a single piece type. "
                   "Within-record design may still be viable — "
                   "inspect the conditional distributions above.")
    else:
        verdict = ("WELD SURVIVES. chi2/dof > 200: definition C does not break "
                   "the coupling. Registered prediction fails. "
                   "Record chess as the fourth null and close this branch.")
    print(f"  {verdict}")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--pgn", type=Path, help="Decompressed Lichess PGN file")
    g.add_argument("--random", action="store_true",
                   help="Random games (UNIT-TEST ONLY — no dependency structure)")
    p.add_argument("--traces", type=int, default=5000)
    p.add_argument("--no-filter", action="store_true",
                   help="Skip time-control filtering")
    p.add_argument("--min-plies", type=int, default=10)
    args = p.parse_args()

    if args.random:
        print("Loading random games (PARSER TEST ONLY)...")
        traces = generate_random_dataset(args.traces, seed=0)
        traces = [t for t in traces if len(t.moves) >= args.min_plies]
        print("  WARNING: random games have no dependency structure.")
    else:
        print(f"Loading {args.pgn} …")
        traces, stats = load_pgn_with_stats(
            args.pgn, n=args.traces, filter_tc=not args.no_filter)
        traces = [t for t in traces if len(t.moves) >= args.min_plies]
        print(f"  {len(traces):,} games after min-plies filter")
        print_load_stats(stats)

    if not traces:
        print("No traces loaded — check path and filter settings.", file=sys.stderr)
        return

    print(f"\nBuilding definition-C consumer maps for {len(traces):,} games …")
    data = analyse(traces)

    per_dof = print_chi2_crosstab(data["near_cells"], data["near_totals"])
    print_k_distribution(data["k_dist"], data["total_producers"])
    print_near_far_joint(data["near_far_dist"])
    print_restricted_cell(data["restricted"], data["restricted_total"],
                          data["total_producers"])
    print_near_far_prov(data["near_prov_dist"], data["far_prov_dist"])
    print_piece_type_table("MOVING PIECE", data["pt_moving_by_class"])
    print_captured_by_class(data["cpt_by_class"])
    print_decision(per_dof, data["restricted_total"])


if __name__ == "__main__":
    main()
