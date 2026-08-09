"""Chess structural diagnostic — before any model is trained.

Measures whether distance-to-consumer and operand provenance are independent in
real chess games, using the same methodology as `m9_decorrelation.py` for the
synthetic arithmetic generator.

## What this measures and what it does not

The PRIMARY question is whether the provenance–distance coupling seen in the
arithmetic tree generator (+374.8 χ²/dof) also appears in chess. The structural
test is χ²/dof on the distance × provenance cross-tab, measured on real games.

## Piece-type confound

Consumer distance in chess is confounded by piece mobility independently of
provenance. Pawns move infrequently → long consumer distances; queens move
frequently → short consumer distances. Since "both-from-earlier" correlates with
mobile pieces, a provenance–distance coupling may be driven by piece type rather
than dependency structure. This script reports piece-type composition per
provenance class and repeats the cross-tab stratified by piece type, so the
confound can be assessed before drawing conclusions.

## Data source

Lichess monthly PGN dumps at https://database.lichess.org/ (standard games).
Download a monthly file, decompress it (zstd -d <file>.zst), and pass the path.
The script filters to classical/rapid by default; see --no-filter to override.

## Composition-only swing

The arithmetic swing used CLASS_GAP values from trained models. Chess has no
trained models yet, so the swing is NOT reported. Report χ²/dof and the
conditional distributions instead — the swing will be computed once chess models
exist.

Usage:
    python scripts/m9_chess_diag.py --pgn /path/to/games.pgn
    python scripts/m9_chess_diag.py --pgn /path/to/games.pgn --traces 5000 --min-plies 20
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import chess

from adze.data.chess import load_pgn, generate_random_dataset
from adze.eval.chess_strata import (
    chess_consumer_distance,
    chess_operand_provenance,
    PIECE_TYPE_NAMES,
)
from adze.eval.strata import PROVENANCE

# Reference points from the arithmetic generator (session 23).
REF_ORIG = 374.8   # original tree generator, per-dof chi2
REF_DEC  =   4.5   # decorrelated synthetic generator, per-dof chi2


def crosstab(traces, min_d: int = 1, max_d: int | None = None):
    """Build distance × provenance counts for all plies with a measured consumer."""
    cells: Counter = Counter()    # (d, provenance) → count
    totals: Counter = Counter()   # d → count
    d_hist: Counter = Counter()   # d → count (marginal, no max_d filter)
    # per-class piece-type composition
    pt_by_class: dict[str, Counter] = {c: Counter() for c in PROVENANCE}

    for tr in traces:
        for k in range(len(tr.moves)):
            mv = tr.moves[k]
            d = chess_consumer_distance(tr, k)
            prov = chess_operand_provenance(mv)
            pt_name = PIECE_TYPE_NAMES.get(mv.piece_type, "?")

            if d is not None:
                d_hist[d] += 1
                pt_by_class[prov][pt_name] += 1
                if d >= min_d and (max_d is None or d <= max_d):
                    cells[(d, prov)] += 1
                    totals[d] += 1

    return cells, totals, d_hist, pt_by_class


def print_distance_histogram(d_hist: Counter, label: str = "") -> None:
    total = sum(d_hist.values())
    print(f"\n{'=' * 72}")
    if label:
        print(f"CONSUMER-DISTANCE DISTRIBUTION — {label}")
    else:
        print("CONSUMER-DISTANCE DISTRIBUTION (marginal, all plies with a consumer)")
    print(f"{'=' * 72}")
    buckets = [
        ("d=1",     [1]),
        ("d=2",     [2]),
        ("d=3",     [3]),
        ("d=4",     [4]),
        ("d=5",     [5]),
        ("d=6..10", list(range(6, 11))),
        ("d=11..20", list(range(11, 21))),
        ("d=21..40", list(range(21, 41))),
        ("d=41+",   None),
    ]
    cum = 0
    for name, ds in buckets:
        if ds is None:
            n = sum(v for k, v in d_hist.items() if k > 40)
        else:
            n = sum(d_hist[d] for d in ds)
        cum += n
        pct = n / total if total else 0
        print(f"  {name:>10}  {n:>8}  {pct:>6.1%}   cumul {cum / total:>6.1%}")
    # 90th percentile
    target = int(0.90 * total)
    cum90 = 0
    for d in sorted(d_hist):
        cum90 += d_hist[d]
        if cum90 >= target:
            print(f"\n  90th percentile: d = {d}")
            break
    print(f"  total plies with a consumer: {total:,}")


def print_conditional_dist(d_hist_by_class: dict[str, Counter]) -> None:
    print(f"\n{'=' * 72}")
    print("CONSUMER-DISTANCE DISTRIBUTION CONDITIONED ON PROVENANCE CLASS")
    print(f"{'=' * 72}")
    print("  This is the diagnostic that matters. If 'both-from-earlier' has")
    print("  substantially shorter consumer distances than 'both-leaves', chess")
    print("  has the same provenance–distance weld as the arithmetic tree")
    print("  generator (but possibly driven by piece mobility — see below).")
    print()
    for cls in PROVENANCE:
        hist = d_hist_by_class[cls]
        total = sum(hist.values())
        if total == 0:
            print(f"  {cls}: no records")
            continue
        cum = 0
        p90_d = None
        buckets_str = []
        for d in [1, 2, 3, 4, 5]:
            n = hist[d]
            buckets_str.append(f"d={d}:{n / total:>5.1%}")
        rest = sum(v for k, v in hist.items() if k > 5)
        buckets_str.append(f"d>5:{rest / total:>5.1%}")
        for d in sorted(hist):
            cum += hist[d]
            if cum >= 0.9 * total and p90_d is None:
                p90_d = d
        print(f"  {cls:<22}  n={total:>6}  " + "  ".join(buckets_str)
              + f"  p90=d{p90_d}")


def print_piece_type_by_class(pt_by_class: dict[str, Counter]) -> None:
    print(f"\n{'=' * 72}")
    print("PIECE-TYPE COMPOSITION PER PROVENANCE CLASS")
    print("  (If 'both-from-earlier' is queen-heavy and 'both-leaves' is pawn-heavy,")
    print("   the coupling is piece mobility rather than dependency structure.)")
    print(f"{'=' * 72}")
    pt_order = ["PAWN", "KNIGHT", "BISHOP", "ROOK", "QUEEN", "KING"]
    header = f"  {'class':<22}" + "".join(f"  {p:>7}" for p in pt_order)
    print(header)
    for cls in PROVENANCE:
        total = sum(pt_by_class[cls].values())
        if total == 0:
            continue
        row = f"  {cls:<22}"
        for p in pt_order:
            row += f"  {pt_by_class[cls][p] / total:>6.1%}"
        row += f"   n={total:,}"
        print(row)


def print_crosstab_and_chi2(cells, totals, min_cell: int = 50,
                             label: str = "") -> float:
    ds = sorted(d for d in totals if totals[d] >= min_cell)
    if not ds:
        print("  (no distance bin with enough records)")
        return 0.0

    tot = sum(totals[d] for d in ds)
    marg = {c: sum(cells[(d, c)] for d in ds) / tot for c in PROVENANCE}

    hdr = f"\n{'=' * 72}"
    print(hdr)
    if label:
        print(f"DISTANCE × PROVENANCE CROSS-TAB   {label}")
    else:
        print("DISTANCE × PROVENANCE CROSS-TAB")
    print(f"{'=' * 72}")
    print(f"  {'d':>4} {'n':>8} " + " ".join(f"{c:>22}" for c in PROVENANCE))

    chi2 = 0.0
    for d in ds:
        row = f"  {d:>4} {totals[d]:>8} "
        for c in PROVENANCE:
            obs = cells[(d, c)]
            exp = totals[d] * marg[c]
            if exp > 0:
                chi2 += (obs - exp) ** 2 / exp
            row += f" {obs / totals[d]:>21.1%}"
        print(row)

    print(f"  {'ALL':>4} {tot:>8} " + " ".join(f"{marg[c]:>21.1%}" for c in PROVENANCE))

    dof = max((len(ds) - 1) * (len(PROVENANCE) - 1), 1)
    per_dof = chi2 / dof
    print(f"\n  chi2 {chi2:.1f} on {dof} dof   per-dof {per_dof:.1f}")
    print(f"  Reference: original arithmetic generator {REF_ORIG:.1f}, "
          f"decorrelated {REF_DEC:.1f}")

    if per_dof < 20:
        verdict = "PROCEED — below 20, near-independent"
    elif per_dof <= 50:
        verdict = "INSPECT — 20–50, coupling present; check whether concentrated in early plies"
    else:
        verdict = "INVESTIGATE — above 50, strongly coupled; consider redesign"

    print(f"\n  VERDICT: {verdict}")
    return per_dof


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--pgn", type=Path,
                   help="Path to a decompressed Lichess PGN file")
    g.add_argument("--random", action="store_true",
                   help="Use random games (UNIT-TEST ONLY — no real dependency structure)")
    p.add_argument("--traces", type=int, default=5000,
                   help="Maximum number of games to load")
    p.add_argument("--no-filter", action="store_true",
                   help="Skip time-control filter (include blitz/bullet)")
    p.add_argument("--min-plies", type=int, default=10,
                   help="Discard games shorter than this many plies")
    p.add_argument("--by-piece", type=str, nargs="+",
                   default=["PAWN", "QUEEN"],
                   help="Piece types to show a stratified cross-tab for")
    args = p.parse_args()

    print(f"Loading {'random games' if args.random else args.pgn}...")
    if args.random:
        traces = generate_random_dataset(args.traces, seed=0)
        print("  WARNING: random games have no dependency structure.")
        print("  Consumer-distance distribution is NOT representative of real chess.")
    else:
        filter_tc = not args.no_filter
        traces = load_pgn(args.pgn, n=args.traces,
                          filter_time_control=filter_tc)

    traces = [t for t in traces if len(t.moves) >= args.min_plies]
    print(f"  {len(traces):,} games loaded (min-plies ≥ {args.min_plies})")
    total_plies = sum(len(t.moves) for t in traces)
    print(f"  {total_plies:,} total plies")

    # ── consumer-distance histogram (marginal) ────────────────────────────────
    cells, totals, d_hist, pt_by_class = crosstab(traces)

    print_distance_histogram(d_hist)

    # ── conditional consumer-distance by provenance class ────────────────────
    # Rebuild per-class histograms from cells+totals.
    d_hist_by_class: dict[str, Counter] = {c: Counter() for c in PROVENANCE}
    for (d, c), n in cells.items():
        d_hist_by_class[c][d] += n

    print_conditional_dist(d_hist_by_class)

    # ── piece-type composition per class ─────────────────────────────────────
    print_piece_type_by_class(pt_by_class)

    # ── full cross-tab and chi2 ───────────────────────────────────────────────
    per_dof = print_crosstab_and_chi2(cells, totals)

    # ── stratified by piece type ──────────────────────────────────────────────
    for pt_name in args.by_piece:
        pt_int = next((k for k, v in PIECE_TYPE_NAMES.items() if v == pt_name.upper()), None)
        if pt_int is None:
            print(f"\n  Unknown piece type: {pt_name!r}, skipping.")
            continue
        pt_traces_cells: Counter = Counter()
        pt_traces_totals: Counter = Counter()
        for tr in traces:
            for k in range(len(tr.moves)):
                mv = tr.moves[k]
                if mv.piece_type != pt_int:
                    continue
                d = chess_consumer_distance(tr, k)
                prov = chess_operand_provenance(mv)
                if d is not None:
                    pt_traces_cells[(d, prov)] += 1
                    pt_traces_totals[d] += 1
        print_crosstab_and_chi2(pt_traces_cells, pt_traces_totals,
                                 label=f"{pt_name} only")

    # ── novelty ──────────────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("TRACE NOVELTY")
    print(f"{'=' * 72}")
    sample = traces[:1000] if len(traces) >= 1000 else traces
    unique = len(set(tuple(m.san for m in t.moves) for t in sample))
    print(f"  {unique} unique SAN sequences in {len(sample)} games "
          f"({unique / len(sample):.1%} unique)")


if __name__ == "__main__":
    main()
