"""Chess structural diagnostic — before any model is trained.

Measures whether distance-to-consumer and operand provenance are independent in
real chess games, using the same methodology as `m9_decorrelation.py`.

## Key questions

1. Is chi2/dof below 20 (natural near-independence) or above 50 (same weld as
   the arithmetic tree generator)?

2. Does the provenance–distance coupling survive stratifying by piece type?
   Both-from-earlier correlates with high-mobility pieces (queens), which also
   have short consumer distances. If the coupling disappears within a single
   piece type, it is piece mobility, not dependency structure.

3. Are the captured piece types in both-from-earlier and one-leaf queen-heavy?
   Both-from-earlier requires TWO mobility draws: a moved piece captures a moved
   piece. If queens are capturing queens, the provenance axis is measuring mobility
   twice over, not dependency.

## Data source

Lichess monthly PGN dumps at https://database.lichess.org/. Download a monthly
file, decompress it with `zstd -d <file>.zst`, and pass the path. Strict
filtering to classical/rapid is applied by default.

Usage:
    python scripts/m9_chess_diag.py --pgn /path/to/games.pgn
    python scripts/m9_chess_diag.py --pgn /path/to/games.pgn --traces 5000
"""

from __future__ import annotations

import argparse
import io
from collections import Counter
from pathlib import Path

import chess
import chess.pgn

from adze.data.chess import (
    _is_classical_or_rapid,
    game_to_trace,
    generate_random_dataset,
)
from adze.eval.chess_strata import (
    chess_consumer_distance,
    chess_operand_provenance,
    PIECE_TYPE_NAMES,
)
from adze.eval.strata import PROVENANCE

# Reference points (session 23).
REF_ORIG = 374.8   # original arithmetic tree generator, per-dof chi2
REF_DEC  =   4.5   # decorrelated synthetic, per-dof chi2

MIN_CELL = 50   # fewer records → report "thin" rather than a rate


# ── loading with stats ────────────────────────────────────────────────────────

def load_pgn_with_stats(path: Path, n: int, filter_tc: bool):
    """Stream games from a PGN file, returning traces and a stats dict.

    Returns:
        traces: list[ChessTrace], len <= n
        stats:  dict with keys
            total_games, kept_games, tc_counts (Counter of TimeControl strings),
            tc_categories (Counter of reject/classical/rapid/blitz/bullet)
    """
    stats = {
        "total_games": 0,
        "kept_games": 0,
        "parse_errors": 0,
        "tc_counts": Counter(),
        "tc_categories": Counter(),
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

            # Categorise the time control.
            cat = _categorise_tc(tc)
            stats["tc_categories"][cat] += 1

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


def _categorise_tc(tc_str: str | None) -> str:
    """Rough Lichess time-control category."""
    if not tc_str or tc_str in ("-", "?"):
        return "unknown"
    try:
        if "+" in tc_str:
            base_s, inc_s = tc_str.split("+")
            base = int(base_s)
            inc = int(inc_s)
        else:
            base = int(tc_str)
            inc = 0
    except ValueError:
        return "unknown"
    total = base + 40 * inc  # rough total seconds
    if total < 30:
        return "ultra-bullet"
    if total < 180:
        return "bullet"
    if total < 480:
        return "blitz"
    if total < 1500:
        return "rapid"
    return "classical"


# ── analysis ─────────────────────────────────────────────────────────────────

def analyse(traces):
    """Compute all cross-tabs and distributions. Returns a dict of counters."""
    # cells[(d, prov)] — count
    cells: Counter = Counter()
    totals: Counter = Counter()   # d → total plies
    d_hist: Counter = Counter()   # d → total (no distance cap)

    # provenance class x distance histogram (for conditional distribution)
    d_by_class: dict[str, Counter] = {c: Counter() for c in PROVENANCE}

    # piece-type composition of each provenance class (MOVING piece)
    pt_moving_by_class: dict[str, Counter] = {c: Counter() for c in PROVENANCE}

    # captured piece type within each provenance class (for captures only)
    cpt_by_class: dict[str, Counter] = {c: Counter() for c in PROVENANCE}

    for tr in traces:
        for k in range(len(tr.moves)):
            mv = tr.moves[k]
            d = chess_consumer_distance(tr, k)
            prov = chess_operand_provenance(mv)
            pt_name = PIECE_TYPE_NAMES.get(mv.piece_type, "?")

            # Moving piece type always tracked.
            pt_moving_by_class[prov][pt_name] += 1

            # Captured piece type — only for captures.
            if mv.is_capture and mv.captured_piece_type is not None:
                cpt_name = PIECE_TYPE_NAMES.get(mv.captured_piece_type, "?")
                cpt_by_class[prov][cpt_name] += 1

            if d is not None:
                d_hist[d] += 1
                d_by_class[prov][d] += 1
                cells[(d, prov)] += 1
                totals[d] += 1

    return {
        "cells": cells,
        "totals": totals,
        "d_hist": d_hist,
        "d_by_class": d_by_class,
        "pt_moving_by_class": pt_moving_by_class,
        "cpt_by_class": cpt_by_class,
    }


# ── printing helpers ──────────────────────────────────────────────────────────

def print_load_stats(stats: dict) -> None:
    print(f"\n{'=' * 72}")
    print("LOADING STATS")
    print(f"{'=' * 72}")
    print(f"  total games in dump (scanned):  {stats['total_games']:>8,}")
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
    print()
    print("  Top-10 time controls (raw):")
    for tc, n in stats["tc_counts"].most_common(10):
        print(f"    {tc:<15} {n:>8,}")


def print_distance_histogram(d_hist: Counter) -> None:
    total = sum(d_hist.values())
    print(f"\n{'=' * 72}")
    print("CONSUMER-DISTANCE DISTRIBUTION (marginal, all plies with a consumer)")
    print(f"{'=' * 72}")
    buckets = [
        ("d=1",     lambda d: d == 1),
        ("d=2",     lambda d: d == 2),
        ("d=3",     lambda d: d == 3),
        ("d=4",     lambda d: d == 4),
        ("d=5",     lambda d: d == 5),
        ("d=6..10", lambda d: 6 <= d <= 10),
        ("d=11..20", lambda d: 11 <= d <= 20),
        ("d=21..40", lambda d: 21 <= d <= 40),
        ("d=41+",   lambda d: d > 40),
    ]
    cum = 0
    for name, pred in buckets:
        n = sum(v for k, v in d_hist.items() if pred(k))
        cum += n
        pct = n / total if total else 0
        print(f"  {name:>10}  {n:>8,}  {pct:>6.1%}   cumul {cum / total:>6.1%}")

    target = int(0.90 * total)
    cum90 = 0
    for d in sorted(d_hist):
        cum90 += d_hist[d]
        if cum90 >= target:
            print(f"\n  90th percentile: d = {d}")
            break
    print(f"  total plies with a consumer: {total:,}")


def print_conditional_dist(d_by_class: dict[str, Counter]) -> None:
    print(f"\n{'=' * 72}")
    print("CONSUMER-DISTANCE DISTRIBUTION CONDITIONED ON PROVENANCE CLASS")
    print("  Key diagnostic: if 'both-from-earlier' has substantially shorter")
    print("  distances than 'both-leaves', check whether piece-type explains it.")
    print(f"{'=' * 72}")
    for cls in PROVENANCE:
        hist = d_by_class[cls]
        total = sum(hist.values())
        if total == 0:
            print(f"  {cls}: no records")
            continue
        cum = 0
        p90_d = None
        parts = []
        for d in [1, 2, 3, 4, 5]:
            n = hist[d]
            parts.append(f"d={d}:{n / total:>5.1%}")
        rest = sum(v for k, v in hist.items() if k > 5)
        parts.append(f"d>5:{rest / total:>5.1%}")
        for d in sorted(hist):
            cum += hist[d]
            if cum >= 0.9 * total and p90_d is None:
                p90_d = d
        print(f"  {cls:<22}  n={total:>7,}  " + "  ".join(parts)
              + f"  p90=d{p90_d}")


def print_piece_type_table(label: str, pt_by_class: dict[str, Counter]) -> None:
    print(f"\n{'=' * 72}")
    print(f"PIECE-TYPE COMPOSITION PER PROVENANCE CLASS — {label}")
    pt_order = ["PAWN", "KNIGHT", "BISHOP", "ROOK", "QUEEN", "KING"]
    header = f"  {'class':<22}" + "".join(f"  {p:>7}" for p in pt_order)
    print(header)
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
    print("  (both-from-earlier requires TWO mobility draws: a moved piece")
    print("   captures a moved piece. Queen-captures-queen = mobility twice over.)")
    pt_order = ["PAWN", "KNIGHT", "BISHOP", "ROOK", "QUEEN"]
    header = f"  {'class':<22}" + "".join(f"  {p:>7}" for p in pt_order) + "      n"
    print(header)
    for cls in ("one-leaf", "both-from-earlier"):
        total = sum(cpt_by_class[cls].values())
        if total < MIN_CELL:
            print(f"  {cls:<22}  thin (n={total} < {MIN_CELL}), not printed")
            continue
        row = f"  {cls:<22}"
        for p in pt_order:
            row += f"  {cpt_by_class[cls][p] / total:>6.1%}"
        row += f"  {total:>7,}"
        print(row)
    # both-leaves: captures exist (e.g. first move is rare capture) but mostly thin.
    cls = "both-leaves"
    total = sum(cpt_by_class[cls].values())
    if total < MIN_CELL:
        print(f"  {cls:<22}  thin (n={total} < {MIN_CELL}), not printed")
    else:
        row = f"  {cls:<22}"
        for p in pt_order:
            row += f"  {cpt_by_class[cls][p] / total:>6.1%}"
        row += f"  {total:>7,}"
        print(row)


def print_crosstab_chi2(cells, totals, label: str = "",
                        min_cell: int = MIN_CELL) -> float:
    """Print cross-tab and chi2. Returns per-dof chi2."""
    ds = sorted(d for d in totals if totals[d] >= min_cell)
    if not ds:
        print(f"  (no distance bin with >= {min_cell} records)")
        return 0.0

    tot = sum(totals[d] for d in ds)
    marg = {c: sum(cells[(d, c)] for d in ds) / tot for c in PROVENANCE}

    hdr = f"\n{'=' * 72}"
    print(hdr)
    print(f"DISTANCE × PROVENANCE CROSS-TAB{' — ' + label if label else ''}")
    print(f"{'=' * 72}")
    print(f"  {'d':>4} {'n':>8} " + " ".join(f"{c:>22}" for c in PROVENANCE))

    chi2 = 0.0
    for d in ds:
        n_d = totals[d]
        row = f"  {d:>4} {n_d:>8,} "
        for c in PROVENANCE:
            obs = cells[(d, c)]
            exp = n_d * marg[c]
            if exp > 0:
                chi2 += (obs - exp) ** 2 / exp
            pct = obs / n_d
            cell_n = cells[(d, c)]
            if cell_n < min_cell:
                row += f"  {'thin':>21}"
            else:
                row += f" {pct:>21.1%}"
        print(row)

    print(f"  {'ALL':>4} {tot:>8,} " + " ".join(f"{marg[c]:>21.1%}" for c in PROVENANCE))

    dof = max((len(ds) - 1) * (len(PROVENANCE) - 1), 1)
    per_dof = chi2 / dof
    print(f"\n  chi2 {chi2:.1f} on {dof} dof   per-dof {per_dof:.1f}")
    print(f"  Reference: arithmetic tree {REF_ORIG}, decorrelated synthetic {REF_DEC}")
    return per_dof


def print_decision(per_dof: float) -> None:
    print(f"\n{'=' * 72}")
    print("DECISION")
    print(f"{'=' * 72}")
    if per_dof < 20:
        verdict = ("PROCEED to tokeniser and VAE. Chi2/dof below 20: "
                   "distance and provenance are near-independent in real games.")
    elif per_dof <= 50:
        verdict = ("INSPECT. Chi2/dof in 20–50 range. Check whether the coupling "
                   "concentrates in the first 10 plies (opening). If so, an "
                   "interior-band cut (exclude first 10 plies) may resolve it — "
                   "decide from the conditional distributions above.")
    else:
        verdict = ("INVESTIGATE. Chi2/dof above 50: chess has the same "
                   "provenance–distance weld as the arithmetic tree generator. "
                   "Report this finding; the domain does not rescue the reach question.")
    print(f"  {verdict}")


# ── per-piece-type cross-tabs ─────────────────────────────────────────────────

def per_piece_type_chi2(traces, piece_types: list[str]) -> None:
    pt_name_to_int = {v: k for k, v in PIECE_TYPE_NAMES.items()}
    for pt_name in piece_types:
        pt_int = pt_name_to_int.get(pt_name.upper())
        if pt_int is None:
            continue
        cells2: Counter = Counter()
        totals2: Counter = Counter()
        for tr in traces:
            for k in range(len(tr.moves)):
                mv = tr.moves[k]
                if mv.piece_type != pt_int:
                    continue
                d = chess_consumer_distance(tr, k)
                prov = chess_operand_provenance(mv)
                if d is not None:
                    cells2[(d, prov)] += 1
                    totals2[d] += 1
        n_total = sum(totals2.values())
        if n_total < MIN_CELL:
            print(f"\n  {pt_name} only — thin ({n_total} records), not printed")
            continue
        print_crosstab_chi2(cells2, totals2, label=f"{pt_name} only")


# ── novelty ──────────────────────────────────────────────────────────────────

def print_novelty(traces) -> None:
    sample = traces[:1000] if len(traces) >= 1000 else traces
    unique = len(set(tuple(m.san for m in t.moves) for t in sample))
    print(f"\n{'=' * 72}")
    print("TRACE NOVELTY")
    print(f"{'=' * 72}")
    print(f"  {unique} unique SAN sequences in {len(sample)} games "
          f"({unique / len(sample):.1%} unique)")


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
    p.add_argument("--by-piece", nargs="+", default=["PAWN", "KNIGHT", "QUEEN"],
                   help="Piece types for per-type cross-tabs")
    args = p.parse_args()

    if args.random:
        print("Loading random games (PARSER TEST ONLY)...")
        traces = generate_random_dataset(args.traces, seed=0)
        traces = [t for t in traces if len(t.moves) >= args.min_plies]
        stats = None
        print("  WARNING: random games have no dependency structure.")
    else:
        print(f"Loading {args.pgn} …")
        traces, stats = load_pgn_with_stats(
            args.pgn, n=args.traces, filter_tc=not args.no_filter)
        traces = [t for t in traces if len(t.moves) >= args.min_plies]
        print(f"  {len(traces):,} games after min-plies filter")
        print_load_stats(stats)

    if not traces:
        print("No traces loaded — check path and filter settings.")
        return

    # ── compute all distributions ─────────────────────────────────────────────
    data = analyse(traces)

    # ── report ────────────────────────────────────────────────────────────────
    print_distance_histogram(data["d_hist"])
    print_conditional_dist(data["d_by_class"])
    print_piece_type_table("MOVING PIECE", data["pt_moving_by_class"])
    print_captured_by_class(data["cpt_by_class"])
    per_dof = print_crosstab_chi2(data["cells"], data["totals"])
    per_piece_type_chi2(traces, args.by_piece)
    print_novelty(traces)
    print_decision(per_dof)


if __name__ == "__main__":
    main()
