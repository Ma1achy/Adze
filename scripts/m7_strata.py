"""M7 — every stratification, from the records dumps. No GPU.

A pooled gap says an effect exists somewhere. These cuts ask whether it is the
effect the design predicts. `scripts/m7_central.py --out` writes the records; all
re-cutting happens here, so asking a new question of the data costs nothing.

What is printed, and why each one is here:

  THE NULL          root corruption — nothing consumes the root, so there is no
                    pin and the gap should be zero. It is NOT zero (-5.6pp), and
                    `scripts/m7_shield.py` establishes why: a condition-level
                    handicap from the 90/10 training mix, not a harness fault.
                    So this table sets the ZERO POINT for every other one.

  THE REDIRECTED    consistent corruption — the pin is moved, not removed. Scored
  PIN               against BOTH targets. Global should lose against the clean
                    step and win against the corrupted one. An effect that tracks
                    the mechanism when the mechanism moves is a stronger claim
                    than one that merely vanishes without it.

  PROVENANCE        the sharp cell is both-from-earlier under consistent
                    corruption: the prefix is untouched, so causal can recompute
                    the CLEAN step exactly while global is pinned to the CORRUPTED
                    one. The arms are aimed at different targets and should
                    disagree in direction.

  DISTANCE          more chain between the corrupted block and its consumer means
                    more to traverse; the gap should decay.

  BLOCK POSITION    earlier blocks have more downstream evidence.

  PAD-FREE SUBSET   when a trace is shorter than B the erased block attends to PAD
                    blocks under GLOBAL and not under CAUSAL. Pads carry a constant
                    latent so they should carry no information, but they change the
                    softmax normalisation. Every table therefore also reports
                    n_steps == B, which is pad-free by construction.

Every rate is printed against its PERMUTATION CHANCE RATE. `none` is not the
baseline: it measures the VAE round-trip, since the VAE was trained on valid
arithmetic only and projects a corrupted step onto the nearest valid one.

Usage:
    python scripts/m7_strata.py runs/*.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from adze.eval.strata import cell, print_cells, stratify


def load(path: Path) -> dict:
    blob = json.loads(path.read_text())
    blob["path"] = path
    return blob


def header(blob: dict) -> None:
    tag = f"  [TARGETED: {blob['targeted']}]" if blob.get("targeted") else ""
    print("\n" + "=" * 96)
    print(f"{blob['path'].name}   --corrupt {blob['corrupt']}{tag}")
    print(f"  {len(blob['records'])} traces, nfe {blob['nfe']}, eta {blob['eta']}")
    print("=" * 96)


def overall(records: list[dict], field: str, target: str, label: str) -> None:
    """The pooled figure and its pad-free subset, side by side.

    Pooled and pad-free are reported as separate rows, never merged: a targeted or
    filtered subset's rate is not comparable to the unbiased one.
    """
    cells = [cell(label, records, field, target)]
    pad_free = [r for r in records if r["full_length"]]
    if pad_free and len(pad_free) != len(records):
        cells.append(cell(f"{label} pad-free", pad_free, field, target))
    print_cells("POOLED", cells, field, target)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dumps", nargs="+", type=Path)
    p.add_argument("--permutations", type=int, default=200)
    args = p.parse_args()

    blobs = [load(d) for d in args.dumps]

    for blob in blobs:
        header(blob)
        records = blob["records"]
        consistent = blob["corrupt"] == "consistent"

        # Under `consistent` the corrupted trace is a genuinely different target,
        # and both scorings are the result. Elsewhere it differs from clean only
        # at the target block by an arbitrary delta, so scoring against it would
        # measure nothing and is not printed.
        targets = ("clean", "corrupt") if consistent else ("clean",)

        for target in targets:
            for field in ("result", "operands"):
                overall(records, field, target, blob["corrupt"])

            print_cells(
                "BY BLOCK POSITION",
                stratify(records, lambda r: f"b{r['block']}", "result", target,
                         args.permutations),
                "result", target,
            )
            print_cells(
                "BY DISTANCE TO CONSUMER",
                stratify(records, lambda r: (f"d={r['distance']}"
                                             if r["distance"] is not None
                                             else "NO CONSUMER"),
                         "result", target, args.permutations),
                "result", target,
            )
            print_cells(
                "BY OPERAND PROVENANCE",
                stratify(records, lambda r: r["provenance"], "result", target,
                         args.permutations),
                "result", target,
            )

        if consistent:
            print("\n  THE DIRECTIONAL READ — same regenerated block, two targets.")
            print("  The mechanism predicts global LOSES against clean and WINS")
            print("  against corrupted. A gap that does not move when the pin moves")
            print("  is not downstream evidence.")
            for prov in ("both-from-earlier", "one-leaf", "both-leaves"):
                sub = [r for r in records if r["provenance"] == prov]
                if not sub:
                    continue
                vs_clean = cell(prov, sub, "result", "clean")
                vs_corr = cell(prov, sub, "result", "corrupt")
                print(f"    {prov:>18}  n={vs_clean.n:<5} "
                      f"vs clean {vs_clean.gap:>+7.1%}   "
                      f"vs corrupted {vs_corr.gap:>+7.1%}   "
                      f"swing {vs_corr.gap - vs_clean.gap:>+7.1%}")

    # --- the recalibration ---------------------------------------------------
    nulls = [b for b in blobs if b["corrupt"] == "final"]
    early = [b for b in blobs if b["corrupt"] == "early"]
    if nulls and early:
        print("\n" + "=" * 96)
        print("RECALIBRATED AGAINST THE NULL")
        print("=" * 96)
        print("  The null's gap is the zero point. It is NOT zero: the model is")
        print("  specialised to regime A's configuration, which it saw in 90% of")
        print("  training steps, so the global arm carries a handicap independent")
        print("  of any downstream evidence. See scripts/m7_shield.py.")
        print("  Additivity is an ASSUMPTION — the null is measured at the root,")
        print("  at a different block position and a different difficulty. Stated")
        print("  so the corrected figure is read as an estimate, not a measurement.")
        for n in nulls:
            for e in early:
                n_gap = cell("null", n["records"], "result", "clean").gap
                e_gap = cell("early", e["records"], "result", "clean").gap
                print(f"\n  {e['path'].name} vs {n['path'].name}")
                print(f"    measured gap {e_gap:>+7.1%}"
                      f"   handicap {n_gap:>+7.1%}"
                      f"   -> mechanism ~ {e_gap - n_gap:>+7.1%}")


if __name__ == "__main__":
    main()
