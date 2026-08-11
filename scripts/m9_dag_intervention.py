"""M9 — the pre-registered multi-consumer intervention (docs/scratchpad-reach.md §11).

One step, consumed twice, at two distances. The step is its own control: same
provenance, same corruption, same erasure — only the DISTANCE of the surviving
consumer evidence changes. That is a WITHIN-record intervention, which is the
design that has held in this project every time a between-record one collapsed.

Restricted to steps with EXACTLY ONE near consumer (d <= 2) and EXACTLY ONE far
consumer (d >= 5), so arms (b) and (c) each erase exactly one consumer block and
the |S| confound cannot arise. Arm (d) erases both and fixes the baseline.

    (a) both consumers visible            — reference
    (b) near erased, far visible          — far evidence only
    (c) far erased, near visible          — near evidence only
    (d) both erased                       — no consumer evidence

Primary comparison (b) vs (c) on RESULT. Robustness [(b)-(d)] vs [(c)-(d)].

The decision tree is printed BEFORE any number, in the order it was registered.
Nothing is snapped, filtered, retried or cleaned. A regenerated block that decodes
to garbage counts as garbage.

Usage:
    PYTHONPATH=src python scripts/m9_dag_intervention.py \\
        --config configs/dag10.yaml \\
        --vae checkpoints/vae_dag10_d16.pt \\
        --denoiser checkpoints/denoiser_dag10_d16_mixedP50.pt \\
        --records 543 --nfe 32 --eta 1.0 --seed 0 \\
        --out runs/dag10_intervention.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from adze.config import load_config
from adze.data.build import make_dataset
from adze.data.corrupt import CorruptedPair, corrupt_step
from adze.data.dataset import LatentCache, TraceDataset
from adze.data.tokeniser import CharTokeniser
from adze.eval.central import _decode, _parse, encode_traces, regenerate, score
from adze.eval.dag_strata import consumer_distances, exactly_one_near_one_far
from adze.eval.load import load_denoiser, load_vae
from adze.eval.strata import chance_rate, operand_provenance
from adze.invariants import MaskMode

CACHE_DIR = Path("data/cache")

ARMS = ("a", "b", "c", "d")
ARM_LABEL = {
    "a": "(a) both visible      — reference",
    "b": "(b) near erased       — FAR evidence only",
    "c": "(c) far erased        — NEAR evidence only",
    "d": "(d) both erased       — no consumer evidence",
}

DECISION_TREE = """\
==============================================================================
PRE-REGISTERED DECISION TREE — docs/scratchpad-reach.md §11
Stated before any number below. Do not reorder.
==============================================================================
  (1) ALL ARMS AT CHANCE
      -> the consumer-provenance confound is moot, the answer is clean,
         report the null and stop. This is the REGISTERED PREDICTION.

  (2) ANY ARM SEPARATES FROM CHANCE
      -> stratify by CONSUMER provenance before interpreting anything.
         The single unconfoundable cell is near=BFE AND far=BFE.

  (3) (b)-(d) >> (c)-(d) ON THE BFE x BFE CELL
      -> genuine distance effect. Report the cell count prominently; it is
         thin (~217 of 543) and needs a second seed before it is a claim.

  (4) (c)-(d) >> (b)-(d)
      -> near beats far. Report as a null for the distance claim, with a
         directional note.

  UNINFORMATIVE REGARDLESS:
      draft quality below 10%, or (d) ~ (b) ~ (c) all well above chance.

  Reference for every arm is CHANCE (permutation null, grouped by block),
  never `none` and never zero.
==============================================================================
"""


def _select(config, want: int, near_max: int, far_min: int, seed_offset: int):
    """Held-out traces, one qualifying step each, grown until the quota fills.

    ONE step per trace, chosen by a per-trace seeded draw among that trace's
    qualifiers. Two steps from the same trace would share a latent context and
    break the independence McNemar assumes. The draw is pre-registered selection,
    not filtering: no record is examined before it is chosen.

    Trace indices are stable as the pool grows — `generate_dataset` seeds each
    trace from `seed * 1_000_003 + i` — so a larger pool is a superset.
    """
    pool = max(want * 4, 1000)
    while True:
        traces = make_dataset(config, pool, config.data.seed + seed_offset)
        chosen = []
        for idx, trace in enumerate(traces):
            quals = [i for i in range(len(trace.steps))
                     if exactly_one_near_one_far(trace, i, near_max, far_min)]
            if not quals:
                continue
            i = random.Random(idx * 1_000_003 + 7).choice(quals)
            cs = consumer_distances(trace, i)
            assert len(cs) == 2, f"expected two consumers, got {cs}"
            chosen.append((trace, i, idx))
            if len(chosen) >= want:
                return chosen
        if pool >= 400_000:
            return chosen
        pool *= 4


def _rate(records: list[dict], arm: str, field: str = "result") -> float:
    want = [r["clean_" + field] for r in records]
    got = [r[arm][field] for r in records]
    hits = sum(g is not None and g == w for g, w in zip(got, want))
    return hits / max(len(records), 1)


def _hits(records: list[dict], arm: str, field: str = "result") -> list[bool]:
    return [r[arm][field] is not None and r[arm][field] == r["clean_" + field]
            for r in records]


def _mcnemar(records: list[dict], first: str, second: str,
             field: str = "result") -> tuple[int, int, float | None]:
    """Discordant pairs between two arms on the same records, and chi2."""
    ha, hb = _hits(records, first, field), _hits(records, second, field)
    only_b = sum(b and not a for a, b in zip(ha, hb))
    only_a = sum(a and not b for a, b in zip(ha, hb))
    disc = only_a + only_b
    chi = None if disc == 0 else (abs(only_b - only_a) - 1) ** 2 / disc
    return only_a, only_b, chi


def _verdict(chi: float | None) -> str:
    return "NOT significant" if chi is None or chi <= 3.84 else "p < 0.05"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/dag10.yaml"))
    p.add_argument("--vae", type=Path, default=Path("checkpoints/vae_dag10_d16.pt"))
    p.add_argument("--denoiser", type=Path,
                   default=Path("checkpoints/denoiser_dag10_d16_mixedP50.pt"))
    p.add_argument("--records", type=int, default=543,
                   help="registered restricted-cell size from the diagnostic")
    p.add_argument("--near-max", type=int, default=2)
    p.add_argument("--far-min", type=int, default=5)
    p.add_argument("--nfe", type=int, default=32)
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0,
                   help="erasure noise seed — SHARED by all four arms")
    p.add_argument("--permutations", type=int, default=200)
    p.add_argument("--thin", type=int, default=50,
                   help="cells at or below this are said, not printed")
    p.add_argument("--out", type=Path, default=Path("runs/dag10_intervention.json"))
    args = p.parse_args()

    config = load_config(args.config)
    device = torch.device(config.device)
    tokeniser = CharTokeniser()
    vae, _ = load_vae(args.vae, device)
    denoiser, arch, scale = load_denoiser(args.denoiser, device)
    blocks, k, d = arch["blocks"], arch["latents_per_block"], arch["latent_dim"]
    if scale is None:
        scale = LatentCache(CACHE_DIR / f"latents_{config.name}_d{d}.pt").scale

    chosen = _select(config, args.records, args.near_max, args.far_min, 909_091)
    if not chosen:
        raise SystemExit("no qualifying steps")

    # M7's corruption discipline: a random signed delta on the RESULT, nothing
    # downstream recomputed. Not an operator swap — swaps produce only
    # {a+b, a-b, a*b}, a distribution a model can exploit without reading evidence.
    pairs, meta = [], []
    for trace, i, idx in chosen:
        pairs.append(CorruptedPair(clean=trace,
                                   corrupted=corrupt_step(trace, i, rng_seed=idx),
                                   block_index=i))
        j_near, j_far = trace.consumer_map[i]
        meta.append({
            "trace_idx": idx,
            "block": i,
            "j_near": j_near,
            "j_far": j_far,
            "d_near": j_near - i,
            "d_far": j_far - i,
            "provenance": operand_provenance(trace.steps[i]),
            "near_provenance": operand_provenance(trace.steps[j_near]),
            "far_provenance": operand_provenance(trace.steps[j_far]),
        })

    ds = TraceDataset(pairs, blocks=blocks, latents_per_block=k, use_corrupted=True)
    items = [ds[n] for n in range(len(ds))]
    tokens = torch.stack([it["tokens"] for it in items]).to(device)
    block_mask = torch.stack([it["block_mask"] for it in items]).to(device)
    latents = encode_traces(vae, tokens, block_mask, scale)
    target_block = torch.stack([it["corrupted_idx"] for it in items]).to(device)
    block_ids = torch.repeat_interleave(torch.arange(blocks), k).to(device)

    n_steps = [len(pr.clean.steps) for pr in pairs]

    def _render(trace):
        return ([s.render() for s in trace.steps]
                + ["<pad>"] * (blocks - len(trace.steps)))

    clean_steps = [_render(pr.clean) for pr in pairs]
    corrupt_steps = [_render(pr.corrupted) for pr in pairs]

    # Erase-sets. The target block is erased in every arm; only which consumers
    # accompany it changes. (b) and (c) erase the same NUMBER of blocks.
    batch = len(pairs)
    erase = {a: torch.zeros(batch, blocks, dtype=torch.bool, device=device)
             for a in ARMS}
    for n, m in enumerate(meta):
        for a in ARMS:
            erase[a][n, m["block"]] = True
        erase["b"][n, m["j_near"]] = True
        erase["c"][n, m["j_far"]] = True
        erase["d"][n, m["j_near"]] = True
        erase["d"][n, m["j_far"]] = True

    print()
    print(f"denoiser      {args.denoiser}  {arch['n_layers']}L x {arch['d_model']}w")
    print(f"vae           {args.vae}")
    print(f"records       {len(pairs)} held-out, one qualifying step per trace")
    print(f"restriction   exactly one consumer at d <= {args.near_max} "
          f"AND exactly one at d >= {args.far_min}")
    print(f"corruption    random signed delta on the result (M7 corrupt_step)")
    print(f"sampling      global mask, nfe {args.nfe}, eta {args.eta}, "
          f"erasure seed {args.seed} (shared across arms)")
    print(f"selection     ORACLE — the corrupted index is known. Upper bound.")
    print()
    print(DECISION_TREE)

    results = {}
    for a in ARMS:
        out = regenerate(denoiser, latents, block_ids, target_block, blocks,
                         args.nfe, MaskMode.GLOBAL, eta=args.eta, seed=args.seed,
                         erase=erase[a])
        decoded = _decode(vae, tokeniser, out, scale, blocks, k)
        results[a] = score(a, decoded, clean_steps, target_block, n_steps)

    # --- records, built before anything is read off them ----------------------
    records = []
    for n, m in enumerate(meta):
        b = m["block"]
        rec = dict(m)
        rec["n_steps"] = n_steps[n]
        rec["clean_text"] = clean_steps[n][b]
        rec["corrupt_text"] = corrupt_steps[n][b]
        for target, text in (("clean", clean_steps[n][b]),
                             ("corrupt", corrupt_steps[n][b])):
            parsed = _parse(text)
            rec[f"{target}_result"] = None if parsed is None else str(parsed[3])
            rec[f"{target}_operands"] = (
                None if parsed is None else f"{parsed[0]} {parsed[1]} {parsed[2]}")
        for a in ARMS:
            got = results[a].texts[n][b]
            parsed = _parse(got)
            rec[a] = {
                "text": got,
                "exact": got,
                "result": None if parsed is None else str(parsed[3]),
                "operands": (None if parsed is None
                             else f"{parsed[0]} {parsed[1]} {parsed[2]}"),
            }
        records.append(rec)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "config": str(args.config),
            "denoiser": str(args.denoiser),
            "vae": str(args.vae),
            "blocks": blocks,
            "near_max": args.near_max,
            "far_min": args.far_min,
            "nfe": args.nfe,
            "eta": args.eta,
            "seed": args.seed,
            "records": records,
        }, indent=1))

    # --- arm table ------------------------------------------------------------
    n = len(records)
    print("=" * 78)
    print("THE FOUR ARMS — RESULT, against a permutation null")
    print("=" * 78)
    print(f"{'arm':<40}{'RESULT':>9}{'chance':>9}{'z':>8}{'formed':>9}")
    chance, zed = {}, {}
    for a in ARMS:
        mean, _sd = chance_rate(records, a, "result", "clean",
                                permutations=args.permutations, seed=0)
        rate = _rate(records, a)
        se = (mean * (1 - mean) / n) ** 0.5 if 0 < mean < 1 else 0.0
        z = (rate - mean) / se if se > 0 else float("nan")
        chance[a], zed[a] = mean, z
        print(f"{ARM_LABEL[a]:<40}{rate * 100:8.1f}%{mean * 100:8.1f}%"
              f"{z:8.2f}{results[a].well_formed * 100:8.1f}%")
    print()
    print(f"  n = {n}. z is against the permutation chance rate with a binomial SE:")
    print("  it is WITHIN-RUN eval noise on one seed, not a between-run bar.")
    print()

    # --- the primary comparison ----------------------------------------------
    only_c, only_b, chi = _mcnemar(records, "c", "b")
    rb, rc, rd = _rate(records, "b"), _rate(records, "c"), _rate(records, "d")
    print("=" * 78)
    print("PRIMARY — (b) far evidence only  vs  (c) near evidence only")
    print("=" * 78)
    print(f"  (b) - (c)  =  {(rb - rc) * 100:+.1f}%")
    print(f"  PAIRED (McNemar) on RESULT: b-only {only_b}, c-only {only_c},")
    print(f"    discordant {only_b + only_c}, chi2(1) cc = "
          f"{'n/a' if chi is None else f'{chi:.2f}'}")
    print(f"    {_verdict(chi)}")
    print()
    print("ROBUSTNESS — difference of differences against (d)")
    print(f"  (b) - (d)  =  {(rb - rd) * 100:+.1f}%")
    print(f"  (c) - (d)  =  {(rc - rd) * 100:+.1f}%")
    print(f"  DiD        =  {((rb - rd) - (rc - rd)) * 100:+.1f}%")
    print("  (d) subtracts out identically from both, so DiD equals (b)-(c) by")
    print("  construction. It fixes the BASELINE, not the treatment composition.")
    print()

    # --- branch selection, per the tree above ---------------------------------
    separated = [a for a in ARMS if abs(zed[a]) > 2.0]
    print("=" * 78)
    print("BRANCH")
    print("=" * 78)
    if not separated:
        print("  (1) No arm separates from chance (all |z| <= 2).")
        print("      The consumer-provenance confound is MOOT: there is no")
        print("      effect for it to explain. This is the registered prediction.")
        print("      Not stratifying — a cut on a null is a cut on noise.")
        print()
        print("  REGISTERED PREDICTION HELD.")
    else:
        print(f"  (2) Arms {', '.join(separated)} separate from chance.")
        print("      Stratifying by CONSUMER provenance before interpretation.")
        print("      The unconfoundable cell is near=BFE AND far=BFE.")
        print()
        cells: dict[tuple[str, str], list[dict]] = {}
        for r in records:
            cells.setdefault((r["near_provenance"], r["far_provenance"]), []).append(r)
        print(f"{'near x far consumer provenance':<44}{'n':>5}"
              + "".join(f"{a:>8}" for a in ARMS))
        thin = []
        for key in sorted(cells, key=lambda c: -len(cells[c])):
            group = cells[key]
            label = f"{key[0]} x {key[1]}"
            if len(group) <= args.thin:
                thin.append((label, len(group)))
                continue
            row = "".join(f"{_rate(group, a) * 100:7.1f}%" for a in ARMS)
            print(f"{label:<44}{len(group):>5}{row}")
        if thin:
            print()
            print("  Cells at or below n = "
                  f"{args.thin} are NOT printed — they cannot support a rate:")
            for label, count in thin:
                print(f"    {label}  (n = {count})")
        print()
        bfe = cells.get(("both-from-earlier", "both-from-earlier"), [])
        if len(bfe) <= args.thin:
            print(f"  The BFE x BFE cell holds n = {len(bfe)}. Too thin to read.")
        else:
            b_b, b_c, b_d = (_rate(bfe, a) for a in ("b", "c", "d"))
            oc, ob, ch = _mcnemar(bfe, "c", "b")
            print(f"  BFE x BFE — n = {len(bfe)}  THE UNCONFOUNDABLE CELL")
            print(f"    (b)-(d) {(b_b - b_d) * 100:+.1f}%   "
                  f"(c)-(d) {(b_c - b_d) * 100:+.1f}%   "
                  f"(b)-(c) {(b_b - b_c) * 100:+.1f}%")
            print(f"    McNemar b-only {ob}, c-only {oc}, chi2 "
                  f"{'n/a' if ch is None else f'{ch:.2f}'} — {_verdict(ch)}")
            print("    ONE SEED. A shape read off this cell is not a claim until")
            print("    a second seed reproduces it.")
    print()

    if args.out is not None:
        print(f"wrote {len(records)} records to {args.out}")

    # A few raw examples, so the numbers have something behind them.
    print()
    print("EXAMPLES — same record, same corruption, same noise; only which")
    print("consumer survives differs")
    print("-" * 78)
    shown = 0
    for nn, r in enumerate(records):
        if shown >= 6 or r["b"]["text"] == r["c"]["text"]:
            continue
        print(f"  block {r['block']}  d_near {r['d_near']}  d_far {r['d_far']}  "
              f"want {r['clean_text']!r}  (corrupt {r['corrupt_text']!r})")
        for a in ("b", "c"):
            mark = "OK " if r[a]["text"] == r["clean_text"] else "   "
            print(f"    {mark}{a}  {r[a]['text']!r}")
        shown += 1


if __name__ == "__main__":
    main()
