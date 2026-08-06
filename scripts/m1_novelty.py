"""Held-out novelty at the STEP level and the TRACE level, separately.

These are different questions and only one of them is a threat.

**Steps repeating across train and test is expected and fine.** The space of short
arithmetic statements is small — at a magnitude cap of 100 there are only ~30k
possible steps against ~800k training steps, so each is seen ~27 times. That is not
a defect. The central experiment does not test arithmetic ability; it tests whether
regenerating a corrupted block GLOBALLY beats regenerating it CAUSALLY, which is a
question about inferring a value from downstream dependency. Arithmetic difficulty
is a nuisance variable, and memorised arithmetic isolates the variable under test
rather than confounding it.

**Whole chains repeating is a different matter and would invalidate M7.** If a
held-out trace appears verbatim in training, then "repair the corrupted block"
is answerable by recall, and the experiment measures nothing. The chain STRUCTURE
— which operations feed which, in what order, over what values — is combinatorially
large regardless of operand range, and that is what has to stay novel.

So this reports both, labelled, and the trace figure is the one with a bar on it.

Usage:
    python scripts/m1_novelty.py --magnitude-cap 100
"""

from __future__ import annotations

import argparse

from adze.data.generate import MAGNITUDE_CAP, fixed_point_leaves, generate_dataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=60_000)
    p.add_argument("--n-held", type=int, default=8_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--operand-max", type=int, default=20)
    p.add_argument("--magnitude-cap", type=int, default=MAGNITUDE_CAP)
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--uniform", action="store_true",
                   help="original uniform leaf sampler, for comparison")
    args = p.parse_args()

    leaves = None if args.uniform else fixed_point_leaves(
        iterations=args.iterations, n=20_000, seed=args.seed,
        max_depth=args.max_depth, operand_max=args.operand_max,
        magnitude_cap=args.magnitude_cap,
    )
    kw = dict(max_depth=args.max_depth, operand_max=args.operand_max,
              leaf_values=leaves, magnitude_cap=args.magnitude_cap)

    train = generate_dataset(n=args.n_train, seed=args.seed, **kw)
    held = generate_dataset(n=args.n_held, seed=args.seed + 909_091, **kw)

    train_steps = {s.render() for t in train for s in t.steps}
    held_steps = [s.render() for t in held for s in t.steps]
    step_unseen = sum(1 for s in held_steps if s not in train_steps)

    # A trace's identity is its rendered chain — the steps AND their order. Two
    # traces with the same steps in a different order are different reasoning.
    train_traces = {t.render() for t in train}
    held_repeated = sum(1 for t in held if t.render() in train_traces)

    # Structure alone, with the values stripped: which op at which position, and
    # which operands come from which earlier step. If even this is exhausted, the
    # generator has no room left regardless of operand range.
    def shape(t) -> tuple:
        return tuple((s.op, s.lhs_from, s.rhs_from) for s in t.steps)

    train_shapes = {shape(t) for t in train}
    held_shapes = [shape(t) for t in held]

    print(f"cap {args.magnitude_cap}, operand_max {args.operand_max}, "
          f"leaves {'uniform' if args.uniform else 'empirical fixed point'}")
    print(f"train {len(train)} traces / {len(train_steps)} distinct steps")
    print()
    print("=" * 74)
    print("NOVELTY — steps vs traces")
    print("=" * 74)
    print(f"  STEP level   {step_unseen}/{len(held_steps)} held-out steps unseen "
          f"({step_unseen / len(held_steps):.1%})")
    print("               Repetition here is EXPECTED and fine — the step space is")
    print("               small by design, and memorised arithmetic isolates the")
    print("               variable the central experiment actually tests.")
    print()
    print(f"  TRACE level  {len(held) - held_repeated}/{len(held)} held-out traces "
          f"novel ({1 - held_repeated / len(held):.2%})")
    print(f"               {held_repeated} appear VERBATIM in training.")
    print("               This is the one with a bar on it: a repeated chain makes")
    print("               'repair the corrupted block' answerable by recall, and")
    print("               M7 would measure nothing.")
    print()
    print(f"  distinct chain structures in train: {len(train_shapes)}")
    print(f"  held-out structures also in train:  "
          f"{sum(1 for h in held_shapes if h in train_shapes)}/{len(held_shapes)}")
    print("               Structure repeating is fine on its own — the VALUES")
    print("               flowing through it are what make the trace novel.")
    print()
    verdict = "OK" if held_repeated == 0 else "CHECK"
    print(f"  {verdict} — trace-level duplication is "
          f"{held_repeated / len(held):.3%}")


if __name__ == "__main__":
    main()
