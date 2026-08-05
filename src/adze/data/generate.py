"""M1 — synthetic arithmetic trace generation.

Pure Python. No torch, no GPU. This is the only component with zero dependencies
and zero uncertainty, which is why it is first.

The generator must produce traces where every intermediate value is recorded
structurally, not just rendered into text. The central experiment depends on
knowing exactly which block holds which value.

Provenance is load-bearing. `Step.lhs_from` / `Step.rhs_from` record which earlier
step produced each operand, and `Trace.is_valid()` checks the chain against them.
That is what makes corruption detectable: change step 0's result and leave step 1's
operand at the original value, and the contradiction is visible from provenance
alone.

Everything is decided once, at construction, and never revisited — no operator is
swapped after its result is computed, no operands are reordered, no result or
provenance index is patched. `is_valid()` on a freshly generated trace is therefore
a genuine check on the builder rather than a formality it was massaged into passing.

See tests/test_m1_generate.py for the acceptance criteria.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

OPS: dict[str, Callable[[int, int], int]] = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
}

# Multiplication operands are drawn from a deliberately small range so that step
# lines stay short and the character statistics that decide K mean something.
MUL_OPERAND_MIN = 2
MUL_OPERAND_MAX = 12

# Probability that a child position becomes a subtree rather than a literal.
BRANCH_P = 0.6

# Traces shorter than the floor are discarded and regenerated, not patched.
MAX_BUILD_ATTEMPTS = 100


@dataclass(frozen=True)
class Step:
    """One reasoning step: a single binary operation and its result.

    `lhs_from` / `rhs_from` hold the index of the step that produced that operand,
    or None if the operand is a literal from the question.
    """

    lhs: int
    op: str          # one of "+", "-", "*"
    rhs: int
    result: int
    lhs_from: int | None = None
    rhs_from: int | None = None

    def render(self) -> str:
        """Render as a single line of the trace, e.g. '3 + 4 = 7'."""
        return f"{self.lhs} {self.op} {self.rhs} = {self.result}"


@dataclass(frozen=True)
class Trace:
    """A complete reasoning trace: a question, ordered steps, and a final answer."""

    question: str
    steps: tuple[Step, ...]
    answer: int

    def render(self) -> str:
        """Render the full trace, one step per line. The question is not a line."""
        return "\n".join(step.render() for step in self.steps)

    def is_valid(self) -> bool:
        """True if every step's arithmetic is correct AND the chain is internally
        consistent (each step's operands trace back to earlier results or to the
        question), and the final step's result equals `answer`.
        """
        if not self.steps:
            return False

        for i, step in enumerate(self.steps):
            if step.op not in OPS:
                return False
            if step.result != OPS[step.op](step.lhs, step.rhs):
                return False

            # Chain consistency: a recorded producer must exist strictly earlier
            # and must actually have produced the value this step consumes.
            for source, value in ((step.lhs_from, step.lhs), (step.rhs_from, step.rhs)):
                if source is None:
                    continue
                if not 0 <= source < i:
                    return False
                if self.steps[source].result != value:
                    return False

        return self.steps[-1].result == self.answer


@dataclass(frozen=True)
class _Node:
    """Internal expression-tree node. Children are _Node or int (a literal leaf)."""

    op: str
    left: "_Node | int"
    right: "_Node | int"


def _operand_range(op: str, operand_max: int) -> tuple[int, int]:
    """Leaf operand range for a given operator. Chosen after the op, never revised."""
    if op == "*":
        return MUL_OPERAND_MIN, MUL_OPERAND_MAX
    return 1, operand_max


def _build(rng: random.Random, depth: int, operand_max: int) -> _Node:
    """Build one expression-tree node, top-down, deciding each thing exactly once.

    Order is shape, then operator, then operands — so the operator constrains the
    operand range rather than the other way round, and nothing needs rewriting
    after the fact.

    Multiplication is only chosen where both operands are literals. Nested
    multiplication compounds: a depth-3 tree of products reaches ~10^8 and would
    dominate the length statistics. Restricting `*` to the frontier holds values
    in the hundreds at construction time, without any post-hoc cap or fallback.
    """
    branch_left = depth > 1 and rng.random() < BRANCH_P
    branch_right = depth > 1 and rng.random() < BRANCH_P

    allowed = ["+", "-", "*"] if not (branch_left or branch_right) else ["+", "-"]
    op = rng.choice(allowed)
    lo, hi = _operand_range(op, operand_max)

    left = _build(rng, depth - 1, operand_max) if branch_left else rng.randint(lo, hi)
    right = _build(rng, depth - 1, operand_max) if branch_right else rng.randint(lo, hi)
    return _Node(op, left, right)


def _emit(node: "_Node | int", steps: list[Step]) -> tuple[int, int | None]:
    """Post-order traversal, appending one Step per internal node in evaluation order.

    Returns (value, index of the step that produced it, or None for a literal).
    """
    if isinstance(node, int):
        return node, None

    lhs, lhs_from = _emit(node.left, steps)
    rhs, rhs_from = _emit(node.right, steps)
    result = OPS[node.op](lhs, rhs)
    steps.append(Step(lhs, node.op, rhs, result, lhs_from, rhs_from))
    return result, len(steps) - 1


def _render_expr(node: "_Node | int") -> str:
    """Render the tree as an infix expression, for the question string."""
    if isinstance(node, int):
        return str(node)
    return f"({_render_expr(node.left)} {node.op} {_render_expr(node.right)})"


def generate_trace(
    rng_seed: int,
    max_depth: int = 3,
    operand_max: int = 100,
    min_steps: int = 3,
) -> Trace:
    """Generate one valid trace from a random expression tree.

    Args:
        rng_seed: seed for this specific trace, so generation is reproducible.
        max_depth: expression tree depth; controls step count.
        operand_max: upper bound on leaf operands.
        min_steps: floor on step count. A trace with too few steps is useless for
            the central experiment — with the final step excluded from corruption,
            there must still be later steps to carry the repair evidence. Clamped
            to what the depth can produce (a depth-d tree holds at most 2^d - 1
            steps), so a depth-1 tree is not asked for the impossible.

    Returns:
        A Trace where `is_valid()` is True.

    Raises:
        RuntimeError: if the step floor could not be met. Fails loudly rather than
            looping, so a bad parameter combination is visible immediately.
    """
    if max_depth < 1:
        raise ValueError(f"max_depth must be >= 1, got {max_depth}")

    floor = min(min_steps, 2 ** max_depth - 1)
    rng = random.Random(rng_seed)

    for _ in range(MAX_BUILD_ATTEMPTS):
        root = _build(rng, max_depth, operand_max)
        steps: list[Step] = []
        answer, _ = _emit(root, steps)
        if len(steps) >= floor:
            # Strip the outermost parens; the expression is unambiguous without them.
            question = _render_expr(root)[1:-1]
            return Trace(question=question, steps=tuple(steps), answer=answer)

    raise RuntimeError(
        f"could not build a trace with >= {floor} steps at max_depth={max_depth} "
        f"in {MAX_BUILD_ATTEMPTS} attempts"
    )


def generate_dataset(
    n: int,
    seed: int = 0,
    max_depth: int = 3,
    operand_max: int = 100,
    min_steps: int = 3,
) -> list[Trace]:
    """Generate `n` traces. Reproducible given `seed`."""
    return [
        generate_trace(
            rng_seed=seed * 1_000_003 + i,
            max_depth=max_depth,
            operand_max=operand_max,
            min_steps=min_steps,
        )
        for i in range(n)
    ]
