"""M1 — synthetic arithmetic trace generation.

Pure Python. No torch, no GPU. This is the only component with zero dependencies
and zero uncertainty, which is why it is first.

The generator must produce traces where every intermediate value is recorded
structurally, not just rendered into text. The central experiment depends on
knowing exactly which block holds which value.

See tests/test_m1_generate.py for the acceptance criteria.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Step:
    """One reasoning step: a single binary operation and its result."""

    lhs: int
    op: str          # one of "+", "-", "*"
    rhs: int
    result: int

    def render(self) -> str:
        """Render as a single line of the trace, e.g. '3 + 4 = 7'."""
        raise NotImplementedError


@dataclass(frozen=True)
class Trace:
    """A complete reasoning trace: a question, ordered steps, and a final answer."""

    question: str
    steps: tuple[Step, ...]
    answer: int

    def render(self) -> str:
        """Render the full trace, one step per line."""
        raise NotImplementedError

    def is_valid(self) -> bool:
        """True if every step's arithmetic is correct AND the chain is internally
        consistent (each step's operands trace back to earlier results or to the
        question), and the final step's result equals `answer`.
        """
        raise NotImplementedError


def generate_trace(
    rng_seed: int,
    max_depth: int = 3,
    operand_max: int = 100,
) -> Trace:
    """Generate one valid trace from a random expression tree.

    Args:
        rng_seed: seed for this specific trace, so generation is reproducible.
        max_depth: expression tree depth; controls step count.
        operand_max: upper bound on leaf operands.

    Returns:
        A Trace where `is_valid()` is True.
    """
    raise NotImplementedError


def generate_dataset(
    n: int,
    seed: int = 0,
    max_depth: int = 3,
    operand_max: int = 100,
) -> list[Trace]:
    """Generate `n` traces. Reproducible given `seed`."""
    raise NotImplementedError
