"""M7 — the central experiment.

    Given the corrupted block's location, does global regeneration repair it more
    reliably than causal regeneration?

Three conditions, otherwise IDENTICAL:

    | condition                    | mask   |
    | no revision                  | ---    |
    | erase + regenerate causally  | causal |
    | erase + regenerate globally  | global |

Use the causal MASK for the comparison arm rather than physically deleting later
blocks. Deletion also changes sequence length and positional context, which would
confound the comparison. "Remove future evidence entirely" is available as an
additional ablation, not as the main arm.

Block selection is ORACLE — the corrupted index is known and erased. This is an
upper bound on what uncertainty-steered selection could achieve, and results must
be labelled as such.

Metrics (design §4). Note the pass-two delta log is NOT useful here: under
complete erasure with oracle selection, selected blocks necessarily move fully
and unselected blocks not at all, so movement is mechanically determined by rho.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ConditionResult:
    """Metrics for one arm of the experiment."""

    condition: str
    repaired_operation_acc: float   # exact match on the corrupted step
    answer_acc: float               # final answer correctness
    preservation_acc: float         # did unselected blocks survive untouched?
    n: int


def run_central_experiment(
    denoiser: torch.nn.Module,
    decoder: torch.nn.Module,
    eval_set: list,
    nfe: int,
) -> list[ConditionResult]:
    """Run all three conditions and return one result per arm.

    The headline number is the gap between the causal and global arms.
    """
    raise NotImplementedError
