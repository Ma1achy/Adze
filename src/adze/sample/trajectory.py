"""M4 — the trajectory printer.

Decode and print at every denoising step. This is the main debugging instrument
for everything after M4 — build it properly, not as a print() in a loop.

Watching noise resolve into structure into content is both the fastest way to
spot a broken model and the actual pleasure of this architecture. If step 40 of
50 is still gibberish you know long before a loss curve tells you.

**Nothing here cleans up the output.** Decoded text is raw argmax, printed
exactly as produced — malformed strings included. A printer that snapped output
to well-formed arithmetic would hide the failure it exists to expose.

**Well-formedness is not evidence.** The decoder emits `N op N = N` from almost
any latent: fed random Gaussian latents it does so 82.6% of the time at D=16 and
87.0% at D=64. So the printer reports the noise floor alongside any
well-formedness figure, and a number below that floor means the sampler is doing
*worse* than noise. Measure the floor for the decoder in hand rather than reusing
a number from another one — it moves with D.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

import torch

STEP_RE = re.compile(r"^(-?\d+) ([+\-*]) (-?\d+) = (-?\d+)$")

# Fallback share of random Gaussian latents that decode to well-formed arithmetic.
# Any well-formedness figure must be read against this, not against zero. It is a
# LAST RESORT: the floor depends on the decoder, and measured values have ranged
# from 82.6% (D=16) to 87.0% (D=64). Measure it for the decoder in hand and pass
# it in; scripts/m4_sample.py does.
DEFAULT_NOISE_FLOOR = 0.87


def classify(text: str) -> str:
    """'true' / 'false' / 'malformed'. Applied to raw output, never to fix it."""
    m = STEP_RE.match(text)
    if not m:
        return "malformed"
    lhs, op, rhs, result = int(m[1]), m[2], int(m[3]), int(m[4])
    return "true" if result == {"+": lhs + rhs, "-": lhs - rhs, "*": lhs * rhs}[op] else "false"


class TrajectoryRecorder:
    """Capture latents at every denoising step, decode on demand.

    Latents are recorded in the denoiser's *scaled* space. `unscale` is applied
    before decoding, because the decoder was trained on unscaled latents — decoding
    scaled ones produces garbage that looks like a broken sampler.
    """

    def __init__(
        self,
        decoder: torch.nn.Module,
        tokeniser: object,
        latents_per_block: int,
        latent_scale: float = 1.0,
        noise_floor: float = DEFAULT_NOISE_FLOOR,
    ) -> None:
        self.decoder = decoder
        self.tokeniser = tokeniser
        self.latents_per_block = latents_per_block
        self.latent_scale = latent_scale
        self.noise_floor = noise_floor
        self._steps: list[tuple[int, float, int | None, torch.Tensor]] = []

    def record(
        self,
        step: int,
        t: float,
        latents: torch.Tensor,
        active_block: int | None = None,
    ) -> None:
        """Store one denoising step.

        Args:
            step: global step index across the whole draft.
            t: the timestep of the block being denoised.
            latents: [N, D] or [1, N, D] scaled latents for a single sample.
            active_block: which block this step is denoising, if any.
        """
        z = latents.detach()
        if z.ndim == 3:
            z = z[0]
        self._steps.append((step, t, active_block, z.clone().cpu()))

    @torch.no_grad()
    def decoded(self) -> Iterator[tuple[int, float, list[str]]]:
        """Yield (step, t, decoded_text_per_block) for each recorded step."""
        for step, t, _active, z in self._steps:
            yield step, t, self._decode(z)

    @torch.no_grad()
    def _decode(self, z: torch.Tensor) -> list[str]:
        """[N, D] scaled latents -> one decoded string per block."""
        device = next(self.decoder.parameters()).device
        blocks = z.shape[0] // self.latents_per_block
        per_block = (z * self.latent_scale).view(blocks, self.latents_per_block, -1)
        logits = self.decoder(per_block.to(device))
        return [self.tokeniser.decode(row) for row in logits.argmax(dim=-1)]

    def stabilised(self) -> list[list[bool]]:
        """Per recorded step, which blocks decode the same as they did the step before.

        A block that stops changing has converged — the useful signal is *when*,
        and whether blocks settle in order under a causal mask.
        """
        texts = [self._decode(z) for _, _, _, z in self._steps]
        out = [[False] * len(texts[0])] if texts else []
        for prev, cur in zip(texts, texts[1:]):
            out.append([a == b for a, b in zip(prev, cur)])
        return out

    def print(self, every: int = 1) -> None:
        """Pretty-print the trajectory to stdout.

        Args:
            every: print one row in `every` recorded steps. The final step is
                always printed regardless, so the endpoint is never elided.
        """
        if not self._steps:
            print("(no steps recorded)")
            return

        rows = list(self.decoded())
        stable = self.stabilised()
        n_blocks = len(rows[0][2])
        last = len(rows) - 1

        width = max(24, max(len(t) for _, _, texts in rows for t in texts) + 2)
        header = "  ".join(f"block {b}".ljust(width) for b in range(n_blocks))
        print(f"{'step':>5} {'t':>6}  {header}")
        print("-" * (13 + (width + 2) * n_blocks))

        for i, (step, t, texts) in enumerate(rows):
            if i % every and i != last:
                continue
            active = self._steps[i][2]
            cells = []
            for b, text in enumerate(texts):
                mark = "*" if b == active else (" " if not stable[i][b] else "=")
                cells.append(f"{mark}{text!r}".ljust(width))
            print(f"{step:>5} {t:>6.3f}  {'  '.join(cells)}")

        print("-" * (13 + (width + 2) * n_blocks))
        print("  * = block being denoised,  = changed this step,  = unchanged")

        final = rows[-1][2]
        kinds = [classify(text) for text in final]
        formed = sum(1 for kind in kinds if kind != "malformed")
        true = sum(1 for kind in kinds if kind == "true")
        print(f"\nfinal: {formed}/{n_blocks} well-formed, {true}/{n_blocks} arithmetically true")
        print(f"       well-formedness floor from random latents is {self.noise_floor:.1%} — "
              f"read the figure against that, not against zero")
