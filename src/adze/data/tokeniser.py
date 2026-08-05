"""M2 — character-level tokenisation.

Character level, deliberately. Arithmetic is the whole point of the dataset and
subword splits on numbers are exactly what this project avoids.

The vocabulary is the 16 characters the M1 statistics actually produced, plus
PAD/BOS/EOS. Nineteen symbols total. That is the entire vocabulary; there is
nothing cleverer to build here.
"""

from __future__ import annotations

import torch

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2

SPECIALS = ("<pad>", "<bos>", "<eos>")

# Exactly the character set reported by scripts/m1_stats.py.
CHARS = "\n *+-0123456789="

# Longest observed step is 18 characters (M1 stats, both configs), + BOS + EOS.
# Fixed rather than data-derived so a checkpoint does not depend on the dataset
# it happened to be trained on. Exceeding it raises rather than truncating.
MAX_STEP_LEN = 24


class CharTokeniser:
    """Character-level tokeniser over the fixed 19-symbol vocabulary."""

    def __init__(self) -> None:
        self.itos: list[str] = [*SPECIALS, *CHARS]
        self.stoi: dict[str, int] = {c: i + len(SPECIALS) for i, c in enumerate(CHARS)}

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str, max_len: int = MAX_STEP_LEN) -> list[int]:
        """One step's text -> [BOS] + chars + [EOS], PAD-filled to `max_len`.

        Raises:
            ValueError: on an unknown character, or text too long for `max_len`.
                Both fail loudly rather than being silently dropped or truncated.
        """
        try:
            body = [self.stoi[c] for c in text]
        except KeyError as exc:
            raise ValueError(f"character {exc.args[0]!r} is not in the vocabulary") from exc

        ids = [BOS_ID, *body, EOS_ID]
        if len(ids) > max_len:
            raise ValueError(
                f"step needs {len(ids)} tokens, max_len is {max_len}: {text!r}"
            )
        return ids + [PAD_ID] * (max_len - len(ids))

    def decode(self, ids: list[int] | torch.Tensor) -> str:
        """Token ids -> text, stopping at the first EOS. Specials are dropped."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        out: list[str] = []
        for i in ids:
            if i == EOS_ID:
                break
            if i in (PAD_ID, BOS_ID):
                continue
            out.append(self.itos[i])
        return "".join(out)

    def encode_batch(self, texts: list[str], max_len: int = MAX_STEP_LEN) -> torch.Tensor:
        """List of step texts -> [batch, max_len] token ids."""
        return torch.tensor([self.encode(t, max_len) for t in texts], dtype=torch.long)
