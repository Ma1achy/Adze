"""Distribution-matched truth — the standard readout.

The property that makes it worth having: a model that is perfect on easy problems
and hopeless on hard ones must NOT score well when the real task is mostly hard.
Raw pooled truth gave it 77.7% in exactly that situation. These tests hold the
difference.
"""

from __future__ import annotations

from adze.eval.readout import magnitude_shares, readout

EASY = "5 + 3 = 8"          # 0-9 bin, true
HARD_OK = "500 + 300 = 800"  # 300+ bin, true
HARD_BAD = "500 + 300 = 801"  # 300+ bin, false


def test_matched_truth_follows_the_real_distribution_not_the_generated_one() -> None:
    generated = [EASY] * 90 + [HARD_BAD] * 10
    real = [EASY] * 10 + [HARD_OK] * 90

    r = readout(generated, real)
    assert abs(r.raw_true - 0.90) < 1e-9      # the inflated figure
    assert abs(r.matched_true - 0.10) < 1e-9  # 10% weight on the bin it gets right


def test_a_bin_the_model_never_generates_scores_zero_not_skipped() -> None:
    """Avoiding a bin is a failure on it, not an absence of evidence about it."""
    r = readout([EASY] * 50, [EASY] * 50 + [HARD_OK] * 50)
    assert abs(r.raw_true - 1.0) < 1e-9
    assert abs(r.matched_true - 0.5) < 1e-9
    assert "300+" not in r.per_bin


def test_matched_equals_raw_when_the_distributions_agree() -> None:
    """No mismatch, no correction — the readout must not move a matched result."""
    texts = [EASY] * 50 + [HARD_BAD] * 50
    r = readout(texts, texts)
    assert abs(r.matched_true - r.raw_true) < 1e-9


def test_matched_ceiling_uses_the_same_weights() -> None:
    """Otherwise the ratio to ceiling compares two different distributions."""
    r = readout([EASY] * 100, [EASY] * 25 + [HARD_OK] * 75,
                ceiling={"0-9": 1.0, "300+": 0.60})
    assert abs(r.matched_ceiling - (0.25 * 1.0 + 0.75 * 0.60)) < 1e-9


def test_unbinnable_text_is_counted_not_dropped() -> None:
    r = readout([EASY] * 8 + ["nonsense"] * 2, [EASY] * 10)
    assert r.unbinnable == 2
    assert r.n == 10
    # Shares are over what could be binned, so the one live bin is all of it.
    assert abs(magnitude_shares([EASY] * 8 + ["nonsense"] * 2)["0-9"] - 1.0) < 1e-9
    # ...but raw truth still has the malformed pair in its denominator.
    assert abs(r.raw_true - 0.8) < 1e-9
