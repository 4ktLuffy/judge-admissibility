"""The exact interval: published small-sample values, and no overflow at sizes a real log reaches."""

from __future__ import annotations

import pytest

from pydantic_evals_admissibility._stats import clopper_pearson


def test_small_samples_match_the_published_exact_values() -> None:
    assert clopper_pearson(0, 10) == (0.0, pytest.approx(0.3085, abs=5e-5))
    assert clopper_pearson(5, 10) == (pytest.approx(0.1871, abs=5e-5), pytest.approx(0.8129, abs=5e-5))
    assert clopper_pearson(19, 20)[0] == pytest.approx(0.7513, abs=5e-5)


def test_large_samples_do_not_overflow() -> None:
    """`math.comb(2000, 1000)` is past a float's range; the direct sum raised OverflowError."""
    low, high = clopper_pearson(1000, 2000)
    assert low == pytest.approx(0.4779, abs=5e-4) and high == pytest.approx(0.5221, abs=5e-4)
    assert clopper_pearson(0, 5000)[1] == pytest.approx(1 - 0.025 ** (1 / 5000), rel=1e-6)
