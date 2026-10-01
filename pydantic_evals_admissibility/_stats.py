"""Interval estimates small enough to read. No scipy: the bounds are closed-form."""

from __future__ import annotations

import math
from collections.abc import Iterable


def wilson(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion; (0, 1) when there are no trials."""
    if trials == 0:
        return 0.0, 1.0
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def cohen_kappa(pairs: Iterable[tuple[bool, bool]]) -> float | None:
    """Agreement between two binary raters beyond chance; None when it is undefined.

    Undefined when there are no pairs, or when both raters gave one label to every item, so
    chance agreement is 1 and kappa divides by zero. Reporting 1.0 there would certify a judge
    on a label set that cannot distinguish it from a constant.
    """
    pairs = list(pairs)
    n = len(pairs)
    if n == 0:
        return None
    observed = sum(a == b for a, b in pairs) / n
    a_yes = sum(a for a, _ in pairs) / n
    b_yes = sum(b for _, b in pairs) / n
    expected = a_yes * b_yes + (1 - a_yes) * (1 - b_yes)
    if expected == 1:
        return None
    return (observed - expected) / (1 - expected)
