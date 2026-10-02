"""How often the `slices` check fails a sound judge, and how often it catches a bad slice. Exact.

    PYTHONPATH=.:bench <venv>/bin/python bench/slice_error.py

No simulation: each slice's pass count is binomial, so the probability that the check FAILs is
computed from the binomial distribution, using the package's own `clopper_pearson` and `wilson`.
The rule is `_certify._slice_check`'s: the check FAILs when any slice's upper bound is below the
bar, with the level (0.05) split across the k slices. Clopper-Pearson is what the package uses;
Wilson, at z for the split level, is what it used before, shown for comparison.

`n` is the number of judgments in a slice (cases times repeats), each assumed independent.
Writes `results/slice_error.json`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

from pydantic_evals_admissibility._certify import DEFAULT_THRESHOLDS
from pydantic_evals_admissibility._stats import clopper_pearson, wilson

ROOT = Path(__file__).parent.parent
OUT = ROOT / 'results' / 'slice_error.json'
BAR = DEFAULT_THRESHOLDS.min_acceptance  # the slices check uses the acceptance bar, 0.7
Z = 1.96  # the certificate's default
SLICES = (2, 3, 4, 6, 8, 12)
SIZES = (10, 14, 20, 30, 40)
METHODS = ('clopper_pearson', 'wilson')


def upper(method: str, successes: int, trials: int, k: int) -> float:
    """A slice's upper bound, with the level split across `k` slices as `_slice_check` splits it."""
    alpha = 2 * (1 - NormalDist().cdf(Z)) / k
    if method == 'clopper_pearson':
        return clopper_pearson(successes, trials, alpha)[1]
    return wilson(successes, trials, NormalDist().inv_cdf(1 - alpha / 2))[1]


def slice_fails(method: str, rate: float, n: int, k: int) -> float:
    """P(this slice's upper bound is below the bar), when its true pass rate is `rate`."""
    return sum(math.comb(n, s) * rate**s * (1 - rate) ** (n - s) for s in range(n + 1) if upper(method, s, n, k) < BAR)


def check_fails(method: str, rates: list[float], n: int) -> float:
    """P(the slices check FAILs): at least one slice's upper bound is below the bar."""
    k = len(rates)
    return 1 - math.prod(1 - slice_fails(method, rate, n, k) for rate in rates)


def main() -> None:
    sound: dict[str, dict[str, float]] = {m: {} for m in METHODS}
    print(f'Sound judge, every slice at exactly the bar ({BAR}): P(slices check FAILs)')
    print(f'{"slices x judgments":>20}  {"Clopper-Pearson":>15}  {"Wilson":>8}')
    for k in SLICES:
        for n in SIZES:
            row = {m: check_fails(m, [BAR] * k, n) for m in METHODS}
            for m in METHODS:
                sound[m][f'{k}x{n}'] = row[m]
            print(f'{f"{k} x {n}":>20}  {row["clopper_pearson"]:>15.2%}  {row["wilson"]:>8.2%}')
    worst = {m: max(sound[m].items(), key=lambda item: item[1]) for m in METHODS}
    for m in METHODS:
        print(f'max {m}: {worst[m][1]:.2%} at {worst[m][0]}')

    power: dict[str, dict[str, float]] = {m: {} for m in METHODS}
    print('\nOne bad slice among 6 (the other five at 0.95): P(slices check FAILs)')
    print(f'{"bad slice rate, judgments":>26}  {"Clopper-Pearson":>15}  {"Wilson":>8}')
    for rate in (0.0, 0.2, 0.3):
        for n in (10, 20):
            row = {m: check_fails(m, [rate] + [0.95] * 5, n) for m in METHODS}
            for m in METHODS:
                power[m][f'rate {rate}, n {n}'] = row[m]
            print(f'{f"{rate}, {n}":>26}  {row["clopper_pearson"]:>15.2%}  {row["wilson"]:>8.2%}')

    result: dict[str, Any] = {
        'rule': 'FAIL iff some slice upper bound < bar; level 0.05 split over k slices; exact binomial',
        'bar': BAR,
        'sound_judge_false_fail': sound,
        'max_false_fail': {m: {'layout': worst[m][0], 'probability': worst[m][1]} for m in METHODS},
        'power_one_bad_slice_of_6': power,
    }
    OUT.write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
