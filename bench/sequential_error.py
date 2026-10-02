"""Exact chance that a sequential certificate fails a sound judge, at the bar, by luck.

    PYTHONPATH=. <venv>/bin/python bench/sequential_error.py

One check whose true rate sits exactly on its threshold, looked at after every batch, failing
as soon as its exact (Clopper-Pearson) upper bound at that look's share of the 5% level is below
the threshold, as `certify_judge(..., batch_size=...)` decides it. Computed by dynamic programming
over the success count, not simulated. The budget for a false FAIL is the 2.5% upper tail.
"""

from __future__ import annotations

import json
from math import comb
from pathlib import Path

from pydantic_evals_admissibility._stats import clopper_pearson

ROOT = Path(__file__).parent.parent
LAYOUTS = ((1, 60), (2, 40), (3, 10), (4, 10), (4, 15), (6, 10), (8, 10))


def false_fail(rate: float, looks: int, per_look: int) -> float:
    alive, failed = {0: 1.0}, 0.0
    for look in range(1, looks + 1):
        grown: dict[int, float] = {}
        for k, p in alive.items():
            for s in range(per_look + 1):
                grown[k + s] = grown.get(k + s, 0.0) + p * comb(per_look, s) * rate**s * (1 - rate) ** (per_look - s)
        alive = {}
        for k, p in grown.items():
            if clopper_pearson(k, look * per_look, 0.05 / looks)[1] < rate:
                failed += p
            else:
                alive[k] = p
    return failed


def main() -> None:
    rows = {
        f'bar {bar}, {looks} looks of {per}': round(false_fail(bar, looks, per), 4)
        for bar in (0.7, 0.8, 0.9)
        for looks, per in LAYOUTS
    }
    for name, value in rows.items():
        print(f'{name:28} {value:.2%}')
    print(f'worst: {max(rows.values()):.2%} (budget 2.5%)')
    (ROOT / 'results' / 'sequential_error.json').write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
