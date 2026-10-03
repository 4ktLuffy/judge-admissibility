"""How often `human_agreement` certifies a judge whose true kappa is just below the bar.

    PYTHONPATH=.:tests <venv>/bin/python bench/kappa_coverage.py

No model calls. Each sample draws human labels (pass with probability p) and a judge that flips
each label with probability f, so the population kappa is known exactly; the bar is set 0.01 above
it, so every PASS is a false PASS. The promise is at most 2.5%. A review measured at least 7% for
the rule before this one (an exact floor applied only where the bootstrap collapsed). Writes
`results/kappa_coverage.json`.
"""

from __future__ import annotations

import json
import random
from multiprocessing import Pool
from pathlib import Path

from pydantic_evals_admissibility._certify import (  # pyright: ignore[reportPrivateUsage]
    Judgment,
    Thresholds,
    _agreement,
)

OUT = Path(__file__).parent.parent / 'results' / 'kappa_coverage.json'
SETTINGS = (('balanced, 40 cases', 40, 0.5, 0.10), ('skewed (90% pass), 80 cases', 80, 0.9, 0.03),
            ('balanced, 100 cases', 100, 0.5, 0.10))  # fmt: skip
TRIALS = 2000


def population_kappa(p_pass: float, flip: float) -> float:
    judge_pass = p_pass * (1 - flip) + (1 - p_pass) * flip
    chance = p_pass * judge_pass + (1 - p_pass) * (1 - judge_pass)
    return ((1 - flip) - chance) / (1 - chance)


def one(args: tuple[int, int, float, float, float]) -> str:
    seed, n, p_pass, flip, bar = args
    rng = random.Random(seed)
    judgments = []
    for i in range(n):
        human = rng.random() < p_pass
        verdict = human if rng.random() >= flip else not human
        judgments.append(Judgment(f'c{i}', f'human:{int(human)}', 'output', verdict))
    return _agreement(judgments, Thresholds(min_kappa=bar), 0.0125).status


def main() -> None:
    out = {}
    for name, n, p_pass, flip in SETTINGS:
        kappa = population_kappa(p_pass, flip)
        bar = round(kappa + 0.01, 3)
        with Pool(8) as pool:
            statuses = pool.map(one, [(seed, n, p_pass, flip, bar) for seed in range(TRIALS)])
        passes = statuses.count('PASS')
        out[name] = {'true_kappa': round(kappa, 3), 'bar': bar, 'trials': TRIALS, 'false_pass': passes,
                     'rate': round(passes / TRIALS, 4)}  # fmt: skip
        print(name, out[name], flush=True)
    OUT.write_text(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
