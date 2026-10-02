"""How many human labels settle a release decision, and does grouping by the judge's verdict help?

    PYTHONPATH=.:bench <venv>/bin/python bench/review_budget.py

No model is called. Ground truth stands in for the human labels.

1. Simulation: finite datasets of N cases with a known true gain, judged by a judge that is wrong
   on each verdict with a given probability. `ReviewPlan` labels in looks of 12, 24, 48 and 96
   and stops at the first decisive look. Compared: grouping by the judge's verdict (the plan) and
   labelling at random (the same plan with the judge ignored, so one group). Reported: wrong
   decisions, how often a decision was reached, labels used.
2. Replay: the baseline and the promoted candidate of the optimization experiment on its 40
   training questions, as the plain (uncertified) judge scored them, with ground truth as labels;
   1,000 random orders of labelling.
"""

from __future__ import annotations

import json
import random
from typing import Any

from judge_bridge import judged, replies
from optimize import BASELINE_PROMPT, ROOT, split, truth

from pydantic_evals_admissibility import ReviewPlan

LOOKS = (12, 24, 48, 96)


def _settle(plan: ReviewPlan, labels: dict[str, tuple[list[bool], list[bool]]]) -> Any:
    while batch := plan.next_batch():
        plan.add_labels({case: labels[case] for case in batch})
        if plan.decision().decision != 'INCONCLUSIVE':
            break
    return plan.decision()


def simulate(n: int, gain: float, error: float, grouped: bool, trials: int = 800) -> dict[str, float]:
    rng = random.Random(2)
    wrong = decided = used = 0
    for t in range(trials):
        labels, judge_b, judge_c = {}, {}, {}
        for i in range(n):
            b, c = rng.random() < 0.6 - gain / 2, rng.random() < 0.6 + gain / 2
            labels[f'c{i}'] = ([b], [c])
            judge_b[f'c{i}'] = [b if rng.random() > error else not b] if grouped else [True]
            judge_c[f'c{i}'] = [c if rng.random() > error else not c] if grouped else [True]
        actual = sum(c[0] - b[0] for b, c in labels.values()) / n
        result = _settle(ReviewPlan.from_verdicts(judge_b, judge_c, looks=LOOKS, seed=t), labels)
        used += result.labels
        decided += result.decision != 'INCONCLUSIVE'
        wrong += (result.decision == 'PROMOTE' and actual <= 0) or (result.decision == 'REJECT' and actual >= 0)
    return {'wrong': wrong / trials, 'decided': decided / trials, 'labels': used / trials}


def replay() -> dict[str, Any]:
    run = json.loads((ROOT / 'results' / 'optimize.json').read_text())
    selected = json.loads((ROOT / 'results' / 'confirm.json').read_text())['selected_index']
    train, _ = split()
    by_name = {q.name: q for q in train}
    texts = {
        'baseline': replies(BASELINE_PROMPT, train),
        'candidate': replies(run['candidates'][selected]['prompt'], train),
    }
    judge = judged('judge', texts, by_name)
    real = {v: truth(t, by_name) for v, t in texts.items()}
    labels = {name: (real['baseline'][name], real['candidate'][name]) for name in by_name}
    actual = sum(sum(c) / len(c) - sum(b) / len(b) for b, c in labels.values()) / len(labels)
    out: dict[str, Any] = {'true_gain': actual, 'cases': len(labels)}
    for grouped in (True, False):
        results = []
        for seed in range(1000):
            verdicts = judge if grouped else {v: {n: [True] for n in by_name} for v in judge}
            plan = ReviewPlan.from_verdicts(verdicts['baseline'], verdicts['candidate'], looks=(12, 24, 40), seed=seed)
            results.append(_settle(plan, labels))
        out['grouped by judge' if grouped else 'random'] = {
            'decisions': {d: sum(r.decision == d for r in results) for d in ('PROMOTE', 'REJECT', 'INCONCLUSIVE')},
            'mean_labels': sum(r.labels for r in results) / len(results),
            'judge_gain': results[0].judge_gain,
        }
    return out


def main() -> None:
    table = {}
    for n, gain, error in ((200, 0.0, 0.1), (200, 0.2, 0.1), (200, 0.2, 0.2), (200, 0.3, 0.05), (400, 0.15, 0.1)):
        for grouped in (True, False):
            key = f'N={n} true gain {gain} judge wrong {error:.0%} ' + ('grouped' if grouped else 'random')
            table[key] = simulate(n, gain, error, grouped)
            print(f'{key:52} ' + '  '.join(f'{k} {v:.3g}' for k, v in table[key].items()), flush=True)
    real = replay()
    print(json.dumps(real, indent=2))
    (ROOT / 'results' / 'review_budget.json').write_text(json.dumps({'simulation': table, 'replay': real}, indent=2))


if __name__ == '__main__':
    main()
