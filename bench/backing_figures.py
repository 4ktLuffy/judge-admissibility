"""Recompute, and save, the figures in README.md and DESIGN.md that come from simulation or arithmetic.

    PYTHONPATH=.:bench:tests <venv>/bin/python bench/backing_figures.py

No model is called. Writes `results/backing_figures.json`. Each entry names where it is cited:

- the gate's behaviour table (README "From a certificate to a decision") and DESIGN's Bonferroni
  figure, with the simulation in `tests/test_gate.py`;
- `detectable_gain` on the real Codex baseline, at the default level and split five ways;
- median reply lengths of the optimization experiment's prompts on the training questions;
- the stratified review plan's interval coverage (README "When the judge cannot decide");
- the external review's two counterexamples, exactly: six slices of ten cases judged three times,
  and four sequential looks under the old rule, each against what the current rules give.
"""

from __future__ import annotations

import json
import random
from math import comb
from pathlib import Path
from statistics import NormalDist, median
from typing import Any

from judge_bridge import replies
from optimize import BASELINE_PROMPT, ROOT, split
from test_gate import FAST, decide, decisions, difficulties, simulate

from pydantic_evals_admissibility import GateRules, ReviewPlan, detectable_gain
from pydantic_evals_admissibility._stats import clopper_pearson, wilson

OUT = Path(ROOT) / 'results' / 'backing_figures.json'


def gate_table() -> dict[str, Any]:
    rows = {'identical versions': (0.0, 200), 'better by 0.1': (0.1, 50), 'better by 0.3': (0.3, 50),
            'worse by 0.1': (-0.1, 50), 'worse by 0.3': (-0.3, 50)}  # fmt: skip
    table = {}
    for name, (shift, trials) in rows.items():
        out = decisions(shift, trials=trials)
        table[name] = {d: out.count(d) for d in ('PROMOTE', 'INCONCLUSIVE', 'REJECT')}
    d = difficulties()
    plain = corrected = 0
    for t in range(200):  # the same simulation as tests/test_gate.py, five identical candidates
        base = simulate(d, 0.0, 1000 + 6 * t)
        cands = [simulate(d, 0.0, 1001 + 6 * t + j) for j in range(5)]
        plain += any(decide(base, c, rules=FAST).decision == 'PROMOTE' for c in cands)
        corrected += any(decide(base, c, rules=FAST.for_candidates(5)).decision == 'PROMOTE' for c in cands)
    return {
        'table': table,
        'any_of_five_identical_promoted': {'plain': f'{plain}/200', 'split_5_ways': f'{corrected}/200'},
    }


def baseline_gain() -> dict[str, Any]:
    rows = json.loads((Path(ROOT) / 'results' / 'task_baseline.json').read_text())['rows']
    outcomes: dict[str, list[bool]] = {}
    for r in rows:
        outcomes.setdefault(r['question'], []).append(bool(r['correct']))
    return {
        'default_level': detectable_gain(outcomes),
        'level_split_five_ways': detectable_gain(outcomes, rules=GateRules().for_candidates(5)),
        'note': 'a shift added to each case success probability, capped at 1; not the mean gain',
    }


def reply_lengths() -> dict[str, Any]:
    run = json.loads((Path(ROOT) / 'results' / 'optimize.json').read_text())
    train, _ = split()
    prompts = {'baseline': BASELINE_PROMPT} | {f'candidate {c["index"]}': c['prompt'] for c in run['candidates']}
    return {name: median(len(r) for rs in replies(p, train).values() for r in rs) for name, p in prompts.items()}


def review_coverage(trials: int = 400) -> dict[str, Any]:
    out = {}
    for n, gain, error in ((200, 0.0, 0.1), (200, 0.2, 0.1), (200, 0.2, 0.2), (200, 0.3, 0.05), (400, 0.15, 0.1)):
        rng, covered, counted = random.Random(2), 0, 0
        for t in range(trials):
            labels, jb, jc = {}, {}, {}
            for i in range(n):
                b, c = rng.random() < 0.6 - gain / 2, rng.random() < 0.6 + gain / 2
                labels[f'c{i}'] = ([b], [c])
                jb[f'c{i}'] = [b if rng.random() > error else not b]
                jc[f'c{i}'] = [c if rng.random() > error else not c]
            actual = sum(c[0] - b[0] for b, c in labels.values()) / n
            plan = ReviewPlan.from_verdicts(jb, jc, looks=(12, 24, 48, 96), seed=t)
            while batch := plan.next_batch():
                plan.add_labels({k: labels[k] for k in batch})
            result = plan.decision()
            if result.interval:
                counted += 1
                covered += result.interval[0] <= actual <= result.interval[1]
        out[f'N={n} gain {gain} judge wrong {error:.0%}'] = round(covered / counted, 4)
    return {'coverage_at_last_look': out, 'minimum': min(out.values()), 'trials': trials}


def review_counterexamples() -> dict[str, Any]:
    # Six slices of ten cases, each case right with probability 0.7, judged three times identically.
    alpha = 0.05 / 6
    fail_at = [k for k in range(31) if clopper_pearson(k, 30, alpha)[1] < 0.7]
    p_slice = sum(comb(10, c) * 0.7**c * 0.3 ** (10 - c) for c in range(11) if 3 * c in fail_at)
    pooled = 1 - (1 - p_slice) ** 6
    p_case = sum(comb(10, c) * 0.7**c * 0.3 ** (10 - c) for c in range(11) if clopper_pearson(c, 10, alpha)[1] < 0.7)
    per_case = 1 - (1 - p_case) ** 6

    # Four looks of 15 at a bar of 0.8, true rate 0.8: Wilson, the last look at the full level.
    def sequential(rule: str) -> float:
        z_early = NormalDist().inv_cdf(1 - 0.05 / 8)
        alive, failed = {0: 1.0}, 0.0
        for look in range(1, 5):
            grown: dict[int, float] = {}
            for k, p in alive.items():
                for s in range(16):
                    grown[k + s] = grown.get(k + s, 0.0) + p * comb(15, s) * 0.8**s * 0.2 ** (15 - s)
            alive = {}
            for k, p in grown.items():
                n = 15 * look
                if rule == 'old':  # Wilson, the budget split over the early looks only
                    upper = wilson(k, n, z_early if look < 4 else 1.96)[1]
                elif rule == 'wilson_split':  # Wilson, the budget split over every look
                    upper = wilson(k, n, z_early)[1]
                else:  # the current rule: exact bounds, the budget split over every look
                    upper = clopper_pearson(k, n, 0.05 / 4)[1]
                if upper < 0.8:
                    failed += p
                else:
                    alive[k] = p
        return failed

    fixed = sum(comb(60, k) * 0.8**k * 0.2 ** (60 - k) for k in range(61) if wilson(k, 60)[1] < 0.8)
    return {
        'slices_6x10_repeated_3x_false_fail': {
            'counting_repeats': round(pooled, 4),
            'counting_cases': round(per_case, 4),
        },
        'sequential_4_looks_false_fail_at_bar': {
            'old_rule': round(sequential('old'), 4),
            'wilson_split_every_look': round(sequential('wilson_split'), 4),
            'current_rule': round(sequential('new'), 4),
            'judging_all_at_once': round(fixed, 4),
        },  # fmt: skip
    }


def main() -> None:
    figures = {
        'gate': gate_table(),
        'detectable_gain': baseline_gain(),
        'median_reply_chars_train': reply_lengths(),
        'review_plan': review_coverage(),
        'external_review': review_counterexamples(),
    }
    print(json.dumps(figures, indent=2))
    OUT.write_text(json.dumps(figures, indent=2))


if __name__ == '__main__':
    main()
