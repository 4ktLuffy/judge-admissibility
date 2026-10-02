"""Replay: would swapping the plain judge for the reference judge have kept the optimizer's gains comparable?

    PYTHONPATH=.:bench <venv>/bin/python bench/judge_bridge.py

No model is called. `bench/optimize.py` cached every agent reply and judge verdict in
`results/optimize-cache.json`; this looks them up by the same keys, for the baseline prompt and
the promoted candidate (`results/confirm.json`'s `selected_index`), and bridges:

- train questions, plain judge (old) -> reference judge (new): both judges' verdicts are saved;
- train questions, reference judge -> ground truth (`task.is_correct`);
- held-out questions, reference judge -> ground truth. The plain judge never scored the held-out
  replies, so plain -> reference cannot be bridged there without new calls.

Writes `results/judge_bridge.json`.
"""

from __future__ import annotations

import json
from typing import Any

from optimize import BASELINE_PROMPT, REPEATS, ROOT, RUBRIC, Cache, split, truth
from task import Question

from pydantic_evals_admissibility import GateRules, compare_judges

CACHE = json.loads((ROOT / 'results' / 'optimize-cache.json').read_text())
MARGIN = 0.05


def replies(prompt: str, qs: list[Question]) -> dict[str, list[str]]:
    return {q.name: [CACHE[Cache.key('agent', prompt, q.name, q.question, str(r))] for r in range(REPEATS)] for q in qs}


def judged(kind: str, by_version: dict[str, dict[str, list[str]]], by_name: dict[str, Question]) -> dict[str, Any]:
    """Saved verdicts of `kind` ('judge' or 'judge-reference'); missing or errored is a fail, as in optimize."""
    return {
        version: {
            n: [bool(CACHE.get(Cache.key(kind, RUBRIC, by_name[n].question, r))) for r in rs] for n, rs in out.items()
        }
        for version, out in by_version.items()
    }


def missing(kind: str, by_version: dict[str, dict[str, list[str]]], by_name: dict[str, Question]) -> int:
    return sum(
        CACHE.get(Cache.key(kind, RUBRIC, by_name[n].question, r)) is None
        for out in by_version.values()
        for n, rs in out.items()
        for r in rs
    )


def main() -> None:
    run = json.loads((ROOT / 'results' / 'optimize.json').read_text())
    selected = json.loads((ROOT / 'results' / 'confirm.json').read_text())['selected_index']
    prompts = {'baseline': BASELINE_PROMPT, 'candidate': run['candidates'][selected]['prompt']}
    train, test = split()
    by_name = {q.name: q for q in train + test}
    rules = GateRules()
    out: dict[str, Any] = {'candidate_index': selected, 'margin': MARGIN, 'bridges': {}}
    for split_name, qs in (('train', train), ('held_out', test)):
        texts = {v: replies(p, qs) for v, p in prompts.items()}
        views = {
            'plain judge': judged('judge', texts, by_name),
            'reference judge': judged('judge-reference', texts, by_name),
            'ground truth': {v: truth(t, by_name) for v, t in texts.items()},
        }
        gaps = {
            'plain judge': missing('judge', texts, by_name),
            'reference judge': missing('judge-reference', texts, by_name),
        }
        pairs = [('reference judge', 'ground truth')]
        if gaps['plain judge'] == 0:
            pairs.insert(0, ('plain judge', 'reference judge'))
        else:
            out[f'{split_name}: plain judge'] = f'{gaps["plain judge"]} verdicts not saved; not bridged'
        for old, new in pairs:
            bridge = compare_judges(views[old], views[new], margin=MARGIN, rules=rules)
            label = f'{split_name}: {old} -> {new}'
            print(f'== {label} ({len(qs)} questions, {REPEATS} replies each)\n{bridge.table()}\n')
            out['bridges'][label] = {
                'cases': bridge.cases,
                'missing_verdicts': {k: v for k, v in gaps.items() if k in (old, new)},
                'rates': {f'{v}/{j}': r for (v, j), r in bridge.rates.items()},
                'gain_old': bridge.decision_old.mean_gain,
                'gain_new': bridge.decision_new.mean_gain,
                'decision_old': bridge.decision_old.summary(),
                'decision_new': bridge.decision_new.summary(),
                'decisions_differ': bridge.decisions_differ,
                'interaction': bridge.interaction.estimate,
                'interaction_interval': bridge.interaction.interval,
                'shift_baseline': [bridge.shift_baseline.estimate, bridge.shift_baseline.interval],
                'shift_candidate': [bridge.shift_candidate.estimate, bridge.shift_candidate.interval],
                'comparable': bridge.verdict,
                'cases_with_interaction': sum(v != 0 for v in bridge.per_case.values()),
            }
    out['certificates'] = {'plain judge': run['judge_certificate'], 'reference judge': run['reference_certificate']}
    (ROOT / 'results' / 'judge_bridge.json').write_text(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
