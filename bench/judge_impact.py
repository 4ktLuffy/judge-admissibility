"""Replay: had the optimizer's judge been replaced, which of its five decisions would have flipped?

    PYTHONPATH=.:bench <venv>/bin/python bench/judge_impact.py

No model is called. `bench/optimize.py` compared the baseline prompt with 5 candidates on the 40
training questions (2 replies each), with the level split five ways (`GateRules().for_candidates(5)`),
and cached every reply and verdict in `results/optimize-cache.json`. Each candidate comparison is a
past decision; `decision_impact` re-decides all five under each pair of views of the same replies:

- plain judge -> reference judge, plain judge -> ground truth: the naive optimizer acted on the plain
  judge's raw scores and promoted its best candidate (`naive_pick`), so that one is recorded as taken
  PROMOTE. The certified gate refused the plain judge (INADMISSIBLE); the decisions here are on its raw
  scores, without a certificate, because those are what the naive pick acted on.
- reference judge -> ground truth: the gate's actual decisions on the reference judge are the taken ones.

Writes `results/judge_impact.json`.
"""

from __future__ import annotations

import json
from typing import Any

from judge_bridge import judged, missing, replies
from optimize import BASELINE_PROMPT, CANDIDATES, ROOT, split, truth

from pydantic_evals_admissibility import ComparisonRecord, GateRules, decision_impact


def main() -> None:
    run = json.loads((ROOT / 'results' / 'optimize.json').read_text())
    train, _ = split()
    by_name = {q.name: q for q in train}
    prompts = {'baseline': BASELINE_PROMPT} | {f'candidate {c["index"]}': c['prompt'] for c in run['candidates']}
    assert len(prompts) == CANDIDATES + 1
    texts = {label: replies(prompt, train) for label, prompt in prompts.items()}
    gaps = {kind: missing(kind, texts, by_name) for kind in ('judge', 'judge-reference')}
    views = {
        'plain judge': judged('judge', texts, by_name),
        'reference judge': judged('judge-reference', texts, by_name),
        'ground truth': {label: truth(t, by_name) for label, t in texts.items()},
    }
    gate_on_reference = [s.split(':')[0] for s in run['gate_on_reference']]
    rules = GateRules().for_candidates(CANDIDATES)  # as optimize.py decided
    out: dict[str, Any] = {
        'rules': {'level': rules.level, 'resamples': rules.resamples, 'seed': rules.seed},
        'questions': len(train),
        'missing_verdicts': gaps,
        'naive_pick': run['naive_pick'],
        'gate_on_reference': gate_on_reference,
        'analyses': {},
    }
    for old, new in (
        ('plain judge', 'reference judge'),
        ('plain judge', 'ground truth'),
        ('reference judge', 'ground truth'),
    ):
        history = {}
        for i in range(CANDIDATES):
            label = f'candidate {i}'
            if old == 'plain judge':
                taken = 'PROMOTE' if run['naive_pick'] == i else None
            else:
                taken = gate_on_reference[i]
            history[label] = ComparisonRecord(
                old={'baseline': views[old]['baseline'], 'candidate': views[old][label]},
                new={'baseline': views[new]['baseline'], 'candidate': views[new][label]},
                taken=taken,  # type: ignore[arg-type]
            )
        report = decision_impact(history, rules=rules)
        print(f'== {old} -> {new} ({len(train)} questions, 2 replies each, level {rules.level:.3f})')
        print(report.table(), '\n')
        out['analyses'][f'{old} -> {new}'] = report.to_dict()
    (ROOT / 'results' / 'judge_impact.json').write_text(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
