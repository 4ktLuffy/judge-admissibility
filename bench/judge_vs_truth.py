"""Certify a Codex `LLMJudge` on the task, with ground truth standing in for human labels.

    PYTHONPATH=.:bench <venv>/bin/python bench/judge_vs_truth.py

Known-good answers are the computed answers written as the agent is asked to write them. The
labels are the agent's 80 real baseline replies, marked right or wrong by `is_correct`, so the
agreement check measures the judge against truth on the outputs it would actually grade.
Writes `results/judge_vs_truth.json` with the certificate and the judge's confusion against truth.
With `--reference` the judge also sees the expected answer (`include_expected_output=True`), and
the result goes to `results/judge_vs_truth.reference.json`.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

from codex_judge import TOKENS, codex_model
from pydantic_evals.evaluators import LLMJudge
from task import questions

from pydantic_evals_admissibility import HumanLabel, JudgeCase, certify_judge

RUBRIC = 'The output correctly answers the question.'


async def main() -> None:
    root = Path(__file__).parent.parent
    baseline = json.loads((root / 'results' / 'task_baseline.json').read_text())
    reference = '--reference' in sys.argv  # the judge also sees the expected answer
    cases = [
        JudgeCase(q.name, q.question, f'Answer: {q.answer}', expected_output=q.answer if reference else None)
        for q in questions()
    ]
    labels = [HumanLabel(r['question'], r['reply'], r['correct']) for r in baseline['rows']]
    judge = LLMJudge(rubric=RUBRIC, model=codex_model(), include_input=True, include_expected_output=reference)
    started = time.monotonic()
    certificate = await certify_judge(judge, cases, human_labels=labels, repeats=2, max_concurrency=4)
    print(f'{time.monotonic() - started:.0f}s, {len(TOKENS)} calls, {sum(TOKENS):,} tokens')
    print(certificate.table())
    confusion = {'pass_correct': 0, 'pass_wrong': 0, 'fail_correct': 0, 'fail_wrong': 0, 'no_verdict': 0}
    for j in certificate.judgments:
        if not j.role.startswith('human:'):
            continue
        truth = j.role == 'human:1'
        if j.passed is None:
            confusion['no_verdict'] += 1
        else:
            confusion[f'{"pass" if j.passed else "fail"}_{"correct" if truth else "wrong"}'] += 1
    print('judge vs ground truth on the 80 real replies:', confusion)
    name = 'judge_vs_truth.reference.json' if reference else 'judge_vs_truth.json'
    (root / 'results' / name).write_text(
        json.dumps({'rubric': RUBRIC, 'confusion': confusion, 'calls': len(TOKENS), **certificate.to_dict()}, indent=2)
    )


if __name__ == '__main__':
    asyncio.run(main())
