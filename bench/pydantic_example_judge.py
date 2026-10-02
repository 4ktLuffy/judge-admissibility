"""Certify the judge in Pydantic's own example evals, with controls written for its rubric.

    PYTHONPATH=.:bench <venv>/bin/python bench/pydantic_example_judge.py

`examples/pydantic_ai_examples/evals/datasets/time_range_v2.yaml` in pydantic/pydantic-ai grades
every case with:

    LLMJudge: Ensure the explanation or error_message fields are truly appropriate for user
    display, in a second-person or friendly style.

That is a style rubric, so the default correctness controls do not apply: another case's friendly
explanation is still friendly. The controls here are written for this rubric. The dataset's own
expected outputs serve as the known-good answers, so this certifies the judge, not their agent.
Judges are Codex `gpt-5.6-luna` with no reasoning and with high reasoning.
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml
from codex_judge import TOKENS, codex_model
from pydantic_evals.evaluators import LLMJudge

from pydantic_evals_admissibility import JudgeCase, MismatchedOutput, Rewrite, certify_judge, diagnose

ROOT = Path(__file__).parent.parent
UPSTREAM = ROOT.parent / 'pydantic-durability' / 'upstream'
DATASET = 'examples/pydantic_ai_examples/evals/datasets/time_range_v2.yaml'
TEXT_FIELDS = ('explanation', 'error_message')


def load() -> tuple[str, str, list[JudgeCase]]:
    commit = subprocess.run(
        ['git', 'rev-parse', '--short', 'origin/main'], cwd=UPSTREAM, capture_output=True, text=True
    ).stdout.strip()
    raw = subprocess.run(
        ['git', 'show', f'origin/main:{DATASET}'], cwd=UPSTREAM, capture_output=True, text=True, check=True
    ).stdout
    data = yaml.safe_load(raw)
    rubric = next(e['LLMJudge'] for e in data['evaluators'] if isinstance(e, dict) and 'LLMJudge' in e)
    cases = [JudgeCase(c['name'], c['inputs'], c['expected_output']) for c in data['cases']]
    return commit, rubric, cases


def edit(output: Any, change: Any) -> Any | None:
    """Apply `change` to the user-facing text field; None if there is none."""
    if not isinstance(output, dict):
        return None
    field = next((f for f in TEXT_FIELDS if f in output and isinstance(output[f], str)), None)
    if field is None:
        return None
    new = change(output[field], output)
    return None if new is None else {**output, field: new}


# Not used as a control: the rubric asks for "second-person or friendly", so a friendly
# third-person explanation satisfies it, and a sound judge may rightly pass one. Kept to show why.
def third_person(text: str, _: Any) -> str | None:
    if not re.search(r'\b(you|your)\b', text, re.I):
        return None
    text = re.sub(r'\bYour\b', "The user's", text)
    text = re.sub(r'\byour\b', "the user's", text)
    text = re.sub(r'\bYou\b', 'The user', text)
    return re.sub(r'\byou\b', 'the user', text)


def debug_string(text: str, output: Any) -> str:
    keys = ','.join(sorted(k for k in output if k not in TEXT_FIELDS))
    return f'ERR_OK keys=[{keys}] status=0x00 trace=none'


CONTROLS = (
    Rewrite(lambda o: edit(o, lambda t, _: ''), 'empty_explanation', 'must_fail'),
    Rewrite(lambda o: edit(o, debug_string), 'debug_string', 'must_fail'),
    MismatchedOutput(kind='must_hold'),
    Rewrite(lambda o: edit(o, lambda t, _: t.replace(' ', '  ') + '\n'), 'extra_spaces', 'must_hold'),
)


async def main() -> None:
    commit, rubric, cases = load()
    print(f'pydantic/pydantic-ai {DATASET} at {commit}: {len(cases)} cases\nrubric: {rubric}')
    results: dict[str, Any] = {'commit': commit, 'rubric': rubric, 'cases': len(cases), 'judges': {}}
    for label, effort in (('no reasoning', 'none'), ('reasoning high', 'high')):
        judge = LLMJudge(rubric=rubric, model=codex_model(effort=effort, timeout=300))
        before = len(TOKENS)
        cert = await certify_judge(judge, cases, controls=CONTROLS, repeats=3, max_concurrency=4)
        advice = diagnose(cert, judge)
        print(f'\n== {label}: {cert.verdict} ({len(TOKENS) - before} calls)\n{cert.table()}')
        for a in advice:
            print('   -', a)
        results['judges'][label] = {**cert.to_dict(), 'advice': advice}
        (ROOT / 'results' / 'pydantic_example_judge.json').write_text(json.dumps(results, indent=2, default=str))


if __name__ == '__main__':
    asyncio.run(main())
