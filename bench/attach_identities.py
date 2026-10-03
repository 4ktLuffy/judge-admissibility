"""Record the judge identity on certificates saved before certificates recorded one.

    PYTHONPATH=.:bench <venv>/bin/python bench/attach_identities.py

Four saved certificates predate `judge_identity`, so `qualify` cannot tell which judge they
cover and refuses them for anything but a report. Their judges are fully defined in the bench
scripts that produced them, so this rebuilds each judge exactly as configured there, records its
identity, and marks where it came from (`identity_source`). No judge is called and no verdict
changes. Re-running the bench scripts would record the identity at certification time instead.
"""

from __future__ import annotations

import json
from pathlib import Path

from codex_judge import codex_model
from judge_vs_truth import RUBRIC as ARITHMETIC_RUBRIC
from pydantic_evals.evaluators import LLMJudge
from support_eval import RUBRIC as SUPPORT_RUBRIC

from pydantic_evals_admissibility import judge_identity

ROOT = Path(__file__).parent.parent
SOURCE = 'recorded after the run from the judge as configured in {}; the run predates identities'

SUPPORT = {
    'weak (no reasoning, sees the question)': LLMJudge(
        rubric=SUPPORT_RUBRIC, model=codex_model(effort='none'), include_input=True
    ),
    'strong (reasoning high, sees the question)': LLMJudge(
        rubric=SUPPORT_RUBRIC, model=codex_model(effort='high', timeout=300), include_input=True
    ),
    'gpt-reserve (no reasoning, sees the question)': LLMJudge(
        rubric=SUPPORT_RUBRIC, model=codex_model('gpt-reserve', effort='none'), include_input=True
    ),
}
REFERENCE = LLMJudge(rubric=ARITHMETIC_RUBRIC, model=codex_model(), include_input=True, include_expected_output=True)


def main() -> None:
    path = ROOT / 'results' / 'support_eval.json'
    data = json.loads(path.read_text())
    for label, judge in SUPPORT.items():
        entry = data['certificates'][label]
        entry['identity'] = judge_identity(judge)
        entry['identity_source'] = SOURCE.format('bench/support_eval.py')
        print(f'support_eval.json {label}: {entry["identity"]["model"]}')
    path.write_text(json.dumps(data, indent=2, default=str))

    path = ROOT / 'results' / 'judge_vs_truth.reference.json'
    data = json.loads(path.read_text())
    data['identity'] = judge_identity(REFERENCE)
    data['identity_source'] = SOURCE.format('bench/judge_vs_truth.py (--reference)')
    print(f'judge_vs_truth.reference.json: {data["identity"]["model"]}')
    path.write_text(json.dumps(data, indent=2, default=str))


if __name__ == '__main__':
    main()
