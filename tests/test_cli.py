"""The `judge-admissibility` command, end to end, on pydantic-ai's offline `test` model."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from pydantic_evals_admissibility._cli import main, only_judges


def write_dataset(path: Path, judges: list[object]) -> Path:
    data = {
        'cases': [
            {
                'name': f'case {i}',
                'inputs': {'question': f'What is {i} + {i}?'},
                'expected_output': f'{i} + {i} is {2 * i}.',
                'evaluators': ['OurOwnCheck'],  # a custom evaluator the command cannot import
            }
            for i in range(12)
        ],
        'evaluators': [*judges, {'AnotherCustom': {'limit': 3}}],
    }
    path.write_text(yaml.safe_dump(data))
    return path


def test_only_judges_drops_every_other_evaluator() -> None:
    data = {
        'cases': [{'name': 'a', 'evaluators': ['IsInstance', {'LLMJudge': 'Is polite.'}, {'Mine': {'x': 1}}]}],
        'evaluators': [{'LLMJudge': {'rubric': 'Is right.', 'include_input': True}}, 'EqualsExpected'],
        'report_evaluators': ['ConfusionMatrix'],
    }
    out = only_judges(data)
    assert out['cases'][0]['evaluators'] == [{'LLMJudge': 'Is polite.'}]
    assert out['evaluators'] == [{'LLMJudge': {'rubric': 'Is right.', 'include_input': True}}]
    assert 'report_evaluators' not in out


def test_certify_fails_ci_on_a_broken_judge_and_report_reads_it_back(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # pydantic-ai's `test` model fails every answer: a constant judge the certificate must refuse.
    dataset = write_dataset(tmp_path / 'cases.yaml', [{'LLMJudge': {'rubric': 'The answer is correct.'}}])
    saved = tmp_path / 'certificates.json'
    code = main(['certify', str(dataset), '--model', 'test', '--repeats', '1', '--json', str(saved)])
    printed = capsys.readouterr().out
    assert code == 1, printed
    assert 'INADMISSIBLE' in printed and 'acceptance             FAIL' in printed
    assert 'It fails answers that are right' in printed  # the doctor's advice, in the CI log

    entry = json.loads(saved.read_text())['judges'][0]
    assert entry['rubric'] == 'The answer is correct.' and entry['certificate']['verdict'] == 'INADMISSIBLE'

    assert main(['report', str(saved)]) == 1
    assert 'acceptance             FAIL' in capsys.readouterr().out


def test_nothing_to_certify_is_its_own_exit_code(tmp_path: Path) -> None:
    assert main(['certify', str(write_dataset(tmp_path / 'none.yaml', [])), '--model', 'test']) == 2
