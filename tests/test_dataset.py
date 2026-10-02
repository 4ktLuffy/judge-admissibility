"""certify_dataset on real pydantic-evals Datasets, with LLMJudges on scripted models."""

from __future__ import annotations

import re

from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import LLMJudge

from pydantic_evals_admissibility import certify_dataset, controls_for, dataset_report, map_prose, rubric_kind

STYLE = 'The explanation is in a second-person or friendly style, appropriate for user display.'


def scripted(decide):  # type: ignore[no-untyped-def]
    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompt = ''.join(p.content for m in messages for p in getattr(m, 'parts', []) if isinstance(p, UserPromptPart))
        out = re.search(r'<Output>\n(.*?)\n</Output>', prompt, re.S)
        passed = decide(out.group(1) if out else '')
        tool = info.output_tools[0]
        return ModelResponse(
            parts=[ToolCallPart(tool.name, {'reason': 'scripted', 'pass': passed, 'score': float(passed)})]
        )

    return FunctionModel(model)


def friendly(output: str) -> bool:
    return bool(re.search(r"'explanation': '\s*You ", output)) or bool(re.search(r'"explanation":\s*"\s*You ', output))


def time_range_dataset(judge_model) -> Dataset:  # type: ignore[no-untyped-def]
    cases = [
        Case(
            name=f'case {i}',
            inputs={'prompt': f'logs from day {i}'},
            expected_output={
                'min_timestamp': f'2021-05-{i % 28 + 1:02d}T00:00:00Z',
                'explanation': f'You asked for day {i}; the whole day is used.',
            },
        )
        for i in range(30)
    ]
    cases[0].evaluators = (LLMJudge(rubric='Mentions the day the user asked for.', model=judge_model),)
    return Dataset(name='time ranges', cases=cases, evaluators=[LLMJudge(rubric=STYLE, model=judge_model)])


def test_rubric_kind_and_controls() -> None:
    assert rubric_kind(STYLE) == 'style' and rubric_kind('The output correctly answers the question.') == 'correctness'
    assert {c.kind for c in controls_for(STYLE) if c.name == 'mismatched_output'} == {'must_hold'}
    assert {c.kind for c in controls_for('The answer is correct.') if c.name == 'mismatched_output'} == {'must_fail'}


def test_controls_touch_prose_never_timestamps() -> None:
    out = {'min_timestamp': '2021-05-08T00:00:00Z', 'explanation': 'You asked for a day.'}
    changed = map_prose(out, lambda t: t.upper())
    assert changed == {'min_timestamp': '2021-05-08T00:00:00Z', 'explanation': 'YOU ASKED FOR A DAY.'}
    assert map_prose({'min_timestamp': '2021-05-08T00:00:00Z'}, str.upper) is None


async def test_one_call_certifies_every_judge_in_a_dataset() -> None:
    results = await certify_dataset(time_range_dataset(scripted(friendly)))
    by_scope = {r.label.split(':')[0]: r for r in results}
    dataset_judge = by_scope['dataset']
    assert dataset_judge.kind == 'style' and dataset_judge.cases == 30
    assert dataset_judge.certificate is not None and dataset_judge.certificate.verdict == 'ADMISSIBLE', dataset_report(
        results
    )
    case_judge = next(r for r in results if r.label.startswith('1 case'))
    assert case_judge.certificate is None and 'at least 10' in (case_judge.skipped or '')


async def test_a_style_judge_that_grades_content_is_caught() -> None:
    def content_sensitive(output: str) -> bool:
        return friendly(output) and 'day 1;' in output

    results = await certify_dataset(time_range_dataset(scripted(content_sensitive)))
    assert results[0].certificate is not None and results[0].certificate.verdict == 'INADMISSIBLE'


async def test_case_judges_with_the_same_rubric_but_different_models_are_certified_apart() -> None:
    calls: list[str] = []

    def tracked(name: str):  # type: ignore[no-untyped-def]
        def decide(output: str) -> bool:
            calls.append(name)
            return friendly(output)

        return scripted(decide)

    dataset = time_range_dataset(scripted(friendly))
    models = {'a': tracked('a'), 'b': tracked('b')}
    for i, case in enumerate(dataset.cases):
        case.evaluators = (LLMJudge(rubric=STYLE, model=models['a' if i % 2 else 'b']),)
    results = await certify_dataset(dataset, min_cases=10)
    case_judges = [r for r in results if not r.label.startswith('dataset')]
    assert len(case_judges) == 2 and all(r.certificate is not None for r in case_judges)
    assert {'a', 'b'} <= set(calls)  # both models were asked, not the first one twice


async def test_unnamed_cases_and_score_only_judges() -> None:
    dataset = time_range_dataset(scripted(friendly))
    for case in dataset.cases:
        case.name = None  # valid in a pydantic-evals Dataset
    dataset.evaluators = [*dataset.evaluators, LLMJudge(rubric=STYLE, assertion=False, score={'include_reason': True})]
    results = await certify_dataset(dataset)
    assert results[0].certificate is not None and results[0].certificate.verdict == 'ADMISSIBLE'
    assert 'scores only' in (results[1].skipped or '')


def test_prose_is_words_not_timestamps_or_ids() -> None:
    assert map_prose({'at': '2026-10-02 10:00:00', 'sku': 'ACME 007'}, str.upper) is None
    assert rubric_kind('The answer is friendly and concise.') == 'style'
