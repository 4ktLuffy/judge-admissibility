"""Only trust a certified judge: a pydantic-evals suite whose judge must earn its place first.

    pytest -p pydantic_evals_admissibility.pytest_plugin examples/test_with_certified_judge.py

(With the package installed, the plugin loads itself and `-p` is not needed.) The judge is
certified once for the session, however many tests use it, and the terminal summary says what
was trusted. In CI, certify once and reuse the file instead of calling the judge on every run:

    pytest --judge-recertify --judge-certificates=judge-certificates.json   # locally, after a judge change
    pytest --judge-certificates=judge-certificates.json                     # in CI: no judge calls

Runs offline: the judge's model is a `FunctionModel` that checks the expected answer is in the
output. Swap in `LLMJudge(rubric=..., model='openai:gpt-5')` and your own known-good answers.

The scripted model is given a `model_name`: that name is its identity, as a real model's is, so its
certificate can be reused and saved. Rename it when you change what it does. A `FunctionModel`
with no name has no reliable identity: the plugin would certify it on every use and never save it.
"""

from __future__ import annotations

import re

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import LLMJudge

from pydantic_evals_admissibility import JudgeCase

CAPITALS = {
    'France': 'Paris', 'Japan': 'Tokyo', 'Kenya': 'Nairobi', 'Peru': 'Lima', 'Canada': 'Ottawa',
    'Egypt': 'Cairo', 'Norway': 'Oslo', 'Chile': 'Santiago', 'Ghana': 'Accra', 'Spain': 'Madrid',
    'India': 'New Delhi', 'Italy': 'Rome', 'Germany': 'Berlin', 'Mexico': 'Mexico City', 'Nigeria': 'Abuja',
    'Vietnam': 'Hanoi', 'Poland': 'Warsaw', 'Turkey': 'Ankara', 'Greece': 'Athens', 'Cuba': 'Havana',
    'Sweden': 'Stockholm', 'Austria': 'Vienna', 'Thailand': 'Bangkok', 'Ireland': 'Dublin', 'Portugal': 'Lisbon',
}  # fmt: skip

# Known-good answers: what the judge must pass. The plugin builds the answers it must fail from them.
KNOWN_GOOD = [
    JudgeCase(name=country, inputs=f'What is the capital of {country}?', output=city, expected_output=city)
    for country, city in CAPITALS.items()
]


def scripted_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Stands in for the judge's LLM: passes an output that contains the expected answer."""
    prompt = ''.join(
        part.content
        for message in messages
        for part in getattr(message, 'parts', [])
        if isinstance(part, UserPromptPart) and isinstance(part.content, str)
    )
    output, expected = (re.search(rf'<{tag}>\n(.*?)\n</{tag}>', prompt, re.S) for tag in ('Output', 'ExpectedOutput'))
    passed = bool(output and expected and output.group(1).strip() and expected.group(1) in output.group(1))
    tool = info.output_tools[0]
    return ModelResponse(
        parts=[ToolCallPart(tool.name, {'reason': 'scripted', 'pass': passed, 'score': float(passed)})]
    )


@pytest.fixture(scope='session')
def capital_judge(judge_certificates):  # type: ignore[no-untyped-def]
    """The judge, returned only once it is certified to gate: otherwise every test using it fails, with the reasons."""
    judge = LLMJudge(
        rubric='The output names the correct capital city.',
        model=FunctionModel(scripted_model, model_name='scripted:capitals'),
        include_expected_output=True,
    )
    return judge_certificates.certify(judge, KNOWN_GOOD, decision='gate', name='capital-judge')


def answer(question: str) -> str:
    """The system under test."""
    country = question.removeprefix('What is the capital of ').removesuffix('?')
    return f'The capital of {country} is {CAPITALS[country]}.'


def test_every_answer_passes_the_certified_judge(capital_judge) -> None:  # type: ignore[no-untyped-def]
    dataset = Dataset[str, str, None](
        name='capitals',
        cases=[Case(name=c.name, inputs=c.inputs, expected_output=c.expected_output) for c in KNOWN_GOOD[:10]],
        evaluators=[capital_judge],
    )
    report = dataset.evaluate_sync(answer, progress=False)
    assert all(result.value for case in report.cases for result in case.assertions.values())


def test_a_wrong_answer_fails_the_certified_judge(capital_judge) -> None:  # type: ignore[no-untyped-def]
    dataset = Dataset[str, str, None](
        name='capitals-wrong',
        cases=[Case(name='France', inputs='What is the capital of France?', expected_output='Paris')],
        evaluators=[capital_judge],
    )
    report = dataset.evaluate_sync(lambda question: 'The capital of France is Lyon.', progress=False)
    assert not any(result.value for case in report.cases for result in case.assertions.values())


@pytest.mark.judge_certified(decision='promote')
def test_an_optimizer_may_keep_what_this_judge_prefers(certify_judge, capital_judge) -> None:  # type: ignore[no-untyped-def]
    """A stronger use than gating: the marker asks for `promote`, which also looks for blind spots by slice."""
    # The session's certificate, qualified again for promote: no new judge calls.
    certify_judge(capital_judge, KNOWN_GOOD, name='capital-judge')
