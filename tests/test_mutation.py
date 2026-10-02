"""Mutation reports: plant known defects, and see which evaluators catch them."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any

import pytest
from judges import judge, no_man, oracle, yes_man
from pydantic import BaseModel
from pydantic_evals.evaluators import Contains, EqualsExpected, Evaluator, EvaluatorContext, IsInstance

from pydantic_evals_admissibility import (
    DEFAULT_MUTANTS,
    EMPTIED,
    NUMBER_CHANGED,
    YES_NO_FLIPPED,
    JudgeCase,
    Mutant,
    UncaughtDefects,
    field_dropped,
    mutation_report,
)


def qa_cases(n: int = 20) -> list[JudgeCase]:
    """Half yes/no, half amounts; every output is right, and says why before the answer."""
    out = []
    for i in range(n):
        answer = ('yes' if i % 4 == 0 else 'no') if i % 2 == 0 else str(10 + i)
        out.append(
            JudgeCase(
                f'q{i}', f'Question {i}', f'I checked the policy carefully. Answer: {answer}', expected_output=answer
            )
        )
    return out


def right(case: JudgeCase, output: Any) -> bool:
    """Ground truth: the output contains the expected answer, and the evidence was not tampered with."""
    return isinstance(output, str) and oracle(output, str(case.expected_output)) and 'tampered' not in case.inputs


TAMPERED = Mutant.of_evidence('evidence_changed', lambda inputs: inputs + ' (tampered)')


@dataclass
class Raises(Evaluator[Any, Any, Any]):
    def evaluate(self, ctx: EvaluatorContext[Any, Any, Any]) -> bool:
        if 'Answer' not in str(ctx.output):
            raise RuntimeError('rate limited')
        return True


async def test_an_evaluator_that_checks_the_answer_kills_every_mutant() -> None:
    report = await mutation_report({'checker': judge(oracle)}, qa_cases(), DEFAULT_MUTANTS, oracle=right)
    rate = report.kill_rate('checker')
    assert rate.caught == rate.judged > 0 and not report.missed('checker'), report.table()
    for mutant in report.mutants:
        assert report.kill_rate('checker', mutant.name).rate in (1.0, None), report.table()
    report.raise_unless_caught('checker', min_rate=0.8)


async def test_is_instance_misses_every_content_mutant() -> None:
    report = await mutation_report(
        {'is str': IsInstance('str')}, qa_cases(), (NUMBER_CHANGED, YES_NO_FLIPPED), oracle=right
    )
    rate = report.kill_rate('is str')
    assert rate.caught == 0 and rate.missed == 20, report.table()
    assert {o.case for o in report.missed('is str', 'number_changed')} == {f'q{i}' for i in range(1, 20, 2)}


async def test_an_output_only_evaluator_cannot_observe_an_evidence_mutant() -> None:
    evaluators = {
        'output only': judge(oracle),  # LLMJudge with include_input=False
        'contains': lambda case: Contains(value=case.expected_output),
        'shown the input, lenient': replace(judge(yes_man), include_input=True),
        'declared output only': Raises(),
    }
    report = await mutation_report(
        evaluators, qa_cases(), (TAMPERED,), oracle=right, observes={'declared output only': ['output']}
    )
    for name in ('output only', 'contains', 'declared output only'):
        rate = report.kill_rate(name, 'evidence_changed')
        assert rate.unobservable == 20 and rate.caught == 0 and rate.rate == 0.0, report.table()
    assert report.kill_rate('shown the input, lenient').missed == 20  # it sees the evidence and passes anyway
    assert 'blind' in report.table()
    with pytest.raises(UncaughtDefects, match='include_input=True'):
        report.raise_unless_caught('output only', 'evidence_changed')


async def test_mutants_the_oracle_still_accepts_are_not_counted() -> None:
    reworded = Mutant.of_output('reworded', lambda out: out.replace('I checked the policy carefully.', 'Checked.'))
    report = await mutation_report({'is str': IsInstance('str')}, qa_cases(), (reworded, EMPTIED), oracle=right)
    summary = {m.name: m for m in report.mutants}
    assert summary['reworded'].not_defect == 20 and summary['reworded'].defects == 0
    assert report.kill_rate('is str', 'reworded').judged == 0  # never held against the evaluator
    assert summary['emptied'].defects == 20

    # Without an oracle, a mutant that lands on the expected output is not a defect either.
    to_expected = Mutant('to_expected', lambda c: JudgeCase(c.name, c.inputs, c.expected_output, c.expected_output))
    report = await mutation_report([IsInstance('str')], qa_cases(), (to_expected,))
    assert report.mutants[0].not_defect == 20 and not report.oracle
    assert 'no oracle' in report.table()


async def test_cases_whose_original_the_oracle_rejects_are_skipped() -> None:
    cases = qa_cases(4) + [JudgeCase('wrong', 'Q', 'Answer: maybe', expected_output='yes')]
    report = await mutation_report([IsInstance('str')], cases, (EMPTIED,), oracle=right)
    assert report.bad_originals == ('wrong',) and report.mutants[0].defects == 4


async def test_errors_are_errors_not_catches() -> None:
    report = await mutation_report({'flaky': Raises(), 'score': _Score()}, qa_cases(), (EMPTIED,), oracle=right)
    flaky = report.kill_rate('flaky')
    assert flaky.errors == 20 and flaky.caught == 0 and flaky.judged == 0, report.table()
    assert report.kill_rate('score').errors == 20  # a number is not a pass/fail verdict
    with pytest.raises(UncaughtDefects, match='no defect was judged'):
        report.raise_unless_caught('flaky')


@dataclass
class _Score(Evaluator[Any, Any, Any]):
    def evaluate(self, ctx: EvaluatorContext[Any, Any, Any]) -> float:
        return 0.5


async def test_an_evaluator_that_fails_the_original_shows_nothing() -> None:
    report = await mutation_report(
        {'always fails': judge(no_man), 'equals': EqualsExpected()}, qa_cases(), (EMPTIED,), oracle=right
    )
    for name in ('always fails', 'equals'):  # EqualsExpected: a full reply never equals 'yes'
        rate = report.kill_rate(name)
        assert rate.original_failed == 20 and rate.caught == 0 and rate.judged == 0
        assert report.originals[name] == (0, 20, 0)


async def test_the_pytest_helper_names_what_was_missed() -> None:
    report = await mutation_report({'is str': IsInstance('str')}, qa_cases(), (NUMBER_CHANGED,), oracle=right)
    with pytest.raises(UncaughtDefects) as raised:
        report.raise_unless_caught('is str', 'number_changed', min_rate=0.5)
    message = str(raised.value)
    assert 'caught 0/10' in message and 'needs a lower bound >= 0.50' in message
    assert "missed number_changed on q1: 'I checked the policy carefully. Answer: 12'" in message
    with pytest.raises(KeyError, match='no mutant named'):
        report.raise_unless_caught('is str', 'nope')


async def test_ten_of_ten_is_not_evidence_for_ninety_percent() -> None:
    report = await mutation_report({'checker': judge(oracle)}, qa_cases(), (NUMBER_CHANGED,), oracle=right)
    assert report.kill_rate('checker', 'number_changed').caught == 10
    report.raise_unless_caught('checker', 'number_changed', min_rate=0.7)
    with pytest.raises(UncaughtDefects, match=r'caught 10/10 \(95% interval \[0\.72'):
        report.raise_unless_caught('checker', 'number_changed', min_rate=0.9)


class Refund(BaseModel):
    approved: bool
    amount: float
    message: str | None


def test_built_in_mutants_on_structured_outputs() -> None:
    case = JudgeCase('r', 'Q', Refund(approved=True, amount=42.5, message='We have refunded your order today.'))
    flipped = YES_NO_FLIPPED.mutate(case)
    assert flipped is not None and flipped.output.approved is False
    bumped = NUMBER_CHANGED.mutate(case)
    assert bumped is not None and bumped.output.amount == 43.5
    dropped = field_dropped('message').mutate(case)
    assert dropped is not None and dropped.output.message is None
    assert field_dropped('missing').mutate(case) is None

    as_dict = JudgeCase('d', 'Q', {'approved': True, 'amount': 3})
    dropped = field_dropped('amount').mutate(as_dict)
    assert dropped is not None and dropped.output == {'approved': True}
    assert EMPTIED.mutate(as_dict) is None  # no prose to empty: does not apply

    text = JudgeCase('t', 'Q', 'Yes, it is covered. Refund: $42.50. Answer: Yes')
    assert YES_NO_FLIPPED.mutate(text).output.endswith('Answer: No')  # type: ignore[union-attr]
    assert NUMBER_CHANGED.mutate(text).output == 'Yes, it is covered. Refund: $43.50. Answer: Yes'  # type: ignore[union-attr]


async def test_the_report_serialises() -> None:
    report = await mutation_report(
        {'checker': judge(oracle), 'is str': IsInstance('str')}, qa_cases(), DEFAULT_MUTANTS + (TAMPERED,), oracle=right
    )
    data = json.loads(json.dumps(report.to_dict()))
    assert data['evaluators']['checker']['by_mutant']['emptied']['rate'] == 1.0
    assert len(data['outcomes']) == len(report.outcomes)
    assert report.table().splitlines()[0].startswith('evaluator')
