"""The doctor's advice on certificates from real scripted-judge runs."""

from __future__ import annotations

from judges import coin, exact, judge, no_man, oracle, yes_man
from pydantic_evals.evaluators import LLMJudge
from test_certify import CASES

from pydantic_evals_admissibility import Certificate, Check, certify_judge, diagnose


async def test_a_sound_judge_gets_no_advice() -> None:
    assert diagnose(await certify_judge(judge(oracle), CASES), judge(oracle)) == []


async def test_a_judge_that_fails_everything_is_told_why() -> None:
    advice = diagnose(await certify_judge(judge(no_man), CASES))
    assert any('cannot confirm anything' in a for a in advice)


def test_blind_rubric_points_at_include_input() -> None:
    cert = Certificate('INADMISSIBLE', (Check('acceptance', 'FAIL', 0, 40, (0.0, 0.09), 0.7),
                                        Check('rejection', 'PASS', 40, 40, (0.91, 1.0), 0.8)), ())  # fmt: skip
    blind = LLMJudge(rubric='The output correctly answers the question.')
    assert any('Set `include_input=True`' in a for a in diagnose(cert, blind))
    sighted = LLMJudge(rubric='The output correctly answers the question.', include_input=True)
    assert not any('include_input=True' in a for a in diagnose(cert, sighted))


async def test_a_judge_that_passes_everything_is_named() -> None:
    advice = diagnose(await certify_judge(judge(yes_man), CASES), judge(yes_man))
    assert any('cannot be right' in a for a in advice)


async def test_formatting_sensitivity_vs_noise_are_told_apart() -> None:
    picky = diagnose(await certify_judge(judge(exact), CASES))
    assert any('sensitive to formatting' in a for a in picky)
    noisy = diagnose(await certify_judge(judge(coin(seed=1)), CASES))
    assert not any('sensitive to formatting' in a for a in noisy)


def test_says_how_many_cases_would_settle_an_open_check() -> None:
    open_check = Check('stability', 'UNVALIDATED', 12, 12, (0.76, 1.0), 0.8, 'interval straddles the threshold')
    advice = diagnose(Certificate('UNVALIDATED', (open_check,), ()))
    assert any('16 would' in a for a in advice), advice


async def test_ci_failure_carries_the_table_and_the_advice() -> None:
    import pytest

    from pydantic_evals_admissibility import InadmissibleJudge

    blind = judge(no_man)
    with pytest.raises(InadmissibleJudge) as raised:
        (await certify_judge(blind, CASES)).raise_unless_admissible(blind)
    message = str(raised.value)
    assert 'INADMISSIBLE' in message and 'acceptance' in message and 'cannot confirm anything' in message
    (await certify_judge(judge(oracle), CASES)).raise_unless_admissible()  # admissible: no error


def test_disagreement_is_not_blamed_on_the_rubric_when_controls_fail() -> None:
    """Found on a real certificate: a judge failing rejection was told it 'passes the controls'."""
    checks = (
        Check('acceptance', 'PASS', 40, 40, (0.91, 1.0), 0.7),
        Check('rejection', 'FAIL', 10, 20, (0.26, 0.74), 0.8, 'mismatched_output 0/10, empty_output 10/10'),
        Check('human_agreement', 'FAIL', 16, 20, (0.52, 0.94), 0.4, 'kappa=0.23'),
    )
    advice = diagnose(Certificate('INADMISSIBLE', checks, ()))
    assert not any('passes the controls' in a for a in advice)
    assert any('fix the failures above first' in a for a in advice)


def test_cases_needed_is_the_smallest_that_works() -> None:
    from pydantic_evals_admissibility import wilson

    check = Check('invariance', 'UNVALIDATED', 10, 10, (0.72, 1.0), 0.8, 'interval straddles the threshold')
    advice = diagnose(Certificate('UNVALIDATED', (check,), ()))
    assert any('16 would' in a for a in advice), advice
    assert wilson(16, 16)[0] >= 0.8 > wilson(15, 15)[0]
