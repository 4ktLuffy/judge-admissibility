"""The certificate's controls: it must certify a sound judge and refuse each kind of broken one."""

from __future__ import annotations

import pytest
from judges import coin, exact, judge, lenient_on_empty, no_man, oracle, yes_man

from pydantic_evals_admissibility import HumanLabel, JudgeCase, Thresholds, certify_judge, cohen_kappa, wilson

CAPITALS = {
    'France': 'Paris',
    'Japan': 'Tokyo',
    'Kenya': 'Nairobi',
    'Peru': 'Lima',
    'Canada': 'Ottawa',
    'Egypt': 'Cairo',
    'Norway': 'Oslo',
    'Chile': 'Santiago',
    'Ghana': 'Accra',
    'Spain': 'Madrid',
    'India': 'New Delhi',
    'Brazil': 'Brasilia',
    'Italy': 'Rome',
    'Germany': 'Berlin',
    'Mexico': 'Mexico City',
    'Nigeria': 'Abuja',
    'Vietnam': 'Hanoi',
    'Poland': 'Warsaw',
    'Turkey': 'Ankara',
    'Greece': 'Athens',
    'Cuba': 'Havana',
    'Iran': 'Tehran',
    'Sweden': 'Stockholm',
    'Austria': 'Vienna',
    'Thailand': 'Bangkok',
    'Morocco': 'Rabat',
    'Ireland': 'Dublin',
    'Portugal': 'Lisbon',
    'Senegal': 'Dakar',
    'Hungary': 'Budapest',
}
CASES = [
    JudgeCase(name=country, inputs=f'Capital of {country}?', output=city, expected_output=city)
    for country, city in CAPITALS.items()
]


def status(certificate, name):  # type: ignore[no-untyped-def]
    return next(check.status for check in certificate.checks if check.name == name)


async def test_a_sound_judge_is_admissible() -> None:
    certificate = await certify_judge(judge(oracle), CASES)
    assert certificate.verdict == 'ADMISSIBLE', certificate.table()


async def test_a_judge_that_passes_everything_fails_rejection() -> None:
    certificate = await certify_judge(judge(yes_man), CASES)
    assert certificate.verdict == 'INADMISSIBLE'
    assert [c.name for c in certificate.failures()] == ['rejection']


async def test_a_judge_that_fails_everything_fails_acceptance() -> None:
    certificate = await certify_judge(judge(no_man), CASES)
    assert certificate.verdict == 'INADMISSIBLE'
    assert status(certificate, 'acceptance') == 'FAIL'
    # It "passes" rejection and invariance, which is why neither alone is a certificate.
    assert status(certificate, 'rejection') == 'PASS'


async def test_a_whitespace_sensitive_judge_fails_invariance() -> None:
    certificate = await certify_judge(judge(exact), CASES)
    assert [c.name for c in certificate.failures()] == ['invariance']


async def test_a_coin_flip_judge_is_inadmissible() -> None:
    certificate = await certify_judge(judge(coin(seed=1)), CASES, repeats=3)
    assert certificate.verdict == 'INADMISSIBLE'
    assert status(certificate, 'stability') == 'FAIL'


async def test_the_failing_control_is_named() -> None:
    certificate = await certify_judge(judge(lenient_on_empty), CASES)
    rejection = next(c for c in certificate.checks if c.name == 'rejection')
    n = len(CASES)
    assert f'empty_output 0/{n}' in rejection.detail
    assert f'mismatched_output {n}/{n}' in rejection.detail


async def test_too_few_cases_is_unvalidated_not_admissible() -> None:
    certificate = await certify_judge(judge(oracle), CASES[:3])
    assert certificate.verdict == 'UNVALIDATED'


async def test_a_perfect_record_on_a_few_cases_is_not_yet_a_pass() -> None:
    """12/12 stable has a Wilson lower bound of 0.76: not evidence of failure, not yet of success."""
    certificate = await certify_judge(judge(oracle), CASES[:12])
    assert status(certificate, 'stability') == 'UNVALIDATED'
    assert certificate.verdict == 'UNVALIDATED'


async def test_human_agreement_counts_labelled_failures() -> None:
    labels = [HumanLabel(case.name, case.output, True) for case in CASES] + [
        HumanLabel(case.name, 'I do not know', False) for case in CASES
    ]
    good = await certify_judge(judge(oracle), CASES, human_labels=labels)
    assert status(good, 'human_agreement') == 'PASS'
    bad = await certify_judge(judge(yes_man), CASES, human_labels=labels)
    assert status(bad, 'human_agreement') == 'FAIL'


async def test_labels_of_one_value_cannot_certify_agreement() -> None:
    labels = [HumanLabel(case.name, case.output, True) for case in CASES]
    certificate = await certify_judge(judge(yes_man), CASES, human_labels=labels, controls=())
    assert status(certificate, 'human_agreement') == 'UNVALIDATED'


async def test_a_judge_that_errors_has_given_no_verdict() -> None:
    def broken(output: str, expected: str) -> bool:
        raise RuntimeError('judge model unavailable')

    certificate = await certify_judge(judge(broken), CASES)
    acceptance = next(c for c in certificate.checks if c.name == 'acceptance')
    assert acceptance.successes == 0 and acceptance.errors == acceptance.trials
    assert certificate.verdict == 'INADMISSIBLE'


async def test_certificates_reproduce_from_a_seed() -> None:
    first = await certify_judge(judge(oracle), CASES, seed=7)
    second = await certify_judge(judge(oracle), CASES, seed=7)
    assert [j.output for j in first.judgments] == [j.output for j in second.judgments]


def test_wilson_and_kappa() -> None:
    assert wilson(0, 0) == (0.0, 1.0)
    low, high = wilson(10, 10)
    assert 0.69 < low < 0.73 and high == 1.0
    assert cohen_kappa([(True, True), (False, False)]) == 1.0
    assert cohen_kappa([(True, True), (True, True)]) is None
    assert cohen_kappa([]) is None


@pytest.mark.parametrize('threshold', [0.5, 0.95])
async def test_thresholds_are_applied_to_the_interval_bound(threshold: float) -> None:
    certificate = await certify_judge(judge(oracle), CASES, thresholds=Thresholds(min_rejection=threshold))
    n = 2 * len(CASES)
    expected = 'PASS' if wilson(n, n)[0] >= threshold else 'UNVALIDATED'
    assert status(certificate, 'rejection') == expected


def test_a_mismatched_answer_is_never_one_that_is_also_right() -> None:
    """Small answer spaces: another case answering "yes" is not a wrong answer to a "yes" question."""
    import random

    from pydantic_evals_admissibility import MismatchedOutput

    cases = [
        JudgeCase(f'q{i}', f'question {i}', f'Sure. Answer: {a}', expected_output=a)
        for i, a in enumerate('yes no yes no yes no'.split())
    ]
    control = MismatchedOutput()
    for case in cases:
        for seed in range(20):
            donor = control.make(case, cases, random.Random(seed))
            assert donor is not None and donor.split()[-1] != case.expected_output
