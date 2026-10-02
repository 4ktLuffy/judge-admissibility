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


async def test_a_saved_certificate_can_be_rebuilt_and_diagnosed_later() -> None:
    import json

    from pydantic_evals_admissibility import Certificate, diagnose

    cert = await certify_judge(judge(yes_man), CASES, repeats=2)
    again = Certificate.from_dict(json.loads(json.dumps(cert.to_dict())))
    assert again == cert
    assert diagnose(again) == diagnose(cert)


async def test_a_blind_spot_hidden_by_the_overall_rate_fails_the_slice_check() -> None:
    """Found on the support task: two judges without reasoning were ADMISSIBLE overall and passed
    none of the correct "no" answers to return-window questions. Here, a judge wrong on every capital
    beginning with A (4 of 30 cases)."""
    from pydantic_evals_admissibility import diagnose

    def blind_to_a(output: str, expected: str) -> bool:
        return oracle(output, expected) and not output.strip().startswith('A')

    def first_letter(case: JudgeCase) -> str:
        return 'starts with A' if str(case.output).startswith('A') else 'other'

    overall = await certify_judge(judge(blind_to_a), CASES)
    assert overall.verdict == 'ADMISSIBLE', overall.table()  # 78/90 accepted: the overall rate hides it

    sliced = await certify_judge(judge(blind_to_a), CASES, slice_by=first_letter)
    assert sliced.verdict == 'INADMISSIBLE' and [c.name for c in sliced.failures()] == ['slices'], sliced.table()
    slices = next(c for c in sliced.checks if c.name == 'slices')
    assert slices.detail == 'worst first: starts with A 0/12, other 78/78', slices
    assert any('one kind of case (starts with A 0/12' in a for a in diagnose(sliced)), diagnose(sliced)

    sound = await certify_judge(judge(oracle), CASES, slice_by=first_letter)
    assert sound.verdict == 'ADMISSIBLE' and status(sound, 'slices') == 'PASS', sound.table()

    # Too few cases of that kind to show it is below the bar: suspicious, not proven.
    one_a = [c for c in CASES if c.output not in ('Athens', 'Ankara', 'Abuja')]  # Accra is left
    few = await certify_judge(judge(blind_to_a), one_a, repeats=1, slice_by=first_letter)
    assert status(few, 'slices') == 'UNVALIDATED', few.table()


async def test_recertify_decides_again_from_saved_judgments() -> None:
    from pydantic_evals_admissibility import recertify

    cert = await certify_judge(judge(oracle), CASES, repeats=2)
    assert recertify(cert).checks == cert.checks
    sliced = recertify(cert, slices={c.name: c.output[0] for c in CASES})
    assert 'slices' in [c.name for c in sliced.checks] and sliced.judgments == cert.judgments
