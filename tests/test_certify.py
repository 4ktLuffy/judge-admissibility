"""The certificate's controls: it must certify a sound judge and refuse each kind of broken one."""

from __future__ import annotations

import pytest
from judges import coin, exact, judge, lenient_on_empty, no_man, oracle, yes_man

from pydantic_evals_admissibility import HumanLabel, JudgeCase, Thresholds, certify_judge, cohen_kappa, wilson
from pydantic_evals_admissibility._certify import Judgment

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
    assert acceptance.successes == acceptance.trials == 0 and acceptance.errors == len(CASES)
    # No verdicts is no evidence either way: not a pass, and not proof of a bad judge.
    assert acceptance.status == 'UNVALIDATED' and 'errored' in acceptance.detail
    assert certificate.verdict == 'UNVALIDATED'


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
    none of the correct "no" answers to return-window questions. Here, a judge wrong on every
    "refund" question, 8 of 60 cases."""
    from pydantic_evals_admissibility import diagnose

    cases = [
        JudgeCase(
            f'{"refund" if i < 8 else "other"} {i}', f'Question {i}?', f'answer {i}', expected_output=f'answer {i}'
        )
        for i in range(60)
    ]
    kinds = {c.name: c.name.split()[0] for c in cases}

    def blind_to_refunds(output: str, expected: str) -> bool:
        return oracle(output, expected) and int(expected.split()[-1]) >= 8

    def kind(case: JudgeCase) -> str:
        return kinds[case.name]

    overall = await certify_judge(judge(blind_to_refunds), cases)
    assert overall.verdict == 'ADMISSIBLE', overall.table()  # 52/60 accepted: the overall rate hides it

    sliced = await certify_judge(judge(blind_to_refunds), cases, slice_by=kind)
    assert sliced.verdict == 'INADMISSIBLE' and [c.name for c in sliced.failures()] == ['slices'], sliced.table()
    slices = next(c for c in sliced.checks if c.name == 'slices')
    assert slices.detail == 'worst first: refund 0/8, other 52/52', slices  # one judgment per case, not per repeat
    assert any('one kind of case (refund 0/8' in a for a in diagnose(sliced)), diagnose(sliced)

    sound = await certify_judge(judge(oracle), cases, slice_by=kind)
    assert sound.verdict == 'ADMISSIBLE' and status(sound, 'slices') == 'PASS', sound.table()

    # Repeats are not new cases: three refunds judged ten times each still cannot show a blind spot.
    few = await certify_judge(judge(blind_to_refunds), cases[5:], repeats=10, slice_by=kind)
    assert status(few, 'slices') == 'UNVALIDATED', few.table()


async def test_recertify_decides_again_from_saved_judgments() -> None:
    from pydantic_evals_admissibility import recertify

    cert = await certify_judge(judge(oracle), CASES, repeats=2)
    assert recertify(cert).checks == cert.checks
    sliced = recertify(cert, slices={c.name: c.output[0] for c in CASES})
    assert 'slices' in [c.name for c in sliced.checks] and sliced.judgments == cert.judgments


async def test_repeats_are_not_new_cases() -> None:
    """Review finding: ten repeats of 20 cases turned 16/20 (UNVALIDATED) into 160/200 (ADMISSIBLE)."""

    def misses_four(output: str, expected: str) -> bool:
        return oracle(output, expected) and expected not in {c.expected_output for c in CASES[:4]}

    once = await certify_judge(judge(misses_four), CASES[:20], repeats=1)
    many = await certify_judge(judge(misses_four), CASES[:20], repeats=10)
    for cert in (once, many):
        acceptance = next(c for c in cert.checks if c.name == 'acceptance')
        assert (acceptance.successes, acceptance.trials) == (16, 20), cert.table()
        assert acceptance.status == 'UNVALIDATED'


async def test_one_control_family_that_always_fails_is_not_averaged_away() -> None:
    """Review finding: nine sound families and one broken one pooled to 270/300 and passed."""
    from pydantic_evals_admissibility import Rewrite

    def accepts_shouting(output: str, expected: str) -> bool:
        return oracle(output, expected) or output.isupper()

    controls = (
        *(Rewrite(lambda o, i=i: f'not {o} {i}', f'wrong_{i}', 'must_fail') for i in range(9)),
        Rewrite(lambda o: f'{o} IS WRONG'.upper() + 'X', 'shouted', 'must_fail'),
    )
    cert = await certify_judge(judge(accepts_shouting), CASES, controls=controls)
    rejection = next(c for c in cert.checks if c.name == 'rejection')
    assert rejection.status == 'FAIL' and 'shouted 0/30 FAIL' in rejection.detail, cert.table()


async def test_a_slice_whose_judgments_all_errored_does_not_vanish() -> None:
    def errors_on_paris(output: str, expected: str) -> bool:
        if expected == 'Paris':
            raise RuntimeError('timeout')
        return oracle(output, expected)

    cert = await certify_judge(
        judge(errors_on_paris), CASES, slice_by=lambda c: 'paris' if c.output == 'Paris' else 'rest'
    )
    slices = next(c for c in cert.checks if c.name == 'slices')
    assert slices.status == 'UNVALIDATED' and 'paris 0/0' in slices.detail and slices.errors == 1, cert.table()


async def test_a_sequential_certificate_keeps_its_looks_when_decided_again() -> None:
    from pydantic_evals_admissibility import Certificate, recertify

    cert = await certify_judge(judge(oracle), CASES, batch_size=10)
    assert cert.looks == 3
    again = recertify(Certificate.from_dict(cert.to_dict()))
    assert again.looks == 3 and again.checks == cert.checks


async def test_agreement_with_many_errored_labels_is_not_a_pass() -> None:
    """Review finding: PASS at kappa 1.0 on ten labels while 90 other labelled judgments errored."""

    def errors_on_most(output: str, expected: str) -> bool:
        if expected not in {c.expected_output for c in CASES[:10]}:
            raise RuntimeError('timeout')
        return oracle(output, expected)

    labels = [HumanLabel(c.name, c.output, True) for c in CASES] + [HumanLabel(c.name, 'wrong', False) for c in CASES]
    cert = await certify_judge(judge(errors_on_most), CASES, human_labels=labels, controls=())
    agreement = next(c for c in cert.checks if c.name == 'human_agreement')
    assert agreement.status == 'UNVALIDATED' and 'errored' in agreement.detail, cert.table()


async def test_control_families_are_recorded_on_the_check_and_saved() -> None:
    from pydantic_evals_admissibility import Certificate, Rewrite

    controls = (Rewrite(lambda o: f'{o}, no longer, after the policy change', 'policy-change', 'must_fail'),)
    cert = await certify_judge(judge(oracle), CASES, controls=controls)
    rejection = next(c for c in cert.checks if c.name == 'rejection')
    [family] = rejection.families
    assert (family.name, family.successes, family.trials, family.errors, family.status) == (
        'policy-change', 0, 30, 0, 'FAIL'
    )  # fmt: skip
    assert Certificate.from_dict(cert.to_dict()).checks == cert.checks
    old = cert.to_dict()
    for c in old['checks']:
        c.pop('families', None)
    assert all(c.families == () for c in Certificate.from_dict(old).checks)  # files saved before still load
    assert 'families' not in next(c for c in cert.to_dict()['checks'] if c['name'] == 'acceptance')


def _labelled(case_and_labels: list[tuple[str, bool, bool]]) -> list[Judgment]:
    """Judgments of human-labelled outputs: (case, the human's label, the judge's verdict)."""
    from pydantic_evals_admissibility._certify import Judgment

    return [Judgment(case, f'human:{int(human)}', 'out', verdict, None) for case, human, verdict in case_and_labels]


def test_human_agreement_is_unvalidated_when_humans_used_one_label() -> None:
    """Found in the eighth review: with every human label `fail`, kappa is 0 for any judge, so a
    judge agreeing on 11 of 12 was INADMISSIBLE. Kappa cannot measure agreement there."""
    from pydantic_evals_admissibility._certify import _agreement  # pyright: ignore[reportPrivateUsage]

    labels = _labelled([(f'c{i}', False, i == 0) for i in range(12)])
    from pydantic_evals_admissibility._certify import DEFAULT_THRESHOLDS

    check = _agreement(labels, DEFAULT_THRESHOLDS, 0.0125)
    assert check.status == 'UNVALIDATED' and 'every output fail' in check.detail and '11/12' in check.detail


def test_human_agreement_counts_cases_not_labels() -> None:
    """Found in the eighth review: twelve labels on one case passed, on an interval of [1, 1]."""
    from pydantic_evals_admissibility._certify import (
        DEFAULT_THRESHOLDS,
        _agreement,  # pyright: ignore[reportPrivateUsage]
    )

    one_case = _labelled([('c0', i % 2 == 0, i % 2 == 0) for i in range(12)])
    check = _agreement(one_case, DEFAULT_THRESHOLDS, 0.0125)
    assert check.status == 'UNVALIDATED' and 'labels on 1 cases' in check.detail
    many = _labelled([(f'c{i}', i % 2 == 0, i % 2 == 0) for i in range(12)])
    assert _agreement(many, DEFAULT_THRESHOLDS, 0.0125).status == 'PASS'


def test_a_slice_with_many_errors_is_unvalidated() -> None:
    """Found in the eighth review: half of one slice errored and the slice check still passed."""
    from pydantic_evals_admissibility._certify import Judgment, _slice_check  # pyright: ignore[reportPrivateUsage]

    first = [Judgment(f'o{i}', 'reference#0', 'out', True, None) for i in range(90)]
    first += [Judgment(f'r{i}', 'reference#0', 'out', None if i < 5 else True,
                       error='context length' if i < 5 else None) for i in range(10)]  # fmt: skip
    slices = {f'o{i}': 'other' for i in range(90)} | {f'r{i}': 'refund' for i in range(10)}
    check = _slice_check(first, slices, ['other', 'refund'], 0.7, 0.0125)
    assert check.status == 'UNVALIDATED' and 'refund: 5 of 10 judgments errored' in check.detail


def test_a_row_below_the_bar_says_so() -> None:
    from pydantic_evals_admissibility._certify import _check  # pyright: ignore[reportPrivateUsage]

    check = _check('acceptance', 15, 30, 0.7, 10, z_fail=2.5)
    assert check.status == 'UNVALIDATED' and 'below the bar' in check.detail


async def test_recertifying_a_certificate_saved_without_judgments_raises() -> None:
    """Found in the eighth review: it silently turned ADMISSIBLE into UNVALIDATED (\"no controls\")."""
    from judges import judge, oracle

    from pydantic_evals_admissibility import Certificate, certify_judge, recertify

    certificate = await certify_judge(judge(oracle), CASES)
    saved = Certificate.from_dict(certificate.to_dict(judgments=False))
    with pytest.raises(ValueError, match='saved without its judgments'):
        recertify(saved)


def test_perfect_agreement_on_few_cases_does_not_certify_a_high_kappa() -> None:
    """Found in review: ten perfectly agreeing labels gave a kappa interval of [1, 1], so a judge
    from a population with kappa 0.8 would be certified at kappa >= 0.99 about a third of the time."""
    from pydantic_evals_admissibility._certify import _agreement  # pyright: ignore[reportPrivateUsage]

    labels = _labelled([(f'c{i}', i % 2 == 0, i % 2 == 0) for i in range(10)])
    check = _agreement(labels, Thresholds(min_kappa=0.99), 0.0125)
    assert check.status == 'UNVALIDATED' and check.interval[0] < 0.5, check
    # Many cases still certify, and the interval is no longer a point.
    many = _labelled([(f'c{i}', i % 2 == 0, i % 2 == 0) for i in range(80)])
    check = _agreement(many, Thresholds(min_kappa=0.4), 0.0125)
    assert check.status == 'PASS' and check.interval[0] < 1.0
