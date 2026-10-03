"""`Jury` and `compare_jury`: several judges vote, and the vote is certified as one judge."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import pytest
from judges import Decide, coin, exact, judge, lenient_on_empty, oracle, yes_man
from pydantic_ai.models.function import FunctionModel
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, LLMJudge

from pydantic_evals_admissibility import (
    Certificate,
    EmptyOutput,
    JudgeCase,
    MismatchedOutput,
    WhitespaceReformat,
    certify_judge,
    judge_identity,
)
from pydantic_evals_admissibility._certify import Judgment
from pydantic_evals_admissibility._identity import fingerprint
from pydantic_evals_admissibility._jury import (
    Jury,
    JuryUndecided,
    _accuracy,  # pyright: ignore[reportPrivateUsage]
    compare_jury,
    evidence_fingerprint,
    jury_certificate,
    jury_verdict,
    summarize_jury,
)
from pydantic_evals_admissibility._stats import clopper_pearson

CASES = [JudgeCase(f'q{i}', f'What is {i} + {i}?', f'{i} + {i} = {2 * i}.', f'= {2 * i}.') for i in range(20)]
CONTROLS = (MismatchedOutput(), EmptyOutput(), WhitespaceReformat())


def named(decide: Decide, name: str) -> LLMJudge:
    """A scripted judge whose model has a name of its own: its identity is that name, so a certificate can be reused."""
    scripted = judge(decide)
    assert isinstance(scripted.model, FunctionModel) and scripted.model.function is not None
    return dataclasses.replace(scripted, model=FunctionModel(scripted.model.function, model_name=name))


def fussy(output: str, expected: str) -> bool:
    """Right on the answers as written, but fails one whose whitespace was changed."""
    return oracle(output, expected) and output == output.strip() and '  ' not in output


@dataclass
class Broken(Evaluator[Any, Any, Any]):
    """A member whose every call errors, like a judge out of credits."""

    def evaluate(self, ctx: EvaluatorContext[Any, Any, Any]) -> bool:
        raise RuntimeError('out of credits')


async def test_majority_outvotes_one_noisy_member_and_says_it_hid_it() -> None:
    jury = Jury([judge(oracle), judge(oracle), judge(coin(3))])
    result = await compare_jury(jury, CASES, controls=CONTROLS, repeats=2)
    noisy = jury.labels[2]
    assert result.jury.verdict == 'ADMISSIBLE', result.table()
    assert result.members[noisy].verdict == 'INADMISSIBLE'
    assert result.accuracy['jury']['right'] == result.accuracy['jury']['judged']
    effect = result.effect[noisy]
    assert effect['fixed'] > 0 and effect['inherited'] == 0 and effect['introduced'] == 0
    assert not result.shared_errors
    assert any('majority hides it' in note and noisy in note for note in result.notes), result.notes
    # Reusing member verdicts: the jury's calls are its members' calls, made once.
    assert result.calls['jury'] == sum(result.calls[label] for label in jury.labels)
    assert 'cost in calls: 3.0x m1, 3.0x m2, 3.0x m3' in result.notes


async def test_a_shared_blind_spot_is_inherited() -> None:
    jury = Jury([judge(lenient_on_empty), judge(lenient_on_empty), judge(oracle)])
    result = await compare_jury(jury, CASES, controls=CONTROLS, repeats=1)
    rejection = next(c for c in result.jury.checks if c.name == 'rejection')
    assert result.jury.verdict == 'INADMISSIBLE' and 'empty_output 0/20 FAIL' in rejection.detail
    lenient = jury.labels[0]
    assert result.effect[lenient] == {'fixed': 0, 'inherited': 20, 'introduced': 0}
    # The sound member is outvoted on every empty answer: the jury introduces errors it did not make.
    assert result.effect[jury.labels[2]]['introduced'] == 20
    assert len(result.shared_errors) == 20 and {e['role'] for e in result.shared_errors} == {'must_fail:empty_output'}


async def test_unanimous_pass_is_strict_where_majority_is_not() -> None:
    members = [judge(oracle), judge(oracle), judge(fussy)]
    strict = await compare_jury(Jury(members, rule='unanimous_pass'), CASES, controls=CONTROLS, repeats=1)
    lenient = await compare_jury(Jury(members, rule='majority'), CASES, controls=CONTROLS, repeats=1)
    assert lenient.jury.verdict == 'ADMISSIBLE', lenient.table()
    invariance = next(c for c in strict.jury.checks if c.name == 'invariance')
    assert strict.jury.verdict == 'INADMISSIBLE' and invariance.successes == 0, strict.table()


def test_the_rules_count_errors_and_ties_as_documented() -> None:
    p, f, e = True, False, None
    assert jury_verdict([p, p, f], 'majority', quorum=2) is True
    assert jury_verdict([p, f, e], 'majority', quorum=2) is False  # tie fails by default
    assert jury_verdict([p, f, e], 'majority', quorum=2, tie='abstain') is None
    assert jury_verdict([p, f, e], 'majority', quorum=2, tie='pass') is True
    assert jury_verdict([p, e, e], 'majority', quorum=2) is None  # one vote is below quorum
    assert jury_verdict([p, p, e], 'any_fail', quorum=2) is True
    assert jury_verdict([p, p, e], 'unanimous_pass', quorum=2) is None  # a missing vote blocks a pass
    assert jury_verdict([f, e, e], 'unanimous_pass', quorum=2) is False  # one fail decides
    assert jury_verdict([f, e, e], 'any_fail', quorum=2) is False
    with pytest.raises(ValueError):
        Jury([judge(oracle)], quorum=2)


async def test_an_errored_member_is_not_a_vote() -> None:
    two_and_broken = Jury([judge(oracle), judge(oracle), Broken()])
    result = await compare_jury(two_and_broken, CASES, controls=CONTROLS, repeats=1)
    assert result.jury.verdict == 'ADMISSIBLE', result.table()
    assert result.members[two_and_broken.labels[2]].verdict == 'UNVALIDATED'
    assert all('m3 Broken: ERROR' in (j.reason or '') for j in result.jury.judgments)
    assert result.accuracy[two_and_broken.labels[2]]['judged'] == 0

    one_and_broken = Jury([judge(oracle), Broken(), Broken()])
    result = await compare_jury(one_and_broken, CASES, controls=CONTROLS, repeats=1)
    assert result.jury.verdict == 'UNVALIDATED'
    assert all(j.passed is None and (j.error or '').startswith('JuryUndecided') for j in result.jury.judgments)
    assert result.accuracy['jury']['undecided'] == 80 and result.accuracy['jury']['judged'] == 0


async def test_evaluate_raises_when_the_jury_has_no_verdict() -> None:
    from pydantic_evals.otel._errors import SpanTreeRecordingError

    ctx = EvaluatorContext[Any, Any, Any](
        name='q', inputs='What is 1 + 1?', metadata=None, expected_output='= 2.', output='1 + 1 = 2.', duration=0.0,
        _span_tree=SpanTreeRecordingError('none'), attributes={}, metrics={},
    )  # fmt: skip
    passed = await Jury([judge(oracle), judge(yes_man), Broken()]).evaluate(ctx)
    assert passed.value is True and passed.reason and passed.reason.startswith('[jury majority: PP?] PASS 2-0')
    with pytest.raises(JuryUndecided, match=r'\[jury unanimous_pass: PP\?\]'):
        await Jury([judge(oracle), judge(oracle), Broken()], rule='unanimous_pass').evaluate(ctx)


async def test_identity_changes_with_any_member_and_is_stable_otherwise() -> None:
    members = [judge(oracle), judge(oracle), judge(coin(1))]
    jury = Jury(members)
    same = Jury([judge(oracle), judge(oracle), judge(coin(1))])
    # Fresh FunctionModels with new function addresses: the same configuration, the same identity.
    assert fingerprint(judge_identity(jury)) == fingerprint(judge_identity(same))
    assert judge_identity(jury)['members'][0]['rubric'] == members[0].rubric

    changed = Jury([members[0], members[1], dataclasses.replace(members[2], include_input=True)])
    certificate = await compare_jury(jury, CASES, controls=CONTROLS, repeats=1)
    assert certificate.jury.covers(jury) and certificate.jury.covers(same)
    assert certificate.jury.differences(changed) == ['members']
    assert certificate.jury.differences(Jury(members, rule='unanimous_pass')) == ['rule']
    assert not certificate.jury.covers(Jury(members[:2]))


async def test_reusing_member_verdicts_matches_calling_the_jury() -> None:
    jury = Jury([judge(oracle), judge(lenient_on_empty), judge(exact)])
    reused = await compare_jury(jury, CASES, controls=CONTROLS, repeats=2)
    called = await compare_jury(jury, CASES, controls=CONTROLS, repeats=2, reuse=False)
    assert [(j.case, j.role, j.passed) for j in reused.jury.judgments] == [
        (j.case, j.role, j.passed) for j in called.jury.judgments
    ]
    assert reused.jury.verdict == called.jury.verdict and reused.effect == called.effect
    assert called.calls['jury'] == 3 * len(called.jury.judgments)


async def test_member_certificates_must_be_on_the_same_plan() -> None:
    jury = Jury([named(oracle, 'oracle'), named(oracle, 'oracle')])
    other_plan = await certify_judge(jury.members[0], CASES[:10], controls=CONTROLS, repeats=1)
    with pytest.raises(ValueError, match='not on this plan'):
        await compare_jury(jury, CASES, controls=CONTROLS, repeats=1, member_certificates=[other_plan, None])
    on_plan = await certify_judge(jury.members[0], CASES, controls=CONTROLS, repeats=1)
    with pytest.raises(ValueError, match='different plan'):
        jury_certificate(jury, [on_plan, other_plan])
    with pytest.raises(ValueError, match='another configuration'):
        await compare_jury(
            Jury([dataclasses.replace(jury.members[0], include_input=True), jury.members[1]]), CASES,
            controls=CONTROLS, repeats=1, member_certificates=[on_plan, None],
        )  # fmt: skip


async def test_a_reused_certificate_must_cover_its_member() -> None:
    """Review P1: an oracle's certificate was accepted as the evidence for a yes-man jury."""
    oracle_cert = await certify_judge(named(oracle, 'oracle'), CASES, controls=CONTROLS, repeats=1)
    with pytest.raises(ValueError, match='does not cover member 1'):
        jury_certificate(Jury([named(yes_man, 'yes-man')]), [oracle_cert])
    anonymous = dataclasses.replace(oracle_cert, identity=None)
    with pytest.raises(ValueError, match='records no judge identity'):
        jury_certificate(Jury([named(oracle, 'oracle')]), [anonymous])
    assert jury_certificate(Jury([named(oracle, 'oracle')]), [oracle_cert]).verdict == 'ADMISSIBLE'


def factory(verdicts: list[bool]) -> LLMJudge:
    """Judges built by one factory, differing only in a list they close over: their identities are the same."""
    return judge(lambda output, expected: verdicts[0])


async def test_a_member_whose_identity_is_not_reliable_cannot_reuse_a_certificate() -> None:
    """Review P1: an oracle's certificate transferred to an always-pass member sharing its (unreliable) identity."""
    failing, passing = factory([False]), factory([True])
    assert judge_identity(failing) == judge_identity(passing)
    certificate = await certify_judge(failing, CASES, controls=CONTROLS, repeats=1)
    with pytest.raises(ValueError, match='identity that does not hold everything'):
        jury_certificate(Jury([passing]), [certificate])
    made_on = evidence_fingerprint(CASES, controls=CONTROLS, repeats=1)
    with pytest.raises(ValueError, match='identity that does not hold everything'):
        await compare_jury(
            Jury([passing]), CASES, controls=CONTROLS, repeats=1, member_certificates=[certificate],
            member_plans=[made_on],
        )  # fmt: skip
    # Certified here, by the member itself, its verdicts are its own: the comparison runs.
    fresh = await compare_jury(Jury([passing]), CASES, controls=CONTROLS, repeats=1)
    assert fresh.members[Jury([passing]).labels[0]].verdict == 'INADMISSIBLE'


def test_the_evidence_fingerprint_keeps_order_and_type() -> None:
    """Review P1: `{'a': 1, 'b': 2}` and `{'b': 2, 'a': 1}` fingerprinted the same; a judge sees them in order."""

    def with_inputs(inputs: object) -> list[JudgeCase]:
        return [dataclasses.replace(c, inputs=inputs) for c in CASES]

    def made_on(inputs: object) -> str:
        return evidence_fingerprint(with_inputs(inputs), controls=CONTROLS, repeats=1)

    assert made_on({'a': 1, 'b': 2}) == made_on({'a': 1, 'b': 2})
    assert made_on({'a': 1, 'b': 2}) != made_on({'b': 2, 'a': 1})
    assert made_on({'a': 1}) != made_on({'a': 1.0}) != made_on({'a': True})
    assert made_on([1, 2]) != made_on((1, 2))
    assert made_on({1, 2, 3}) == made_on({3, 2, 1})  # a set has no order to keep


async def test_an_order_sensitive_judge_cannot_reuse_a_certificate_made_on_reordered_inputs() -> None:
    """Review P1: reused, the jury stayed ADMISSIBLE; certified fresh on the reordered inputs, INADMISSIBLE."""

    ordered = [dataclasses.replace(c, inputs={'a': i, 'b': 2 * i}) for i, c in enumerate(CASES)]
    reordered = [dataclasses.replace(c, inputs={'b': 2 * i, 'a': i}) for i, c in enumerate(CASES)]
    # Shown the inputs, a judge reads their items in the order the prompt lists them: other evidence.
    member = dataclasses.replace(named(oracle, 'oracle'), include_input=True)
    certificate = await certify_judge(member, ordered, controls=CONTROLS, repeats=1)
    made_on = evidence_fingerprint(ordered, controls=CONTROLS, repeats=1)
    with pytest.raises(ValueError, match='made on other evidence'):
        await compare_jury(
            Jury([member]), reordered, controls=CONTROLS, repeats=1, member_certificates=[certificate],
            member_plans=[made_on],
        )  # fmt: skip


async def test_reuse_refuses_certificates_made_on_other_evidence() -> None:
    """Review P1: same case names and outputs, other inputs and expected answers, reused as ADMISSIBLE."""
    jury = Jury([named(oracle, 'oracle'), named(oracle, 'oracle')])
    certs = [await certify_judge(member, CASES, controls=CONTROLS, repeats=1) for member in jury.members]
    made_on = evidence_fingerprint(CASES, controls=CONTROLS, repeats=1)
    changed = [
        dataclasses.replace(c, inputs=f'{c.inputs} Answer in words.', expected_output=f'{c.expected_output} In words.')
        for c in CASES
    ]
    with pytest.raises(ValueError, match='made on other evidence'):
        await compare_jury(
            jury, changed, controls=CONTROLS, repeats=1, member_certificates=certs, member_plans=[made_on, made_on]
        )
    with pytest.raises(ValueError, match='member_plans'):
        await compare_jury(jury, changed, controls=CONTROLS, repeats=1, member_certificates=certs)
    same = await compare_jury(
        jury, CASES, controls=CONTROLS, repeats=1, member_certificates=certs, member_plans=[made_on, made_on]
    )
    assert same.jury.verdict == 'ADMISSIBLE' and same.evidence == made_on
    # The plan is compared in order, judgment for judgment, not as a set of (case, role).
    shuffled = dataclasses.replace(certs[1], judgments=tuple(reversed(certs[1].judgments)))
    with pytest.raises(ValueError, match='different plan'):
        jury_certificate(jury, [certs[0], shuffled])
    with pytest.raises(ValueError, match='not on this plan'):
        await compare_jury(
            jury, CASES, controls=CONTROLS, repeats=1, member_certificates=[certs[0], shuffled],
            member_plans=[made_on, made_on],
        )  # fmt: skip


async def test_a_jury_of_sequential_certificates_keeps_their_looks() -> None:
    jury = Jury([named(oracle, 'oracle'), named(oracle, 'oracle')])
    certs = [await certify_judge(m, CASES, controls=CONTROLS, repeats=1, batch_size=5) for m in jury.members]
    assert certs[0].looks == 4
    assert jury_certificate(jury, certs).looks == 4


def _hand_made(member: Any, wrong_case: str | None) -> Certificate:
    """Three cases, six judgments each with a known right verdict; wrong on every one of `wrong_case`."""
    roles = ['reference#0', 'must_fail:a', 'must_fail:b', 'must_fail:c', 'must_hold:d', 'must_hold:e']
    judgments = []
    for case in ('c0', 'c1', 'c2'):
        for role in roles:
            right = not role.startswith('must_fail')
            passed = (not right) if case == wrong_case else right
            judgments.append(Judgment(case, role, f'{case} {role}', passed, 'because'))
    return Certificate('UNVALIDATED', (), tuple(judgments), calls=18, planned=18, identity=judge_identity(member))


def test_the_comparison_counts_cases_not_judgments() -> None:
    """Review P2: six wins on one case of three were reported 'better than' at McNemar p=0.03."""
    members = [named(coin(1), 'coin-1'), named(oracle, 'oracle'), named(exact, 'exact')]
    jury = Jury(members)
    certs = [_hand_made(members[0], 'c0'), _hand_made(members[1], 'c1'), _hand_made(members[2], None)]
    result = summarize_jury(jury, jury_certificate(jury, certs), certs, reused=True)
    weak = jury.labels[0]
    assert not any('better than' in note for note in result.notes), result.notes
    versus = result.against_members[weak]
    assert (versus['jury_right_member_wrong'], versus['member_right_jury_wrong']) == (6, 0)
    assert versus['cases'] == 3 and versus['decision'] == 'INCONCLUSIVE', versus
    assert versus['level'] == pytest.approx(0.05 / 3)  # each member is a comparison: the level is split
    assert result.accuracy[weak]['per_case'] == {'c0': [0, 6], 'c1': [6, 6], 'c2': [6, 6]}
    assert result.accuracy[weak]['cases'] == 3 and result.accuracy[weak]['cases_all_right'] == 2


def test_the_accuracy_interval_is_exact_when_resampling_cases_cannot_show_uncertainty() -> None:
    """Review P2: three cases, all right, reported [1, 1]: every resample of identical cases is the same."""
    few = _accuracy({'a': [True] * 6, 'b': [True] * 6, 'c': [True] * 6})
    low, high = few['interval']
    assert high == 1.0 and low == pytest.approx(0.025 ** (1 / 3), abs=1e-6)  # Clopper-Pearson, 3 of 3 cases
    assert few['interval_method'].startswith('exact')
    uniform = _accuracy({f'c{i}': [True, False] for i in range(30)})  # many cases, every one the same rate
    assert uniform['interval'] == pytest.approx(list(clopper_pearson(15, 30)))  # each case one trial at rate 0.5
    assert 'same rate' in uniform['interval_method']
    varied = _accuracy({f'c{i}': [True] * (6 - i % 3) + [False] * (i % 3) for i in range(30)})
    assert varied['interval_method'].startswith('cases resampled')
    assert varied['interval'][0] < varied['rate'] < varied['interval'][1]
