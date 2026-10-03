"""`qualify`: one rule per concern, each shown to refuse what it is for and to allow what it is not."""

from __future__ import annotations

import dataclasses
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from judges import judge, oracle
from pydantic_evals.evaluators import LLMJudge
from test_certify import CASES
from test_evidence import grades_the_claim, grades_the_work, scripted

from pydantic_evals_admissibility import Certificate, Check, certify_judge, judge_identity
from pydantic_evals_admissibility._qualify import UnqualifiedJudge, evidence_families_of, qualify

sys.path.insert(0, str(Path(__file__).parent.parent / 'bench'))
from evidence_task import CONTROLS, RUBRIC, episodes  # noqa: E402

IDENTITY = {'evaluator': 'LLMJudge', 'rubric': 'r', 'include_input': True, 'assertion': {'include_reason': True}}


def check(name: str, status: str = 'PASS', detail: str = '', threshold: float = 0.7) -> Check:
    return Check(name, status, 10, 10, (0.72, 1.0), threshold, detail)  # type: ignore[arg-type]


def cert(*checks: Check, verdict: str = 'ADMISSIBLE', identity: dict | None = IDENTITY) -> Certificate:  # type: ignore[type-arg]
    checks = checks or (check('acceptance'), check('rejection', detail='empty_output 10/10'), check('invariance'))
    return Certificate(verdict, checks, (), 'LLMJudge(test)', identity=identity)  # type: ignore[arg-type]


def rule(q, name):  # type: ignore[no-untyped-def]
    return next(r for r in q.rules if r.name == name)


async def test_coverage_refuses_a_certificate_for_another_configuration() -> None:
    sound = judge(oracle)
    certificate = await certify_judge(sound, CASES)
    assert qualify(sound, certificate, decision='gate').qualified
    other = dataclasses.replace(sound, rubric='The answer is polite.')
    for decision in ('report', 'gate', 'promote', 'steer'):
        q = qualify(other, certificate, decision=decision)  # type: ignore[arg-type]
        assert not q.qualified and 'rubric' in rule(q, 'coverage').detail
    # A harness can pass the identity it logged instead of the judge object.
    assert qualify(judge_identity(sound), certificate, decision='gate').qualified


def test_a_certificate_without_identity_is_enough_for_a_report_only() -> None:
    old = cert(identity=None)
    assert rule(qualify(None, old, decision='report'), 'coverage').outcome == 'warn'
    assert qualify(None, old, decision='report').qualified
    assert not qualify(None, old, decision='gate').qualified


def test_report_accepts_unvalidated_with_a_warning_and_nothing_else_does() -> None:
    unvalidated = cert(check('acceptance', 'UNVALIDATED'), check('rejection'), verdict='UNVALIDATED')
    report = qualify(IDENTITY, unvalidated, decision='report')
    assert report.qualified and any('UNVALIDATED' in w for w in report.warnings)
    for decision in ('gate', 'promote', 'steer'):
        assert not qualify(IDENTITY, unvalidated, decision=decision).qualified  # type: ignore[arg-type]
    inadmissible = cert(check('acceptance', 'FAIL'), check('rejection'), verdict='INADMISSIBLE')
    assert not qualify(IDENTITY, inadmissible, decision='report').qualified


def test_gate_needs_both_directions_even_when_the_verdict_says_admissible() -> None:
    """A hand-built or old certificate can say ADMISSIBLE without a rejection check."""
    acceptance_only = cert(check('acceptance'), check('invariance'))
    q = qualify(IDENTITY, acceptance_only, decision='gate')
    assert not q.qualified and 'missing rejection' in rule(q, 'checks').detail
    pairwise = cert(check('accuracy'), check('order_consistency'))
    assert qualify(IDENTITY, pairwise, decision='gate').qualified


def test_promote_refuses_a_failing_slice_and_warns_without_slices() -> None:
    base = (check('acceptance'), check('rejection'))
    blind = cert(*base, check('slices', 'FAIL', 'worst first: refund:no 0/5, other 70/75'))
    assert not qualify(IDENTITY, blind, decision='promote').qualified
    assert qualify(IDENTITY, blind, decision='gate').qualified  # the slice rule is promote's
    unsliced = qualify(IDENTITY, cert(*base), decision='promote')
    assert unsliced.qualified and rule(unsliced, 'slices').outcome == 'warn'


def test_a_context_slice_must_have_been_measured_and_above_the_bar() -> None:
    slices = check('slices', 'UNVALIDATED', 'worst first: refund:no 3/5, shipping 18/20; refund:no is at or below ...')
    c = cert(check('acceptance'), check('rejection'), slices)
    assert qualify(IDENTITY, c, decision='gate', context={'slice': 'shipping'}).qualified
    low = qualify(IDENTITY, c, decision='gate', context={'slice': 'refund:no'})
    assert not low.qualified and '3/5' in low.reasons[0]
    assert not qualify(IDENTITY, c, decision='gate', context={'slice': 'returns'}).qualified
    assert qualify(IDENTITY, c, decision='report', context={'slice': 'returns'}).qualified  # a warning only


async def test_steering_on_traces_needs_evidence_controls_that_pass() -> None:
    work = LLMJudge(rubric=RUBRIC, model=scripted(grades_the_work), include_input=True)
    good = await certify_judge(work, episodes(40), controls=CONTROLS, repeats=1)
    assert evidence_families_of(good) == {'tool_failed', 'amount_differs'}
    assert qualify(work, good, decision='steer', context={'sees_traces': True}).qualified

    claim = LLMJudge(rubric=RUBRIC, model=scripted(grades_the_claim), include_input=True)
    bad = await certify_judge(claim, episodes(40), controls=CONTROLS, repeats=1)
    q = qualify(claim, bad, decision='steer', context={'sees_traces': True})
    assert not q.qualified and 'tool_failed' in rule(q, 'evidence').detail

    # Admissible on output controls alone is not enough to read traces and steer.
    plain = cert()
    assert qualify(IDENTITY, plain, decision='steer').qualified
    q = qualify(IDENTITY, plain, decision='steer', context={'sees_traces': True})
    assert not q.qualified and 'no evidence controls' in rule(q, 'evidence').detail
    blind = {**IDENTITY, 'include_input': False}
    q = qualify(blind, cert(identity=blind), decision='steer', context={'sees_traces': True}, evidence_families=['x'])
    assert 'include_input=False' in rule(q, 'evidence').detail


def test_steering_without_reasons_is_a_warning() -> None:
    bare = {**IDENTITY, 'assertion': {'include_reason': False}}
    q = qualify(bare, cert(identity=bare), decision='steer')
    assert q.qualified and rule(q, 'reasons').outcome == 'warn'


def test_freshness() -> None:
    now = datetime(2026, 10, 2, tzinfo=timezone.utc)
    c = cert()
    assert qualify(IDENTITY, c, decision='gate', max_age_days=30, issued_at=now - timedelta(days=3), now=now).qualified
    old = qualify(IDENTITY, c, decision='gate', max_age_days=30, issued_at='2026-07-01T00:00:00+00:00', now=now)
    assert not old.qualified and 'older than 30' in old.reasons[0]
    assert not qualify(IDENTITY, c, decision='gate', max_age_days=30, now=now).qualified  # age unknown
    assert qualify(IDENTITY, c, decision='gate', now=now).qualified  # no limit asked for


def test_the_record_is_machine_readable_and_the_raise_lists_every_reason() -> None:
    c = cert(check('acceptance', 'FAIL'), verdict='INADMISSIBLE', identity=None)
    q = qualify(None, c, decision='promote', context={'region': 'eu'})
    record = q.to_dict()
    assert record['qualified'] is False and record['decision'] == 'promote'
    assert {r['name'] for r in record['rules']} >= {'coverage', 'verdict', 'checks', 'slices', 'context'}
    assert any('region' in w for w in record['warnings'])  # an unknown key is not silently trusted
    with pytest.raises(UnqualifiedJudge) as raised:
        q.raise_unless_qualified()
    assert all(reason in str(raised.value) for reason in q.reasons)
    with pytest.raises(ValueError):
        qualify(None, c, decision='deploy')  # type: ignore[arg-type]


def test_family_statuses_come_from_the_check_not_its_prose() -> None:
    from pydantic_evals_admissibility._certify import FamilyResult

    families = (FamilyResult('policy-change', 0, 30, 0, 'FAIL'), FamilyResult('tool.result:changed', 30, 30, 0, 'PASS'))
    misleading = 'policy-change 30/30, tool.result:changed 30/30'  # the structured record wins over the detail
    rejection = Check('rejection', 'FAIL', 30, 60, (0.4, 0.6), 0.8, misleading, families=families)
    c = cert(check('acceptance'), rejection, check('invariance'), verdict='INADMISSIBLE')
    q = qualify(IDENTITY, c, decision='steer', context={'sees_traces': True}, evidence_families=['policy-change'])
    assert rule(q, 'evidence').detail == 'evidence controls not passed: policy-change FAIL'
    old = Check('rejection', 'FAIL', 30, 60, (0.4, 0.6), 0.8, 'policy-change 0/30 FAIL, tool.result:changed 30/30')
    c = cert(check('acceptance'), old, check('invariance'), verdict='INADMISSIBLE')
    q = qualify(IDENTITY, c, decision='steer', context={'sees_traces': True}, evidence_families=['tool.result:changed'])
    assert rule(q, 'evidence').outcome == 'ok'


async def test_a_judge_that_cannot_be_identified_reliably_is_warned_about_not_refused() -> None:
    """Certified just now from this object, coverage holds; the warning says not to reuse it elsewhere."""
    from judges import judge, oracle
    from test_certify import CASES

    from pydantic_evals_admissibility import certify_judge, qualify

    unnamed = judge(oracle, named=False)
    certificate = await certify_judge(unnamed, CASES)
    result = qualify(unnamed, certificate, decision='gate')
    assert result.qualified, result.reasons
    assert any('cannot be identified reliably' in w for w in result.warnings), result.warnings
