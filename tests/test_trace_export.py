"""A certificate exported to Logfire: the span tree, its attributes, its caps, and the judge calls under it."""

from __future__ import annotations

import dataclasses
from typing import Any

import logfire
import pytest
from judges import exact, judge, lenient_on_empty, oracle, yes_man
from logfire.testing import CaptureLogfire
from pydantic_ai.models.function import FunctionModel

from pydantic_evals_admissibility import JudgeCase, certify_judge
from pydantic_evals_admissibility._certify import Certificate
from pydantic_evals_admissibility._trace_export import certificate_attributes, certify_judge_traced, log_certificate

CASES = [
    JudgeCase(
        name=f'case-{i:02d}', inputs=f'Which code is item {i}?', output=f'item-{i:02d}', expected_output=f'item-{i:02d}'
    )
    for i in range(20)
]


def spans(capfire: CaptureLogfire) -> list[dict[str, Any]]:
    return capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)


def named(capfire: CaptureLogfire, template: str) -> list[dict[str, Any]]:
    return [s for s in spans(capfire) if s['attributes'].get('logfire.msg_template') == template]


def children(capfire: CaptureLogfire, parent: dict[str, Any]) -> list[dict[str, Any]]:
    return [s for s in spans(capfire) if s['parent'] and s['parent']['span_id'] == parent['context']['span_id']]


async def test_a_certificate_is_one_span_with_one_child_per_check(capfire: CaptureLogfire) -> None:
    certificate = await certify_judge(judge(oracle), CASES)
    log_certificate(certificate, dataset='items')
    [root] = named(capfire, 'judge certificate {judge}: {verdict}')
    attrs = root['attributes']
    assert root['name'] == 'judge certificate {judge}: {verdict}'
    assert attrs['logfire.msg'] == f'judge certificate {certificate.judge}: ADMISSIBLE'
    assert attrs['verdict'] == 'ADMISSIBLE' and attrs['admissible'] is True
    assert attrs['fingerprint'] == certificate.fingerprint
    assert attrs['dataset'] == 'items' and attrs['calls'] == certificate.calls and attrs['looks'] == 1
    assert attrs['judge.evaluator'] == 'LLMJudge' and attrs['judge.include_expected_output'] is True
    assert attrs['failed_checks'] == []
    for check in certificate.checks:
        assert attrs[f'check.{check.name}.status'] == check.status
        assert attrs[f'check.{check.name}.low'] == check.interval[0]
        assert attrs[f'check.{check.name}.trials'] == check.trials
    kids = children(capfire, root)
    assert [k['attributes']['check'] for k in kids] == [c.name for c in certificate.checks]
    assert all(k['name'] == 'check {check}: {status}' for k in kids)
    assert not named(capfire, '{check} {family}: {wrong} of {judged} judged wrongly')  # nothing failed


async def test_a_failed_family_leads_to_the_judgments_that_failed_it(capfire: CaptureLogfire) -> None:
    certificate = await certify_judge(judge(lenient_on_empty), CASES)
    log_certificate(certificate)
    [root] = named(capfire, 'judge certificate {judge}: {verdict}')
    assert root['attributes']['verdict'] == 'INADMISSIBLE'
    assert root['attributes']['failed_checks'] == ['rejection']
    assert root['attributes']['logfire.level_num'] == 13  # warn
    [rejection] = [k for k in children(capfire, root) if k['attributes']['check'] == 'rejection']
    families = {f['family']: f for f in rejection['attributes']['families']}
    assert families['empty_output'] == {'family': 'empty_output', 'successes': 0, 'trials': 20, 'status': 'FAIL'}
    assert families['mismatched_output']['status'] == 'PASS'
    [event] = children(capfire, rejection)
    attrs = event['attributes']
    assert attrs['logfire.msg'] == 'rejection empty_output: 20 of 20 judged wrongly'
    assert attrs['examples_omitted'] == 15 and len(attrs['examples']) == 5
    example = attrs['examples'][0]
    assert example['role'] == 'must_fail:empty_output' and example['passed'] is True
    assert example['reason'] == 'scripted' and example['case'] == 'case-00'


async def test_a_failed_invariance_family_shows_verdicts_that_moved(capfire: CaptureLogfire) -> None:
    certificate = await certify_judge(judge(exact), CASES)
    log_certificate(certificate, max_examples=2)
    [event] = named(capfire, '{check} {family}: {wrong} of {judged} judged wrongly')
    attrs = event['attributes']
    assert attrs['check'] == 'invariance' and attrs['wrong'] > 0
    assert len(attrs['examples']) == 2
    assert all(e['role'].startswith('must_hold:') and e['passed'] is False for e in attrs['examples'])


async def test_examples_and_text_are_capped(capfire: CaptureLogfire) -> None:
    certificate = await certify_judge(judge(yes_man), CASES)
    long = [dataclasses.replace(j, reason='x' * 5000, output='y' * 5000) for j in certificate.judgments]
    log_certificate(dataclasses.replace(certificate, judgments=tuple(long)), max_examples=1, text_limit=50)
    events = named(capfire, '{check} {family}: {wrong} of {judged} judged wrongly')
    assert {e['attributes']['family'] for e in events} == {'empty_output', 'mismatched_output'}
    for event in events:
        [example] = event['attributes']['examples']
        assert len(example['reason']) == 50 and len(example['output']) == 50
        assert event['attributes']['examples_omitted'] == event['attributes']['wrong'] - 1


async def test_a_certificate_saved_without_judgments_still_exports_its_failures(capfire: CaptureLogfire) -> None:
    certificate = await certify_judge(judge(yes_man), CASES)
    rebuilt = Certificate.from_dict(certificate.to_dict(judgments=False))
    log_certificate(rebuilt)
    events = named(capfire, '{check} {family}: {wrong} of {judged} judged wrongly')
    assert events and all(e['attributes']['judgments_saved'] is False for e in events)
    assert all(e['attributes']['examples'] == [] for e in events)


async def test_the_certificate_says_whether_it_covers_the_judge_now() -> None:
    certified = judge(oracle)
    certificate = await certify_judge(certified, CASES)
    assert certificate_attributes(certificate, judge=certified)['covers_judge'] is True
    edited = dataclasses.replace(certified, rubric='Be nice.')
    attrs = certificate_attributes(certificate, judge=edited)
    assert attrs['covers_judge'] is False and attrs['judge_changed'] == ['rubric']


def test_attributes_need_no_numbers_that_are_absent() -> None:
    attrs = certificate_attributes(Certificate('UNVALIDATED', (), ()))
    assert 'fingerprint' not in attrs and 'calls' not in attrs and 'dataset' not in attrs


def test_max_examples_must_not_be_negative(capfire: CaptureLogfire) -> None:
    with pytest.raises(ValueError):
        log_certificate(Certificate('UNVALIDATED', (), ()), max_examples=-1)


async def test_traced_certification_nests_the_judge_calls_and_the_certificate(capfire: CaptureLogfire) -> None:
    scored = judge(lenient_on_empty)
    assert isinstance(scored.model, FunctionModel)
    scored = dataclasses.replace(scored, model=logfire.instrument_pydantic_ai(scored.model))
    certificate = await certify_judge_traced(scored, CASES[:10], dataset='items', repeats=1)
    [outer] = named(capfire, 'certify judge {judge}')
    assert outer['attributes']['logfire.msg'] == f'certify judge {certificate.judge}: {certificate.verdict}'
    assert outer['attributes']['fingerprint'] == certificate.fingerprint
    assert outer['attributes']['verdict'] == certificate.verdict and outer['attributes']['cases'] == 10
    kids = children(capfire, outer)
    [cert] = [k for k in kids if k['name'] == 'judge certificate {judge}: {verdict}']
    assert cert['attributes']['covers_judge'] is True and cert['attributes']['dataset'] == 'items'
    model_calls = [k for k in kids if k is not cert]
    assert len(model_calls) == certificate.calls  # one instrumented model request per judge call


async def test_a_hyphenated_control_family_survives_certify_diagnose_export_and_qualify(
    capfire: CaptureLogfire,
) -> None:
    """Regression: names were read from the detail with `\\w+`, so `policy-change 0/30 FAIL` became `change`."""
    from pydantic_evals_admissibility import Rewrite, diagnose
    from pydantic_evals_admissibility._qualify import qualify

    controls = (Rewrite(lambda o: f'{o}, no longer, after the policy change', 'policy-change', 'must_fail'),)
    # Passes anything that contains the right answer, the rewrite included; shown the input, so it can steer.
    scored = dataclasses.replace(judge(oracle), include_input=True)
    certificate = await certify_judge(scored, CASES, controls=controls, repeats=1)
    rejection = next(c for c in certificate.checks if c.name == 'rejection')
    assert [(f.name, f.successes, f.trials, f.status) for f in rejection.families] == [('policy-change', 0, 20, 'FAIL')]
    assert any('(policy-change)' in a for a in diagnose(certificate, scored)), diagnose(certificate, scored)

    old = certificate.to_dict()
    for c in old['checks']:
        c.pop('families', None)  # saved before `Check.families` existed: names come from the detail
    for saved in (certificate, Certificate.from_dict(certificate.to_dict()), Certificate.from_dict(old)):
        capfire.exporter.clear()
        log_certificate(saved)
        [kid] = [
            s for s in spans(capfire) if s['attributes'].get('check') == 'rejection' and 'families' in s['attributes']
        ]
        assert [f['family'] for f in kid['attributes']['families']] == ['policy-change']
        [event] = named(capfire, '{check} {family}: {wrong} of {judged} judged wrongly')
        assert event['attributes']['family'] == 'policy-change' and event['attributes']['wrong'] == 20
        assert len(event['attributes']['examples']) == 5
        assert all(e['role'] == 'must_fail:policy-change' for e in event['attributes']['examples'])
        q = qualify(scored, saved, decision='steer', context={'sees_traces': True}, evidence_families=['policy-change'])
        assert 'policy-change FAIL' in next(r for r in q.rules if r.name == 'evidence').detail


async def test_a_control_family_named_reference_exports_its_failures(capfire: CaptureLogfire) -> None:
    """Review P2: a must-fail family named 'reference' exported no examples: acceptance used the same name."""
    from pydantic_evals_admissibility import Rewrite

    controls = (Rewrite(lambda o: o + ' extra', 'reference'),)
    certificate = await certify_judge(judge(yes_man), CASES, controls=controls, repeats=1)
    rejection = next(c for c in certificate.checks if c.name == 'rejection')
    assert [(f.name, f.status) for f in rejection.families] == [('reference', 'FAIL')]
    log_certificate(certificate, max_examples=3)
    [event] = named(capfire, '{check} {family}: {wrong} of {judged} judged wrongly')
    attrs = event['attributes']
    assert (attrs['check'], attrs['family'], attrs['wrong']) == ('rejection', 'reference', 20)
    assert len(attrs['examples']) == 3 and attrs['examples_omitted'] == 17
    assert all(e['role'] == 'must_fail:reference' and e['passed'] is True for e in attrs['examples'])
