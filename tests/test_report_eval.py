"""The certificate as report analyses: shown in the report, refused for another judge, scores left alone."""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import pytest
from judges import judge, oracle, yes_man
from pydantic import TypeAdapter
from pydantic_ai.models.function import FunctionModel
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import LLMJudge
from pydantic_evals.reporting import EvaluationReportAdapter
from pydantic_evals.reporting.analyses import ReportAnalysis, TableResult

from pydantic_evals_admissibility._certify import InadmissibleJudge, Thresholds
from pydantic_evals_admissibility._controls import EmptyOutput
from pydantic_evals_admissibility._identity import judge_identity
from pydantic_evals_admissibility._report_eval import (
    UNVERIFIABLE,
    CertifiedJudgeReport,
    CertifyJudgeReport,
    certify_for_report,
    result_names,
)

WRONG = {3, 8, 15}


def make_dataset(llm_judge: Any, n: int = 20) -> Dataset[dict[str, int], str, Any]:
    cases = [Case(name=f'q{i}', inputs={'a': i}, expected_output=f'The answer is {i + 7}.') for i in range(n)]
    return Dataset(name='sums', cases=cases, evaluators=[llm_judge])


def task(inputs: dict[str, int]) -> str:
    a = inputs['a']
    return f'The answer is {a + 7 + (a in WRONG)}.'


def tables(report: Any) -> dict[str, TableResult]:
    out: dict[str, TableResult] = {}
    for a in report.analyses:
        assert isinstance(a, TableResult)
        out[a.title.split(':')[0]] = a
    return out


async def test_certificate_appears_in_report_analyses() -> None:
    dataset = make_dataset(judge(oracle))
    attach = await certify_for_report(dataset)
    dataset.report_evaluators.append(attach)
    report = await dataset.evaluate(task, progress=False)

    assert not report.report_evaluator_failures
    found = tables(report)
    cert = found['Judge certificate']
    assert 'ADMISSIBLE' in cert.title and str(attach.certified.fingerprint) in cert.title
    assert [row[0] for row in cert.rows] == [c.name for c in attach.certified.checks]
    assert all(row[1] == 'PASS' for row in cert.rows)
    assert found['Certificate coverage'].rows[0][2] == 'yes'
    evidence = found['Judge results in this report']
    assert evidence.rows == [
        ['LLMJudge', 'assertion', 20, '17/20 passed', 'evidence', 'ADMISSIBLE, and the certificate covers this judge']
    ]
    rendered = report.render(width=200)
    assert 'Judge certificate' in rendered and 'evidence' in rendered


async def test_mismatched_judge_is_flagged_or_refused() -> None:
    sound = judge(oracle)
    certificate = (await certify_for_report(make_dataset(sound))).certified
    changed = dataclasses.replace(sound, rubric='The answer is correct and polite.')

    flagged = make_dataset(changed)
    flagged.report_evaluators.append(CertifiedJudgeReport(certificate, judge=changed))
    report = await flagged.evaluate(task, progress=False)
    found = tables(report)
    assert found['Certificate coverage'].rows[0][2] == 'no'
    assert 'rubric' in str(found['Certificate coverage'].rows[0][3])
    assert found['Judge results in this report'].rows[0][4] == 'NOT EVIDENCE'

    refused = make_dataset(changed)
    refused.report_evaluators.append(CertifiedJudgeReport(certificate, judge=changed, on_mismatch='raise'))
    report = await refused.evaluate(task, progress=False)
    assert not report.analyses
    assert 'different configuration' in report.report_evaluator_failures[0].error_message


async def test_mismatch_seen_from_the_report_without_a_judge() -> None:
    """With no judge given, the rubric the report records is still compared with the certified one, and a
    match is not taken as coverage: the scores are not evidence until `judge=` lets coverage be checked."""
    certificate = (await certify_for_report(make_dataset(judge(oracle)))).certified
    other = dataclasses.replace(judge(oracle), rubric='Something else entirely.', include_input=True)
    dataset = make_dataset(other)
    dataset.report_evaluators.append(CertifiedJudgeReport(certificate, evaluation_name='LLMJudge'))
    found = tables(await dataset.evaluate(task, progress=False))
    assert found['Certificate coverage'].rows[0][2] == 'no'
    assert found['Judge results in this report'].rows[0][4] == 'NOT EVIDENCE'

    same = make_dataset(judge(oracle))
    same.report_evaluators.append(CertifiedJudgeReport(certificate))
    found = tables(await same.evaluate(task, progress=False))
    assert found['Certificate coverage'].rows[0][2] == 'not checked'
    assert found['Judge results in this report'].rows[0][4] == 'NOT EVIDENCE (coverage not checked: pass judge=)'


async def test_inadmissible_judge_marked_not_evidence_scores_untouched() -> None:
    broken = judge(yes_man)
    dataset = make_dataset(broken)
    dataset.report_evaluators.append(CertifyJudgeReport(broken))
    report = await dataset.evaluate(task, progress=False)

    found = tables(report)
    assert 'INADMISSIBLE' in found['Judge certificate'].title
    row = found['Judge results in this report'].rows[0]
    assert row[4] == 'NOT EVIDENCE' and 'rejection' in str(row[5])
    # The judge's verdicts are left exactly as it gave them: every answer passed, the wrong ones too.
    assert all(c.assertions['LLMJudge'].value is True for c in report.cases)
    assert all(c.assertions['LLMJudge'].source.name == 'LLMJudge' for c in report.cases)


async def test_unvalidated_is_not_evidence_and_too_few_cases_is_said() -> None:
    small = make_dataset(judge(oracle), n=12)  # 12/12 rejected is a lower bound of 0.76: not yet 0.8
    small.report_evaluators.append(CertifyJudgeReport(small.evaluators[0]))
    found = tables(await small.evaluate(task, progress=False))
    assert 'UNVALIDATED' in found['Judge certificate'].title
    assert found['Judge results in this report'].rows[0][4] == 'NOT EVIDENCE'

    tiny = make_dataset(judge(oracle), n=5)
    tiny.report_evaluators.append(CertifyJudgeReport(tiny.evaluators[0]))
    found = tables(await tiny.evaluate(task, progress=False))
    assert found['Judge certificate'].rows[0][2].startswith('NOT EVIDENCE')  # type: ignore[union-attr]


async def test_serialization_round_trips() -> None:
    dataset = make_dataset(judge(oracle))
    attach = await certify_for_report(dataset)
    dataset.report_evaluators.append(attach)
    report = await dataset.evaluate(task, progress=False)

    dumped = TypeAdapter(list[ReportAnalysis]).dump_json(report.analyses)
    assert TypeAdapter(list[ReportAnalysis]).validate_json(dumped) == report.analyses
    whole = json.loads(EvaluationReportAdapter.dump_json(report))
    assert [a['title'].split(':')[0] for a in whole['analyses']] == [
        'Judge certificate',
        'Certificate coverage',
        'Judge results in this report',
    ]

    # The evaluator's own spec: the certificate without judgments, the judge as its identity.
    spec = attach.as_spec()
    assert spec.name == 'CertifiedJudgeReport'
    args = json.loads(json.dumps(spec.arguments))
    rebuilt = CertifiedJudgeReport(**args)
    assert rebuilt.certified.verdict == 'ADMISSIBLE'
    assert rebuilt.certified.fingerprint == attach.certified.fingerprint
    assert rebuilt.analyses(report)[2].rows[0][4] == 'evidence'  # type: ignore[union-attr]


def test_result_names_follow_llm_judge() -> None:
    assert result_names(LLMJudge(rubric='r')) == ['LLMJudge']
    assert result_names(LLMJudge(rubric='r', score={'include_reason': True})) == ['LLMJudge_score', 'LLMJudge_pass']
    named = LLMJudge(rubric='r', assertion={'evaluation_name': 'correct'})
    assert result_names(named) == ['correct']


async def test_certify_for_report_needs_one_judge() -> None:
    dataset = make_dataset(judge(oracle))
    dataset.evaluators.append(judge(yes_man))
    with pytest.raises(ValueError, match='2 LLMJudges'):
        await certify_for_report(dataset)
    picked = await certify_for_report(dataset, judge=dataset.evaluators[1])
    assert picked.certified.verdict == 'INADMISSIBLE'


async def test_coverage_is_checked_against_the_judge_that_scored_not_the_one_supplied() -> None:
    """Given the certified judge, the results of another judge in the report are still not evidence."""
    sound = judge(oracle)
    certificate = (await certify_for_report(make_dataset(sound))).certified
    other = dataclasses.replace(sound, rubric='Something else entirely.')
    dataset = make_dataset(other)
    dataset.report_evaluators.append(CertifiedJudgeReport(certificate, judge=sound))
    found = tables(await dataset.evaluate(task, progress=False))
    assert found['Certificate coverage'].rows[0][2] == 'no'
    assert 'rubric' in str(found['Certificate coverage'].rows[0][3])
    assert found['Judge results in this report'].rows[0][4] == 'NOT EVIDENCE'


def named(llm_judge: Any, name: str) -> Any:
    return dataclasses.replace(llm_judge, assertion={'evaluation_name': name, 'include_reason': True})


async def test_round_trip_keeps_which_results_the_certificate_is_about() -> None:
    """Two judges with one rubric: after a spec round trip, only the certified judge's results are evidence."""
    certified, lenient = named(judge(oracle), 'correct'), named(judge(yes_man), 'lenient')
    dataset = make_dataset(certified)
    dataset.evaluators.append(lenient)
    attach = await certify_for_report(dataset, judge=certified)
    report = await dataset.evaluate(task, progress=False)

    args = json.loads(json.dumps(attach.as_spec().arguments))
    rows = CertifiedJudgeReport(**args).analyses(report)[2].rows  # type: ignore[union-attr]
    assert [(row[0], row[4]) for row in rows] == [('correct', 'evidence')]

    # A spec that does not say which results are the judge's: both judges could be it, so it says so.
    del args['results']
    with pytest.raises(ValueError, match='could be the certified judge'):
        CertifiedJudgeReport(**args).analyses(report)


async def test_identical_judges_in_one_report_are_ambiguous() -> None:
    sound = judge(oracle)
    certificate = (await certify_for_report(make_dataset(sound))).certified
    dataset = make_dataset(sound)
    twin = dataclasses.replace(sound)
    dataset.evaluators.append(twin)  # its results are named `LLMJudge_2`
    report = await dataset.evaluate(task, progress=False)
    for attach in (CertifiedJudgeReport(certificate, judge=twin), CertifiedJudgeReport(certificate)):
        with pytest.raises(ValueError, match='could be the certified judge'):
            attach.analyses(report)


async def test_score_only_judge_is_refused_before_any_judge_call() -> None:
    calls: list[str] = []

    def counted(output: str, expected: str) -> bool:
        calls.append(output)
        return oracle(output, expected)

    score_only = dataclasses.replace(judge(counted), score={'include_reason': True}, assertion=False)
    with pytest.raises(ValueError, match='score-only'):
        CertifyJudgeReport(score_only)
    with pytest.raises(ValueError, match='score-only'):
        await certify_for_report(make_dataset(score_only))
    assert calls == []


async def test_certify_judge_report_round_trips_through_a_dataset_file() -> None:
    llm_judge = LLMJudge(
        rubric='The output answers the question correctly.', model='test', include_expected_output=True
    )
    evaluator = CertifyJudgeReport(llm_judge, repeats=1, thresholds=Thresholds(min_acceptance=0.6))
    args = json.loads(json.dumps(evaluator.as_spec().arguments))
    rebuilt = CertifyJudgeReport(**args)
    assert isinstance(rebuilt.judge, LLMJudge)
    assert judge_identity(rebuilt.judge) == judge_identity(llm_judge)
    assert rebuilt.thresholds == Thresholds(min_acceptance=0.6) and rebuilt.repeats == 1

    dataset = make_dataset(llm_judge)
    dataset.report_evaluators.append(evaluator)
    loaded = Dataset[dict[str, int], str, Any].from_dict(
        json.loads(json.dumps(dataset.model_dump(mode='json', by_alias=True))),
        custom_report_evaluator_types=[CertifyJudgeReport],
    )
    report = await loaded.evaluate(task, progress=False)
    assert not report.report_evaluator_failures
    assert 'Judge certificate' in tables(report)


def test_certify_judge_report_refuses_what_it_cannot_rebuild() -> None:
    args = json.loads(json.dumps(CertifyJudgeReport(judge(oracle)).as_spec().arguments))
    with pytest.raises(ValueError, match='model'):
        CertifyJudgeReport(**args)
    llm_judge = LLMJudge(rubric='r', model='test')
    args = json.loads(json.dumps(CertifyJudgeReport(llm_judge, controls=[EmptyOutput()]).as_spec().arguments))
    with pytest.raises(ValueError, match='controls'):
        CertifyJudgeReport(**args)


def function_of(llm_judge: LLMJudge) -> Any:
    assert isinstance(llm_judge.model, FunctionModel) and llm_judge.model.function is not None
    return llm_judge.model.function


def unnamed(llm_judge: Any) -> Any:
    """The judge on a FunctionModel with no name of its own: the report records only `function:...`."""
    return dataclasses.replace(llm_judge, model=FunctionModel(function_of(llm_judge)))


async def test_coverage_of_an_unnamed_function_model_cannot_be_verified() -> None:
    """Review P1: certified on judge(oracle), scored by judge(yes_man), attached with judge=oracle: coverage was
    'yes' and 20/20 yes-man passes were 'evidence'. The report records both models by the same generic name."""
    sound = unnamed(judge(oracle))
    certificate = (await certify_for_report(make_dataset(sound))).certified
    assert certificate.verdict == 'ADMISSIBLE'
    impostor = make_dataset(unnamed(judge(yes_man)))
    report = await impostor.evaluate(task, progress=False)
    for attach in (CertifiedJudgeReport(certificate, judge=sound), CertifiedJudgeReport(certificate)):
        coverage, evidence = attach.analyses(report)[1:]
        assert isinstance(coverage, TableResult) and isinstance(evidence, TableResult)
        assert coverage.title == 'Certificate coverage: unverifiable' and coverage.rows[0][2] == 'unverifiable'
        assert 'cannot be told' in str(coverage.rows[0][3])
        assert evidence.rows[0][3] == '20/20 passed'
        assert evidence.rows[0][4] == UNVERIFIABLE
        assert UNVERIFIABLE == 'NOT EVIDENCE (coverage cannot be verified)'
    with pytest.raises(InadmissibleJudge, match='cannot be verified'):
        CertifiedJudgeReport(certificate, judge=sound, on_mismatch='raise').analyses(report)
    # Its own results are no better off: the report cannot show they are the oracle's.
    honest = make_dataset(sound)
    honest.report_evaluators.append(CertifiedJudgeReport(certificate, judge=sound))
    found = tables(await honest.evaluate(task, progress=False))
    assert found['Certificate coverage'].rows[0][2] == 'unverifiable'


async def test_named_models_are_compared_by_name() -> None:
    """A model with a name of its own is recorded in full: another judge is 'no', the certified one 'yes'."""
    sound = named(judge(oracle), 'correct')
    sound = dataclasses.replace(sound, model=FunctionModel(function_of(sound), model_name='oracle'))
    certificate = (await certify_for_report(make_dataset(sound))).certified
    lenient = dataclasses.replace(sound, model=FunctionModel(function_of(judge(yes_man)), model_name='yes-man'))
    impostor = make_dataset(lenient)
    impostor.report_evaluators.append(CertifiedJudgeReport(certificate, judge=sound))
    found = tables(await impostor.evaluate(task, progress=False))
    assert found['Certificate coverage'].rows[0][2] == 'no' and 'model' in str(found['Certificate coverage'].rows[0][3])
    assert found['Judge results in this report'].rows[0][4] == 'NOT EVIDENCE'
    same = make_dataset(sound)
    same.report_evaluators.append(CertifiedJudgeReport(certificate, judge=sound))
    found = tables(await same.evaluate(task, progress=False))
    assert found['Certificate coverage'].rows[0][2] == 'yes'
    assert found['Judge results in this report'].rows[0][4] == 'evidence'


async def test_an_unreliable_certified_identity_is_unverifiable() -> None:
    """An identity with something it could not identify can be shared by another judge: never 'yes'."""
    sound = judge(oracle)
    certificate = (await certify_for_report(make_dataset(sound))).certified
    assert certificate.identity is not None
    identity = {**certificate.identity, 'opaque': ['model_settings.x: list']}
    opaque = dataclasses.replace(certificate, identity=identity)
    dataset = make_dataset(sound)
    dataset.report_evaluators.append(CertifiedJudgeReport(opaque, judge=identity))
    found = tables(await dataset.evaluate(task, progress=False))
    assert found['Certificate coverage'].rows[0][2] == 'unverifiable'
    assert 'model_settings.x: list' in str(found['Certificate coverage'].rows[0][3])
    assert found['Judge results in this report'].rows[0][4] == UNVERIFIABLE


async def test_a_custom_evaluators_certificate_is_unverifiable_in_a_report() -> None:
    """Found in the sixth review: a report records a custom evaluator by class name and arguments
    only, so a certificate for one configuration marked another's results as evidence."""
    from pydantic_evals_admissibility._report_eval import _custom_source  # pyright: ignore[reportPrivateUsage]

    assert _custom_source(judge_identity(judge(oracle))) == []
    assert _custom_source({'evaluator': 'CodeJudge', '!code': 'abc'})


async def test_a_mistyped_evaluation_name_says_so() -> None:
    """Found in the eighth review: a typo was reported as an ambiguous attribution."""
    certificate = (await certify_for_report(make_dataset(judge(oracle)))).certified
    dataset = make_dataset(judge(oracle))
    dataset.report_evaluators.append(CertifiedJudgeReport(certificate, evaluation_name='LLMJudge_typo'))
    report = await dataset.evaluate(task, progress=False)
    message = report.report_evaluator_failures[0].error_message
    assert "no result in the report is named 'LLMJudge_typo'" in message and "'LLMJudge'" in message
