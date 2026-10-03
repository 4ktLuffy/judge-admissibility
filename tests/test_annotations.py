"""Logfire run annotations as human labels: every row accounted for, agreement decided as a certificate decides it."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from judges import judge, oracle, yes_man
from pydantic_evals.evaluators import EvaluationResult, EvaluatorSpec
from pydantic_evals.reporting import EvaluationReport, ReportCase

from pydantic_evals_admissibility import JudgeCase, Thresholds, certify_judge, cohen_kappa
from pydantic_evals_admissibility._annotations import (
    AnnotationFormat,
    annotation_agreement,
    read_annotations,
    verdicts_from_certificate,
    verdicts_from_report,
)
from pydantic_evals_admissibility._certify import Certificate, Judgment
from pydantic_evals_admissibility._stats import clopper_pearson

FIXTURES = Path(__file__).parent / 'fixtures' / 'annotations'


# Parsing


@pytest.mark.parametrize('name', ['export.jsonl', 'export.csv'])
def test_every_row_of_an_export_is_accounted_for(name: str) -> None:
    got = read_annotations(FIXTURES / name)
    assert got.labels == {
        'b7ad6b7169203331': True,
        'c7ad6b7169203332': False,
        'f7ad6b7169203335': True,  # ben's later edit replaces his fail
        'a8ad6b7169203336': False,  # ana's fail; ben's neutral is not a label
    }
    assert got.rows == 10
    assert got.neutral == 2
    assert got.neutral_runs == ('d7ad6b7169203333',)
    assert got.conflicts == {'e7ad6b7169203334': ('pass', 'fail')}
    assert got.superseded == 1
    assert got.missing_id == (8,)
    s = got.summary()
    labelled, left_out = s['labelled'], s['neutral_runs'] + s['conflicts']
    assert labelled + left_out == s['runs_annotated'] == 6


def test_jsonl_and_csv_read_the_same() -> None:
    a, b = read_annotations(FIXTURES / 'export.jsonl'), read_annotations(FIXTURES / 'export.csv')
    assert a.labels == b.labels
    assert [x.tags for x in a.annotations] == [x.tags for x in b.annotations]
    fail = next(x for x in a.annotations if x.run == 'a8ad6b7169203336' and x.verdict == 'fail')
    assert fail.tags == ('tool-error', 'escalate')
    assert fail.category == 'wrong tool'
    assert fail.expected_output == 'Answer: no'
    assert fail.reviewer == 'ana@example.com'


def test_runs_can_be_identified_by_trace() -> None:
    got = read_annotations(FIXTURES / 'export.jsonl', fields=AnnotationFormat(run_id='trace'))
    assert got.missing_id == ()  # the row with no span id has a trace id
    assert got.labels['0af7651916cd43dd8448eb211c8031a1'] is False


def test_field_names_and_verdict_values_are_configurable() -> None:
    rows = [
        {'runId': 'r1', 'rating': 'GOOD', 'who': 'a'},
        {'runId': 'r2', 'rating': 'bad', 'who': 'a'},
        {'runId': 'r3', 'rating': 'meh', 'who': 'a'},
    ]
    fields = AnnotationFormat(
        verdict=('rating',),
        reviewer=('who',),
        verdicts={'good': 'pass', 'bad': 'fail', 'meh': 'neutral'},
        run_id=lambda row: row.get('runId'),
    )
    got = read_annotations(rows, fields=fields)
    assert got.labels == {'r1': True, 'r2': False}
    assert got.neutral_runs == ('r3',)


def test_an_unknown_verdict_is_refused_not_dropped() -> None:
    with pytest.raises(ValueError, match="verdict 'thumbs-up'"):
        read_annotations([{'span_id': 's', 'verdict': 'thumbs-up'}])


def test_a_row_without_a_verdict_field_is_refused() -> None:
    with pytest.raises(ValueError, match='no verdict'):
        read_annotations([{'span_id': 's', 'rating': 'pass'}])


def test_an_export_with_no_recognised_run_id_is_refused() -> None:
    with pytest.raises(ValueError, match='no row has a run id'):
        read_annotations([{'id': 'x', 'verdict': 'pass'}, {'id': 'y', 'verdict': 'fail'}])


def test_an_empty_export_is_empty() -> None:
    got = read_annotations([])
    assert got.labels == {} and got.rows == 0


def test_majority_resolves_conflicts_and_keeps_ties_out() -> None:
    rows = [
        {'span_id': 'a', 'verdict': 'pass', 'reviewer': 'x'},
        {'span_id': 'a', 'verdict': 'pass', 'reviewer': 'y'},
        {'span_id': 'a', 'verdict': 'fail', 'reviewer': 'z'},
        {'span_id': 'b', 'verdict': 'pass', 'reviewer': 'x'},
        {'span_id': 'b', 'verdict': 'fail', 'reviewer': 'y'},
    ]
    assert read_annotations(rows).conflicts.keys() == {'a', 'b'}
    got = read_annotations(rows, conflicts='majority')
    assert got.labels == {'a': True}
    assert got.conflicts == {'b': ('pass', 'fail')}


def test_rows_without_a_reviewer_are_separate_reviewers() -> None:
    rows = [{'span_id': 'a', 'verdict': 'pass'}, {'span_id': 'a', 'verdict': 'pass'}]
    got = read_annotations(rows)
    assert got.labels == {'a': True} and got.superseded == 0
    rows.append({'span_id': 'a', 'verdict': 'fail'})
    assert read_annotations(rows).conflicts == {'a': ('pass', 'pass', 'fail')}


def test_without_readable_timestamps_the_last_row_wins() -> None:
    rows = [
        {'span_id': 'a', 'verdict': 'pass', 'reviewer': 'x', 'updated_at': 'yesterday'},
        {'span_id': 'a', 'verdict': 'fail', 'reviewer': 'x', 'updated_at': 'today'},
    ]
    assert read_annotations(rows).labels == {'a': False}
    rows[0]['updated_at'], rows[1]['updated_at'] = '2026-09-02T00:00:00Z', '2026-09-01T00:00:00Z'
    assert read_annotations(rows).labels == {'a': True}


def test_an_unknown_suffix_needs_a_kind(tmp_path: Path) -> None:
    path = tmp_path / 'annotations.txt'
    path.write_text(json.dumps({'span_id': 'a', 'verdict': 'pass'}) + '\n')
    with pytest.raises(ValueError, match='kind='):
        read_annotations(path)
    assert read_annotations(path, kind='jsonl').labels == {'a': True}


def test_a_bad_jsonl_line_says_where(tmp_path: Path) -> None:
    path = tmp_path / 'a.jsonl'
    path.write_text('{"span_id": "a", "verdict": "pass"}\nnot json\n')
    with pytest.raises(ValueError, match='line 2'):
        read_annotations(path)


# The join and the statistics


def labelled(n_pass: int, n_fail: int) -> dict[str, bool]:
    return {**{f'p{i}': True for i in range(n_pass)}, **{f'f{i}': False for i in range(n_fail)}}


def test_a_judge_that_agrees_passes() -> None:
    labels = labelled(12, 12)
    result = annotation_agreement(labels, dict(labels))
    assert result.status == 'PASS'
    assert result.kappa == 1.0
    assert result.agreement == result.trials == 24
    assert result.agreement_interval == clopper_pearson(24, 24)
    assert result.confusion.both_pass == 12 and result.confusion.both_fail == 12


def test_a_judge_that_passes_everything_fails() -> None:
    labels = labelled(12, 12)
    result = annotation_agreement(labels, dict.fromkeys(labels, True))
    assert result.status == 'FAIL'
    assert result.kappa == 0.0
    assert result.confusion.judge_pass_human_fail == 12
    assert len(result.disagreements) == 12


def test_the_join_reports_both_sides_left_out() -> None:
    labels = labelled(12, 12)
    verdicts: dict[str, bool | None] = {run: passed for run, passed in labels.items() if run != 'p0'}
    verdicts['unlabelled'] = True
    result = annotation_agreement(labels, verdicts)
    assert result.human_only == ('p0',)
    assert result.judge_only == ('unlabelled',)
    assert result.trials == 23


def test_agreement_from_an_export_carries_what_was_left_out() -> None:
    annotations = read_annotations(FIXTURES / 'export.jsonl')
    verdicts = {'b7ad6b7169203331': True, 'c7ad6b7169203332': True, 'e7ad6b7169203334': True}
    result = annotation_agreement(annotations, verdicts)
    assert result.judge_only == ('e7ad6b7169203334',)  # a conflict is not a label
    assert result.annotations['conflicts'] == 1 and result.annotations['neutral_annotations'] == 2
    assert result.status == 'UNVALIDATED'  # 2 labelled runs is fewer than min_trials
    assert 'left out' in result.table()


def test_nothing_matched_is_unvalidated_and_says_why() -> None:
    result = annotation_agreement(labelled(12, 12), {'other': True})
    assert result.status == 'UNVALIDATED'
    assert 'same run id' in result.check.detail
    assert result.trials == 0 and result.agreement_interval == (0.0, 1.0)


def test_people_must_use_both_labels() -> None:
    labels = labelled(24, 0)
    result = annotation_agreement(labels, dict(labels))
    assert result.status == 'UNVALIDATED'
    assert 'labelled every output pass' in result.check.detail


def test_a_constant_judge_against_both_labels_has_kappa_zero() -> None:
    labels = labelled(12, 12)
    assert annotation_agreement(labels, dict.fromkeys(labels, False)).kappa == 0.0


def test_judge_errors_beyond_a_tenth_leave_it_unvalidated() -> None:
    labels = labelled(12, 12)
    verdicts: dict[str, bool | None] = dict(labels)
    for run in ('p0', 'p1', 'f0'):
        verdicts[run] = None
    result = annotation_agreement(labels, verdicts)
    assert result.status == 'UNVALIDATED'
    assert result.judge_errors == ('p0', 'p1', 'f0')
    assert result.trials == 21


def test_min_trials_counts_cases_not_runs() -> None:
    labels = labelled(12, 12)
    result = annotation_agreement(labels, dict(labels), case_of=lambda run: run[0])  # two cases
    assert result.status == 'UNVALIDATED'
    assert 'labels on 2 cases' in result.check.detail
    assert result.cases == 2
    with pytest.raises(KeyError, match='no case'):
        annotation_agreement(labels, dict(labels), case_of={'p0': 'a'})


def test_the_kappa_is_the_same_as_cohen_kappa() -> None:
    labels = labelled(15, 15)
    verdicts = {run: (passed if i % 5 else not passed) for i, (run, passed) in enumerate(labels.items())}
    result = annotation_agreement(labels, verdicts)
    assert result.kappa == pytest.approx(cohen_kappa((labels[r], verdicts[r]) for r in labels))
    low, high = result.kappa_interval
    assert low <= (result.kappa or 0) <= high


def test_thresholds_are_respected() -> None:
    labels = labelled(4, 4)
    assert annotation_agreement(labels, dict(labels)).status == 'UNVALIDATED'  # fewer than min_trials
    # Lowering min_trials does not make 8 perfectly agreeing labels a certified kappa: the exact
    # bound on agreement by case keeps the interval honest (it was [1, 1] before review).
    small = annotation_agreement(labels, dict(labels), thresholds=Thresholds(min_trials=5))
    assert small.status == 'UNVALIDATED' and small.kappa_interval[0] < 0.4
    many = labelled(30, 30)
    assert annotation_agreement(many, dict(many), thresholds=Thresholds(min_trials=5)).status == 'PASS'


def test_to_dict_is_json() -> None:
    labels = labelled(12, 12)
    json.dumps(annotation_agreement(labels, dict(labels)).to_dict())


# Sources of judge verdicts


def report_case(name: str, span_id: str | None, verdict: bool | None) -> ReportCase[Any, Any, Any]:
    spec = EvaluatorSpec(name='LLMJudge', arguments=None)
    assertions = (
        {}
        if verdict is None
        else {'LLMJudge': EvaluationResult(name='LLMJudge', value=verdict, reason=None, source=spec)}
    )
    return ReportCase(
        name=name, inputs=None, metadata=None, expected_output=None, output=None, metrics={}, attributes={},
        scores={'LLMJudge_score': EvaluationResult(name='LLMJudge_score', value=1.0, reason=None, source=spec)},
        labels={},
        assertions=assertions, task_duration=0.0, total_duration=0.0, trace_id='t', span_id=span_id,
    )  # fmt: skip


def test_verdicts_from_a_report() -> None:
    report = EvaluationReport(
        name='r', cases=[report_case('a', 's1', True), report_case('b', 's2', False), report_case('c', 's3', None)]
    )
    assert verdicts_from_report(report, 'LLMJudge', key='span_id') == {'s1': True, 's2': False, 's3': None}
    assert verdicts_from_report(report, 'LLMJudge', key='name') == {'a': True, 'b': False, 'c': None}
    assert verdicts_from_report(report, 'LLMJudge', key=lambda c: c.name.upper())['A'] is True
    with pytest.raises(ValueError, match='does not identify a run'):
        verdicts_from_report(report, 'LLMJudge', key='trace_id')
    with pytest.raises(TypeError, match='not a pass/fail'):
        verdicts_from_report(report, 'LLMJudge_score', key='span_id')
    with pytest.raises(ValueError, match='has no span_id'):
        verdicts_from_report(
            EvaluationReport(name='r', cases=[report_case('a', None, True)]), 'LLMJudge', key='span_id'
        )


def test_verdicts_from_a_saved_certificate() -> None:
    judgments = (
        Judgment('a', 'reference#0', 'x', True),
        Judgment('a', 'human:1', 'x', True),
        Judgment('a', 'human:0', 'y', True),
        Judgment('b', 'human:0', 'z', None, error='boom'),
    )
    cert = Certificate('UNVALIDATED', (), judgments)
    expected = {'a#0': True, 'a#1': True, 'b#0': None}
    assert verdicts_from_certificate(cert) == expected
    assert verdicts_from_certificate(cert.to_dict()) == expected
    assert verdicts_from_certificate(cert, role='reference#', key=lambda j, i: j.case) == {'a': True}
    with pytest.raises(ValueError, match='without its judgments'):
        verdicts_from_certificate(cert.to_dict(judgments=False))


CASES = [JudgeCase(f'q{i}', f'What is {i} + 1?', str(i + 1), expected_output=str(i + 1)) for i in range(24)]


async def test_annotations_as_human_labels_decide_like_annotation_agreement() -> None:
    """The same labels give the same kappa through `certify_judge` and through `annotation_agreement`."""
    rows = [{'span_id': f'{c.name}-good', 'verdict': 'pass'} for c in CASES]
    rows += [{'span_id': f'{c.name}-bad', 'verdict': 'fail'} for c in CASES]
    annotations = read_annotations(rows)
    runs = {f'{c.name}-good': (c.name, c.output) for c in CASES} | {f'{c.name}-bad': (c.name, 'no idea') for c in CASES}
    labels = annotations.human_labels(runs)
    assert len(labels) == 48 and labels[0].labels['run'] == 'q0-good'

    for decide, expected in ((oracle, 'PASS'), (yes_man, 'FAIL')):
        cert = await certify_judge(judge(decide), CASES, human_labels=labels, controls=(), repeats=1)
        check = next(c for c in cert.checks if c.name == 'human_agreement')
        assert check.status == expected
        order = [label.labels['run'] for label in labels]
        human = [j for j in cert.judgments if j.role.startswith('human:')]
        verdicts = {run: j.passed for run, j in zip(order, human, strict=True)}
        # Acceptance and agreement can each fail this certificate: each has half the 5% FAIL budget.
        cases = {r: c for r, (c, _) in runs.items()}
        result = annotation_agreement(annotations, verdicts, case_of=cases, alpha_fail=0.025)
        assert result.kappa == check.estimate
        assert result.kappa_interval == check.interval
        assert result.status == check.status


def test_human_labels_refuse_runs_without_an_output() -> None:
    annotations = read_annotations([{'span_id': 'a', 'verdict': 'pass'}, {'span_id': 'b', 'verdict': 'fail'}])
    with pytest.raises(KeyError, match='1 labelled runs'):
        annotations.human_labels({'a': ('q0', '1')})
    assert [x.case for x in annotations.human_labels({'a': ('q0', '1')}, skip_missing=True)] == ['q0']


def test_the_bench_export_reads_as_eighty_labels() -> None:
    """`bench/annotations_agreement.py` writes this from saved verdicts; ground truth stands in for reviewers."""
    got = read_annotations(FIXTURES / 'support_eval_truth.jsonl')
    assert got.summary()['labelled'] == 80
    assert sum(got.labels.values()) == 72
    assert got.neutral == 0 and not got.conflicts and not got.missing_id
