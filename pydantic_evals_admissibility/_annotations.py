"""Logfire run annotations as the human labels a judge is measured against.

Pydantic's own guidance is that "a judge nobody has checked against a person is a number, not
a measurement" (https://pydantic.dev/logfire/llm-as-a-judge), and that the comparison is a
confusion matrix or Cohen's kappa against reviewer labels
(https://pydantic.dev/articles/three-layer-evals-logfire). Logfire records those labels as run
annotations; pydantic-evals has no API to compare a judge with them. This module reads an
annotation export, joins it to a judge's verdicts by run, and decides agreement exactly as
`certify_judge`'s `human_agreement` check does.

What the export format is, as documented (read 2026-10-03):

- https://pydantic.dev/docs/logfire/evaluate/annotate-agent-runs/ ("Export annotations for
  reuse"): on an agent's Runs tab, "Export annotations" downloads `annotations.jsonl`, one JSON
  object per line. Each line "carries the run's trace and span IDs, agent name, verdict,
  category, expected output, comment, tags, the reviewer's email, and the timestamps". The
  export covers the runs in the table and respects its verdict, category and tag filters.
- Same page and https://pydantic.dev/docs/logfire/evaluate/human-review/: the verdict is Pass,
  Neutral or Fail. Neutral means "neither clearly good nor clearly bad, or does not have enough
  information". Category (Neutral or Fail only) is one of hallucination, wrong tool, off topic,
  refused, format error, slow. Expected output appears for Fail. Annotations can be edited.
- https://pydantic.dev/articles/logfire-annotations: run annotations are in beta; "multiple
  reviewers can annotate the same run"; annotations can be exported "to JSONL or CSV".

What is not documented: the JSON key (or CSV column) names, the spelling of verdict values in
the file (the UI says Pass/Neutral/Fail), how tags are encoded in CSV, the timestamp format, and
the CSV layout itself. The format source for the docs page
(github.com/pydantic/logfire, docs/evaluate/annotate-agent-runs.md) has the same sentence and
no schema. `logfire.experimental.annotations.record_feedback` in the Logfire SDK is a different,
older mechanism (feedback spans), not this export. So nothing here assumes one spelling:
`AnnotationFormat` lists the candidate names for each field (snake_case and camelCase of the
documented fields by default), matches verdicts case-insensitively, and both can be replaced.
A file whose rows have no recognised run id or verdict field is refused with the keys it has,
rather than read as empty.

How a run is identified: by its span id (default) or trace id, as `AnnotationFormat.run_id`
says, or by any function of the row. A judge's verdicts must be keyed the same way. In a
pydantic-evals report every case span shares the experiment's trace, and an annotated agent run
is usually a child of the case span, so neither id of a `ReportCase` is guaranteed to be the
annotated run's: `verdicts_from_report` makes the caller say which key to use, and refuses keys
that do not identify one case.

The rules, so nothing is dropped silently:

- Neutral is not a label. Each neutral annotation is counted; a run whose only verdicts are
  neutral is listed in `neutral_runs` and left out of agreement.
- A row with no run id is left out and its row number listed in `missing_id`.
- One reviewer, several annotations of a run (an edited annotation exported twice, say): the
  latest counts, by timestamp when every one has a readable timestamp and by file order
  otherwise; the rest are counted as `superseded`. Rows with no reviewer are each their own
  reviewer.
- Several reviewers who disagree (pass against fail, neutral aside): by default the run is a
  conflict, listed in `conflicts` with its verdicts and left out, since either label would
  credit or blame the judge for a choice the people did not agree on. `conflicts='majority'`
  takes a strict majority instead and keeps ties as conflicts.
"""

from __future__ import annotations

import csv
import io
import json
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from ._cases import HumanLabel
from ._certify import (
    DEFAULT_THRESHOLDS,
    Certificate,
    Check,
    CheckStatus,
    Judgment,
    Thresholds,
    _agreement,  # pyright: ignore[reportPrivateUsage]
)
from ._stats import clopper_pearson

if TYPE_CHECKING:
    from pydantic_evals.reporting import EvaluationReport, ReportCase

Verdict = Literal['pass', 'neutral', 'fail']
_VERDICTS: dict[str, Verdict] = {'pass': 'pass', 'neutral': 'neutral', 'fail': 'fail'}


@dataclass(frozen=True)
class AnnotationFormat:
    """Where each documented field is in an export row, and how a run is identified.

    Each field is a tuple of candidate keys (JSON keys, or CSV column headers), tried in order;
    the first one present with a non-empty value is used. The defaults are the documented field
    names in snake_case and camelCase, because the docs name the fields but not their keys.
    Replace any of them when a real export uses other names.
    """

    trace_id: tuple[str, ...] = ('trace_id', 'traceId')
    span_id: tuple[str, ...] = ('span_id', 'spanId')
    verdict: tuple[str, ...] = ('verdict',)
    category: tuple[str, ...] = ('category',)
    expected_output: tuple[str, ...] = ('expected_output', 'expectedOutput')
    comment: tuple[str, ...] = ('comment',)
    tags: tuple[str, ...] = ('tags',)
    reviewer: tuple[str, ...] = ('reviewer_email', 'reviewerEmail', 'reviewer', 'email')
    agent: tuple[str, ...] = ('agent_name', 'agentName', 'agent')
    timestamp: tuple[str, ...] = ('updated_at', 'updatedAt', 'created_at', 'createdAt')
    """Used only to order one reviewer's annotations of a run; the first readable one wins."""
    verdicts: Mapping[str, Verdict] = field(default_factory=lambda: dict(_VERDICTS))
    """Verdict values in the file, lower-cased and stripped, to pass, neutral or fail."""
    run_id: Literal['span', 'trace'] | Callable[[Mapping[str, Any]], str | None] = 'span'
    """What identifies a run: its span id, its trace id, or a function of the raw row."""


DEFAULT_FORMAT = AnnotationFormat()


@dataclass(frozen=True)
class Annotation:
    """One row of an export, read."""

    run: str
    verdict: Verdict
    row: int
    """Its 1-based position among the export's rows (CSV header not counted)."""
    reviewer: str | None = None
    trace_id: str | None = None
    span_id: str | None = None
    agent: str | None = None
    category: str | None = None
    expected_output: Any = None
    comment: str | None = None
    tags: tuple[str, ...] = ()
    timestamp: str | None = None


@dataclass(frozen=True)
class Annotations:
    """An export read into one pass/fail label per run, with everything left out accounted for."""

    labels: dict[str, bool]
    """Run id to the people's verdict: True for pass, False for fail."""
    annotations: tuple[Annotation, ...] = field(repr=False)
    """Every row that had a run id, including neutral, superseded and conflicting ones."""
    rows: int = 0
    neutral: int = 0
    """Neutral annotations still counting after superseded ones were removed."""
    neutral_runs: tuple[str, ...] = ()
    """Runs whose only verdicts were neutral: annotated, but not labelled."""
    conflicts: dict[str, tuple[Verdict, ...]] = field(default_factory=dict[str, tuple[Verdict, ...]])
    """Runs whose reviewers disagreed, with each reviewer's verdict; left out."""
    superseded: int = 0
    """Annotations replaced by a later one from the same reviewer of the same run."""
    missing_id: tuple[int, ...] = ()
    """Row numbers with no run id; left out."""
    by_run: dict[str, tuple[Annotation, ...]] = field(default_factory=dict[str, tuple[Annotation, ...]], repr=False)
    """The annotations that decided each run, after superseded ones were removed."""

    def summary(self) -> dict[str, Any]:
        """Counts of what was read and what was left out, for a report."""
        return {
            'rows': self.rows,
            'runs_annotated': len(self.by_run),
            'labelled': len(self.labels),
            'pass': sum(self.labels.values()),
            'fail': sum(not v for v in self.labels.values()),
            'neutral_annotations': self.neutral,
            'neutral_runs': len(self.neutral_runs),
            'conflicts': len(self.conflicts),
            'superseded': self.superseded,
            'missing_id': len(self.missing_id),
        }

    def human_labels(self, runs: Mapping[str, tuple[str, Any]], *, skip_missing: bool = False) -> list[HumanLabel]:
        """The labels as `HumanLabel`s for `certify_judge`, which judges each output itself.

        The export does not carry the run's inputs or output, so `runs` maps each run id to the
        name of the `JudgeCase` it belongs to and the output the reviewer judged. A labelled run
        missing from `runs` raises, unless `skip_missing`; a run listed in `runs` that has no label
        is not a label and is ignored.
        """
        missing = sorted(set(self.labels) - set(runs))
        if missing and not skip_missing:
            raise KeyError(
                f'{len(missing)} labelled runs have no case and output in `runs` '
                f'(first: {missing[:3]}); pass them, or skip_missing=True'
            )
        out: list[HumanLabel] = []
        for run, passed in self.labels.items():
            if run not in runs:
                continue
            case, output = runs[run]
            decided = self.by_run[run]
            out.append(
                HumanLabel(
                    case,
                    output,
                    passed,
                    {
                        'run': run,
                        'reviewers': [a.reviewer for a in decided if a.reviewer],
                        'category': next((a.category for a in decided if a.category), None),
                        'expected_output': next((a.expected_output for a in decided if a.expected_output), None),
                        'comments': [a.comment for a in decided if a.comment],
                        'tags': sorted({t for a in decided for t in a.tags}),
                    },
                )
            )
        return out


def _first(row: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and value != '':
            return value
    return None


def _text(value: Any) -> str | None:
    return None if value is None else str(value)


def _tags(value: Any) -> tuple[str, ...]:
    """Tags as a list (JSONL), or in a CSV cell as a JSON list or comma-separated text."""
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        if text.startswith('['):
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                pass
        if isinstance(value, str):
            return tuple(t.strip() for t in value.split(',') if t.strip())
    if isinstance(value, list | tuple):
        return tuple(str(t) for t in value)  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
    return (str(value),)


def _rows(
    source: str | Path | Iterable[Mapping[str, Any]], kind: Literal['jsonl', 'csv'] | None
) -> list[dict[str, Any]]:
    if not isinstance(source, str | Path):
        return [dict(row) for row in source]
    path = Path(source)
    if kind is None:
        suffix = path.suffix.lower()
        if suffix in ('.jsonl', '.ndjson', '.json'):
            kind = 'jsonl'
        elif suffix == '.csv':
            kind = 'csv'
        else:
            raise ValueError(f'cannot tell the format of {path.name!r} from its suffix; pass kind="jsonl" or "csv"')
    text = path.read_text(encoding='utf-8-sig')
    if kind == 'csv':
        return [dict(row) for row in csv.DictReader(io.StringIO(text))]
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f'{path.name} line {number} is not JSON: {error}') from error
        if not isinstance(row, dict):
            raise ValueError(f'{path.name} line {number} is a {type(row).__name__}, not an object')
        rows.append(row)  # pyright: ignore[reportUnknownArgumentType]
    return rows


def _read(row: Mapping[str, Any], number: int, fmt: AnnotationFormat) -> Annotation | None:
    trace_id, span_id = _text(_first(row, fmt.trace_id)), _text(_first(row, fmt.span_id))
    if fmt.run_id == 'span':
        run = span_id
    elif fmt.run_id == 'trace':
        run = trace_id
    else:
        run = fmt.run_id(row)
    raw = _first(row, fmt.verdict)
    if raw is None:
        raise ValueError(f'row {number} has no verdict under {list(fmt.verdict)}; it has {sorted(row)}')
    verdict = fmt.verdicts.get(str(raw).strip().lower())
    if verdict is None:
        raise ValueError(
            f'row {number} has verdict {raw!r}, which is none of {sorted(fmt.verdicts)}; '
            'pass AnnotationFormat(verdicts=...) to map it'
        )
    if run is None:
        return None
    return Annotation(
        str(run),
        verdict,
        number,
        reviewer=_text(_first(row, fmt.reviewer)),
        trace_id=trace_id,
        span_id=span_id,
        agent=_text(_first(row, fmt.agent)),
        category=_text(_first(row, fmt.category)),
        expected_output=_first(row, fmt.expected_output),
        comment=_text(_first(row, fmt.comment)),
        tags=_tags(_first(row, fmt.tags)),
        timestamp=_text(_first(row, fmt.timestamp)),
    )


def _when(stamp: str | None) -> datetime | None:
    if stamp is None:
        return None
    try:
        when = datetime.fromisoformat(stamp.replace('Z', '+00:00'))
    except ValueError:
        return None
    return when if when.tzinfo is not None else None  # naive and aware times do not compare


def _latest(annotations: list[Annotation]) -> Annotation:
    """One reviewer's latest annotation of a run: by timestamp if all are readable, else by row."""
    times = [_when(a.timestamp) for a in annotations]
    if all(t is not None for t in times):
        return max(zip(times, annotations, strict=True), key=lambda p: (p[0], p[1].row))[1]
    return max(annotations, key=lambda a: a.row)


def read_annotations(
    source: str | Path | Iterable[Mapping[str, Any]],
    *,
    kind: Literal['jsonl', 'csv'] | None = None,
    fields: AnnotationFormat = DEFAULT_FORMAT,
    conflicts: Literal['exclude', 'majority'] = 'exclude',
) -> Annotations:
    """Read a Logfire run annotation export into one pass/fail label per run.

    Args:
        source: A `.jsonl` or `.csv` export, or rows already parsed (mappings).
        kind: The file format, when the suffix does not say.
        fields: Which keys hold which field, and what identifies a run (see the module docstring
            for what Logfire documents and what it does not).
        conflicts: Reviewers who disagree on a run: 'exclude' leaves the run out (the default),
            'majority' takes a strict majority of pass and fail and leaves ties out.

    Every row is accounted for in the result: labelled, neutral, superseded, conflicting, or
    missing a run id. A verdict that is not pass, neutral or fail raises.
    """
    if conflicts not in ('exclude', 'majority'):
        raise ValueError(f"conflicts must be 'exclude' or 'majority', got {conflicts!r}")
    rows = _rows(source, kind)
    read: list[Annotation] = []
    missing: list[int] = []
    for number, row in enumerate(rows, 1):
        annotation = _read(row, number, fields)
        if annotation is None:
            missing.append(number)
        else:
            read.append(annotation)
    if rows and not read:
        raise ValueError(
            f'no row has a run id (AnnotationFormat.run_id={fields.run_id!r}); the first row has keys {sorted(rows[0])}'
        )

    grouped: dict[str, dict[object, list[Annotation]]] = defaultdict(lambda: defaultdict(list))
    for a in read:
        grouped[a.run][a.reviewer if a.reviewer is not None else ('row', a.row)].append(a)

    labels: dict[str, bool] = {}
    by_run: dict[str, tuple[Annotation, ...]] = {}
    neutral_runs: list[str] = []
    conflicted: dict[str, tuple[Verdict, ...]] = {}
    superseded = neutral = 0
    for run, reviewers in grouped.items():
        decided = tuple(_latest(group) for group in reviewers.values())
        superseded += sum(len(group) for group in reviewers.values()) - len(decided)
        by_run[run] = decided
        neutral += sum(a.verdict == 'neutral' for a in decided)
        votes = Counter(a.verdict for a in decided if a.verdict != 'neutral')
        if not votes:
            neutral_runs.append(run)
        elif len(votes) == 1:
            labels[run] = 'pass' in votes
        elif conflicts == 'majority' and votes['pass'] != votes['fail']:
            labels[run] = votes['pass'] > votes['fail']
        else:
            conflicted[run] = tuple(a.verdict for a in decided)
    return Annotations(
        labels,
        tuple(read),
        rows=len(rows),
        neutral=neutral,
        neutral_runs=tuple(neutral_runs),
        conflicts=conflicted,
        superseded=superseded,
        missing_id=tuple(missing),
        by_run=by_run,
    )


@dataclass(frozen=True)
class Confusion:
    """Labelled runs with a judge verdict, by what the people said and what the judge said."""

    both_pass: int = 0
    both_fail: int = 0
    judge_pass_human_fail: int = 0
    """The judge passed what people failed: the error that lets bad outputs through."""
    judge_fail_human_pass: int = 0


@dataclass(frozen=True)
class AgreementResult:
    """How a judge's verdicts agree with annotated runs, decided as `certify_judge` decides it."""

    check: Check
    """The `human_agreement` check, with the same rules and interval as in a certificate."""
    agreement: int
    trials: int
    agreement_interval: tuple[float, float]
    """Exact (Clopper-Pearson) 95% interval on raw agreement; runs treated as independent."""
    confusion: Confusion
    cases: int
    """Distinct cases among the matched runs: the unit the kappa interval resamples."""
    human_only: tuple[str, ...] = ()
    """Labelled runs the judge has no verdict for."""
    judge_only: tuple[str, ...] = ()
    """Runs the judge gave a verdict on that nobody labelled (neutral and conflicting runs included)."""
    judge_errors: tuple[str, ...] = ()
    """Labelled runs where the judge gave no verdict (it errored)."""
    disagreements: tuple[tuple[str, bool, bool], ...] = ()
    """(run, human passed, judge passed) for every run where they differ."""
    annotations: dict[str, Any] = field(default_factory=dict[str, Any])
    """`Annotations.summary()` of the labels used: neutral, conflicts, superseded, missing ids."""

    @property
    def status(self) -> CheckStatus:
        return self.check.status

    @property
    def kappa(self) -> float | None:
        return self.check.estimate

    @property
    def kappa_interval(self) -> tuple[float, float]:
        return self.check.interval

    def table(self) -> str:
        c = self.confusion
        kappa = '-' if self.kappa is None else f'{self.kappa:.2f}'
        low, high = self.kappa_interval
        a_low, a_high = self.agreement_interval
        a = self.annotations
        return '\n'.join(
            [
                f'human_agreement: {self.status}  kappa={kappa} [{low:.2f}, {high:.2f}] '
                f'(needs >= {self.check.threshold:.2f})',
                f'  {self.check.detail}',
                f'  raw agreement {self.agreement}/{self.trials} [{a_low:.2f}, {a_high:.2f}] over {self.cases} cases',
                '                 judge pass  judge fail',
                f'  human pass     {c.both_pass:>10}  {c.judge_fail_human_pass:>10}',
                f'  human fail     {c.judge_pass_human_fail:>10}  {c.both_fail:>10}',
                f'  left out: {len(self.human_only)} labelled runs with no verdict, '
                f'{len(self.judge_only)} verdicts with no label, {len(self.judge_errors)} judge errors, '
                f'{a.get("neutral_runs", 0)} neutral runs, {a.get("conflicts", 0)} conflicts, '
                f'{a.get("missing_id", 0)} rows without a run id, {a.get("superseded", 0)} superseded',
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        c = self.check
        return {
            'status': c.status,
            'kappa': c.estimate,
            'kappa_interval': list(c.interval),
            'min_kappa': c.threshold,
            'detail': c.detail,
            'agreement': self.agreement,
            'trials': self.trials,
            'agreement_interval': list(self.agreement_interval),
            'cases': self.cases,
            'confusion': {
                'both_pass': self.confusion.both_pass,
                'both_fail': self.confusion.both_fail,
                'judge_pass_human_fail': self.confusion.judge_pass_human_fail,
                'judge_fail_human_pass': self.confusion.judge_fail_human_pass,
            },
            'human_only': list(self.human_only),
            'judge_only': list(self.judge_only),
            'judge_errors': list(self.judge_errors),
            'disagreements': [list(d) for d in self.disagreements],
            'annotations': self.annotations,
        }


def annotation_agreement(
    annotations: Annotations | Mapping[str, bool],
    verdicts: Mapping[str, bool | None],
    *,
    case_of: Mapping[str, str] | Callable[[str], str] | None = None,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    alpha_fail: float = 0.05,
) -> AgreementResult:
    """Join people's labels to a judge's verdicts by run id and decide agreement.

    Args:
        annotations: From `read_annotations`, or a plain mapping of run id to passed.
        verdicts: Run id to the judge's verdict, None where it errored; see `verdicts_from_report`
            and `verdicts_from_certificate`.
        case_of: The case each run belongs to. Runs of one case share its difficulty, so the
            kappa interval resamples cases, not runs, and `min_trials` counts cases. By default
            each run is its own case. A run missing from a mapping raises.
        thresholds: `min_kappa` and `min_trials`, as for `certify_judge`.
        alpha_fail: The error budget for FAIL. A certificate shares 5% among all its checks;
            alone, this check has all of it. Pass a certificate's share to decide identically.

    The status follows `certify_judge`'s `human_agreement` check: UNVALIDATED when more than a
    tenth of matched runs have no judge verdict, when people used only one label, when kappa is
    undefined, or with fewer than `min_trials` labelled runs or distinct cases; otherwise PASS when
    the case-bootstrap 95% lower bound clears `min_kappa`, FAIL when the upper bound at
    `alpha_fail` is below it.
    """
    summary = annotations.summary() if isinstance(annotations, Annotations) else {}
    labels = annotations.labels if isinstance(annotations, Annotations) else dict(annotations)

    def case(run: str) -> str:
        if case_of is None:
            return run
        if callable(case_of):
            return case_of(run)
        if run not in case_of:
            raise KeyError(f'run {run!r} has no case in `case_of`')
        return case_of[run]

    matched = [run for run in labels if run in verdicts]
    human_only = tuple(run for run in labels if run not in verdicts)
    judge_only = tuple(run for run in verdicts if run not in labels)
    errors = tuple(run for run in matched if verdicts[run] is None)
    done = [(run, labels[run], bool(verdicts[run])) for run in matched if verdicts[run] is not None]
    confusion = Confusion(
        both_pass=sum(h and j for _, h, j in done),
        both_fail=sum(not h and not j for _, h, j in done),
        judge_pass_human_fail=sum(j and not h for _, h, j in done),
        judge_fail_human_pass=sum(h and not j for _, h, j in done),
    )
    agree = confusion.both_pass + confusion.both_fail
    if matched:
        judgments = [Judgment(case(run), f'human:{int(labels[run])}', run, verdicts[run]) for run in matched]
        check = _agreement(judgments, thresholds, alpha_fail)
    else:
        check = Check(
            'human_agreement', 'UNVALIDATED', 0, 0, (0.0, 1.0), thresholds.min_kappa,
            'no labelled run has a judge verdict: check that both are keyed by the same run id',
        )  # fmt: skip
    return AgreementResult(
        check=check,
        agreement=agree,
        trials=len(done),
        agreement_interval=clopper_pearson(agree, len(done)),
        confusion=confusion,
        cases=len({case(run) for run, _, _ in done}),
        human_only=human_only,
        judge_only=judge_only,
        judge_errors=errors,
        disagreements=tuple((run, h, j) for run, h, j in done if h != j),
        annotations=summary,
    )


def verdicts_from_report(
    report: EvaluationReport[Any, Any, Any],
    evaluation: str,
    *,
    key: Literal['span_id', 'trace_id', 'name'] | Callable[[ReportCase[Any, Any, Any]], str | None],
) -> dict[str, bool | None]:
    """A judge's pass/fail verdicts from a pydantic-evals report, keyed by run.

    `evaluation` is the assertion's name in the report (`result_kinds(judge)` lists an
    `LLMJudge`'s). `key` must say how a case is matched to an annotated run: there is no safe
    default, since a report's cases share one trace and the annotated agent span is usually a
    child of the case span (see the module docstring). A case where the judge failed is None.
    A key that is missing or shared by two cases raises.
    """
    out: dict[str, bool | None] = {}
    for case in report.cases:
        run = key(case) if callable(key) else getattr(case, key)
        if run is None:
            raise ValueError(f'case {case.name!r} has no {key if isinstance(key, str) else "key"}')
        if run in out:
            raise ValueError(f'two cases have the key {run!r}: it does not identify a run; choose another key')
        if evaluation in case.assertions:
            out[run] = bool(case.assertions[evaluation].value)
        elif evaluation in case.scores or evaluation in case.labels:
            raise TypeError(f'{evaluation!r} is a score or label in case {case.name!r}, not a pass/fail assertion')
        else:
            out[run] = None  # the judge gave no verdict on this case (an evaluator failure, or not run)
    return out


def verdicts_from_certificate(
    certificate: Certificate | Mapping[str, Any],
    *,
    role: str = 'human:',
    key: Callable[[Judgment, int], str] | None = None,
) -> dict[str, bool | None]:
    """A judge's verdicts from a certificate's saved judgments, keyed by run.

    By default the labelled outputs (roles starting `human:`), keyed `case#i` for the case's
    i-th such judgment in order. `key(judgment, i)` names runs any other way. A certificate may be
    passed as saved (`to_dict()`); one saved without judgments raises.
    """
    cert = certificate if isinstance(certificate, Certificate) else Certificate.from_dict(dict(certificate))
    if not cert.judgments:
        raise ValueError('this certificate was saved without its judgments, so it has no verdicts to compare')
    seen: Counter[str] = Counter()
    out: dict[str, bool | None] = {}
    for j in cert.judgments:
        if not j.role.startswith(role):
            continue
        i = seen[j.case]
        seen[j.case] += 1
        run = key(j, i) if key is not None else f'{j.case}#{i}'
        if run in out:
            raise ValueError(f'two judgments have the key {run!r}')
        out[run] = j.passed
    return out
