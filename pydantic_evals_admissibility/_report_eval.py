"""Put a judge's certificate in the pydantic-evals report its scores appear in.

A report shows an `LLMJudge`'s pass rate next to every other number, with nothing to say whether
the judge was shown to be evidence. `CertifiedJudgeReport` is a `ReportEvaluator`: added to a
`Dataset`'s `report_evaluators`, it attaches the certificate as report analyses (the verdict,
every check on its interval, the fingerprint of the certified configuration), says whether the
certificate covers the judge that scored, and marks the judge's results in the report as not
evidence unless the certificate is ADMISSIBLE and about that judge. The scores themselves are
left exactly as the judge gave them.

`CertifyJudgeReport` certifies the judge while the report is built, on the report's own cases
(their `expected_output` as the known-good answers, as `certify_dataset` does), and
`certify_for_report` does it before the run, from a `Dataset`.
"""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic_evals.evaluators import (
    EvaluationResult,
    Evaluator,
    EvaluatorSpec,
    ReportEvaluator,
    ReportEvaluatorContext,
)
from pydantic_evals.evaluators.common import DEFAULT_EVALUATORS
from pydantic_evals.reporting import EvaluationReport
from pydantic_evals.reporting.analyses import ReportAnalysis, TableResult

from ._cases import JudgeCase
from ._certify import DEFAULT_THRESHOLDS, Certificate, InadmissibleJudge, Thresholds, certify_judge
from ._controls import Control
from ._dataset import _judges, controls_for  # pyright: ignore[reportPrivateUsage]
from ._identity import differences, fingerprint, identity_reliable, judge_identity

UNVERIFIABLE = 'NOT EVIDENCE (coverage cannot be verified)'
UNCHECKED = 'NOT EVIDENCE (coverage not checked: pass judge=)'
Evidence = Literal[
    'evidence',
    'NOT EVIDENCE (coverage not checked: pass judge=)',
    'NOT EVIDENCE',
    'NOT EVIDENCE (coverage cannot be verified)',
]
Coverage = Literal['yes', 'no', 'unverifiable', 'not checked']
ResultKind = Literal['assertion', 'score', 'label']

# The settings an `LLMJudge` result's `EvaluatorSpec` records, compared with the certified identity.
# Defaults are left out of a spec, so a missing key means the default.
_LLM_JUDGE_DEFAULTS: dict[str, Any] = {
    'model': None,
    'include_input': False,
    'include_expected_output': False,
    'model_settings': None,
    'score': False,
    'assertion': {'include_reason': True},
}
# For another evaluator, only these are compared, and only when its spec records them.
_OTHER_SETTINGS = ('rubric', 'model', 'include_input', 'include_expected_output')


def _rubric(spec: Any) -> str | None:
    args = spec.arguments
    if isinstance(args, tuple | list) and args:
        return args[0] if isinstance(args[0], str) else None  # pyright: ignore[reportUnknownVariableType]
    if isinstance(args, Mapping):
        rubric = args.get('rubric')  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        return rubric if isinstance(rubric, str) else None
    return None


def _kwargs(spec: Any) -> dict[str, Any]:
    return dict(spec.arguments) if isinstance(spec.arguments, Mapping) else {}  # pyright: ignore[reportUnknownArgumentType]


def _spec_settings(spec: Any) -> dict[str, Any]:
    """What a result's source records about the evaluator that produced it."""
    args = _kwargs(spec)
    if spec.name == 'LLMJudge':
        return {'rubric': _rubric(spec), **{k: args.get(k, v) for k, v in _LLM_JUDGE_DEFAULTS.items()}}
    out = {k: args[k] for k in _OTHER_SETTINGS if k in args}
    if (rubric := _rubric(spec)) is not None:
        out['rubric'] = rubric
    return out


def _generic_model(identity: Mapping[str, Any], spec: Any) -> str | None:
    """The model name a result's source records, when it is only the generic part of the certified model.

    A `FunctionModel` with no name of its own is identified by its name and a digest of its
    function (`function:function:model:#3f2a...`); the report records the name alone, which every
    such model shares. None when the source records the certified model in full, or another model.
    """
    certified, recorded = identity.get('model'), _spec_settings(spec).get('model')
    if not isinstance(certified, str) or not isinstance(recorded, str) or certified == recorded:
        return None
    return recorded if re.fullmatch(re.escape(recorded) + '#[0-9a-f]+', certified) else None


def _source_changes(identity: Mapping[str, Any], spec: Any) -> list[str]:
    """Settings in which the evaluator a result records as its source differs from the certified one.

    A model the source records only by its generic name (`_generic_model`) is not counted as a
    change here: it cannot be shown to differ, nor to be the same; coverage says so separately.
    """
    changed: set[str] = set()
    if (identity.get('evaluator') == 'LLMJudge') != (spec.name == 'LLMJudge'):
        changed.add('evaluator')
    for key, value in _spec_settings(spec).items():
        if key not in identity:
            continue
        if key == 'model' and _generic_model(identity, spec) is not None:
            continue
        if identity[key] != value:
            changed.add(key)
    return sorted(changed)


def _unreliable(identity: Mapping[str, Any] | None, whose: str) -> list[str]:
    """Why `identity` cannot show which judge it is (`identity_reliable`): another judge can share it."""
    if identity is None or identity_reliable(dict(identity)):
        return []
    opaque = ', '.join(str(item) for item in identity.get('opaque', ())) or 'an unnamed function-backed model'
    return [f'the {whose} identity is not reliable ({opaque}): another judge can share it']


def _custom_source(identity: Mapping[str, Any] | None) -> list[str]:
    """Why a report cannot show that a custom evaluator produced its results, if the certified judge is one.

    A report records an evaluator by its class name and the arguments it serializes; for anything
    but pydantic-evals' own `LLMJudge`, that leaves out its module, its code and any state it does
    not serialize, so two different judges can leave the same record.
    """
    if identity is None or '!code' not in identity:
        return []
    return [
        f'the report records a {identity.get("evaluator")} only by its class name and serialized arguments, '
        'not its module, code or other state, so which one produced these results cannot be told'
    ]


def _outputs(spec: Any) -> int | None:
    """How many results one run of the evaluator `spec` describes produces, when that is known."""
    if spec.name != 'LLMJudge':
        return None
    args = _kwargs(spec)
    return (args.get('score', False) is not False) + (args.get('assertion', {'include_reason': True}) is not False)


def _source_key(spec: Any) -> str:
    return json.dumps({'name': spec.name, 'arguments': spec.arguments}, sort_keys=True, default=str)


def result_names(judge: Any) -> list[str]:
    """The names an `LLMJudge`'s results get in a report, as `LLMJudge.evaluate` names them."""
    return list(result_kinds(judge))


def result_kinds(judge: Any) -> dict[str, ResultKind | None]:
    """The names a judge's results get in a report, each with the kind of result it is.

    For an `LLMJudge`, as `LLMJudge.evaluate` names them; for another evaluator, its default
    evaluation name, of a kind not known in advance (None).
    """
    base: str = judge.get_default_evaluation_name()
    configs: list[tuple[Any, str, ResultKind]] = [
        (getattr(judge, 'score', False), 'score', 'score'),
        (getattr(judge, 'assertion', False), 'pass', 'assertion'),
    ]
    used: list[tuple[Any, str, ResultKind]] = [
        (config, suffix, kind) for config, suffix, kind in configs if config is not False
    ]
    if not used:
        return {base: None}
    out: dict[str, ResultKind | None] = {}
    for config, suffix, kind in used:
        custom = config.get('evaluation_name') if isinstance(config, Mapping) else None  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        out[str(custom) if custom else f'{base}_{suffix}' if len(used) == 2 else base] = kind
    return out


def _check_certifiable(judge: Any) -> None:
    """Refuse, before any judge call, a judge whose verdicts certification cannot read."""
    if type(judge).__name__ == 'LLMJudge' and getattr(judge, 'assertion', None) is False:
        raise ValueError(
            'the judge is score-only (`assertion=False`): certification reads a pass/fail verdict from every '
            'judgment, and this judge gives none. Give it `assertion=` (it can keep `score=`).'
        )


@dataclass(frozen=True)
class _Found:
    name: str
    kind: str
    results: list[EvaluationResult[Any]]


def _groups(case: Any) -> list[tuple[str, Mapping[str, EvaluationResult[Any]]]]:
    return [('assertion', case.assertions), ('score', case.scores), ('label', case.labels)]


def _judge_results(
    report: EvaluationReport[Any, Any, Any], keys: Mapping[str, str | None] | None, identity: Mapping[str, Any] | None
) -> list[_Found]:
    """The judge's results in the report, by name, from every case.

    With `keys` (result name to kind, None for any kind), exactly those. Without, every `LLMJudge`
    result whose rubric is the certified one. Either way, when more than one evaluator in the
    report could be the certified judge, this raises rather than guess which.
    """
    certified_rubric = identity.get('rubric') if identity else None
    found: dict[tuple[str, str], _Found] = {}
    for case in report.cases:
        for kind, results in _groups(case):
            for name, result in results.items():
                if keys is not None:
                    keep = name in keys and keys[name] in (None, kind)
                else:
                    keep = result.source.name == 'LLMJudge' and (
                        certified_rubric is None or _rubric(result.source) == certified_rubric
                    )
                if keep:
                    found.setdefault((name, kind), _Found(name, kind, [])).results.append(result)
    _check_attribution(report, list(found.values()), keys, identity)
    return list(found.values())


def _check_attribution(
    report: EvaluationReport[Any, Any, Any],
    found: Sequence[_Found],
    keys: Mapping[str, str | None] | None,
    identity: Mapping[str, Any] | None,
) -> None:
    """Raise when the report holds results of more than one evaluator that could be the certified judge."""
    sources = {_source_key(r.source): r.source for f in found for r in f.results}

    def ambiguous(detail: str) -> ValueError:
        return ValueError(
            f'more than one evaluator in the report could be the certified judge ({detail}), so which results '
            'the certificate is about cannot be told. Give each judge its own `evaluation_name` and pass it, '
            'or pass the judge object.'
        )

    if keys is not None and not found:  # a mistyped `evaluation_name`, most likely: say so, not "ambiguous"
        present = sorted({name for case in report.cases for _, results in _groups(case) for name in results})
        raise ValueError(
            f'no result in the report is named {", ".join(map(repr, keys))}; '
            f'its results are named {", ".join(map(repr, present)) or "nothing"}'
        )
    if keys is None and len(sources) > 1:
        names = sorted({f.name for f in found})
        raise ambiguous(f'{len(sources)} LLMJudges with results {", ".join(names)}')
    selected = {(f.name, f.kind) for f in found}
    kinds = {s.name for s in sources.values()}
    if identity is not None and identity.get('evaluator') == 'LLMJudge':
        kinds.add('LLMJudge')
    for case in report.cases:
        per_source: dict[str, list[str]] = {}
        for kind, results in _groups(case):
            for name, result in results.items():
                key = _source_key(result.source)
                if key in sources:
                    per_source.setdefault(key, []).append(name)
                elif (
                    (name, kind) not in selected
                    and identity is not None
                    and result.source.name in kinds
                    and not _source_changes(identity, result.source)
                ):
                    raise ambiguous(f'{name!r} in case {case.name!r} also matches the certified configuration')
        for key, names in per_source.items():
            expected = _outputs(sources[key])
            if expected is not None and len(names) > expected:
                raise ambiguous(f'{", ".join(sorted(names))} in case {case.name!r} come from identical judges')


def _summary(found: _Found) -> str:
    values = [r.value for r in found.results]
    if found.kind == 'assertion':
        return f'{sum(bool(v) for v in values)}/{len(values)} passed'
    if found.kind == 'score':
        return f'mean {sum(values) / len(values):.2f}' if values else '-'
    counts: dict[str, int] = {}
    for v in values:
        counts[str(v)] = counts.get(str(v), 0) + 1
    return ', '.join(f'{k}: {n}' for k, n in sorted(counts.items()))


@dataclass(repr=False)
class CertifiedJudgeReport(ReportEvaluator[Any, Any, Any]):
    """Attach `certificate` to the report, and say whether the judge's results in it are evidence.

    Adds three tables to `report.analyses`: the certificate (verdict and fingerprint in the
    title, one row per check), its coverage of the judge that scored, and the judge's results in
    this report, each marked `evidence` or `NOT EVIDENCE` with the reason. Results are evidence
    only when the certificate is ADMISSIBLE and covers the judge; an UNVALIDATED certificate has
    shown neither that the judge works nor that it does not, so its scores are not evidence either.

    Coverage is always checked on what the report records about the evaluator behind each of the
    judge's results (rubric, model name, what it is shown, its settings): the certificate covers
    the results only if they were produced by the certified configuration, whatever judge is
    passed here. When more than one evaluator in the report could be the certified judge, this
    raises rather than guess which.

    Coverage is `unverifiable`, and the results NOT EVIDENCE, when the provenance is incomplete:
    the certified (or given) judge's identity is not reliable, or its model is a `FunctionModel`
    with no name of its own, which the report records only by a generic name (`function:...`), so
    the results of any other function would look the same. Give the model a `model_name` (or use
    a provider model) and coverage is checked as above.

    Args:
        certificate: The judge's certificate, or its `to_dict()` (so a dataset file can carry it).
        judge: The judge that scored the report: an evaluator, or its `judge_identity`. Given, a
            certificate for another configuration of it is flagged (or refused, see
            `on_mismatch`), and its result names pick its results out of the report.
        evaluation_name: The name of the judge's result in the report. By default, the names the
            given judge produces, or every `LLMJudge` result with the certified rubric.
        results: The judge's result names, each with its kind (`assertion`, `score`, `label`, or
            None for any). Filled in from `judge` when this evaluator is serialized, so a rebuilt
            one still picks out the same results.
        on_mismatch: `flag` marks the results NOT EVIDENCE and says what changed (or why coverage
            cannot be verified); `raise` fails this report evaluator in either case, which
            pydantic-evals records under the report's evaluator failures, so no certificate is
            shown for a judge it is not shown to cover.
    """

    certificate: Certificate | dict[str, Any]
    judge: Any = None
    evaluation_name: str | None = None
    on_mismatch: Literal['flag', 'raise'] = 'flag'
    results: Mapping[str, str | None] | None = None
    _cert: Certificate = field(init=False, repr=False)

    def __post_init__(self) -> None:
        cert = self.certificate
        self._cert = cert if isinstance(cert, Certificate) else Certificate.from_dict(cert)

    @property
    def certified(self) -> Certificate:
        return self._cert

    def build_serialization_arguments(self) -> dict[str, Any]:
        """The certificate without its judgments, the judge as its identity and its result names: plain data."""
        out: dict[str, Any] = {'certificate': self._cert.to_dict(judgments=False)}
        if self.judge is not None:
            out['judge'] = self._identity()
        if self.evaluation_name is not None:
            out['evaluation_name'] = self.evaluation_name
        if self.on_mismatch != 'flag':
            out['on_mismatch'] = self.on_mismatch
        keys = self._keys()
        if self.evaluation_name is None and keys is not None:
            out['results'] = dict(keys)
        return out

    def _identity(self) -> dict[str, Any] | None:
        if self.judge is None:
            return None
        return dict(self.judge) if isinstance(self.judge, Mapping) else judge_identity(self.judge)  # pyright: ignore[reportUnknownArgumentType]

    def _keys(self) -> dict[str, str | None] | None:
        """The judge's results in the report, by name and kind, when they can be told."""
        if self.evaluation_name:
            return {self.evaluation_name: None}
        if self.results is not None:
            return dict(self.results)
        if self.judge is not None and not isinstance(self.judge, Mapping):
            return dict(result_kinds(self.judge))
        return None

    def evaluate(self, ctx: ReportEvaluatorContext[Any, Any, Any]) -> list[ReportAnalysis]:
        return self.analyses(ctx.report)

    def analyses(self, report: EvaluationReport[Any, Any, Any]) -> list[ReportAnalysis]:
        """The tables `evaluate` adds, for a report already built."""
        cert = self._cert
        found = _judge_results(report, self._keys(), cert.identity)

        identity = self._identity()
        judge_changed = differences(cert.identity, identity) if identity is not None and cert.identity else []
        spec_changed = self._spec_changes(found)
        changed = sorted({*judge_changed, *spec_changed})
        unverifiable = [
            *_unreliable(cert.identity, 'certified'),
            *(_unreliable(identity, "given judge's") if identity != cert.identity else []),
            *self._generic_sources(found),
            *_custom_source(cert.identity),
        ]
        coverage: Coverage
        if changed:
            coverage = 'no'
        elif unverifiable and cert.identity is not None:
            coverage = 'unverifiable'
        elif identity is not None and cert.identity is not None:
            coverage = 'yes'
        else:
            coverage = 'not checked'
        if changed and self.on_mismatch == 'raise':
            raise InadmissibleJudge(
                f'the certificate ({cert.fingerprint}) is for a different configuration of the judge: '
                f'{", ".join(changed)} changed. Certify the judge as it is now.'
            )
        if coverage == 'unverifiable' and self.on_mismatch == 'raise':
            raise InadmissibleJudge(
                f'whether the certificate ({cert.fingerprint}) covers the judge that scored cannot be verified: '
                f"{'; '.join(unverifiable)}. Give the judge's model a name of its own."
            )
        return [
            self._certificate_table(),
            self._coverage_table(identity, coverage, judge_changed, spec_changed, unverifiable),
            self._evidence_table(found, coverage, changed, unverifiable),
        ]

    def _generic_sources(self, found: Sequence[_Found]) -> list[str]:
        """Why the report's record of the judge's model cannot tell it from another function's, if it cannot."""
        identity = self._cert.identity
        if identity is None:
            return []
        generic = sorted({g for f in found for r in f.results if (g := _generic_model(identity, r.source))})
        return [
            f'the report records the model only as {name!r}, the name every unnamed FunctionModel of that function '
            'name shares, so which function produced these results cannot be told'
            for name in generic
        ]

    def _spec_changes(self, found: Sequence[_Found]) -> list[str]:
        """Settings in which the evaluators that produced the results, as the report records them, differ."""
        identity = self._cert.identity
        if identity is None:
            return []
        changed: set[str] = set()
        for f in found:
            for result in f.results:
                changed.update(_source_changes(identity, result.source))
        return sorted(changed)

    def _certificate_table(self) -> TableResult:
        cert = self._cert
        rows: list[list[str | int | float | bool | None]] = []
        for c in cert.checks:
            low, high = c.interval
            rows.append(
                [c.name, c.status, None if c.rate is None else round(c.rate, 3), f'[{low:.2f}, {high:.2f}]',
                 c.threshold, c.trials, c.detail]
            )  # fmt: skip
        calls = f'{cert.calls} of {cert.planned} judge calls' if cert.calls is not None else 'calls not recorded'
        name, fp = cert.judge or 'judge', cert.fingerprint or 'none'
        return TableResult(
            title=f'Judge certificate: {name} {cert.verdict} (fingerprint {fp})',
            description=f'{calls}; a check passes on its 95% lower bound and fails on an exact upper bound.',
            columns=['check', 'status', 'rate', '95% interval', 'needs', 'n', 'detail'],
            rows=rows,
        )

    def _coverage_table(
        self,
        identity: dict[str, Any] | None,
        coverage: Coverage,
        judge_changed: list[str],
        spec_changed: list[str],
        unverifiable: list[str],
    ) -> TableResult:
        cert = self._cert
        if cert.identity is None:
            why = 'the certificate recorded no judge configuration, so what it covers cannot be shown'
        elif judge_changed:
            why = f'{", ".join(judge_changed)} changed since certification'
        elif spec_changed:
            why = f'the report records a judge whose {", ".join(spec_changed)} differ from the certified one'
        elif coverage == 'unverifiable':
            why = '; '.join(unverifiable) + ". Give the judge's model a name of its own"
        elif identity is not None:
            why = "same configuration as certified, and the report records the results as that judge's"
        else:
            why = 'no judge given: only rubric, inputs shown and model name were compared, from the report'
        return TableResult(
            title=f'Certificate coverage: {coverage}',
            columns=['certified fingerprint', 'judge fingerprint', 'covers', 'why'],
            rows=[[cert.fingerprint, fingerprint(identity) if identity is not None else None, coverage, why]],
        )

    def _evidence_table(
        self, found: Sequence[_Found], coverage: Coverage, changed: list[str], unverifiable: list[str]
    ) -> TableResult:
        cert = self._cert
        status: Evidence
        if changed:
            status, why = 'NOT EVIDENCE', 'the certificate is about another configuration of the judge'
        elif cert.verdict == 'INADMISSIBLE':
            failed = ', '.join(c.name for c in cert.failures())
            status, why = 'NOT EVIDENCE', f'the judge failed certification ({failed})'
        elif cert.verdict == 'UNVALIDATED':
            open_checks = ', '.join(c.name for c in cert.checks if c.status != 'PASS')
            status, why = 'NOT EVIDENCE', f'certification did not decide ({open_checks} UNVALIDATED)'
        elif coverage == 'unverifiable':
            status, why = UNVERIFIABLE, '; '.join(unverifiable)
        elif coverage == 'yes':
            status, why = 'evidence', 'ADMISSIBLE, and the certificate covers this judge'
        else:
            # A matching rubric is not a matching judge (the model, its function, its settings), and a
            # certificate with no identity says nothing of which judge it was: never evidence on trust.
            status, why = UNCHECKED, 'ADMISSIBLE, but nothing shows it covers this judge; pass judge= to check'
        rows: list[list[str | int | float | bool | None]] = [
            [f.name, f.kind, len(f.results), _summary(f), status, why] for f in found
        ]
        if not rows:
            rows = [[self.evaluation_name, None, 0, 'no results from this judge in the report', status, why]]
        return TableResult(
            title=f'Judge results in this report: {status}',
            description='The scores are as the judge gave them; this says only whether they count as evidence.',
            columns=['result', 'kind', 'runs', 'in this report', 'status', 'why'],
            rows=rows,
        )


def _known_cases(cases: Sequence[Any]) -> list[JudgeCase]:
    """Each case with an expected output, once, as a known-good answer (repeats share a source case)."""
    seen: dict[str, JudgeCase] = {}
    for i, c in enumerate(cases):
        if c.expected_output is None:
            continue
        name = getattr(c, 'source_case_name', None) or c.name or f'case {i}'
        seen.setdefault(name, JudgeCase(name, c.inputs, c.expected_output, c.expected_output, c.metadata))
    return list(seen.values())


def _load_judge(spec: Any, custom_types: Sequence[type[Evaluator[Any, Any, Any]]]) -> Evaluator[Any, Any, Any]:
    """Rebuild an evaluator from its spec, as a dataset file does: pydantic-evals' defaults, then `custom_types`."""
    registry: dict[str, type[Evaluator[Any, Any, Any]]] = {}
    for cls in (*custom_types, *DEFAULT_EVALUATORS):
        registry.setdefault(cls.get_serialization_name(), cls)
    parsed = EvaluatorSpec.model_validate(spec)
    cls = registry.get(parsed.name)
    if cls is None:
        raise ValueError(
            f'the judge {parsed.name!r} is not a pydantic-evals evaluator; pass its class in `custom_evaluator_types`'
        )
    return cls(*parsed.args, **parsed.kwargs)


@dataclass(repr=False)
class CertifyJudgeReport(ReportEvaluator[Any, Any, Any]):
    """Certify `judge` on the report's cases while the report is built, then attach the result.

    The known-good answers are the cases' `expected_output`, not the task's outputs: the
    certificate is about the judge, not the task. This calls the judge again on every case and
    control (`repeats` times per case), so it costs more than the evaluation itself. Fewer than
    `min_cases` cases with an expected output and nothing is certified: the report says so and
    marks the judge's results NOT EVIDENCE.

    Arguments as for `certify_judge`; `controls` defaults to `controls_for(judge.rubric)`. A
    score-only `LLMJudge` (`assertion=False`) is refused here, before any judge call.

    Serialized (in a dataset file), the judge is written as its evaluator spec and rebuilt from
    pydantic-evals' default evaluators, or from `custom_evaluator_types` for a judge of your own.
    Only a judge that comes back as the same judge can be rebuilt: an `LLMJudge` with a model
    name, not a model object. A judge that cannot, or custom `controls` (code, not data), is
    written with the reason, and rebuilding it raises with that reason.
    """

    judge: Any
    controls: Sequence[Control] | None = None
    repeats: int = 2
    min_cases: int = 10
    thresholds: Thresholds = DEFAULT_THRESHOLDS
    batch_size: int | None = None
    max_concurrency: int = 8
    evaluation_name: str | None = None
    custom_evaluator_types: Sequence[type[Evaluator[Any, Any, Any]]] = ()
    unserializable: str | None = None
    """Set only by serialization: why this evaluator cannot be rebuilt from its spec."""

    def __post_init__(self) -> None:
        if self.unserializable is not None:
            raise ValueError(f'CertifyJudgeReport cannot be rebuilt from its spec: {self.unserializable}')
        if isinstance(self.judge, str | Mapping):
            self.judge = _load_judge(self.judge, self.custom_evaluator_types)
        if isinstance(self.thresholds, Mapping):
            self.thresholds = Thresholds(**self.thresholds)  # pyright: ignore[reportUnknownArgumentType]
        _check_certifiable(self.judge)

    def build_serialization_arguments(self) -> dict[str, Any]:
        """The judge as its evaluator spec and the settings that differ from the defaults, or why it cannot be rebuilt.

        Never raises: pydantic-evals serializes a report evaluator when recording its failure.
        """
        why = self._unserializable()
        out: dict[str, Any] = (
            {'judge': None, 'unserializable': why} if why else {'judge': self.judge.as_spec().model_dump(mode='json')}
        )
        defaults = {'repeats': 2, 'min_cases': 10, 'batch_size': None, 'max_concurrency': 8, 'evaluation_name': None}
        out.update({k: getattr(self, k) for k, v in defaults.items() if getattr(self, k) != v})
        if self.thresholds != DEFAULT_THRESHOLDS:
            out['thresholds'] = dataclasses.asdict(self.thresholds)
        return out

    def _unserializable(self) -> str | None:
        if self.controls is not None:
            return 'custom controls are code, not data; construct it in code with its controls'
        try:
            spec = json.loads(json.dumps(self.judge.as_spec().model_dump(mode='json')))
            rebuilt = _load_judge(spec, (*self.custom_evaluator_types, type(self.judge)))
            before, after = judge_identity(self.judge), judge_identity(rebuilt)
        except Exception as e:
            return f'the judge cannot be serialized ({type(e).__name__}: {e}); construct it in code'
        model = getattr(rebuilt, 'model', None)
        if isinstance(model, str) and model.startswith('function:'):
            # A named FunctionModel keeps its identity in the spec, but a name is not its function:
            # no model of that name can be built from the spec.
            return f'its model {model!r} is a FunctionModel, code that a spec cannot hold; construct it in code'
        changed = differences(before, after)
        if not changed:
            return None
        detail = '; '.join(f'{k} {before.get(k)!r} would come back as {after.get(k)!r}' for k in changed)
        return (
            f'the judge does not survive serialization ({detail}). Give it a model name, not a model '
            'object, or construct it in code.'
        )

    async def evaluate(self, ctx: ReportEvaluatorContext[Any, Any, Any]) -> list[ReportAnalysis]:
        known = _known_cases(ctx.report.cases)
        if len(known) < self.min_cases:
            return [
                TableResult(
                    title='Judge certificate: not certified',
                    columns=['cases with an expected output', 'needed', 'status'],
                    rows=[[len(known), self.min_cases, 'NOT EVIDENCE: the judge was not certified']],
                )
            ]
        rubric = getattr(self.judge, 'rubric', '')
        certificate = await certify_judge(
            self.judge,
            known,
            controls=self.controls if self.controls is not None else controls_for(rubric),
            repeats=self.repeats,
            thresholds=self.thresholds,
            batch_size=self.batch_size,
            max_concurrency=self.max_concurrency,
        )
        return CertifiedJudgeReport(certificate, self.judge, self.evaluation_name).analyses(ctx.report)


async def certify_for_report(
    dataset: Any,
    *,
    judge: Any = None,
    evaluation_name: str | None = None,
    controls: Sequence[Control] | None = None,
    repeats: int = 2,
    min_cases: int = 10,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    batch_size: int | None = None,
    max_concurrency: int = 8,
) -> CertifiedJudgeReport:
    """Certify one `LLMJudge` of `dataset` and return a report evaluator that attaches its certificate.

    The judge is `judge` (the object in the dataset), or the one whose results are named
    `evaluation_name`, or the dataset's only `LLMJudge`. It is certified on the cases it grades
    that have an expected output, as `certify_dataset` does. Add the result to
    `dataset.report_evaluators` before `dataset.evaluate(task)`.
    """
    candidates = _judges(dataset)
    if judge is not None:
        picked = [c for c in candidates if c[1] is judge]
    elif evaluation_name is not None:
        picked = [c for c in candidates if evaluation_name in result_names(c[1])]
    else:
        picked = candidates
    if len(picked) != 1:
        raise ValueError(
            f'{len(picked)} LLMJudges of the dataset match; pass the judge object or a unique `evaluation_name`'
        )
    _, found, cases = picked[0]
    _check_certifiable(found)
    known = _known_cases(cases)
    if len(known) < min_cases:
        raise ValueError(f'the judge grades {len(known)} case(s) with an expected output; at least {min_cases} needed')
    certificate = await certify_judge(
        found,
        known,
        controls=controls if controls is not None else controls_for(found.rubric),
        repeats=repeats,
        thresholds=thresholds,
        batch_size=batch_size,
        max_concurrency=max_concurrency,
    )
    return CertifiedJudgeReport(certificate, found, evaluation_name)


__all__ = ['CertifiedJudgeReport', 'CertifyJudgeReport', 'certify_for_report', 'result_kinds', 'result_names']
