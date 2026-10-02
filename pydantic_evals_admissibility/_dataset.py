"""Certify every `LLMJudge` in a pydantic-evals dataset with one call.

`certify_dataset(dataset)` finds the dataset's judges, those that apply to every case and those
attached to single cases, uses each case's `expected_output` as its known-good answer, picks
controls from what the rubric grades, and returns a certificate and the doctor's advice per judge.

Two things it decides for you, and says so in the result:

- **What the rubric grades.** A rubric about correctness gets the default controls; one about
  style or tone gets controls for that (another case's answer must not change the verdict). The
  guess is read from the rubric's words; pass `controls=` to override it.
- **Where to apply controls in structured outputs.** Empty and whitespace controls change only the
  prose fields (strings with a space in them), never timestamps or ids, so a "whitespace only"
  change cannot change the answer.
"""

from __future__ import annotations

import dataclasses
import random
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from ._cases import JudgeCase
from ._certify import DEFAULT_THRESHOLDS, Certificate, Thresholds, certify_judge
from ._controls import Control, ControlKind, MismatchedOutput
from ._diagnose import _ABOUT_STYLE, diagnose

RubricKind = Literal['correctness', 'style']


def _is_prose(value: Any) -> bool:
    """Words, not a timestamp or an id: at least three tokens, two of them alphabetic words.

    `"2026-10-02 10:00:00"` and `"ACME 007"` are not prose; `"You asked for May."` is.
    """
    if not isinstance(value, str):
        return False
    tokens = value.split()
    return len(tokens) >= 3 and sum(t.strip('.,;:!?\'"()').isalpha() for t in tokens) >= 2


def map_prose(output: Any, change: Callable[[str], str]) -> Any | None:
    """Apply `change` to the prose in `output`: the string itself, or its prose fields.

    Works on strings, dicts, Pydantic models and dataclasses. Returns None when there is no prose.
    """
    if isinstance(output, str):
        return change(output) if output.strip() else None
    if isinstance(output, dict):
        keys = [k for k, v in output.items() if _is_prose(v)]
        return {**output, **{k: change(output[k]) for k in keys}} if keys else None
    if hasattr(output, 'model_dump') and hasattr(output, 'model_copy'):
        fields = {k: v for k, v in output.model_dump().items() if _is_prose(v)}
        return output.model_copy(update={k: change(v) for k, v in fields.items()}) if fields else None
    if dataclasses.is_dataclass(output) and not isinstance(output, type):
        fields = {
            f.name: getattr(output, f.name) for f in dataclasses.fields(output) if _is_prose(getattr(output, f.name))
        }
        return dataclasses.replace(output, **{k: change(v) for k, v in fields.items()}) if fields else None
    return None


@dataclass(frozen=True)
class EmptyProse:
    """Every prose field emptied: nothing left to read. Must fail under any rubric."""

    name: str = 'empty_output'
    kind: ControlKind = 'must_fail'

    def make(self, case: JudgeCase, cases: Sequence[JudgeCase], rng: random.Random) -> Any | None:
        return map_prose(case.output, lambda text: '')


@dataclass(frozen=True)
class ProseWhitespace:
    """Every prose field with doubled spaces and a trailing newline. The verdict must not change."""

    name: str = 'whitespace_reformat'
    kind: ControlKind = 'must_hold'

    def make(self, case: JudgeCase, cases: Sequence[JudgeCase], rng: random.Random) -> Any | None:
        return map_prose(case.output, lambda text: text.replace(' ', '  ') + '\n')


_ABOUT_BEING_RIGHT = re.compile(r'\b(correct\w*|accura\w*|right|true|factual\w*|valid|wrong|incorrect)\b', re.I)


def rubric_kind(rubric: str) -> RubricKind:
    """'style' when the rubric talks about style, tone or format and not about being right.

    A guess from the rubric's words, reported with the result; pass `controls=` to decide it
    yourself. The word "answer" says nothing either way ("the answer is friendly" is about style).
    """
    style = bool(_ABOUT_STYLE.search(rubric))
    correctness = bool(_ABOUT_BEING_RIGHT.search(rubric))
    return 'style' if style and not correctness else 'correctness'


def controls_for(rubric: str) -> tuple[Control, ...]:
    if rubric_kind(rubric) == 'style':
        return (EmptyProse(), MismatchedOutput(kind='must_hold'), ProseWhitespace())
    return (MismatchedOutput(), EmptyProse(), ProseWhitespace())


@dataclass
class DatasetJudgeCertificate:
    """One judge of a dataset, the cases it grades, and what certifying it found."""

    label: str
    rubric: str
    kind: RubricKind
    cases: int
    certificate: Certificate | None
    advice: list[str] = field(default_factory=list)
    skipped: str | None = None
    include_input: bool = False
    include_expected_output: bool = False

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready, with every verdict; `from_dict` rebuilds it to re-read or re-diagnose later."""
        out: dict[str, Any] = {
            'label': self.label,
            'rubric': self.rubric,
            'kind': self.kind,
            'cases': self.cases,
            'include_input': self.include_input,
            'include_expected_output': self.include_expected_output,
            'skipped': self.skipped,
            'advice': self.advice,
        }
        if self.certificate is not None:
            out['certificate'] = self.certificate.to_dict()
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetJudgeCertificate:
        cert = data.get('certificate')
        return cls(
            data['label'],
            data.get('rubric', ''),
            data.get('kind', 'correctness'),
            data.get('cases', 0),
            Certificate.from_dict(cert) if cert else None,
            list(data.get('advice', [])),
            data.get('skipped'),
            data.get('include_input', False),
            data.get('include_expected_output', False),
        )


def _judges(dataset: Any) -> list[tuple[str, Any, list[Any]]]:
    """(label, judge, cases it applies to) for every LLMJudge in the dataset."""
    from pydantic_evals.evaluators import LLMJudge

    found: list[tuple[str, Any, list[Any]]] = []
    for judge in dataset.evaluators:
        if isinstance(judge, LLMJudge):
            found.append(('dataset', judge, list(dataset.cases)))
    by_config: dict[tuple[Any, ...], tuple[Any, list[Any]]] = {}
    for case in dataset.cases:
        for judge in case.evaluators:
            if isinstance(judge, LLMJudge):
                by_config.setdefault(_identity(judge), (judge, []))[1].append(case)
    for judge, cases in by_config.values():
        found.append((f'{len(cases)} case(s): ' + ', '.join(str(c.name) for c in cases[:3]), judge, cases))
    return found


def _identity(judge: Any) -> tuple[Any, ...]:
    """Case-level judges are certified together only when they are the same judge in every setting."""
    model = judge.model if isinstance(judge.model, str | type(None)) else id(judge.model)
    return (
        judge.rubric,
        model,
        judge.include_input,
        judge.include_expected_output,
        repr(judge.model_settings),
        repr(judge.score),
        repr(judge.assertion),
    )


async def certify_dataset(
    dataset: Any,
    *,
    model: Any = None,
    controls: Sequence[Control] | None = None,
    repeats: int = 2,
    min_cases: int = 10,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    batch_size: int | None = None,
    max_concurrency: int = 8,
) -> list[DatasetJudgeCertificate]:
    """Certify every `LLMJudge` in `dataset`, using each case's `expected_output` as a good answer.

    Args:
        dataset: A `pydantic_evals.Dataset`.
        model: Certify every judge with this model instead of its own (for example a cheaper one).
        controls: Use these controls for every judge instead of choosing them from each rubric.
        repeats: Judgments per good answer, for the stability check.
        min_cases: Judges that grade fewer cases with an expected output are reported, not run.
        thresholds, batch_size, max_concurrency: As for `certify_judge`.
    """
    results: list[DatasetJudgeCertificate] = []
    for scope, judge, cases in _judges(dataset):
        if model is not None:
            judge = dataclasses.replace(judge, model=model)
        kind = rubric_kind(judge.rubric)
        position = {id(c): i for i, c in enumerate(dataset.cases)}
        known = [
            JudgeCase(
                c.name or f'case {position[id(c)]}',  # unnamed cases are valid in a Dataset
                c.inputs,
                c.expected_output,
                expected_output=c.expected_output,
                metadata=c.metadata,
            )
            for c in cases
            if c.expected_output is not None
        ]
        label = f'{scope}: {judge.rubric[:60]}'
        flags = {'include_input': judge.include_input, 'include_expected_output': judge.include_expected_output}
        if judge.assertion is False:
            results.append(
                DatasetJudgeCertificate(
                    label,
                    judge.rubric,
                    kind,
                    len(known),
                    None,
                    skipped='scores only (assertion=False): there is no pass/fail verdict to certify',
                    **flags,
                )  # fmt: skip
            )
            continue
        if len(known) < min_cases:
            results.append(
                DatasetJudgeCertificate(
                    label,
                    judge.rubric,
                    kind,
                    len(known),
                    None,
                    skipped=f'grades {len(known)} case(s) with an expected output; at least {min_cases} are needed',
                    **flags,
                )
            )
            continue
        cert = await certify_judge(
            judge,
            known,
            controls=controls or controls_for(judge.rubric),
            repeats=repeats,
            thresholds=thresholds,
            batch_size=batch_size,
            max_concurrency=max_concurrency,
        )
        results.append(
            DatasetJudgeCertificate(label, judge.rubric, kind, len(known), cert, diagnose(cert, judge), **flags)
        )
    return results


def dataset_report(results: Sequence[DatasetJudgeCertificate]) -> str:
    """Plain text: one block per judge, its verdict, table and advice."""
    blocks = []
    for r in results:
        head = f'{r.label}  [{r.kind} rubric, {r.cases} cases]'
        if r.certificate is None:
            blocks.append(f'{head}\n  not certified: {r.skipped}')
            continue
        lines = [head, r.certificate.table()] + [f'  - {a}' for a in r.advice]
        blocks.append('\n'.join(lines))
    return '\n\n'.join(blocks)


__all__ = [
    'DatasetJudgeCertificate',
    'EmptyProse',
    'ProseWhitespace',
    'certify_dataset',
    'controls_for',
    'dataset_report',
    'map_prose',
    'rubric_kind',
]
