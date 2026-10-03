"""Put a certificate in Logfire, next to the traces of the judge it certifies.

A certificate lives in a JSON file; the judge's scores live in traces. `log_certificate` emits it
as spans, so whoever is looking at a judge's verdicts in Logfire (or any OpenTelemetry backend)
can find whether that judge was certified, on what evidence, and for which configuration:

    judge certificate {judge}: {verdict}       verdict, fingerprint, identity, every check's numbers
      check {check}: {status}                  one per check, its interval and its control families
        {check} {family}: {wrong} of {judged} judged wrongly   one per FAILed family, with examples

The examples are the judgments that made a family fail (case, role, reason, the output shown),
capped at `max_examples` and truncated, so a failing certificate leads to its evidence without
putting every judgment in a trace. The full record stays in `Certificate.to_dict()`.

`certify_judge_traced` runs `certify_judge` inside a span, so each judge call's own spans (an
`LLMJudge`'s agent runs, when pydantic-ai is instrumented) sit under it, beside the certificate
they produced.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from ._cases import JudgeCase
from ._certify import Certificate, Check, Judgment, _describe, certify_judge
from ._diagnose import _failed_families, _family_results
from ._identity import fingerprint, judge_identity

_FAMILY_CHECKS = {'rejection': 'must_fail:', 'invariance': 'must_hold:'}


def _clip(text: Any, limit: int) -> Any:
    if not isinstance(text, str) or len(text) <= limit:
        return text
    return text[: limit - 1] + '…'


def _bounded(value: Any, limit: int) -> Any:
    """A setting as an attribute: strings clipped, structures kept unless their JSON is too long."""
    if isinstance(value, str):
        return _clip(value, limit)
    if isinstance(value, (dict, list, tuple)):
        dumped = json.dumps(value, sort_keys=True, default=str)
        return value if len(dumped) <= limit else _clip(dumped, limit)
    return value


def _present(attributes: dict[str, Any]) -> dict[str, Any]:
    """Logfire records None as the string 'null'; an absent attribute says the same, more cheaply."""
    return {k: v for k, v in attributes.items() if v is not None}


def _dataset_name(dataset: Any) -> str | None:
    if dataset is None or isinstance(dataset, str):
        return dataset
    return getattr(dataset, 'name', None) or repr(dataset)


def _families(check: Check) -> list[dict[str, Any]]:
    """The per-family counts of a rejection or invariance check, from `check.families` or, saved before it
    existed, from its detail.

    Certificates saved before families were decided one by one carry no status marks; for those
    the status is left out rather than guessed.
    """
    if check.name not in _FAMILY_CHECKS:
        return []
    return [
        _present(
            {
                'family': f.name,
                'successes': f.successes,
                'trials': f.trials,
                'errors': f.errors or None,
                'status': f.status,
            }
        )
        for f in _family_results(check)
    ]


def _majorities(judgments: Sequence[Judgment]) -> dict[str, bool]:
    """Each case's majority verdict on its known-good answer: what a must-hold control must match."""
    verdicts: dict[str, list[bool]] = defaultdict(list)
    for j in judgments:
        if j.role.startswith('reference#') and j.passed is not None:
            verdicts[j.case].append(j.passed)
    return {case: sum(v) * 2 > len(v) for case, v in verdicts.items()}


def _wrong(certificate: Certificate, check: str, family: str | None) -> list[Judgment]:
    """The judgments that count against `check` (and `family`), in the order they were made.

    `family` is a control family's name for rejection and invariance, and None for acceptance,
    which has none: a control family may be called anything, 'reference' included.
    """
    judgments = certificate.judgments
    if check == 'acceptance':
        return [j for j in judgments if j.role == 'reference#0' and j.passed is False]
    if check == 'rejection':
        return [j for j in judgments if j.role == f'must_fail:{family}' and j.passed is True]
    if check == 'invariance':
        majority = _majorities(judgments)
        return [
            j
            for j in judgments
            if j.role == f'must_hold:{family}'
            and j.passed is not None
            and j.case in majority
            and j.passed != majority[j.case]
        ]
    return []


def _example(judgment: Judgment, text_limit: int) -> dict[str, Any]:
    output = judgment.output if isinstance(judgment.output, str) else repr(judgment.output)
    return _present(
        {
            'case': judgment.case,
            'role': judgment.role,
            'passed': judgment.passed,
            'reason': _clip(judgment.reason, text_limit),
            'output': _clip(output, text_limit),
        }
    )


def _evidence(certificate: Certificate, check: Check, max_examples: int, text_limit: int) -> list[dict[str, Any]]:
    """One record per failed family of `check` (or one for a failed acceptance), with capped examples."""
    if check.status != 'FAIL':
        return []
    # (label shown, family the judgments are filtered by, wrong, judged). Acceptance has no family
    # (None), kept apart from the family names, which a user's control may give any value.
    failed: list[tuple[str, str | None, int, int]]
    if check.name == 'acceptance':
        failed = [('reference', None, check.trials - check.successes, check.trials)]
    elif check.name in _FAMILY_CHECKS:
        counts = {f['family']: f for f in _families(check)}
        failed = [
            (name, name, counts[name]['trials'] - counts[name]['successes'], counts[name]['trials'])
            for name in _failed_families(check)
            if name in counts
        ]
    else:
        return []
    records = []
    for family, selected, wrong, judged in failed:
        found = _wrong(certificate, check.name, selected)
        records.append(
            {
                'check': check.name,
                'family': family,
                'wrong': wrong,
                'judged': judged,
                'judgments_saved': bool(certificate.judgments),
                'examples': [_example(j, text_limit) for j in found[:max_examples]],
                'examples_omitted': max(0, len(found) - max_examples),
            }
        )
    return records


def certificate_attributes(
    certificate: Certificate,
    *,
    judge: Any = None,
    dataset: Any = None,
    text_limit: int = 1000,
) -> dict[str, Any]:
    """The certificate as flat, JSON-serializable span attributes, one key per number.

    Flat keys (`check.rejection.status`, `judge.model`) so they can be filtered on directly in
    Logfire. With `judge`, also whether the certificate covers that judge as configured now, and
    what changed if not. `dataset` is a name, or anything with a `.name` (a pydantic-evals Dataset).
    """
    out: dict[str, Any] = {
        'judge': certificate.judge,
        'verdict': certificate.verdict,
        'admissible': certificate.admissible,
        'fingerprint': certificate.fingerprint,
        'calls': certificate.calls,
        'planned': certificate.planned,
        'looks': certificate.looks,
        'failed_checks': [c.name for c in certificate.failures()],
    }
    out['dataset'] = _dataset_name(dataset)
    for key, value in (certificate.identity or {}).items():
        out[f'judge.{key}'] = _bounded(value, text_limit)
    if judge is not None:
        changed = certificate.differences(judge)
        out['covers_judge'] = certificate.identity is not None and not changed
        out['judge_changed'] = changed
    for check in certificate.checks:
        low, high = check.interval
        out.update(
            {
                f'check.{check.name}.status': check.status,
                f'check.{check.name}.rate': check.rate,
                f'check.{check.name}.low': low,
                f'check.{check.name}.high': high,
                f'check.{check.name}.threshold': check.threshold,
                f'check.{check.name}.trials': check.trials,
                f'check.{check.name}.errors': check.errors,
            }
        )
    return _present(out)


def log_certificate(
    certificate: Certificate,
    *,
    judge: Any = None,
    dataset: Any = None,
    max_examples: int = 5,
    text_limit: int = 300,
) -> None:
    """Emit `certificate` as a Logfire span tree under the current span, if any.

    One `judge certificate {judge}: {verdict}` span with every number as an attribute; under it
    one `check {check}: {status}` span per check; under a FAILed check, one warning per failed
    control family (or the failed acceptance) with up to `max_examples` of the judgments that
    failed it, their reason and output clipped to `text_limit` characters.

    Args:
        certificate: The certificate, fresh or rebuilt with `Certificate.from_dict`.
        judge: The judge as configured now; records whether the certificate still covers it.
        dataset: The dataset, or its name, the judge was certified on.
        max_examples: Example judgments per failed family. The count of the rest is recorded.
        text_limit: Characters kept of each reason, output and identity setting.
    """
    import logfire

    if max_examples < 0:
        raise ValueError('max_examples must be >= 0')
    attributes = certificate_attributes(certificate, judge=judge, dataset=dataset, text_limit=max(text_limit, 1000))
    level = 'warn' if certificate.verdict == 'INADMISSIBLE' else 'info'
    with logfire.span('judge certificate {judge}: {verdict}', _level=level, **attributes):
        for check in certificate.checks:
            low, high = check.interval
            check_attributes = _present(
                {
                    'check': check.name,
                    'status': check.status,
                    'rate': check.rate,
                    'low': low,
                    'high': high,
                    'threshold': check.threshold,
                    'successes': check.successes,
                    'trials': check.trials,
                    'errors': check.errors,
                    'detail': _clip(check.detail, text_limit) or None,
                    'families': _families(check) or None,
                }
            )
            with logfire.span(
                'check {check}: {status}', _level='warn' if check.status == 'FAIL' else 'info', **check_attributes
            ):
                for record in _evidence(certificate, check, max_examples, text_limit):
                    logfire.log('warn', '{check} {family}: {wrong} of {judged} judged wrongly', attributes=record)


async def certify_judge_traced(
    judge: Any,
    cases: Sequence[JudgeCase],
    *,
    dataset: Any = None,
    max_examples: int = 5,
    **options: Any,
) -> Certificate:
    """`certify_judge`, inside a `certify judge {judge}` span that ends with the certificate under it.

    The judge's own spans nest under the same span: turn on `logfire.instrument_pydantic_ai()`, or
    pass an `LLMJudge` a model wrapped by `logfire.instrument_pydantic_ai(model)`, and every judge
    call is a child, next to the `judge certificate` span that says what those calls showed.
    `options` are `certify_judge`'s keyword arguments.
    """
    import logfire

    name = _describe(judge)
    with logfire.span(
        'certify judge {judge}',
        judge=name,
        fingerprint=fingerprint(judge_identity(judge)),
        cases=len(cases),
        **_present({'dataset': _dataset_name(dataset)}),
    ) as span:
        certificate = await certify_judge(judge, cases, **options)
        log_certificate(certificate, judge=judge, dataset=dataset, max_examples=max_examples)
        span.set_attributes(_present({'verdict': certificate.verdict, 'calls': certificate.calls}))
        span.message = f'certify judge {name}: {certificate.verdict}'
        if certificate.verdict == 'INADMISSIBLE':
            span.set_level('warn')
    return certificate
