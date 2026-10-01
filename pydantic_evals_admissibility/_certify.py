"""Certify a judge before its scores are used: it must pass what is good and fail what is not.

`certify_judge` runs any pydantic-evals `Evaluator` that produces a pass/fail assertion, such as
`LLMJudge`, over known-good answers and over controls built from them, and returns a
`Certificate`. Every rate is reported with a Wilson interval, and each check is decided on the
interval's conservative bound, not the point estimate, so ten lucky verdicts do not certify a
judge.

A judge whose verdicts are all "pass" clears every invariance and stability check; it is caught
by rejection. One whose verdicts are all "fail" clears rejection; it is caught by acceptance.
Neither check alone is a certificate, which is why both are required.
"""

from __future__ import annotations

import asyncio
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext
from pydantic_evals.otel._errors import SpanTreeRecordingError

from ._cases import HumanLabel, JudgeCase
from ._controls import DEFAULT_CONTROLS, Control
from ._stats import cohen_kappa, wilson

CheckStatus = Literal['PASS', 'FAIL', 'UNVALIDATED']
Verdict = Literal['ADMISSIBLE', 'INADMISSIBLE', 'UNVALIDATED']


@dataclass(frozen=True)
class Thresholds:
    """What a judge must show. Rates are compared on their Wilson interval, never the estimate.

    A check passes when the interval's lower bound clears the threshold and fails when its upper
    bound is below it. In between, the cases cannot tell, and the check is UNVALIDATED.
    """

    min_acceptance: float = 0.7
    """Lower bound on the share of known-good answers the judge passes."""
    min_rejection: float = 0.8
    """Lower bound on the share of `must_fail` controls the judge fails."""
    min_invariance: float = 0.8
    """Lower bound on the share of `must_hold` controls whose verdict matches the original's."""
    min_stability: float = 0.8
    """Lower bound on the share of cases judged identically on every repeat."""
    min_kappa: float = 0.4
    """Cohen's kappa against human labels, when labels are given."""
    min_trials: int = 10
    """Fewer judgments than this and a check is UNVALIDATED, whatever its rate."""


@dataclass(frozen=True)
class Check:
    name: str
    status: CheckStatus
    successes: int
    trials: int
    interval: tuple[float, float]
    threshold: float
    detail: str = ''
    errors: int = 0

    @property
    def rate(self) -> float | None:
        return self.successes / self.trials if self.trials else None


@dataclass(frozen=True)
class Judgment:
    """One verdict, kept so a certificate can be audited case by case."""

    case: str
    role: str
    output: Any
    passed: bool | None
    reason: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class Certificate:
    verdict: Verdict
    checks: tuple[Check, ...]
    judgments: tuple[Judgment, ...] = field(repr=False)
    judge: str = ''

    @property
    def admissible(self) -> bool:
        return self.verdict == 'ADMISSIBLE'

    def failures(self) -> list[Check]:
        return [check for check in self.checks if check.status == 'FAIL']

    def table(self) -> str:
        """A plain-text summary, one row per check."""
        rows = [f'{self.judge}: {self.verdict}', '']
        rows.append(f'{"check":<22} {"status":<12} {"rate":>6}  {"95% interval":<15} {"needs":>6}  n')
        for check in self.checks:
            rate = '-' if check.rate is None else f'{check.rate:.2f}'
            low, high = check.interval
            bound = 'kappa>=' if check.name == 'human_agreement' else '>='
            rows.append(
                f'{check.name:<22} {check.status:<12} {rate:>6}  [{low:.2f}, {high:.2f}]    '
                f'{bound} {check.threshold:.2f}  {check.trials}'
                + (f'  ({check.errors} errors)' if check.errors else '')
                + (f'  {check.detail}' if check.detail else '')
            )
        return '\n'.join(rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            'judge': self.judge,
            'verdict': self.verdict,
            'checks': [
                {
                    'name': c.name,
                    'status': c.status,
                    'successes': c.successes,
                    'trials': c.trials,
                    'rate': c.rate,
                    'interval': list(c.interval),
                    'threshold': c.threshold,
                    'errors': c.errors,
                    'detail': c.detail,
                }
                for c in self.checks
            ],
        }


def assertion_of(output: Any, assertion: str | None = None) -> tuple[bool, str | None]:
    """The pass/fail verdict in an evaluator's output, and its reason when there is one.

    Evaluators return a bool, an `EvaluationReason`, or a mapping of named results. With a
    mapping, `assertion` names the entry to use; without it, the single boolean entry is used,
    and more than one is ambiguous.
    """
    if isinstance(output, EvaluationReason):
        if isinstance(output.value, bool):
            return output.value, output.reason
        raise TypeError(f'the judge returned a {type(output.value).__name__}, not a pass/fail assertion')
    if isinstance(output, bool):
        return output, None
    if isinstance(output, Mapping):
        booleans = {
            name: value
            for name, value in output.items()
            if isinstance(value, bool) or (isinstance(value, EvaluationReason) and isinstance(value.value, bool))
        }
        if assertion is not None:
            if assertion not in booleans:
                raise KeyError(f'no boolean result named {assertion!r}; the judge returned {sorted(output)}')
            return assertion_of(booleans[assertion])
        if len(booleans) == 1:
            return assertion_of(next(iter(booleans.values())))
        raise ValueError(
            f'the judge returned {len(booleans)} boolean results ({sorted(booleans)}); pass `assertion=` to pick one'
        )
    raise TypeError(f'the judge returned a {type(output).__name__}, not a pass/fail assertion')


def _context(case: JudgeCase, output: Any) -> EvaluatorContext[Any, Any, Any]:
    return EvaluatorContext(
        name=case.name,
        inputs=case.inputs,
        metadata=case.metadata,
        expected_output=case.expected_output,
        output=output,
        duration=0.0,
        _span_tree=SpanTreeRecordingError('certify_judge runs the judge outside a task span'),
        attributes={},
        metrics={},
    )


async def _judge(
    judge: Evaluator[Any, Any, Any],
    case: JudgeCase,
    output: Any,
    role: str,
    assertion: str | None,
    limit: asyncio.Semaphore,
) -> Judgment:
    async with limit:
        try:
            raw = await judge.evaluate_async(_context(case, output))
            passed, reason = assertion_of(raw, assertion)
        except Exception as error:  # a judge that errors has not given a verdict
            return Judgment(case.name, role, output, None, error=f'{type(error).__name__}: {error}'[:300])
    return Judgment(case.name, role, output, passed, reason)


def _check(
    name: str, successes: int, trials: int, threshold: float, minimum: int, *, detail: str = '', errors: int = 0
) -> Check:
    interval = wilson(successes, trials)
    low, high = interval
    if trials < minimum:
        status: CheckStatus = 'UNVALIDATED'
        detail = (detail + '; ' if detail else '') + f'fewer than {minimum} judgments'
    elif low >= threshold:
        status = 'PASS'
    elif high < threshold:
        # Evidence the judge is below the bar, not merely a lack of evidence that it is above it.
        status = 'FAIL'
    else:
        status = 'UNVALIDATED'
        detail = (detail + '; ' if detail else '') + 'interval straddles the threshold: needs more cases'
    return Check(name, status, successes, trials, interval, threshold, detail, errors)


async def certify_judge(
    judge: Evaluator[Any, Any, Any],
    cases: Sequence[JudgeCase],
    *,
    controls: Sequence[Control] = DEFAULT_CONTROLS,
    human_labels: Sequence[HumanLabel] = (),
    repeats: int = 3,
    thresholds: Thresholds = Thresholds(),
    assertion: str | None = None,
    max_concurrency: int = 8,
    seed: int = 0,
) -> Certificate:
    """Run `judge` over `cases`, controls built from them, and any human labels.

    Args:
        judge: Any evaluator with a pass/fail assertion, for example `LLMJudge(rubric=...)`.
        cases: Known-good answers. The judge should pass each one.
        controls: Outputs with a known correct verdict, built from `cases`.
        human_labels: People's verdicts on outputs, for an agreement check.
        repeats: How many times each known-good answer is judged, for the stability check.
        thresholds: What the judge must show.
        assertion: The result to read when the judge returns several boolean results.
        max_concurrency: Judgments in flight at once.
        seed: Seeds the choice of mismatched answers, so a certificate can be reproduced.
    """
    if repeats < 1:
        raise ValueError(f'repeats must be >= 1, got {repeats}')
    names = [case.name for case in cases]
    if len(set(names)) != len(names):
        raise ValueError('case names must be unique: human labels are matched to cases by name')
    rng = random.Random(seed)
    limit = asyncio.Semaphore(max_concurrency)
    by_name = {case.name: case for case in cases}

    planned: list[tuple[JudgeCase, Any, str]] = []
    for case in cases:
        planned.extend((case, case.output, f'reference#{i}') for i in range(repeats))
    for control in controls:
        for case in cases:
            made = control.make(case, cases, rng)
            if made is not None:
                planned.append((case, made, f'{control.kind}:{control.name}'))
    for label in human_labels:
        case = by_name.get(label.case)
        if case is None:
            raise KeyError(f'human label for unknown case {label.case!r}')
        planned.append((case, label.output, f'human:{int(label.passed)}'))

    judgments = await asyncio.gather(*(_judge(judge, c, out, role, assertion, limit) for c, out, role in planned))

    references: dict[str, list[Judgment]] = defaultdict(list)
    for judgment in judgments:
        if judgment.role.startswith('reference#'):
            references[judgment.case].append(judgment)

    def majority(case: str) -> bool | None:
        verdicts = [j.passed for j in references[case] if j.passed is not None]
        if not verdicts:
            return None
        return sum(verdicts) * 2 > len(verdicts)

    checks: list[Check] = []
    reference_judgments = [j for js in references.values() for j in js]
    checks.append(
        _check(
            'acceptance',
            sum(j.passed is True for j in reference_judgments),
            len(reference_judgments),
            thresholds.min_acceptance,
            thresholds.min_trials,
            errors=sum(j.error is not None for j in reference_judgments),
        )
    )

    must_fail = [j for j in judgments if j.role.startswith('must_fail:')]
    checks.append(
        _check(
            'rejection',
            sum(j.passed is False for j in must_fail),
            len(must_fail),
            thresholds.min_rejection,
            thresholds.min_trials,
            detail=_breakdown(must_fail, lambda j: j.passed is False),
            errors=sum(j.error is not None for j in must_fail),
        )
    )

    must_hold = [j for j in judgments if j.role.startswith('must_hold:')]
    held = [j for j in must_hold if majority(j.case) is not None]
    checks.append(
        _check(
            'invariance',
            sum(j.passed is not None and j.passed == majority(j.case) for j in held),
            len(held),
            thresholds.min_invariance,
            thresholds.min_trials,
            detail=_breakdown(held, lambda j: j.passed == majority(j.case)),
            errors=sum(j.error is not None for j in held),
        )
    )

    if repeats > 1:
        stable = [
            all(j.passed is not None for j in js) and len({j.passed for j in js}) == 1 for js in references.values()
        ]
        checks.append(
            _check(
                'stability',
                sum(stable),
                len(stable),
                thresholds.min_stability,
                thresholds.min_trials,
                detail=f'{repeats} judgments per case',
            )
        )

    labelled = [j for j in judgments if j.role.startswith('human:')]
    if labelled:
        pairs = [(j.role == 'human:1', j.passed) for j in labelled if j.passed is not None]
        kappa = cohen_kappa(pairs)  # type: ignore[arg-type]
        agree = sum(a == b for a, b in pairs)
        interval = wilson(agree, len(pairs))
        if kappa is None or len(pairs) < thresholds.min_trials:
            status: CheckStatus = 'UNVALIDATED'
            detail = 'kappa undefined: labels or verdicts are all one value' if kappa is None else ''
        else:
            status = 'PASS' if kappa >= thresholds.min_kappa else 'FAIL'
            detail = f'kappa={kappa:.2f}'
        checks.append(
            Check(
                'human_agreement',
                status,
                agree,
                len(pairs),
                interval,
                thresholds.min_kappa,
                detail,
                errors=len(labelled) - len(pairs),
            )
        )

    if any(check.status == 'FAIL' for check in checks):
        verdict: Verdict = 'INADMISSIBLE'
    elif any(check.status == 'UNVALIDATED' for check in checks):
        verdict = 'UNVALIDATED'
    else:
        verdict = 'ADMISSIBLE'
    return Certificate(verdict, tuple(checks), tuple(judgments), judge=_describe(judge))


def _breakdown(judgments: Sequence[Judgment], ok: Any) -> str:
    """Per-control success counts, so a failing aggregate says which control failed."""
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for judgment in judgments:
        name = judgment.role.split(':', 1)[1]
        counts[name][1] += 1
        counts[name][0] += bool(ok(judgment))
    return ', '.join(f'{name} {k}/{n}' for name, (k, n) in counts.items())


def _describe(judge: Evaluator[Any, Any, Any]) -> str:
    name = type(judge).__name__
    model = getattr(judge, 'model', None)
    model_name = getattr(model, 'model_name', model)
    return f'{name}({model_name})' if model_name else name
