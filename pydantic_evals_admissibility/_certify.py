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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any, Literal

from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext
from pydantic_evals.otel._errors import SpanTreeRecordingError

from ._cases import HumanLabel, JudgeCase
from ._controls import DEFAULT_CONTROLS, Control
from ._stats import clopper_pearson, cohen_kappa, wilson

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


DEFAULT_THRESHOLDS = Thresholds()


class InadmissibleJudge(AssertionError):
    """A judge's certificate is not ADMISSIBLE; the message says why and what to try."""


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
    calls: int | None = None
    """Judge calls made; fewer than `planned` when a sequential certificate stopped early."""
    planned: int | None = None

    @property
    def admissible(self) -> bool:
        return self.verdict == 'ADMISSIBLE'

    def raise_unless_admissible(self, judge: Any = None) -> None:
        """Raise `InadmissibleJudge` with the table and the doctor's advice unless ADMISSIBLE.

        For a test or a CI step: the failure says what is wrong with the judge and what to try.
        """
        if self.admissible:
            return
        from ._diagnose import diagnose

        advice = diagnose(self, judge)
        lines = [self.table(), ''] + [f'- {a}' for a in advice]
        raise InadmissibleJudge('\n'.join(lines))

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

    def to_dict(self, *, judgments: bool = True) -> dict[str, Any]:
        out = self._summary()
        if judgments:
            # Every verdict, so a certificate can be audited case by case after the fact.
            out['judgments'] = [
                {'case': j.case, 'role': j.role, 'output': j.output if isinstance(j.output, str) else repr(j.output),
                 'passed': j.passed, 'reason': j.reason, 'error': j.error}
                for j in self.judgments
            ]  # fmt: skip
        return out

    def _summary(self) -> dict[str, Any]:
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
            'calls': self.calls,
            'planned': self.planned,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Certificate:
        """Rebuild a certificate saved with `to_dict`, to audit or `diagnose` it later without calling the judge.

        Outputs come back as saved: strings, or the `repr` of structured outputs.
        """
        checks = tuple(
            Check(
                c['name'],
                c['status'],
                c['successes'],
                c['trials'],
                (c['interval'][0], c['interval'][1]),
                c['threshold'],
                c.get('detail', ''),
                c.get('errors', 0),
            )  # fmt: skip
            for c in data['checks']
        )
        judgments = tuple(
            Judgment(j['case'], j['role'], j['output'], j['passed'], j.get('reason'), j.get('error'))
            for j in data.get('judgments', ())
        )
        return cls(data['verdict'], checks, judgments, data.get('judge', ''), data.get('calls'), data.get('planned'))


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
    name: str,
    successes: int,
    trials: int,
    threshold: float,
    minimum: int,
    *,
    detail: str = '',
    errors: int = 0,
    z: float = 1.96,
) -> Check:
    interval = wilson(successes, trials, z)
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
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    assertion: str | None = None,
    max_concurrency: int = 8,
    seed: int = 0,
    batch_size: int | None = None,
    slice_by: Callable[[JudgeCase], str] | None = None,
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
        batch_size: Judge this many cases at a time and stop early if the judge has clearly
            failed (intervals widened for the number of looks). A judge that is not failing runs
            to the end and is judged as if all at once. None judges everything at once.
        slice_by: Names the kind of each case (for example its category and expected answer). Adds a
            `slices` check that fails when the judge is shown to be below the acceptance bar on one
            kind of case, which an overall rate can hide. Each slice's interval is widened for the
            number of slices.
    """
    if repeats < 1:
        raise ValueError(f'repeats must be >= 1, got {repeats}')
    names = [case.name for case in cases]
    if len(set(names)) != len(names):
        raise ValueError('case names must be unique: human labels are matched to cases by name')
    rng = random.Random(seed)
    limit = asyncio.Semaphore(max_concurrency)
    planned = _plan(cases, controls, human_labels, repeats, rng)
    slices = {case.name: slice_by(case) for case in cases} if slice_by else None
    if batch_size is None:
        judgments = await asyncio.gather(*(_judge(judge, c, out, role, assertion, limit) for c, out, role in planned))
        verdict, checks = _assess(judgments, repeats, thresholds, slices=slices)
        return Certificate(
            verdict, checks, tuple(judgments), judge=_describe(judge), calls=len(planned), planned=len(planned)
        )
    return await _certify_sequentially(
        judge, cases, planned, repeats, thresholds, assertion, limit, rng, batch_size=batch_size, slices=slices
    )


async def _certify_sequentially(
    judge: Evaluator[Any, Any, Any],
    cases: Sequence[JudgeCase],
    planned: list[tuple[JudgeCase, Any, str]],
    repeats: int,
    thresholds: Thresholds,
    assertion: str | None,
    limit: asyncio.Semaphore,
    rng: random.Random,
    *,
    batch_size: int,
    slices: dict[str, str] | None = None,
) -> Certificate:
    """Judge `batch_size` cases at a time and stop early only when the judge has clearly failed.

    A judge that is broken shows it fast: after the first batch it is already failing controls
    beyond any doubt, and there is no reason to pay for the rest. A judge that is doing well is
    not stopped early: it runs to the end and is judged there with the usual interval, so stopping
    early costs a sound judge nothing. Early failures use a wider interval, the confidence level
    split across the looks (Bonferroni), so peeking does not fail a sound judge by chance.
    (Stopping early for success too was measured and rejected: it certified sound judges less
    often, 74/100 against 87/100, for an 18% saving.)
    """
    order = [case.name for case in cases]
    rng.shuffle(order)
    batches = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
    z_early = NormalDist().inv_cdf(1 - 0.05 / (2 * len(batches)))
    judgments: list[Judgment] = []
    verdict: Verdict = 'UNVALIDATED'
    checks: tuple[Check, ...] = ()
    for index, batch in enumerate(batches):
        names = set(batch)
        todo = [(c, out, role) for c, out, role in planned if c.name in names]
        judgments += await asyncio.gather(*(_judge(judge, c, out, role, assertion, limit) for c, out, role in todo))
        if index < len(batches) - 1:
            verdict, checks = _assess(judgments, repeats, thresholds, z_early, slices=slices)
            if verdict == 'INADMISSIBLE':
                break
        else:
            verdict, checks = _assess(judgments, repeats, thresholds, slices=slices)
    return Certificate(
        verdict, checks, tuple(judgments), judge=_describe(judge), calls=len(judgments), planned=len(planned)
    )


def recertify(
    certificate: Certificate,
    *,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    slices: Mapping[str, str] | None = None,
) -> Certificate:
    """Decide a certificate again from its saved judgments, without calling the judge.

    For new thresholds or a new `slices` mapping (case name to kind of case) on a certificate
    already paid for, for example one rebuilt with `Certificate.from_dict`.
    """
    refs = [j.role for j in certificate.judgments if j.role.startswith('reference#')]
    repeats = max((int(role.split('#')[1]) + 1 for role in refs), default=1)
    verdict, checks = _assess(certificate.judgments, repeats, thresholds, slices=dict(slices) if slices else None)
    return Certificate(
        verdict, checks, certificate.judgments, certificate.judge, certificate.calls, certificate.planned
    )


def _plan(
    cases: Sequence[JudgeCase],
    controls: Sequence[Control],
    human_labels: Sequence[HumanLabel],
    repeats: int,
    rng: random.Random,
) -> list[tuple[JudgeCase, Any, str]]:
    """Every judgment the certificate needs: known-good answers, controls, labelled outputs."""
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

    return planned


def _assess(
    judgments: Sequence[Judgment],
    repeats: int,
    thresholds: Thresholds,
    z: float = 1.96,
    *,
    slices: dict[str, str] | None = None,
) -> tuple[Verdict, tuple[Check, ...]]:
    """Every check, and the verdict, from the judgments made so far."""
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
            z=z,
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
            z=z,
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
            z=z,
            detail=_breakdown(held, lambda j: j.passed == majority(j.case)),
            errors=sum(j.error is not None for j in held),
        )
    )

    if slices:
        checks.append(_slice_check(reference_judgments, slices, thresholds.min_acceptance, z))

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
                z=z,
                detail=f'{repeats} judgments per case',
            )
        )

    labelled = [j for j in judgments if j.role.startswith('human:')]
    if labelled:
        pairs = [(j.role == 'human:1', j.passed) for j in labelled if j.passed is not None]
        kappa = cohen_kappa(pairs)  # type: ignore[arg-type]
        agree = sum(a == b for a, b in pairs)
        interval = wilson(agree, len(pairs), z)
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
    return verdict, tuple(checks)


def _slice_check(references: Sequence[Judgment], slices: dict[str, str], threshold: float, z: float) -> Check:
    """Acceptance on each kind of case; fails when one kind is shown to be below the bar.

    It looks for a blind spot, it does not certify each slice. FAIL: a slice is shown to be below
    the bar. UNVALIDATED: a slice is at or below the bar but has too few cases to show it. PASS:
    every slice is above the bar and none is shown below it. To certify a slice, certify its cases
    alone.
    """
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for j in references:
        if j.passed is not None:
            counts[slices[j.case]][0] += j.passed
            counts[slices[j.case]][1] += 1
    # One test per slice: split the level across them, as `GateRules.for_candidates` does. Exact
    # intervals, because Wilson's undercoverage on small slices failed a sound judge 6.3% of the time.
    alpha = 2 * (1 - NormalDist().cdf(z)) / max(len(counts), 1)
    bounds = {name: clopper_pearson(k, n, alpha) for name, (k, n) in counts.items()}
    order = sorted(counts, key=lambda name: (bounds[name][1], name))
    if not order:
        return Check('slices', 'UNVALIDATED', 0, 0, (0.0, 1.0), threshold, 'no judgments yet')
    worst = order[0]
    k, n = counts[worst]
    detail = 'worst first: ' + ', '.join(f'{name} {counts[name][0]}/{counts[name][1]}' for name in order)
    low = [name for name in order if counts[name][0] <= threshold * counts[name][1]]
    status: CheckStatus
    if bounds[worst][1] < threshold:
        status = 'FAIL'
    elif low:
        status, worst = 'UNVALIDATED', low[0]
        k, n = counts[worst]
        detail += f'; {worst} is at or below the bar but too small to show it'
    else:
        status = 'PASS'
    return Check('slices', status, k, n, bounds[worst], threshold, detail)


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
