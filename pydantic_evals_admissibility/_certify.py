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
    estimate: float | None = None
    """The statistic decided on when it is not `successes / trials` (Cohen's kappa)."""

    @property
    def rate(self) -> float | None:
        if self.estimate is not None:
            return self.estimate
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
    looks: int = 1
    """How many times the evidence was looked at with the chance to stop (sequential batches)."""

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
                    'estimate': c.estimate,
                }
                for c in self.checks
            ],
            'calls': self.calls,
            'planned': self.planned,
            'looks': self.looks,
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
                c.get('estimate'),
            )
            for c in data['checks']
        )
        judgments = tuple(
            Judgment(j['case'], j['role'], j['output'], j['passed'], j.get('reason'), j.get('error'))
            for j in data.get('judgments', ())
        )
        return cls(
            data['verdict'],
            checks,
            judgments,
            data.get('judge', ''),
            data.get('calls'),
            data.get('planned'),
            data.get('looks', 1),
        )


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
    z_fail: float = 1.96,
    max_missing: float = 0.1,
) -> Check:
    """PASS on the 95% lower bound; FAIL on an upper bound widened for every other chance to fail.

    A certificate needs every check to pass, so a PASS needs no correction for the number of
    checks. It is INADMISSIBLE if any check fails, so the FAILs share one error budget: `z_fail`
    is widened for the number of checks and looks. Errored judgments are left out of the rate, and
    more than `max_missing` of them leaves the check UNVALIDATED: a judge that errors on the hard
    cases must not pass on the easy ones.
    """
    interval = wilson(successes, trials)
    # The FAIL bound is exact (Clopper-Pearson): Wilson's undercoverage at small n leaked through
    # the split budget (3.4% false FAILs at the bar over four looks, against 2.5%).
    alpha_fail = 2 * (1 - NormalDist().cdf(z_fail))
    low, high = interval[0], clopper_pearson(successes, trials, alpha_fail)[1]
    if errors > max_missing * (trials + errors):
        status: CheckStatus = 'UNVALIDATED'
        detail = (detail + '; ' if detail else '') + f'{errors} of {trials + errors} judgments errored'
    elif trials < minimum:
        status = 'UNVALIDATED'
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
    if max_concurrency < 1 or (batch_size is not None and batch_size < 1):
        raise ValueError('max_concurrency and batch_size must be >= 1')
    if not cases:
        raise ValueError('no cases to certify the judge on')
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
    not stopped early: it runs to the end. A FAIL at any look, the last included, uses an interval
    widened for the number of looks (Bonferroni), so peeking does not raise the chance of failing a
    sound judge; a PASS, only possible at the end, uses the usual interval.
    (Stopping early for success too was tried and rejected: it certified sound judges less often
    for a small saving in calls.)
    """
    order = [case.name for case in cases]
    rng.shuffle(order)
    batches = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
    judgments: list[Judgment] = []
    verdict: Verdict = 'UNVALIDATED'
    checks: tuple[Check, ...] = ()
    for batch in batches:
        names = set(batch)
        todo = [(c, out, role) for c, out, role in planned if c.name in names]
        judgments += await asyncio.gather(*(_judge(judge, c, out, role, assertion, limit) for c, out, role in todo))
        # Every look, the last included, spends its share of the FAIL budget.
        verdict, checks = _assess(judgments, repeats, thresholds, looks=len(batches), slices=slices)
        if verdict == 'INADMISSIBLE':
            break
    return Certificate(
        verdict,
        checks,
        tuple(judgments),
        judge=_describe(judge),
        calls=len(judgments),
        planned=len(planned),
        looks=len(batches),
    )


def recertify(
    certificate: Certificate,
    *,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    slices: Mapping[str, str] | None = None,
) -> Certificate:
    """Decide a certificate again from its saved judgments, without calling the judge.

    For new thresholds or a new `slices` mapping (case name to kind of case) on a certificate
    already paid for, for example one rebuilt with `Certificate.from_dict`. A sequential
    certificate keeps its number of looks, so its error budget is not spent twice.
    """
    refs = [j.role for j in certificate.judgments if j.role.startswith('reference#')]
    repeats = max((int(role.split('#')[1]) + 1 for role in refs), default=1)
    verdict, checks = _assess(
        certificate.judgments, repeats, thresholds, looks=certificate.looks, slices=dict(slices) if slices else None
    )
    return Certificate(
        verdict,
        checks,
        certificate.judgments,
        certificate.judge,
        certificate.calls,
        certificate.planned,
        certificate.looks,
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
    *,
    looks: int = 1,
    slices: dict[str, str] | None = None,
) -> tuple[Verdict, tuple[Check, ...]]:
    """Every check, and the verdict, from the judgments made so far.

    The unit of evidence is the case. Repeats of one case share its difficulty, so they are not
    new evidence: acceptance and slices use each case's first judgment, and the repeats are used
    only for stability. Control families (`empty_output`, `mismatched_output`, ...) are decided
    separately, since a pooled rate can hide one that always fails.
    """
    references: dict[str, list[Judgment]] = defaultdict(list)
    for judgment in judgments:
        if judgment.role.startswith('reference#'):
            references[judgment.case].append(judgment)
    first = [j for js in references.values() for j in js if j.role == 'reference#0']

    def majority(case: str) -> bool | None:
        verdicts = [j.passed for j in references[case] if j.passed is not None]
        if not verdicts:
            return None
        return sum(verdicts) * 2 > len(verdicts)

    def family(prefix: str) -> dict[str, list[Judgment]]:
        out: dict[str, list[Judgment]] = defaultdict(list)
        for j in judgments:
            if j.role.startswith(prefix) and (prefix == 'must_fail:' or majority(j.case) is not None):
                out[j.role.split(':', 1)[1]].append(j)
        return out

    must_fail, must_hold = family('must_fail:'), family('must_hold:')
    labelled = [j for j in judgments if j.role.startswith('human:')]
    slice_names = sorted(set(slices[c] for c in references if c in slices)) if slices else []
    # Every test that can fail the certificate, at every look, shares one 2.5% upper tail.
    tests = 1 + len(must_fail) + len(must_hold) + len(slice_names) + (repeats > 1) + bool(labelled)
    alpha_fail = 0.05 / (tests * looks)
    z_fail = NormalDist().inv_cdf(1 - alpha_fail / 2)

    checks: list[Check] = [
        _check(
            'acceptance',
            sum(j.passed is True for j in first),
            sum(j.passed is not None for j in first),
            thresholds.min_acceptance,
            thresholds.min_trials,
            z_fail=z_fail,
            detail='first judgment of each case' if repeats > 1 else '',
            errors=sum(j.passed is None for j in first),
        ),
        _families('rejection', must_fail, lambda j: j.passed is False, thresholds.min_rejection, thresholds, z_fail),
        _families(
            'invariance',
            must_hold,
            lambda j: j.passed == majority(j.case),
            thresholds.min_invariance,
            thresholds,
            z_fail,
        ),
    ]
    if slices:
        checks.append(_slice_check(first, slices, slice_names, thresholds.min_acceptance, alpha_fail))

    if repeats > 1:
        complete = [js for js in references.values() if all(j.passed is not None for j in js)]
        checks.append(
            _check(
                'stability',
                sum(len({j.passed for j in js}) == 1 for js in complete),
                len(complete),
                thresholds.min_stability,
                thresholds.min_trials,
                z_fail=z_fail,
                detail=f'{repeats} judgments per case',
                errors=len(references) - len(complete),  # a case with an errored repeat is missing, not unstable
            )
        )

    if labelled:
        checks.append(_agreement(labelled, thresholds, alpha_fail))

    if any(check.status == 'FAIL' for check in checks):
        verdict: Verdict = 'INADMISSIBLE'
    elif any(check.status == 'UNVALIDATED' for check in checks):
        verdict = 'UNVALIDATED'
    else:
        verdict = 'ADMISSIBLE'
    return verdict, tuple(checks)


def _families(
    name: str,
    families: dict[str, list[Judgment]],
    ok: Callable[[Judgment], bool],
    threshold: float,
    thresholds: Thresholds,
    z_fail: float,
) -> Check:
    """One check per control family, reported as one row: it passes only if every family passes."""
    parts = {
        family: _check(
            family,
            sum(j.passed is not None and ok(j) for j in js),
            sum(j.passed is not None for j in js),
            threshold,
            thresholds.min_trials,
            z_fail=z_fail,
            errors=sum(j.passed is None for j in js),
        )
        for family, js in families.items()
    }
    successes = sum(c.successes for c in parts.values())
    trials = sum(c.trials for c in parts.values())
    statuses = {c.status for c in parts.values()}
    status: CheckStatus = 'FAIL' if 'FAIL' in statuses else 'PASS' if statuses == {'PASS'} else 'UNVALIDATED'
    detail = ', '.join(
        f'{family} {c.successes}/{c.trials}' + ('' if c.status == 'PASS' else f' {c.status}')
        for family, c in parts.items()
    )
    if not parts:
        detail = 'no controls'
    return Check(
        name, status, successes, trials, wilson(successes, trials), threshold, detail,
        sum(c.errors for c in parts.values()),
    )  # fmt: skip


def _agreement(labelled: Sequence[Judgment], thresholds: Thresholds, alpha_fail: float) -> Check:
    """Cohen's kappa against human labels, decided on an interval resampled by case.

    Labelled outputs of one case are not independent, so the bootstrap resamples cases.
    """
    done = [j for j in labelled if j.passed is not None]
    errors = len(labelled) - len(done)
    by_case: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for j in done:
        by_case[j.case].append((j.role == 'human:1', bool(j.passed)))
    pairs = [p for ps in by_case.values() for p in ps]
    kappa = cohen_kappa(pairs)
    agree = sum(a == b for a, b in pairs)
    if errors > 0.1 * len(labelled):
        status: CheckStatus = 'UNVALIDATED'
        detail, interval = f'{errors} of {len(labelled)} labelled judgments errored', (0.0, 1.0)
    elif kappa is None or len(pairs) < thresholds.min_trials:
        status, interval = 'UNVALIDATED', (0.0, 1.0)
        detail = 'kappa undefined: no variation in labels or verdicts' if kappa is None else 'too few labels'
    else:
        interval = _bootstrap_kappa(list(by_case.values()), 0.05)
        low, high = interval[0], _bootstrap_kappa(list(by_case.values()), alpha_fail)[1]
        status = 'PASS' if low >= thresholds.min_kappa else 'FAIL' if high < thresholds.min_kappa else 'UNVALIDATED'
        detail = f'kappa={kappa:.2f}, interval by case; agreement {agree}/{len(pairs)}'
    return Check(
        'human_agreement', status, agree, len(pairs), interval, thresholds.min_kappa, detail, errors, estimate=kappa
    )


def _bootstrap_kappa(
    clusters: Sequence[Sequence[tuple[bool, bool]]], alpha: float, resamples: int = 2000
) -> tuple[float, float]:
    """Percentile interval for kappa, resampling whole cases; undefined resamples are dropped."""
    rng = random.Random(0)
    values = []
    for _ in range(resamples):
        sample = [p for _ in clusters for p in clusters[rng.randrange(len(clusters))]]
        k = cohen_kappa(sample)
        if k is not None:
            values.append(k)
    if not values:
        return 0.0, 1.0
    values.sort()
    lo = values[max(0, int(alpha / 2 * len(values)))]
    hi = values[min(len(values) - 1, int((1 - alpha / 2) * len(values)))]
    return lo, hi


def _slice_check(
    first: Sequence[Judgment], slices: dict[str, str], names: Sequence[str], threshold: float, alpha: float
) -> Check:
    """Acceptance on each kind of case; fails when one kind is shown to be below the bar.

    One judgment per case, as for acceptance. It looks for a blind spot, it does not certify each
    slice. FAIL: a slice is shown to be below the bar. UNVALIDATED: a slice is at or below the
    bar but too small to show it, or has no completed judgment. PASS: no slice is shown or
    estimated below the bar. To certify one kind, certify its cases alone. Each slice's FAIL is
    decided with an exact interval at its share `alpha` of the error budget.
    """
    counts: dict[str, list[int]] = {name: [0, 0] for name in names}
    errors = 0
    for j in first:
        if j.case not in slices:
            continue
        if j.passed is None:
            errors += 1
            continue
        counts[slices[j.case]][0] += j.passed
        counts[slices[j.case]][1] += 1
    if not counts:
        return Check('slices', 'UNVALIDATED', 0, 0, (0.0, 1.0), threshold, 'no judgments yet')
    bounds = {name: clopper_pearson(k, n, alpha) for name, (k, n) in counts.items()}
    order = sorted(counts, key=lambda name: (bounds[name][1], name))
    worst = order[0]
    detail = 'worst first: ' + ', '.join(f'{name} {counts[name][0]}/{counts[name][1]}' for name in order)
    low = [name for name in order if counts[name][1] == 0 or counts[name][0] <= threshold * counts[name][1]]
    status: CheckStatus
    if counts[worst][1] and bounds[worst][1] < threshold:
        status = 'FAIL'
    elif low:
        status, worst = 'UNVALIDATED', low[0]
        detail += f'; {worst} is at or below the bar but too small to show it'
    else:
        status = 'PASS'
    k, n = counts[worst]
    return Check('slices', status, k, n, clopper_pearson(k, n), threshold, detail, errors)


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
